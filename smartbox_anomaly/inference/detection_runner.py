"""
Two-pass detection runner for anomaly detection pipeline.

This module handles the core detection logic including:
- Pass 1: Initial detection without dependency context
- Pass 2: Re-analysis with dependency context for latency anomalies
- Metrics collection and validation
- Fallback detection when enhanced detection fails
"""

import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from smartbox_anomaly.core import (
    DependencyContext,
    DependencyStatus,
)
from smartbox_anomaly.detection import TimeAwareAnomalyDetector
from vmclient import VictoriaMetricsClient

from .model_manager import EnhancedModelManager
from .time_aware import EnhancedTimeAwareDetector

logger = logging.getLogger(__name__)


class DetectionRunner:
    """Handles two-pass anomaly detection logic.

    This class encapsulates the detection pipeline:
    - Metrics collection from VictoriaMetrics
    - Pass 1 detection without dependency context
    - Pass 2 detection with dependency context for latency anomalies
    - Fallback detection when enhanced detection fails
    """

    def __init__(
        self,
        vm_client: VictoriaMetricsClient,
        model_manager: EnhancedModelManager,
        time_aware_detector: EnhancedTimeAwareDetector,
        dependency_graph: Dict[str, List[str]],
        check_drift: bool = False,
        verbose: bool = False,
        max_workers: int = 3,
    ):
        """Initialize the detection runner.

        Args:
            vm_client: VictoriaMetrics client for metrics collection.
            model_manager: Model manager for loading ML models.
            time_aware_detector: Time-aware detector for enhanced detection.
            dependency_graph: Service dependency graph for cascade analysis.
            check_drift: Whether to check for model drift.
            verbose: Enable verbose logging.
            max_workers: Maximum parallel workers for metrics collection.
        """
        self.vm_client = vm_client
        self.model_manager = model_manager
        self.time_aware_detector = time_aware_detector
        self.dependency_graph = dependency_graph
        self.check_drift = check_drift
        self.verbose = verbose
        self.max_workers = max_workers

    # Validation bounds (class-level constants)
    MAX_REQUEST_RATE = 1_000_000.0  # 1M req/s
    MAX_LATENCY_MS = 300_000.0  # 5 minutes
    MAX_ERROR_RATE = 1.0  # 100%

    def validate_metrics(
        self,
        metrics_dict: Dict[str, float],
        service_name: str,
    ) -> Tuple[Dict[str, float], List[str]]:
        """Validate and sanitize metrics before inference.

        Checks for:
        - NaN, inf, None values
        - Negative rates and latencies
        - Extreme outliers beyond reasonable bounds

        Args:
            metrics_dict: Raw metrics from VictoriaMetrics
            service_name: Name of the service (for logging)

        Returns:
            Tuple of (cleaned_metrics, validation_warnings)
        """
        warnings: List[str] = []
        cleaned: Dict[str, float] = {}

        for metric_name, value in metrics_dict.items():
            cleaned_value, warning = self._validate_single_metric(metric_name, value)
            if warning:
                warnings.append(warning)
            if cleaned_value is not None:
                cleaned[metric_name] = cleaned_value

        self._log_validation_warnings(service_name, warnings)
        return cleaned, warnings

    def _validate_single_metric(
        self, metric_name: str, value: Any
    ) -> Tuple[Optional[float], Optional[str]]:
        """Validate a single metric value.

        Args:
            metric_name: Name of the metric.
            value: Raw metric value.

        Returns:
            Tuple of (cleaned_value or None, warning message or None).
        """
        # Check for non-numeric values
        if not isinstance(value, (int, float)):
            return None, f"{metric_name}: non-numeric value {type(value).__name__}, skipping"

        # Check for invalid values (NaN, inf, None)
        if not self._is_valid_numeric(value):
            return 0.0, f"{metric_name}: invalid value {value}, using 0.0"

        # Route to type-specific validators
        metric_lower = metric_name.lower()

        if "error" in metric_lower and "rate" in metric_lower:
            return self._validate_error_rate(metric_name, value)

        if "request" in metric_lower and "rate" in metric_lower:
            return self._validate_request_rate(metric_name, value)

        if "rate" in metric_lower:
            return self._validate_generic_rate(metric_name, value)

        if "latency" in metric_lower:
            return self._validate_latency(metric_name, value)

        # No special validation needed
        return value, None

    def _is_valid_numeric(self, value: Any) -> bool:
        """Check if a value is a valid numeric (not None, NaN, or inf).

        Args:
            value: Value to check.

        Returns:
            True if valid, False otherwise.
        """
        if value is None:
            return False
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return False
        return True

    def _validate_error_rate(
        self, metric_name: str, value: float
    ) -> Tuple[float, Optional[str]]:
        """Validate error rate metric (should be between 0 and 1).

        Args:
            metric_name: Name of the metric.
            value: Error rate value.

        Returns:
            Tuple of (cleaned_value, warning or None).
        """
        if value < 0:
            return 0.0, f"{metric_name}: negative rate {value}, using 0.0"
        if value > self.MAX_ERROR_RATE:
            return self.MAX_ERROR_RATE, f"{metric_name}: error rate {value} > 1.0, capping at 1.0"
        return value, None

    def _validate_request_rate(
        self, metric_name: str, value: float
    ) -> Tuple[float, Optional[str]]:
        """Validate request rate metric.

        Args:
            metric_name: Name of the metric.
            value: Request rate value.

        Returns:
            Tuple of (cleaned_value, warning or None).
        """
        if value < 0:
            return 0.0, f"{metric_name}: negative rate {value}, using 0.0"
        if value > self.MAX_REQUEST_RATE:
            return self.MAX_REQUEST_RATE, f"{metric_name}: extreme rate {value}, capping at {self.MAX_REQUEST_RATE}"
        return value, None

    def _validate_generic_rate(
        self, metric_name: str, value: float
    ) -> Tuple[float, Optional[str]]:
        """Validate generic rate metric (should not be negative).

        Args:
            metric_name: Name of the metric.
            value: Rate value.

        Returns:
            Tuple of (cleaned_value, warning or None).
        """
        if value < 0:
            return 0.0, f"{metric_name}: negative rate {value}, using 0.0"
        return value, None

    def _validate_latency(
        self, metric_name: str, value: float
    ) -> Tuple[float, Optional[str]]:
        """Validate latency metric (should not be negative, cap extreme values).

        Args:
            metric_name: Name of the metric.
            value: Latency value in ms.

        Returns:
            Tuple of (cleaned_value, warning or None).
        """
        if value < 0:
            return 0.0, f"{metric_name}: negative latency {value}, using 0.0"
        if value > self.MAX_LATENCY_MS:
            return self.MAX_LATENCY_MS, f"{metric_name}: extreme latency {value}ms, capping at {self.MAX_LATENCY_MS}ms"
        return value, None

    def _log_validation_warnings(self, service_name: str, warnings: List[str]) -> None:
        """Log validation warnings if verbose mode is enabled.

        Args:
            service_name: Name of the service.
            warnings: List of warning messages.
        """
        if warnings and self.verbose:
            logger.warning(f"Metrics validation for {service_name}: {len(warnings)} issues")
            for warning in warnings[:5]:
                logger.warning(f"  {warning}")
            if len(warnings) > 5:
                logger.warning(f"  ... and {len(warnings) - 5} more")

    def has_latency_anomaly(self, result: Dict) -> bool:
        """Check if a result contains latency-related anomalies.

        Args:
            result: Detection result dictionary.

        Returns:
            True if latency anomaly detected, False otherwise.
        """
        if "error" in result:
            return False
        anomalies = result.get("anomalies", {})
        for anomaly_name, anomaly_data in anomalies.items():
            # Check if it's a latency-related anomaly
            root_metric = anomaly_data.get("root_metric", "")
            if "latency" in root_metric.lower() or "latency" in anomaly_name.lower():
                return True
            # Check contributing metrics
            for metric in anomaly_data.get("contributing_metrics", []):
                if "latency" in metric.lower():
                    return True
        return False

    def build_dependency_context(
        self,
        service_name: str,
        all_results: Dict[str, Dict],
    ) -> Optional[DependencyContext]:
        """Build dependency context from Pass 1 results.

        Args:
            service_name: The service to build context for
            all_results: Results from Pass 1 for all services

        Returns:
            DependencyContext if dependencies exist, None otherwise
        """
        if not self.dependency_graph:
            return None

        service_deps = self.dependency_graph.get(service_name, [])
        if not service_deps:
            return None

        dependencies = {}
        for dep_service in service_deps:
            if dep_service in all_results:
                result = all_results[dep_service]
                if "error" not in result:
                    anomalies = result.get("anomalies", {})
                    has_anomaly = len(anomalies) > 0

                    # Get primary anomaly info
                    primary_anomaly_type = None
                    severity = None
                    latency_percentile = None

                    if anomalies:
                        first_anomaly = list(anomalies.values())[0]
                        primary_anomaly_type = first_anomaly.get("pattern_name", list(anomalies.keys())[0])
                        severity = first_anomaly.get("severity")

                        # Try to extract latency percentile
                        for signal in first_anomaly.get("detection_signals", []):
                            if "latency" in signal.get("method", "").lower():
                                latency_percentile = signal.get("percentile")
                                break

                        # Fallback: check comparison_data
                        if latency_percentile is None:
                            comp_data = first_anomaly.get("comparison_data", {})
                            lat_data = comp_data.get("application_latency", {})
                            latency_percentile = lat_data.get("percentile_estimate")

                    dependencies[dep_service] = DependencyStatus(
                        service_name=dep_service,
                        has_anomaly=has_anomaly,
                        anomaly_type=primary_anomaly_type,
                        severity=severity,
                        latency_percentile=latency_percentile,
                        timestamp=result.get("timestamp"),
                    )

        if not dependencies:
            return None

        return DependencyContext(
            dependencies=dependencies,
            graph=self.dependency_graph,
            detection_timestamp=datetime.now().isoformat(),
        )

    def _collect_single_service_metrics(self, service_name: str) -> Tuple[str, Any]:
        """Collect metrics for a single service (thread-safe helper).

        Args:
            service_name: Name of the service to collect metrics for.

        Returns:
            Tuple of (service_name, metrics_or_None).
        """
        try:
            if self.verbose:
                logger.info(f"Collecting metrics from VictoriaMetrics for {service_name}")
            return (service_name, self.vm_client.collect_service_metrics(service_name))
        except (ConnectionError, TimeoutError, OSError) as e:
            logger.error(f"Network error collecting metrics for {service_name}: {e}")
            return (service_name, None)
        except ValueError as e:
            logger.error(f"Invalid metrics data for {service_name}: {e}")
            return (service_name, None)
        except Exception as e:
            logger.error(f"Failed to collect metrics for {service_name}: {e}", exc_info=True)
            return (service_name, None)

    def collect_metrics_for_services(
        self,
        service_names: List[str],
    ) -> Dict[str, Any]:
        """Collect metrics from VictoriaMetrics for all services in parallel.

        Args:
            service_names: List of service names to collect metrics for.

        Returns:
            Dictionary mapping service names to their collected metrics.
        """
        metrics_cache: Dict[str, Any] = {}

        # Use ThreadPoolExecutor for parallel I/O-bound metrics collection
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all collection tasks
            futures = {
                executor.submit(self._collect_single_service_metrics, name): name
                for name in service_names
            }

            # Collect results as they complete
            for future in as_completed(futures):
                service_name, metrics = future.result()
                if metrics is not None:
                    metrics_cache[service_name] = metrics

        return metrics_cache

    def run_pass1_detection(
        self,
        service_names: List[str],
        metrics_cache: Dict[str, Any],
    ) -> Tuple[Dict[str, Dict], Dict[str, List[str]]]:
        """Run Pass 1 detection without dependency context.

        Args:
            service_names: List of service names to process.
            metrics_cache: Pre-collected metrics for each service.

        Returns:
            Tuple of (pass1_results, validation_warnings_by_service).
        """
        pass1_results: Dict[str, Dict] = {}
        validation_warnings_by_service: Dict[str, List[str]] = {}

        for service_name in service_names:
            if service_name not in metrics_cache:
                pass1_results[service_name] = {'service': service_name, 'error': 'No metrics collected'}
                continue

            service_timestamp = datetime.now()

            if self.verbose:
                logger.info(f"Pass 1: Analyzing {service_name}")

            try:
                result = self._detect_service_anomalies(
                    service_name, metrics_cache[service_name], service_timestamp, validation_warnings_by_service
                )
                pass1_results[service_name] = result
            except Exception as model_error:
                result = self._fallback_detection(
                    service_name, metrics_cache, service_timestamp, validation_warnings_by_service, model_error
                )
                pass1_results[service_name] = result

        return pass1_results, validation_warnings_by_service

    def _detect_service_anomalies(
        self,
        service_name: str,
        metrics: Any,
        service_timestamp: datetime,
        validation_warnings_by_service: Dict[str, List[str]],
    ) -> Dict[str, Any]:
        """Detect anomalies for a single service using enhanced detection.

        Args:
            service_name: Name of the service.
            metrics: Collected metrics object.
            service_timestamp: Timestamp for detection.
            validation_warnings_by_service: Dict to store validation warnings.

        Returns:
            Detection result dictionary.
        """
        # Check if metrics are reliable enough for detection
        if not metrics.is_reliable_for_detection():
            failure_summary = metrics.get_failure_summary()
            logger.warning(f"Skipping detection for {service_name}: metrics unreliable - {failure_summary}")
            return {
                'service': service_name,
                'alert_type': 'metrics_unavailable',
                'error': f'Metrics collection failed: {failure_summary}',
                'failed_metrics': metrics.failed_metrics,
                'collection_errors': metrics.collection_errors,
                'timestamp': service_timestamp.isoformat(),
                'anomalies': {},
                'anomaly_count': 0,
                'overall_severity': 'none',
                'skipped_reason': 'critical_metrics_unavailable',
            }

        metrics_dict = metrics.to_dict()
        validated_metrics, validation_warnings = self.validate_metrics(metrics_dict, service_name)
        if validation_warnings:
            validation_warnings_by_service[service_name] = validation_warnings

        enhanced_result = self.time_aware_detector.detect_with_explainability(
            service_name, validated_metrics, service_timestamp, self.verbose,
            check_drift=self.check_drift
        )

        if validation_warnings:
            enhanced_result['validation_warnings'] = validation_warnings

        if metrics.has_any_failures():
            enhanced_result['partial_metrics_failure'] = {
                'failed_metrics': metrics.failed_metrics,
                'failure_summary': metrics.get_failure_summary(),
            }

        return enhanced_result

    def _fallback_detection(
        self,
        service_name: str,
        metrics_cache: Dict[str, Any],
        service_timestamp: datetime,
        validation_warnings_by_service: Dict[str, List[str]],
        original_error: Exception,
    ) -> Dict[str, Any]:
        """Fallback to standard detection when enhanced detection fails.

        Args:
            service_name: Name of the service.
            metrics_cache: Pre-collected metrics cache.
            service_timestamp: Timestamp for detection.
            validation_warnings_by_service: Dict to store validation warnings.
            original_error: The original exception that triggered fallback.

        Returns:
            Detection result dictionary.
        """
        if self.verbose:
            logger.warning(f"Enhanced detection failed for {service_name}: {str(original_error)[:50]}...")

        try:
            metrics = metrics_cache[service_name]
            metrics_dict = metrics.to_dict()
            validated_metrics, validation_warnings = self.validate_metrics(metrics_dict, service_name)

            detector = TimeAwareAnomalyDetector.load_models(
                str(self.model_manager.models_directory), service_name, False
            )
            anomalies = detector.detect_anomalies(validated_metrics, service_timestamp)
            period = detector.get_time_period(service_timestamp)

            if isinstance(anomalies, dict):
                anomaly_count = len(anomalies)
            else:
                anomaly_count = len(anomalies) if hasattr(anomalies, '__len__') else 0
                anomalies = {}

            fallback_result = {
                'service': service_name,
                'time_period': period,
                'model_type': 'time_aware_fallback',
                'anomaly_count': anomaly_count,
                'anomalies': anomalies,
                'metrics': validated_metrics,
                'timestamp': service_timestamp.isoformat(),
                'explainable': False
            }

            if validation_warnings:
                fallback_result['validation_warnings'] = validation_warnings

            return fallback_result

        except (FileNotFoundError, OSError) as e:
            logger.error(f"Model not found for {service_name} during fallback: {e}")
            return {'service': service_name, 'error': str(e)}
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Data error during fallback for {service_name}: {e}")
            return {'service': service_name, 'error': str(e)}
        except Exception as fallback_error:
            logger.error(f"Both enhanced and standard models failed for {service_name}: {fallback_error}", exc_info=True)
            return {'service': service_name, 'error': str(fallback_error)}

    def run_pass2_detection(
        self,
        service_names: List[str],
        pass1_results: Dict[str, Dict],
        metrics_cache: Dict[str, Any],
        validation_warnings_by_service: Dict[str, List[str]],
    ) -> Dict[str, Dict]:
        """Run Pass 2 detection with dependency context for latency anomalies.

        Args:
            service_names: List of service names.
            pass1_results: Results from Pass 1 detection.
            metrics_cache: Pre-collected metrics cache.
            validation_warnings_by_service: Validation warnings from Pass 1.

        Returns:
            Updated results dictionary with Pass 2 results merged in.
        """
        results = dict(pass1_results)

        if not self.dependency_graph:
            return results

        services_for_pass2 = [
            svc for svc in service_names
            if svc in pass1_results
            and 'error' not in pass1_results[svc]
            and self.has_latency_anomaly(pass1_results[svc])
            and svc in self.dependency_graph
        ]

        if not services_for_pass2:
            return results

        if self.verbose:
            logger.info(f"Pass 2: Re-analyzing {len(services_for_pass2)} services with dependency context")

        for service_name in services_for_pass2:
            pass2_timestamp = datetime.now()

            if self.verbose:
                logger.info(f"Pass 2: Re-analyzing {service_name} with dependency context")

            try:
                dep_context = self.build_dependency_context(service_name, pass1_results)

                if dep_context and any(status.has_anomaly for status in dep_context.dependencies.values()):
                    metrics = metrics_cache[service_name]
                    metrics_dict = metrics.to_dict()
                    validated_metrics, _ = self.validate_metrics(metrics_dict, service_name)

                    enhanced_result = self.time_aware_detector.detect_with_explainability(
                        service_name, validated_metrics, pass2_timestamp, self.verbose,
                        dependency_context=dep_context,
                        check_drift=self.check_drift
                    )

                    # Preserve validation warnings from Pass 1
                    if service_name in validation_warnings_by_service:
                        enhanced_result['validation_warnings'] = validation_warnings_by_service[service_name]

                    results[service_name] = enhanced_result

                    if self.verbose:
                        logger.info(f"Pass 2 complete for {service_name}")
                        if dep_context:
                            upstream_anomalies = [
                                dep for dep, status in dep_context.dependencies.items()
                                if status.has_anomaly
                            ]
                            if upstream_anomalies:
                                logger.info(f"  Upstream anomalies: {upstream_anomalies}")

            except Exception as e:
                logger.warning(f"Pass 2 failed for {service_name}, keeping Pass 1 result: {e}")

        return results
