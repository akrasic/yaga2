"""
Main ML inference pipeline for Smartbox anomaly detection.

This module provides the main orchestration layer that composes:
- DetectionRunner: Two-pass anomaly detection
- EnrichmentRunner: SLO evaluation and context enrichment
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from smartbox_anomaly.core import (
    AnomalySeverity,
    ModelLoadError,
    MetricsCollectionError,
    get_config,
)
from smartbox_anomaly.enrichment import (
    EnvoyEnrichmentService,
    ExceptionEnrichmentService,
    ServiceGraphEnrichmentService,
)
from smartbox_anomaly.slo import SLOEvaluator
from vmclient import InferenceMetrics, VictoriaMetricsClient

from .detection_engine import EnhancedAnomalyDetectionEngine
from .detection_runner import DetectionRunner
from .enrichment_runner import EnrichmentRunner
from .model_manager import EnhancedModelManager
from .models import AnomalyResult, ServiceInferenceResult
from .results_processor import EnhancedResultsProcessor
from .time_aware import EnhancedTimeAwareDetector

logger = logging.getLogger(__name__)


class SmartboxMLInferencePipeline:
    """Enhanced production-grade ML inference pipeline with explainability.

    This class orchestrates the full inference pipeline by composing:
    - DetectionRunner for two-pass anomaly detection
    - EnrichmentRunner for SLO evaluation and context enrichment
    """

    def __init__(
        self,
        vm_endpoint: str | None = None,
        models_directory: str | None = None,
        alerts_directory: str | None = None,
        max_workers: int | None = None,
        verbose: bool = False,
        check_drift: bool | None = None,
    ):
        # Load configuration
        config = get_config()

        # Use config values as defaults, allow overrides
        vm_endpoint = vm_endpoint or config.victoria_metrics.endpoint
        models_directory = models_directory or config.model.models_directory
        alerts_directory = alerts_directory or config.inference.alerts_directory
        max_workers = max_workers if max_workers is not None else config.inference.max_workers
        self.check_drift = check_drift if check_drift is not None else config.inference.check_drift

        # Initialize core components
        self.vm_client = VictoriaMetricsClient(vm_endpoint)
        self.model_manager = EnhancedModelManager(models_directory)
        self.detection_engine = EnhancedAnomalyDetectionEngine(self.model_manager)
        self.results_processor = EnhancedResultsProcessor(alerts_directory, verbose)

        # Get excluded periods from training config for holiday variant model selection
        excluded_periods = config.training.excluded_periods if config.training else []
        self.time_aware_detector = EnhancedTimeAwareDetector(
            models_directory,
            excluded_periods=excluded_periods,
        )
        self.max_workers = max_workers
        self.verbose = verbose

        # Load dependency graph from config
        self.dependency_graph = self._load_dependency_graph()

        # Initialize SLO evaluator if configured
        self.slo_evaluator: SLOEvaluator | None = None
        if config.slo.enabled:
            self.slo_evaluator = SLOEvaluator(config.slo)
            logger.info("SLO-aware severity evaluation enabled")

        # Initialize enrichment services
        enrichment_enabled = getattr(config.inference, 'exception_enrichment_enabled', True)
        self.exception_enrichment = ExceptionEnrichmentService(
            vm_client=self.vm_client,
            lookback_minutes=5,
            enabled=enrichment_enabled,
        )
        self.service_graph_enrichment = ServiceGraphEnrichmentService(
            vm_client=self.vm_client,
            lookback_minutes=5,
            enabled=enrichment_enabled,
        )

        # Initialize Envoy enrichment service (queries Mimir for edge metrics)
        envoy_config = getattr(config, 'envoy_enrichment', None)
        if envoy_config and getattr(envoy_config, 'enabled', False):
            self.envoy_enrichment = EnvoyEnrichmentService(
                mimir_endpoint=getattr(envoy_config, 'mimir_endpoint', 'https://mimir.sbxtest.net/prometheus'),
                lookback_minutes=getattr(envoy_config, 'lookback_minutes', 5),
                timeout_seconds=getattr(envoy_config, 'timeout_seconds', 10),
                cluster_mapping=getattr(envoy_config, 'cluster_mapping', None),
                enabled=True,
            )
            logger.info("Envoy enrichment enabled for edge/ingress context")
        else:
            self.envoy_enrichment = None

        if enrichment_enabled:
            logger.info("Exception enrichment enabled for error-related anomalies")
            logger.info("Service graph enrichment enabled for client latency anomalies")

        # Initialize runners
        self._detection_runner = DetectionRunner(
            vm_client=self.vm_client,
            model_manager=self.model_manager,
            time_aware_detector=self.time_aware_detector,
            dependency_graph=self.dependency_graph,
            check_drift=self.check_drift,
            verbose=verbose,
            max_workers=max_workers,
        )

        self._enrichment_runner = EnrichmentRunner(
            slo_evaluator=self.slo_evaluator,
            exception_enrichment=self.exception_enrichment,
            service_graph_enrichment=self.service_graph_enrichment,
            envoy_enrichment=self.envoy_enrichment,
            verbose=verbose,
        )

        logger.info("Enhanced Yaga2 ML Inference Pipeline initialized with explainability")
        if verbose:
            logger.info(f"  VM Endpoint: {vm_endpoint}")
            logger.info(f"  Models Directory: {models_directory}")
            logger.info(f"  Observability API: {config.observability.base_url}")
            if self.dependency_graph:
                logger.info(f"  Dependency graph loaded: {len(self.dependency_graph)} services")
            if self.slo_evaluator:
                logger.info(f"  SLO evaluation: enabled")
            if excluded_periods:
                logger.info(f"  Holiday variant selection: {len(excluded_periods)} excluded periods configured")

    def _load_dependency_graph(self) -> Dict[str, List[str]]:
        """Load dependency graph from config.json."""
        try:
            config_path = Path("config.json")
            if config_path.exists():
                with open(config_path) as f:
                    config_data = json.load(f)
                deps = config_data.get("dependencies", {})
                return deps.get("graph", {})
        except Exception as e:
            logger.warning(f"Failed to load dependency graph: {e}")
        return {}

    # Delegate validation to detection runner for backward compatibility
    def _validate_metrics(
        self,
        metrics_dict: Dict[str, float],
        service_name: str,
    ) -> tuple[Dict[str, float], List[str]]:
        """Validate and sanitize metrics before inference."""
        return self._detection_runner.validate_metrics(metrics_dict, service_name)

    # Delegate detection methods to detection runner for backward compatibility
    def _has_latency_anomaly(self, result: Dict) -> bool:
        """Check if a result contains latency-related anomalies."""
        return self._detection_runner.has_latency_anomaly(result)

    def _build_dependency_context(
        self,
        service_name: str,
        all_results: Dict[str, Dict],
    ):
        """Build dependency context from Pass 1 results."""
        return self._detection_runner.build_dependency_context(service_name, all_results)

    def run_enhanced_time_aware_inference(self) -> Dict[str, Dict]:
        """Run the full enhanced time-aware inference pipeline.

        This is a two-pass detection system:
        - Pass 1: Detect anomalies for all services without dependency context
        - Pass 2: Re-analyze services with latency anomalies using dependency context

        After detection, applies SLO evaluation and enrichment.

        Returns:
            Dictionary of service_name -> detection result.
        """
        # Get services to analyze
        config_services = self.model_manager.load_services_from_config()
        if config_services:
            service_names = config_services
            if self.verbose:
                logger.info(f"Using {len(service_names)} services from config.json")
        else:
            service_names = self.model_manager.get_base_services()
            if self.verbose:
                logger.info(f"Using {len(service_names)} base services from models directory")

        if not service_names:
            logger.warning("No services to analyze")
            return {}

        # Phase 1: Collect all metrics upfront
        if self.verbose:
            logger.info(f"Collecting metrics for {len(service_names)} services...")
        metrics_cache = self._detection_runner.collect_metrics_for_services(service_names)

        # Phase 2: Run Pass 1 detection
        if self.verbose:
            logger.info("Running Pass 1 detection...")
        pass1_results, validation_warnings = self._detection_runner.run_pass1_detection(
            service_names, metrics_cache
        )

        # Phase 3: Run Pass 2 detection with dependency context
        if self.verbose:
            logger.info("Running Pass 2 detection with dependency context...")
        results = self._detection_runner.run_pass2_detection(
            service_names, pass1_results, metrics_cache, validation_warnings
        )

        # Phase 4: Apply SLO evaluation
        if self.slo_evaluator:
            if self.verbose:
                logger.info("Applying SLO evaluation...")
            results = self._enrichment_runner.apply_slo_evaluation(results)

        # Phase 5: Apply exception enrichment
        if self.verbose:
            logger.info("Applying exception enrichment...")
        results = self._enrichment_runner.apply_exception_enrichment(results)

        # Phase 6: Apply service graph enrichment
        if self.verbose:
            logger.info("Applying service graph enrichment...")
        results = self._enrichment_runner.apply_service_graph_enrichment(results)

        # Phase 7: Apply Envoy edge metrics enrichment
        if self.envoy_enrichment:
            if self.verbose:
                logger.info("Applying Envoy edge metrics enrichment...")
            results = self._enrichment_runner.apply_envoy_enrichment(results)

        # Log summary
        self._enrichment_runner.process_and_log_results(results)

        return results

    def _resolve_service_names(self, services: Optional[List[str]] = None) -> List[str]:
        """Resolve service names to use for inference.

        Args:
            services: Optional list of specific service names.

        Returns:
            List of service names to process.
        """
        if services:
            return services

        # Try config first, then model directory
        config_services = self.model_manager.load_services_from_config()
        if config_services:
            return config_services

        return self.model_manager.get_base_services()

    def run_inference(self, services: Optional[List[str]] = None) -> Dict[str, ServiceInferenceResult]:
        """Run standard inference pipeline with optional service filtering.

        This is a simpler interface that returns structured ServiceInferenceResult objects.

        Args:
            services: Optional list of specific services to analyze.

        Returns:
            Dictionary mapping service_name -> ServiceInferenceResult.
        """
        service_names = self._resolve_service_names(services)

        if not service_names:
            logger.warning("No services to analyze")
            return {}

        results = {}

        for service_name in service_names:
            start_time = datetime.now()

            try:
                result = self._run_enhanced_service_inference(service_name, start_time)
                results[service_name] = result

            except ModelLoadError as e:
                logger.warning(f"MODEL_UNAVAILABLE - {service_name}: {e}")
                results[service_name] = ServiceInferenceResult(
                    service_name=service_name,
                    timestamp=start_time,
                    input_metrics=InferenceMetrics(service_name, datetime.now(), 0.0),
                    anomalies=[],
                    model_version="unknown",
                    inference_time_ms=0.0,
                    status='no_model',
                    error_message=str(e)
                )

            except MetricsCollectionError as e:
                logger.error(f"METRICS_FAILED - {service_name}: {e}")
                results[service_name] = ServiceInferenceResult(
                    service_name=service_name,
                    timestamp=start_time,
                    input_metrics=InferenceMetrics(service_name, datetime.now(), 0.0),
                    anomalies=[],
                    model_version="unknown",
                    inference_time_ms=0.0,
                    status='error',
                    error_message=str(e)
                )

            except Exception as e:
                logger.error(f"INFERENCE_ERROR - {service_name}: {e}", exc_info=True)
                results[service_name] = ServiceInferenceResult(
                    service_name=service_name,
                    timestamp=start_time,
                    input_metrics=InferenceMetrics(service_name, datetime.now(), 0.0),
                    anomalies=[],
                    model_version="unknown",
                    inference_time_ms=0.0,
                    status='error',
                    error_message=str(e)
                )

        return results

    def _run_enhanced_service_inference(
        self,
        service_name: str,
        start_time: datetime,
    ) -> ServiceInferenceResult:
        """Run enhanced inference for a single service.

        Args:
            service_name: Name of the service.
            start_time: Inference start timestamp.

        Returns:
            ServiceInferenceResult with detection results.
        """
        # Collect metrics
        metrics = self.vm_client.collect_service_metrics(service_name)
        metrics_dict = metrics.to_dict()

        # Validate metrics
        validated_metrics, validation_warnings = self._detection_runner.validate_metrics(
            metrics_dict, service_name
        )

        # Run time-aware detection
        enhanced_result = self.time_aware_detector.detect_with_explainability(
            service_name, validated_metrics, start_time, self.verbose,
            check_drift=self.check_drift
        )

        # Convert anomalies
        raw_anomalies = enhanced_result.get("anomalies", {})
        if isinstance(raw_anomalies, dict):
            anomalies_list = list(raw_anomalies.values())
        else:
            anomalies_list = raw_anomalies

        anomaly_results = self._convert_explainable_anomalies(anomalies_list)

        # Enrich with exception context if applicable
        exception_context = self._enrichment_runner.enrich_with_exceptions(
            service_name, anomaly_results, metrics, start_time
        )

        inference_time = (datetime.now() - start_time).total_seconds() * 1000

        return ServiceInferenceResult(
            service_name=service_name,
            timestamp=start_time,
            input_metrics=metrics,
            anomalies=anomaly_results,
            model_version=enhanced_result.get("model_type", "time_aware"),
            inference_time_ms=inference_time,
            status='success',
            historical_context=enhanced_result.get("historical_context"),
            metric_analysis=enhanced_result.get("metric_analysis"),
            explanation=enhanced_result.get("explanation"),
            recommended_actions=enhanced_result.get("recommended_actions"),
            exception_context=exception_context,
        )

    def _convert_explainable_anomalies(self, explainable_anomalies: List[Dict]) -> List[AnomalyResult]:
        """Convert explainable anomalies to AnomalyResult objects."""
        results = []

        for anomaly_data in explainable_anomalies:
            if isinstance(anomaly_data, dict):
                result = AnomalyResult(
                    anomaly_type=anomaly_data.get('type', 'unknown'),
                    severity=self.detection_engine._map_severity(anomaly_data.get('severity', 'medium')),
                    confidence_score=anomaly_data.get('score', 0.5),
                    description=anomaly_data.get('description', 'Anomaly detected'),
                    threshold_value=anomaly_data.get('threshold_value'),
                    actual_value=anomaly_data.get('actual_value'),
                    metadata=anomaly_data.get('metadata', {}),
                    comparison_data=anomaly_data.get('comparison_data'),
                    business_impact=anomaly_data.get('business_impact'),
                    percentile_position=anomaly_data.get('percentile_position')
                )
                results.append(result)

        return results

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status with explainability info."""
        available_services = self.model_manager.get_available_services()
        explainable_services = self._count_explainable_services(available_services)

        return {
            'timestamp': datetime.now().isoformat(),
            'status': 'ready' if available_services else 'no_models',
            'available_services': len(available_services),
            'explainable_services': explainable_services,
            'explainability_coverage': f"{explainable_services}/{len(available_services)}" if available_services else "0/0",
            'services': available_services,
            'vm_endpoint': self.vm_client.endpoint,
            'vm_status': self._check_vm_connectivity(),
            'models_directory': str(self.model_manager.models_directory),
            'circuit_breaker_status': 'open' if self.vm_client.is_circuit_open() else 'closed',
            'features': self._get_feature_status()
        }

    def _check_vm_connectivity(self) -> str:
        """Test VictoriaMetrics connectivity with proper error handling."""
        try:
            base_services = self.model_manager.get_base_services()
            test_service = base_services[0] if base_services else "test"
            self.vm_client.collect_service_metrics(test_service)
            return "connected"
        except (ConnectionError, TimeoutError, requests.RequestException) as e:
            logger.debug(f"VictoriaMetrics connectivity test failed: {e}")
            return "disconnected"
        except Exception as e:
            logger.warning(f"Unexpected error testing VictoriaMetrics connectivity: {e}")
            return "error"

    def _count_explainable_services(self, available_services: List[str]) -> int:
        """Count services with explainability support."""
        explainable_count = 0
        for service in available_services:
            try:
                model = self.model_manager.load_model(service)
                if hasattr(model, 'detect_anomalies_with_context') and model.training_statistics:
                    explainable_count += 1
            except (ModelLoadError, FileNotFoundError, json.JSONDecodeError) as e:
                logger.debug(f"Could not check explainability for {service}: {e}")
            except Exception as e:
                logger.debug(f"Unexpected error checking explainability for {service}: {e}")
        return explainable_count

    def _get_feature_status(self) -> Dict[str, bool]:
        """Get current feature availability status."""
        return {
            'explainable_anomaly_detection': True,
            'time_aware_detection': True,
            'historical_context': True,
            'business_impact_analysis': True,
            'actionable_recommendations': True
        }
