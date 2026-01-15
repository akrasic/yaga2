"""
Enrichment runner for anomaly detection pipeline.

This module handles post-detection enrichment including:
- SLO-aware severity evaluation
- Exception context enrichment for error anomalies
- Service graph enrichment for latency anomalies
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from smartbox_anomaly.core import AnomalySeverity
from smartbox_anomaly.enrichment import ExceptionEnrichmentService, ServiceGraphEnrichmentService
from smartbox_anomaly.slo import SLOEvaluator
from vmclient import InferenceMetrics

from .models import AnomalyResult

logger = logging.getLogger(__name__)


class EnrichmentRunner:
    """Handles post-detection enrichment of anomaly results.

    This class encapsulates:
    - SLO-aware severity evaluation and adjustment
    - Exception context enrichment for error-related anomalies
    - Service graph enrichment for latency-related anomalies
    """

    def __init__(
        self,
        slo_evaluator: Optional[SLOEvaluator],
        exception_enrichment: ExceptionEnrichmentService,
        service_graph_enrichment: ServiceGraphEnrichmentService,
        verbose: bool = False,
    ):
        """Initialize the enrichment runner.

        Args:
            slo_evaluator: SLO evaluator for severity adjustment (may be None if disabled).
            exception_enrichment: Service for exception context enrichment.
            service_graph_enrichment: Service for service graph enrichment.
            verbose: Enable verbose logging.
        """
        self.slo_evaluator = slo_evaluator
        self.exception_enrichment = exception_enrichment
        self.service_graph_enrichment = service_graph_enrichment
        self.verbose = verbose

    def apply_slo_evaluation(
        self,
        results: Dict[str, Dict],
    ) -> Dict[str, Dict]:
        """Apply SLO-aware severity evaluation to results.

        Args:
            results: Detection results dictionary.

        Returns:
            Updated results with SLO evaluation applied.
        """
        if not self.slo_evaluator:
            return results

        for service_name, result in results.items():
            if "error" in result or result.get("alert_type") == "metrics_unavailable":
                continue

            anomalies = result.get("anomalies", {})
            if not anomalies:
                continue

            try:
                # Get current metrics for SLO evaluation
                current_metrics = result.get("current_metrics", result.get("metrics", {}))
                original_severity = result.get("overall_severity", "none")
                time_period = result.get("time_period", "business_hours")

                # Evaluate against SLOs
                slo_result = self.slo_evaluator.evaluate_metrics(
                    current_metrics,
                    service_name,
                    original_severity,
                    timestamp=datetime.fromisoformat(result["timestamp"]) if "timestamp" in result else None,
                    time_period=time_period,
                )

                # Apply SLO evaluation result
                result["slo_evaluation"] = slo_result.to_dict()

                # Update overall severity if changed
                if slo_result.severity_changed:
                    result["original_severity"] = original_severity
                    result["overall_severity"] = slo_result.adjusted_severity

                    if self.verbose:
                        logger.info(
                            f"SLO adjustment for {service_name}: "
                            f"{original_severity} -> {slo_result.adjusted_severity} "
                            f"(status: {slo_result.slo_status})"
                        )

            except Exception as e:
                logger.warning(f"SLO evaluation failed for {service_name}: {e}")

        return results

    def apply_exception_enrichment(
        self,
        results: Dict[str, Dict],
    ) -> Dict[str, Dict]:
        """Apply exception enrichment to error-related anomalies.

        Args:
            results: Detection results dictionary.

        Returns:
            Updated results with exception context added where applicable.
        """
        for service_name, result in results.items():
            if "error" in result or result.get("alert_type") == "metrics_unavailable":
                continue

            anomalies = result.get("anomalies", {})
            if not anomalies:
                continue

            # Check if we should enrich based on SLO evaluation
            slo_eval = result.get("slo_evaluation", {})
            error_eval = slo_eval.get("error_rate_evaluation", {})

            # Only enrich if error rate is elevated according to SLO
            if error_eval.get("status") not in ("warning", "critical"):
                continue

            try:
                # Get timestamp for aligned query
                timestamp_str = result.get("timestamp")
                if timestamp_str:
                    anomaly_timestamp = datetime.fromisoformat(timestamp_str)
                else:
                    anomaly_timestamp = datetime.now()

                # Query exception breakdown
                breakdown = self.exception_enrichment.get_exception_breakdown(
                    service_name=service_name,
                    anomaly_timestamp=anomaly_timestamp,
                )

                if breakdown.query_successful and breakdown.has_exceptions:
                    result["exception_context"] = breakdown.to_dict()

                    if self.verbose:
                        top_exc = breakdown.top_exception
                        logger.info(
                            f"Exception enrichment for {service_name}: "
                            f"{len(breakdown.exceptions)} types, "
                            f"top: {top_exc.short_name if top_exc else 'none'}"
                        )

            except Exception as e:
                logger.warning(f"Exception enrichment failed for {service_name}: {e}")

        return results

    def apply_service_graph_enrichment(
        self,
        results: Dict[str, Dict],
    ) -> Dict[str, Dict]:
        """Apply service graph enrichment to latency-related anomalies.

        Args:
            results: Detection results dictionary.

        Returns:
            Updated results with service graph context added where applicable.
        """
        for service_name, result in results.items():
            if "error" in result or result.get("alert_type") == "metrics_unavailable":
                continue

            # Check if we should enrich based on SLO evaluation
            slo_eval = result.get("slo_evaluation", {})
            latency_eval = slo_eval.get("latency_evaluation", {})

            # Only enrich if latency is elevated according to SLO
            if latency_eval.get("status") not in ("warning", "critical"):
                continue

            try:
                # Get timestamp for aligned query
                timestamp_str = result.get("timestamp")
                if timestamp_str:
                    anomaly_timestamp = datetime.fromisoformat(timestamp_str)
                else:
                    anomaly_timestamp = datetime.now()

                # Query service graph
                graph_result = self.service_graph_enrichment.get_service_graph(
                    service_name=service_name,
                    anomaly_timestamp=anomaly_timestamp,
                )

                if graph_result.query_successful and graph_result.routes:
                    result["service_graph_context"] = graph_result.to_dict()

                    if self.verbose:
                        logger.info(
                            f"Service graph enrichment for {service_name}: "
                            f"{len(graph_result.routes)} routes, "
                            f"top: {graph_result.top_route.server if graph_result.top_route else 'none'}"
                        )

            except Exception as e:
                logger.warning(f"Service graph enrichment failed for {service_name}: {e}")

        return results

    def enrich_with_exceptions(
        self,
        service_name: str,
        anomalies: List[AnomalyResult],
        metrics: InferenceMetrics,
        anomaly_timestamp: datetime,
    ) -> Optional[Dict[str, Any]]:
        """Enrich anomaly result with exception context if applicable.

        Queries OpenTelemetry exception metrics when error-related anomalies
        are detected. The query window is aligned with the anomaly timestamp.

        Args:
            service_name: Name of the service.
            anomalies: List of detected anomalies.
            metrics: Current metrics for the service.
            anomaly_timestamp: When the anomaly was detected.

        Returns:
            Exception context dictionary if enrichment was performed, None otherwise.
        """
        # Check if any anomaly is error-related and high/critical severity
        error_related = False
        high_severity = False

        for anomaly in anomalies:
            # Check severity
            if anomaly.severity in (AnomalySeverity.HIGH, AnomalySeverity.CRITICAL):
                high_severity = True

            # Check if error-related (by type or description)
            anomaly_type = anomaly.anomaly_type.lower()
            description = (anomaly.description or "").lower()

            if any(term in anomaly_type or term in description for term in
                   ["error", "failure", "outage", "fail"]):
                error_related = True

        # Also check error_rate from metrics
        error_rate = metrics.error_rate if metrics.error_rate else 0.0
        if error_rate > 0.01:  # More than 1% errors
            error_related = True

        # Only enrich if high/critical severity AND error-related
        if not (high_severity and error_related):
            return None

        try:
            breakdown = self.exception_enrichment.get_exception_breakdown(
                service_name=service_name,
                anomaly_timestamp=anomaly_timestamp,
            )

            if breakdown.query_successful and breakdown.has_exceptions:
                logger.info(
                    f"EXCEPTION_ENRICHMENT - {service_name}: "
                    f"{len(breakdown.exceptions)} exception types, "
                    f"top: {breakdown.top_exception.short_name if breakdown.top_exception else 'none'}"
                )
                return breakdown.to_dict()
            elif not breakdown.query_successful:
                logger.warning(f"Exception enrichment query failed for {service_name}: {breakdown.error_message}")
            else:
                logger.debug(f"No exceptions found for {service_name} in enrichment window")

            return None

        except Exception as e:
            logger.warning(f"Exception enrichment failed for {service_name}: {e}")
            return None

    def process_and_log_results(
        self,
        results: Dict[str, Dict],
    ) -> None:
        """Process results and log summary information.

        Args:
            results: Final detection results.
        """
        anomaly_count = 0
        services_with_anomalies = []

        for service_name, result in results.items():
            if "error" in result:
                continue
            anomalies = result.get("anomalies", {})
            if anomalies:
                anomaly_count += len(anomalies)
                services_with_anomalies.append(service_name)

        if self.verbose:
            logger.info(f"Detection complete: {anomaly_count} anomalies across {len(services_with_anomalies)} services")
