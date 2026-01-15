"""
Alert formatter for converting detection results to API payloads.

This module handles formatting anomaly detection results into the structured
JSON format documented in docs/INFERENCE_API_PAYLOAD.md.
"""

from datetime import datetime
from typing import Any, Dict, List

from smartbox_anomaly.api import (
    AlertType,
    Anomaly,
    AnomalyDetectedPayload,
    CurrentMetrics,
    FingerprintingMetadata,
    PayloadMetadata,
    Severity,
)

from .anomaly_builder import AnomalyBuilder
from .models import ServiceInferenceResult


class AlertFormatter:
    """Formats detection results into API payloads.

    This class handles the conversion of raw detection results into the
    structured JSON format expected by the API, using Pydantic models
    for validation and consistency.
    """

    def __init__(self):
        """Initialize the alert formatter."""
        self._anomaly_builder = AnomalyBuilder()

    def format_time_aware_alert(self, result: dict) -> dict:
        """Format time-aware anomaly as structured JSON matching API specification.

        Produces payload format documented in docs/INFERENCE_API_PAYLOAD.md.
        Uses Pydantic models for validation and consistency.

        Args:
            result: Raw detection result dictionary.

        Returns:
            Formatted alert payload as dictionary.
        """
        # Normalize anomalies to dict format
        anomalies_raw = self._normalize_anomalies_to_dict(result.get("anomalies", {}))

        # Format anomalies using Pydantic models
        formatted_anomalies: Dict[str, Anomaly] = {}
        severities: List[Severity] = []

        for anomaly_name, anomaly_data in anomalies_raw.items():
            if isinstance(anomaly_data, dict):
                anomaly_model = self._anomaly_builder.build_anomaly_model(
                    anomaly_name, anomaly_data
                )
                formatted_anomalies[anomaly_name] = anomaly_model
                severities.append(anomaly_model.severity)

        # Extract time period and model info
        time_period = result.get("time_period", "unknown")
        model_name = result.get("model_name", time_period)
        service_name = result.get("service", result.get("service_name", "unknown"))
        timestamp = result.get("timestamp", datetime.now().isoformat())

        # Determine alert type and overall severity
        alert_type = (
            AlertType.ANOMALY_DETECTED if formatted_anomalies else AlertType.NO_ANOMALY
        )
        overall_severity = self._determine_overall_severity(result, severities)

        # Build current metrics model
        metrics_data = result.get("metrics", result.get("current_metrics", {}))
        current_metrics = (
            CurrentMetrics(**metrics_data) if metrics_data else CurrentMetrics()
        )

        # Build fingerprinting metadata if present
        fingerprinting = self._build_fingerprinting_metadata(result)

        # Build payload metadata
        models_used = self._anomaly_builder.extract_models_used(
            {k: v.model_dump() for k, v in formatted_anomalies.items()}
        )
        metadata = PayloadMetadata(
            service_name=service_name,
            detection_timestamp=timestamp,
            models_used=models_used,
            enhanced_messaging=True,
            features={
                "contextual_severity": True,
                "named_patterns": True,
                "recommendations": True,
                "interpretations": True,
                "anomaly_correlation": True,
            },
        )

        # Build the AnomalyDetectedPayload using Pydantic model
        payload = AnomalyDetectedPayload(
            alert_type=alert_type,
            service_name=service_name,
            timestamp=timestamp,
            time_period=time_period,
            model_name=model_name,
            model_type=result.get("model_type", "time_aware_5period"),
            anomalies=formatted_anomalies,
            anomaly_count=len(formatted_anomalies),
            overall_severity=overall_severity,
            current_metrics=current_metrics,
            fingerprinting=(
                fingerprinting
                if isinstance(fingerprinting, FingerprintingMetadata)
                else None
            ),
            performance_info=result.get("performance_info"),
            exception_context=result.get("exception_context"),
            service_graph_context=result.get("service_graph_context"),
            metadata=metadata,
        )

        # Convert to dict and add additional fields
        response = payload.model_dump(mode="json", exclude_none=True)

        # Add fingerprinting as dict if it wasn't a valid model
        if fingerprinting and not isinstance(fingerprinting, FingerprintingMetadata):
            response["fingerprinting"] = fingerprinting

        # Add optional fields not in Pydantic model
        self._add_optional_fields(response, result)

        return response

    def format_explainable_alert(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Format explainable anomaly result as enhanced JSON.

        Args:
            result: Raw explainable detection result.

        Returns:
            Formatted alert payload.
        """
        enhanced_anomalies = []

        # Process anomalies with explainability data
        for anomaly_data in result.get("anomalies", []):
            if isinstance(anomaly_data, dict):
                enhanced_anomaly = {
                    "type": anomaly_data.get("type", "unknown"),
                    "severity": anomaly_data.get("severity", "medium"),
                    "confidence_score": anomaly_data.get("score", 0.0),
                    "description": anomaly_data.get("description", "Anomaly detected"),
                    "detection_method": anomaly_data.get(
                        "detection_method", anomaly_data.get("type", "unknown")
                    ),
                    "threshold_value": anomaly_data.get("threshold_value"),
                    "actual_value": anomaly_data.get("actual_value"),
                    "comparison_data": anomaly_data.get("comparison_data"),
                    "feature_contributions": anomaly_data.get("feature_contributions"),
                    "business_impact": anomaly_data.get("business_impact"),
                    "metadata": anomaly_data.get("metadata", {}),
                }
                enhanced_anomalies.append(enhanced_anomaly)

        base_alert = {
            "alert_type": "anomaly_detected",
            "service": result["service"],
            "timestamp": result["timestamp"],
            "overall_severity": result["overall_severity"],
            "anomaly_count": result["anomaly_count"],
            "current_metrics": result["current_metrics"],
            "anomalies": enhanced_anomalies,
            "model_type": "explainable_ml",
        }

        # Add explainability context if available
        optional_fields = [
            "historical_context",
            "metric_analysis",
            "explanation",
            "recommended_actions",
            "model_metadata",
        ]
        for field in optional_fields:
            if field in result:
                base_alert[field] = result[field]

        return base_alert

    def format_standard_alert(self, result: ServiceInferenceResult) -> dict:
        """Format regular ServiceInferenceResult as structured JSON.

        Args:
            result: ServiceInferenceResult object.

        Returns:
            Formatted alert payload.
        """
        formatted_anomalies = []
        for anomaly in result.anomalies:
            anomaly_dict = {
                "type": anomaly.anomaly_type,
                "severity": anomaly.severity.value,
                "confidence_score": anomaly.confidence_score,
                "description": anomaly.description,
                "threshold_value": anomaly.threshold_value,
                "actual_value": anomaly.actual_value,
                "metadata": anomaly.metadata or {},
            }

            # Add explainability fields if available
            if anomaly.comparison_data:
                anomaly_dict["comparison_data"] = anomaly.comparison_data
            if anomaly.business_impact:
                anomaly_dict["business_impact"] = anomaly.business_impact
            if anomaly.percentile_position:
                anomaly_dict["percentile_position"] = anomaly.percentile_position

            formatted_anomalies.append(anomaly_dict)

        alert_data = {
            "alert_type": "anomaly_detected",
            "service": result.service_name,
            "timestamp": result.timestamp.isoformat(),
            "overall_severity": result.max_severity.value,
            "anomaly_count": len(result.anomalies),
            "model_version": result.model_version,
            "inference_time_ms": result.inference_time_ms,
            "current_metrics": result.input_metrics.to_dict(),
            "anomalies": formatted_anomalies,
            "model_type": "standard_ml",
        }

        # Add explainability context if available
        optional_fields = [
            ("historical_context", result.historical_context),
            ("metric_analysis", result.metric_analysis),
            ("explanation", result.explanation),
            ("recommended_actions", result.recommended_actions),
        ]
        for field_name, field_value in optional_fields:
            if field_value:
                alert_data[field_name] = field_value

        return alert_data

    def _normalize_anomalies_to_dict(self, anomalies_raw: Any) -> Dict[str, Any]:
        """Normalize anomalies to dict format.

        Args:
            anomalies_raw: Anomalies as dict or list.

        Returns:
            Anomalies as dictionary keyed by anomaly name.
        """
        if isinstance(anomalies_raw, list):
            anomalies_dict = {}
            for i, anomaly in enumerate(anomalies_raw):
                if isinstance(anomaly, dict):
                    anomaly_name = (
                        anomaly.get("anomaly_name")
                        or anomaly.get("pattern_name")
                        or anomaly.get("type")
                        or f"anomaly_{i}"
                    )
                    anomalies_dict[anomaly_name] = anomaly
            return anomalies_dict
        return anomalies_raw

    def _determine_overall_severity(
        self, result: dict, severities: List[Severity]
    ) -> Severity:
        """Determine overall severity from result and individual severities.

        Args:
            result: Raw detection result.
            severities: List of individual anomaly severities.

        Returns:
            Overall severity level.
        """
        # Use SLO-adjusted severity if available
        slo_eval = result.get("slo_evaluation", {})
        if slo_eval.get("severity_changed") and result.get("overall_severity"):
            severity_str = result["overall_severity"]
            if severity_str in [s.value for s in Severity]:
                return Severity(severity_str)

        # No SLO adjustment - calculate from individual anomaly severities
        return Severity.max_severity(severities) if severities else Severity.NONE

    def _build_fingerprinting_metadata(
        self, result: dict
    ) -> FingerprintingMetadata | dict | None:
        """Build fingerprinting metadata from result.

        Args:
            result: Raw detection result.

        Returns:
            FingerprintingMetadata, dict, or None.
        """
        if "fingerprinting" not in result or not result["fingerprinting"]:
            return None

        try:
            return FingerprintingMetadata(**result["fingerprinting"])
        except Exception:
            # Fall back to dict if validation fails
            return result["fingerprinting"]

    def _add_optional_fields(self, response: dict, result: dict) -> None:
        """Add optional fields to the response that aren't in Pydantic model.

        Args:
            response: Response dictionary to modify.
            result: Original result dictionary.
        """
        # SLO evaluation
        if result.get("slo_evaluation"):
            response["slo_evaluation"] = result["slo_evaluation"]

        # Drift analysis
        if result.get("drift_analysis"):
            response["drift_analysis"] = result["drift_analysis"]
        if result.get("drift_warning"):
            response["drift_warning"] = result["drift_warning"]

        # Validation warnings
        if result.get("validation_warnings"):
            response["validation_warnings"] = result["validation_warnings"]
