"""
Anomaly model builder for constructing Pydantic models from raw anomaly data.

This module handles the conversion of raw anomaly dictionaries into validated
Pydantic models for the API payload.
"""

from typing import Any, Dict, List, Optional

from smartbox_anomaly.api import (
    Anomaly,
    CascadeInfo,
    DetectionSignal,
    Severity,
)


class AnomalyBuilder:
    """Builds Pydantic Anomaly models from raw anomaly data.

    This class encapsulates the logic for:
    - Converting detection signals to Pydantic models
    - Determining root metrics from various sources
    - Building cascade analysis info
    - Extracting models used from anomalies
    """

    def build_anomaly_model(self, anomaly_name: str, anomaly_data: dict) -> Anomaly:
        """Build an Anomaly Pydantic model from raw anomaly data.

        Handles detection_signals array format from the sequential IF → Pattern pipeline.

        Args:
            anomaly_name: Name/key of the anomaly.
            anomaly_data: Raw anomaly dictionary.

        Returns:
            Validated Anomaly Pydantic model.
        """
        # Build detection signals
        detection_signals = self._build_detection_signals(anomaly_data)

        # Determine root metric
        root_metric = self._determine_root_metric(anomaly_name, anomaly_data)

        # Build cascade info if present
        cascade_info = self._build_cascade_info(anomaly_data)

        # Use type directly (already API-compatible from detector)
        output_type = anomaly_data.get("type", "ml_isolation")

        # Build the Anomaly model
        return Anomaly(
            type=output_type,
            severity=Severity(anomaly_data.get("severity", "medium")),
            confidence=float(
                anomaly_data.get("confidence", anomaly_data.get("score", 0.5))
            ),
            score=float(anomaly_data.get("score", anomaly_data.get("confidence", 0.0))),
            description=anomaly_data.get(
                "description", anomaly_name.replace("_", " ").title()
            ),
            root_metric=root_metric,
            signal_count=anomaly_data.get("signal_count"),
            pattern_name=anomaly_data.get("pattern_name"),
            interpretation=anomaly_data.get("interpretation"),
            value=anomaly_data.get("value") or anomaly_data.get("actual_value"),
            detection_signals=detection_signals,
            possible_causes=anomaly_data.get("possible_causes"),
            recommended_actions=anomaly_data.get("recommended_actions"),
            checks=anomaly_data.get("checks"),
            comparison_data=anomaly_data.get("comparison_data"),
            business_impact=anomaly_data.get("business_impact"),
            cascade_analysis=cascade_info,
            fingerprint_id=anomaly_data.get("fingerprint_id"),
            fingerprint_action=anomaly_data.get("fingerprint_action"),
            incident_id=anomaly_data.get("incident_id"),
            incident_action=anomaly_data.get("incident_action"),
            incident_duration_minutes=anomaly_data.get("incident_duration_minutes"),
            first_seen=anomaly_data.get("first_seen"),
            last_updated=anomaly_data.get("last_updated"),
            occurrence_count=anomaly_data.get("occurrence_count"),
            time_confidence=anomaly_data.get("time_confidence"),
            detected_by_model=anomaly_data.get("detected_by_model"),
        )

    def _build_detection_signals(self, anomaly_data: dict) -> List[DetectionSignal]:
        """Build detection signals from raw data.

        Args:
            anomaly_data: Raw anomaly dictionary containing detection_signals.

        Returns:
            List of DetectionSignal Pydantic models.
        """
        detection_signals_raw = anomaly_data.get("detection_signals", [])
        detection_signals = []

        if detection_signals_raw:
            for signal in detection_signals_raw:
                if isinstance(signal, dict):
                    try:
                        detection_signals.append(
                            DetectionSignal(
                                method=signal.get("method", "unknown"),
                                type=signal.get("type", "ml_isolation"),
                                severity=Severity(signal.get("severity", "medium")),
                                score=float(signal.get("score", 0.0)),
                                direction=signal.get("direction"),
                                percentile=signal.get("percentile"),
                                pattern=signal.get("pattern"),
                            )
                        )
                    except Exception:
                        pass

        # Fallback: create a single detection signal from the anomaly data
        if not detection_signals:
            detection_signals.append(
                DetectionSignal(
                    method=anomaly_data.get("detection_method", "isolation_forest"),
                    type=anomaly_data.get("type", "ml_isolation"),
                    severity=Severity(anomaly_data.get("severity", "medium")),
                    score=float(anomaly_data.get("score", 0.0)),
                    direction=anomaly_data.get("direction"),
                    percentile=anomaly_data.get("percentile_position")
                    or anomaly_data.get("percentile"),
                    pattern=anomaly_data.get("pattern_name"),
                )
            )

        return detection_signals

    def _determine_root_metric(
        self, anomaly_name: str, anomaly_data: dict
    ) -> Optional[str]:
        """Determine the root metric for an anomaly.

        Args:
            anomaly_name: Name of the anomaly.
            anomaly_data: Raw anomaly dictionary.

        Returns:
            Root metric name or None.
        """
        root_metric = anomaly_data.get("root_metric")
        if root_metric:
            return root_metric

        # Try to get from contributing_metrics
        contributing = anomaly_data.get("contributing_metrics", [])
        if contributing:
            # Prioritize latency > error_rate > request_rate
            priority = [
                "application_latency",
                "error_rate",
                "request_rate",
                "database_latency",
                "dependency_latency",
            ]
            for metric in priority:
                if metric in contributing:
                    return metric
            return contributing[0]

        # Fallback: infer from anomaly name
        name_lower = anomaly_name.lower()
        if "latency" in name_lower:
            return "application_latency"
        elif "error" in name_lower:
            return "error_rate"
        elif "traffic" in name_lower or "request" in name_lower:
            return "request_rate"

        return None

    def _build_cascade_info(self, anomaly_data: dict) -> Optional[CascadeInfo]:
        """Build cascade analysis info if present.

        Args:
            anomaly_data: Raw anomaly dictionary.

        Returns:
            CascadeInfo model or None.
        """
        cascade_analysis = anomaly_data.get("cascade_analysis")
        if not cascade_analysis:
            return None

        return CascadeInfo(
            is_cascade=cascade_analysis.get("is_cascade", False),
            root_cause_service=cascade_analysis.get("root_cause_service"),
            affected_chain=cascade_analysis.get("affected_chain", []),
            cascade_type=cascade_analysis.get("cascade_type", "none"),
            confidence=cascade_analysis.get("confidence", 0.0),
            propagation_path=cascade_analysis.get("propagation_path"),
        )

    def extract_models_used(self, anomalies: Dict[str, Any]) -> List[str]:
        """Extract list of detection models used from anomalies.

        Args:
            anomalies: Dictionary of anomaly name -> anomaly data.

        Returns:
            List of unique model/method names used.
        """
        models = set()
        for anomaly in anomalies.values():
            if isinstance(anomaly, dict):
                signals = anomaly.get("detection_signals", [])
                for signal in signals:
                    if isinstance(signal, dict) and signal.get("method"):
                        models.add(signal["method"])
        return list(models) if models else []
