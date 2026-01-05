"""
SLO-aware severity evaluation.

This module provides a post-processing layer that evaluates anomaly detection
results against operational SLO thresholds. It adjusts severity based on
actual operational impact, not just statistical deviation.

The key insight: ML detects "is this unusual?" while SLOs determine "does it matter?"

Severity Matrix:
                    | Within Acceptable | Approaching SLO | Breaching SLO |
    Anomaly Detected|   informational   |     warning     |    critical   |
    No Anomaly      |      normal       |  warning (SLO)  | critical (SLO)|
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from smartbox_anomaly.core.config import SLOConfig, ServiceSLOConfig

logger = logging.getLogger(__name__)


@dataclass
class SLOEvaluationResult:
    """Result of SLO evaluation for a single anomaly or service."""

    # Original ML-based severity
    original_severity: str

    # Adjusted severity after SLO evaluation
    adjusted_severity: str

    # SLO status: 'ok', 'warning', 'critical', 'breached'
    slo_status: str

    # How close to SLO breach (0.0 = far from breach, 1.0 = at threshold, >1.0 = breached)
    slo_proximity: float

    # Operational impact assessment
    operational_impact: str  # 'none', 'informational', 'actionable', 'critical'

    # Whether this is during a busy period
    is_busy_period: bool = False

    # Detailed breakdown
    latency_evaluation: dict[str, Any] = field(default_factory=dict)
    error_rate_evaluation: dict[str, Any] = field(default_factory=dict)

    # Human-readable explanation
    explanation: str = ""

    # Whether severity was changed
    severity_changed: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "original_severity": self.original_severity,
            "adjusted_severity": self.adjusted_severity,
            "severity_changed": self.severity_changed,
            "slo_status": self.slo_status,
            "slo_proximity": round(self.slo_proximity, 3),
            "operational_impact": self.operational_impact,
            "is_busy_period": self.is_busy_period,
            "latency_evaluation": self.latency_evaluation,
            "error_rate_evaluation": self.error_rate_evaluation,
            "explanation": self.explanation,
        }


class SLOEvaluator:
    """
    Evaluates anomaly detection results against SLO thresholds.

    This is a post-processing layer that takes ML detection results and
    adjusts severity based on operational significance.

    Usage:
        evaluator = SLOEvaluator(slo_config)
        adjusted_result = evaluator.evaluate_result(detection_result)
    """

    def __init__(self, config: SLOConfig):
        """
        Initialize the SLO evaluator.

        Args:
            config: SLO configuration containing thresholds and settings.
        """
        self.config = config
        self._evaluation_count = 0
        self._adjustments_made = 0

    def evaluate_result(
        self,
        result: dict[str, Any],
        timestamp: datetime | None = None,
    ) -> dict[str, Any]:
        """
        Evaluate a detection result and adjust severity based on SLOs.

        Args:
            result: Detection result dict from inference pipeline.
            timestamp: Optional timestamp for busy period check.

        Returns:
            Updated result dict with SLO evaluation added.
        """
        if not self.config.enabled:
            return result

        # Skip if there's an error or no service name
        if "error" in result or "service" not in result:
            return result

        service_name = result.get("service", "")
        slo = self.config.get_service_slo(service_name)
        is_busy = self.config.is_busy_period(timestamp)

        # Apply busy period factor if applicable
        effective_slo = self._apply_busy_period_factor(slo, is_busy)

        # Get current metrics from result
        metrics = result.get("metrics", {})
        anomalies = result.get("anomalies", {})
        current_severity = result.get("overall_severity", "none")

        # Evaluate against SLOs
        evaluation = self._evaluate_against_slo(
            service_name=service_name,
            metrics=metrics,
            anomalies=anomalies,
            original_severity=current_severity,
            slo=effective_slo,
            is_busy_period=is_busy,
        )

        self._evaluation_count += 1

        # Update result with evaluation
        result["slo_evaluation"] = evaluation.to_dict()

        # Adjust severity if needed
        if evaluation.severity_changed:
            result["original_severity"] = current_severity
            result["overall_severity"] = evaluation.adjusted_severity
            self._adjustments_made += 1

            logger.info(
                f"SLO adjustment for {service_name}: "
                f"{current_severity} -> {evaluation.adjusted_severity} "
                f"(SLO proximity: {evaluation.slo_proximity:.2f}, "
                f"impact: {evaluation.operational_impact})"
            )

        # Also evaluate individual anomalies if present
        if anomalies:
            result["anomalies"] = self._evaluate_anomalies(
                anomalies, metrics, effective_slo, is_busy
            )

        return result

    def _apply_busy_period_factor(
        self, slo: ServiceSLOConfig, is_busy: bool
    ) -> ServiceSLOConfig:
        """Apply busy period relaxation factor to SLO thresholds."""
        if not is_busy:
            return slo

        factor = slo.busy_period_factor
        return ServiceSLOConfig(
            latency_acceptable_ms=slo.latency_acceptable_ms * factor,
            latency_warning_ms=slo.latency_warning_ms * factor,
            latency_critical_ms=slo.latency_critical_ms * factor,
            error_rate_acceptable=min(slo.error_rate_acceptable * factor, 0.1),
            error_rate_warning=min(slo.error_rate_warning * factor, 0.15),
            error_rate_critical=min(slo.error_rate_critical * factor, 0.2),
            min_traffic_rps=slo.min_traffic_rps,
            busy_period_factor=factor,
        )

    def _evaluate_against_slo(
        self,
        service_name: str,
        metrics: dict[str, Any],
        anomalies: dict[str, Any],
        original_severity: str,
        slo: ServiceSLOConfig,
        is_busy_period: bool,
    ) -> SLOEvaluationResult:
        """
        Evaluate metrics and anomalies against SLO thresholds.

        Returns:
            SLOEvaluationResult with adjusted severity and explanation.
        """
        # Extract relevant metrics
        latency = metrics.get("server_latency_avg", 0.0) or metrics.get("latency_avg", 0.0)
        error_rate = metrics.get("error_rate", 0.0)
        request_rate = metrics.get("request_rate", 0.0)

        # Evaluate latency against SLO
        latency_eval = self._evaluate_latency(latency, slo)

        # Evaluate error rate against SLO
        error_eval = self._evaluate_error_rate(error_rate, slo)

        # Determine overall SLO status (worst of latency and error)
        slo_status = self._combine_slo_status(latency_eval["status"], error_eval["status"])
        slo_proximity = max(latency_eval["proximity"], error_eval["proximity"])

        # Determine operational impact
        has_anomaly = len(anomalies) > 0 or original_severity not in ("none", "informational")
        operational_impact = self._determine_operational_impact(
            has_anomaly=has_anomaly,
            slo_status=slo_status,
            slo_proximity=slo_proximity,
            request_rate=request_rate,
            min_traffic=slo.min_traffic_rps,
        )

        # Compute adjusted severity
        adjusted_severity = self._compute_adjusted_severity(
            original_severity=original_severity,
            slo_status=slo_status,
            operational_impact=operational_impact,
            has_anomaly=has_anomaly,
        )

        # Generate explanation
        explanation = self._generate_explanation(
            service_name=service_name,
            original_severity=original_severity,
            adjusted_severity=adjusted_severity,
            latency=latency,
            error_rate=error_rate,
            slo=slo,
            slo_status=slo_status,
            is_busy_period=is_busy_period,
        )

        return SLOEvaluationResult(
            original_severity=original_severity,
            adjusted_severity=adjusted_severity,
            slo_status=slo_status,
            slo_proximity=slo_proximity,
            operational_impact=operational_impact,
            is_busy_period=is_busy_period,
            latency_evaluation=latency_eval,
            error_rate_evaluation=error_eval,
            explanation=explanation,
            severity_changed=adjusted_severity != original_severity,
        )

    def _evaluate_latency(
        self, latency_ms: float, slo: ServiceSLOConfig
    ) -> dict[str, Any]:
        """Evaluate latency against SLO thresholds."""
        if latency_ms <= 0:
            return {"status": "ok", "proximity": 0.0, "value": latency_ms, "threshold": slo.latency_critical_ms}

        proximity = latency_ms / slo.latency_critical_ms

        if latency_ms >= slo.latency_critical_ms:
            status = "breached"
        elif latency_ms >= slo.latency_warning_ms:
            status = "warning"
        elif latency_ms >= slo.latency_acceptable_ms:
            status = "elevated"
        else:
            status = "ok"

        return {
            "status": status,
            "proximity": proximity,
            "value": latency_ms,
            "threshold_acceptable": slo.latency_acceptable_ms,
            "threshold_warning": slo.latency_warning_ms,
            "threshold_critical": slo.latency_critical_ms,
        }

    def _evaluate_error_rate(
        self, error_rate: float, slo: ServiceSLOConfig
    ) -> dict[str, Any]:
        """Evaluate error rate against SLO thresholds."""
        if error_rate <= 0:
            return {"status": "ok", "proximity": 0.0, "value": error_rate, "threshold": slo.error_rate_critical}

        proximity = error_rate / slo.error_rate_critical if slo.error_rate_critical > 0 else 0.0

        if error_rate >= slo.error_rate_critical:
            status = "breached"
        elif error_rate >= slo.error_rate_warning:
            status = "warning"
        elif error_rate >= slo.error_rate_acceptable:
            status = "elevated"
        else:
            status = "ok"

        return {
            "status": status,
            "proximity": proximity,
            "value": error_rate,
            "value_percent": f"{error_rate * 100:.2f}%",
            "threshold_acceptable": slo.error_rate_acceptable,
            "threshold_warning": slo.error_rate_warning,
            "threshold_critical": slo.error_rate_critical,
        }

    def _combine_slo_status(self, latency_status: str, error_status: str) -> str:
        """Combine latency and error SLO status to get overall status."""
        status_order = ["ok", "elevated", "warning", "breached"]

        latency_idx = status_order.index(latency_status) if latency_status in status_order else 0
        error_idx = status_order.index(error_status) if error_status in status_order else 0

        return status_order[max(latency_idx, error_idx)]

    def _determine_operational_impact(
        self,
        has_anomaly: bool,
        slo_status: str,
        slo_proximity: float,
        request_rate: float,
        min_traffic: float,
    ) -> str:
        """
        Determine operational impact based on anomaly presence and SLO status.

        Returns one of: 'none', 'informational', 'actionable', 'critical'
        """
        # Low traffic services get reduced impact
        is_low_traffic = request_rate < min_traffic

        if slo_status == "breached":
            return "critical"
        elif slo_status == "warning":
            return "actionable" if has_anomaly else "actionable"
        elif slo_status == "elevated":
            if has_anomaly:
                return "informational" if is_low_traffic else "actionable"
            return "none"
        else:  # ok
            if has_anomaly:
                return "informational"
            return "none"

    def _compute_adjusted_severity(
        self,
        original_severity: str,
        slo_status: str,
        operational_impact: str,
        has_anomaly: bool,
    ) -> str:
        """
        Compute adjusted severity based on SLO status and operational impact.

        The key logic:
        - If SLO breached -> critical (regardless of ML severity)
        - If SLO warning -> at least medium
        - If anomaly but within acceptable -> downgrade to informational (if allowed)
        - If no anomaly but SLO issue -> use SLO-based severity
        """
        # SLO breach always results in critical
        if slo_status == "breached":
            return "critical"

        # SLO warning results in at least high
        if slo_status == "warning":
            if original_severity == "critical":
                return "critical"
            return "high"

        # Anomaly detected but within acceptable thresholds
        if has_anomaly and slo_status in ("ok", "elevated"):
            if self.config.allow_downgrade_to_informational and slo_status == "ok":
                # Anomaly is statistically real but operationally fine
                if original_severity in ("critical", "high"):
                    return "medium"  # Don't downgrade too aggressively
                elif original_severity == "medium":
                    return "low"
                return "informational"
            elif slo_status == "elevated":
                # Elevated but not warning - keep or slight downgrade
                if original_severity == "critical":
                    return "high"
                return original_severity

        # No anomaly - return based on SLO status
        if not has_anomaly:
            if slo_status == "elevated":
                return "low"
            return "none"

        # Default: keep original severity
        return original_severity

    def _evaluate_anomalies(
        self,
        anomalies: dict[str, Any],
        metrics: dict[str, Any],
        slo: ServiceSLOConfig,
        is_busy_period: bool,
    ) -> dict[str, Any]:
        """Add SLO context to individual anomalies."""
        evaluated_anomalies = {}

        for anomaly_name, anomaly_data in anomalies.items():
            if not isinstance(anomaly_data, dict):
                evaluated_anomalies[anomaly_name] = anomaly_data
                continue

            # Add SLO context to each anomaly
            anomaly_copy = dict(anomaly_data)

            # Check if this anomaly is latency-related
            if "latency" in anomaly_name.lower():
                latency = metrics.get("server_latency_avg", 0.0)
                anomaly_copy["slo_context"] = {
                    "current_value_ms": latency,
                    "acceptable_threshold_ms": slo.latency_acceptable_ms,
                    "critical_threshold_ms": slo.latency_critical_ms,
                    "within_acceptable": latency <= slo.latency_acceptable_ms,
                    "is_busy_period": is_busy_period,
                }

            # Check if this anomaly is error-related
            elif "error" in anomaly_name.lower():
                error_rate = metrics.get("error_rate", 0.0)
                anomaly_copy["slo_context"] = {
                    "current_value": error_rate,
                    "current_value_percent": f"{error_rate * 100:.2f}%",
                    "acceptable_threshold": slo.error_rate_acceptable,
                    "critical_threshold": slo.error_rate_critical,
                    "within_acceptable": error_rate <= slo.error_rate_acceptable,
                    "is_busy_period": is_busy_period,
                }

            evaluated_anomalies[anomaly_name] = anomaly_copy

        return evaluated_anomalies

    def _generate_explanation(
        self,
        service_name: str,
        original_severity: str,
        adjusted_severity: str,
        latency: float,
        error_rate: float,
        slo: ServiceSLOConfig,
        slo_status: str,
        is_busy_period: bool,
    ) -> str:
        """Generate human-readable explanation of the SLO evaluation."""
        parts = []

        if original_severity != adjusted_severity:
            parts.append(
                f"Severity adjusted from {original_severity} to {adjusted_severity} "
                f"based on SLO evaluation."
            )

        if is_busy_period:
            parts.append(
                f"Busy period active - thresholds relaxed by {slo.busy_period_factor}x."
            )

        if slo_status == "breached":
            if latency >= slo.latency_critical_ms:
                parts.append(
                    f"Latency SLO BREACHED: {latency:.0f}ms exceeds "
                    f"critical threshold of {slo.latency_critical_ms:.0f}ms."
                )
            if error_rate >= slo.error_rate_critical:
                parts.append(
                    f"Error rate SLO BREACHED: {error_rate*100:.2f}% exceeds "
                    f"critical threshold of {slo.error_rate_critical*100:.1f}%."
                )
        elif slo_status == "warning":
            if latency >= slo.latency_warning_ms:
                parts.append(
                    f"Latency approaching SLO: {latency:.0f}ms "
                    f"(warning at {slo.latency_warning_ms:.0f}ms, "
                    f"critical at {slo.latency_critical_ms:.0f}ms)."
                )
            if error_rate >= slo.error_rate_warning:
                parts.append(
                    f"Error rate approaching SLO: {error_rate*100:.2f}% "
                    f"(warning at {slo.error_rate_warning*100:.1f}%)."
                )
        elif slo_status == "ok" and original_severity not in ("none", "informational"):
            parts.append(
                f"Anomaly detected but metrics within acceptable SLO thresholds "
                f"(latency: {latency:.0f}ms < {slo.latency_acceptable_ms:.0f}ms, "
                f"errors: {error_rate*100:.2f}% < {slo.error_rate_acceptable*100:.1f}%)."
            )

        return " ".join(parts) if parts else "No SLO concerns."

    def get_stats(self) -> dict[str, Any]:
        """Get evaluation statistics."""
        return {
            "evaluations_performed": self._evaluation_count,
            "severity_adjustments": self._adjustments_made,
            "adjustment_rate": (
                self._adjustments_made / self._evaluation_count
                if self._evaluation_count > 0
                else 0.0
            ),
        }
