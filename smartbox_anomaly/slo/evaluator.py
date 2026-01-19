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

from smartbox_anomaly.core.config import (
    SLOConfig,
    ServiceSLOConfig,
    RequestRateEvaluationConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class SLOEvaluationInput:
    """Input context for SLO evaluation.

    Groups related parameters that flow through the evaluation process,
    reducing parameter count in method signatures.
    """

    service_name: str
    metrics: dict[str, Any]
    anomalies: dict[str, Any]
    original_severity: str
    slo: "ServiceSLOConfig"
    is_busy_period: bool = False
    time_period: str = "business_hours"


@dataclass
class ExplanationContext:
    """Context for generating human-readable explanations.

    Groups parameters needed for explanation generation, extracted
    from evaluation results to reduce method parameter count.
    """

    service_name: str
    original_severity: str
    adjusted_severity: str
    latency: float
    error_rate: float
    slo: "ServiceSLOConfig"
    slo_status: str
    is_busy_period: bool
    db_latency_eval: dict[str, Any] | None = None
    request_rate_eval: dict[str, Any] | None = None


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
    database_latency_evaluation: dict[str, Any] = field(default_factory=dict)
    request_rate_evaluation: dict[str, Any] = field(default_factory=dict)

    # Human-readable explanation
    explanation: str = ""

    # Whether severity was changed
    severity_changed: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
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
        # Only include optional evaluations if present
        if self.database_latency_evaluation:
            result["database_latency_evaluation"] = self.database_latency_evaluation
        if self.request_rate_evaluation:
            result["request_rate_evaluation"] = self.request_rate_evaluation
        return result


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

        # Skip if there's an actual error (not just null/None) or no service name
        if result.get("error") or "service" not in result:
            return result

        service_name = result.get("service", "")
        slo = self.config.get_service_slo(service_name)
        is_busy = self.config.is_busy_period(timestamp)

        # Apply busy period factor if applicable
        effective_slo = self._apply_busy_period_factor(slo, is_busy)

        # Get current metrics from result
        # Handle case where 'metrics' key exists but is None, fallback to 'current_metrics'
        metrics = result.get("metrics") or result.get("current_metrics") or {}
        anomalies = result.get("anomalies") or {}
        current_severity = result.get("overall_severity", "none")
        time_period = result.get("time_period", "business_hours")

        # Evaluate against SLOs
        evaluation = self._evaluate_against_slo(
            service_name=service_name,
            metrics=metrics,
            anomalies=anomalies,
            original_severity=current_severity,
            slo=effective_slo,
            is_busy_period=is_busy,
            time_period=time_period,
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
        # This may filter out anomalies that are below operational thresholds
        if anomalies:
            filtered_anomalies = self._evaluate_anomalies(
                anomalies, metrics, effective_slo, is_busy
            )
            result["anomalies"] = filtered_anomalies

            # Update anomaly count and severity if anomalies were filtered out
            original_count = len(anomalies)
            filtered_count = len(filtered_anomalies)

            if filtered_count != original_count:
                result["anomaly_count"] = filtered_count

                # If all anomalies were filtered, set to no_anomaly
                if filtered_count == 0:
                    result["alert_type"] = "no_anomaly"
                    result["overall_severity"] = "none"
                    result["slo_evaluation"]["adjusted_severity"] = "none"
                    result["slo_evaluation"]["explanation"] = (
                        f"All {original_count} anomalies suppressed - metrics below operational thresholds. "
                        + result["slo_evaluation"].get("explanation", "")
                    )
                    logger.info(
                        f"All anomalies suppressed for {service_name}: "
                        f"metrics below operational thresholds"
                    )
                else:
                    # Recalculate overall severity from remaining anomalies
                    severities = [
                        a.get("severity", "none")
                        for a in filtered_anomalies.values()
                        if isinstance(a, dict)
                    ]
                    if severities:
                        severity_order = ["none", "low", "medium", "high", "critical"]
                        max_severity = max(severities, key=lambda s: severity_order.index(s) if s in severity_order else 0)
                        result["overall_severity"] = max_severity

        return result

    def evaluate_metrics(
        self,
        metrics: dict[str, Any],
        service_name: str,
        original_severity: str = "none",
        timestamp: datetime | None = None,
        time_period: str = "business_hours",
    ) -> SLOEvaluationResult:
        """
        Evaluate metrics against SLO thresholds (standalone evaluation).

        This is used for standalone metric evaluation when building resolution
        context or other non-detection scenarios where you have metrics but
        not a full detection result.

        Unlike evaluate_result(), this method:
        - Does not modify any input dict
        - Does not filter anomalies
        - Returns just the SLOEvaluationResult

        Args:
            metrics: Current metric values (request_rate, application_latency, etc.)
            service_name: Service name for SLO config lookup.
            original_severity: Original ML severity (default: "none" for no anomaly).
            timestamp: Optional for busy period checking.
            time_period: Current time period (default: "business_hours").

        Returns:
            SLOEvaluationResult with evaluation details.
        """
        if not self.config.enabled:
            # Return a minimal result when SLO is disabled
            return SLOEvaluationResult(
                original_severity=original_severity,
                adjusted_severity=original_severity,
                slo_status="ok",
                slo_proximity=0.0,
                operational_impact="none",
                is_busy_period=False,
                explanation="SLO evaluation disabled",
            )

        slo = self.config.get_service_slo(service_name)
        is_busy = self.config.is_busy_period(timestamp)

        # Apply busy period factor if applicable
        effective_slo = self._apply_busy_period_factor(slo, is_busy)

        # Evaluate against SLOs (empty anomalies dict for standalone evaluation)
        return self._evaluate_against_slo(
            service_name=service_name,
            metrics=metrics,
            anomalies={},  # No anomalies for standalone evaluation
            original_severity=original_severity,
            slo=effective_slo,
            is_busy_period=is_busy,
            time_period=time_period,
        )

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
            # Database latency floor is relaxed during busy periods
            database_latency_floor_ms=slo.database_latency_floor_ms * factor,
            database_latency_ratios=slo.database_latency_ratios,
        )

    def _evaluate_against_slo(
        self,
        service_name: str,
        metrics: dict[str, Any],
        anomalies: dict[str, Any],
        original_severity: str,
        slo: ServiceSLOConfig,
        is_busy_period: bool,
        time_period: str = "business_hours",
    ) -> SLOEvaluationResult:
        """
        Evaluate metrics and anomalies against SLO thresholds.

        Args:
            service_name: Name of the service being evaluated.
            metrics: Current metrics including latency, error_rate, request_rate.
            anomalies: Detected anomalies from the ML model.
            original_severity: ML-assigned severity before SLO evaluation.
            slo: Service SLO configuration with thresholds.
            is_busy_period: Whether we're in a configured busy period.
            time_period: Current time period (business_hours, night_hours, etc.).

        Returns:
            SLOEvaluationResult with adjusted severity and explanation.
        """
        # Extract relevant metrics
        # Support multiple key naming conventions used across the codebase
        latency = (
            metrics.get("server_latency_avg", 0.0)
            or metrics.get("application_latency", 0.0)  # From InferenceMetrics
            or metrics.get("latency_avg", 0.0)
        )
        error_rate = metrics.get("error_rate", 0.0)
        request_rate = metrics.get("request_rate", 0.0)

        # Extract baseline metrics from training
        request_rate_baseline = (
            metrics.get("request_rate_mean", 0.0)       # Training baseline
            or metrics.get("traffic_mean", 0.0)
        )

        # Extract database latency metrics (current value and baseline)
        # Support multiple key naming conventions
        db_latency = (
            metrics.get("database_latency", 0.0)        # From InferenceMetrics
            or metrics.get("db_latency_avg", 0.0)
            or metrics.get("database_latency_avg", 0.0)
        )
        db_latency_baseline = (
            metrics.get("database_latency_mean", 0.0)   # Training baseline
            or metrics.get("db_latency_mean", 0.0)
        )

        # Evaluate latency against SLO
        latency_eval = self._evaluate_latency(latency, slo)

        # Evaluate error rate against SLO
        error_eval = self._evaluate_error_rate(error_rate, slo)

        # Evaluate database latency if we have data
        db_latency_eval: dict[str, Any] = {}
        db_latency_status: str | None = None
        if db_latency > 0:
            db_latency_eval = self._evaluate_database_latency(
                db_latency_ms=db_latency,
                baseline_mean_ms=db_latency_baseline,
                slo=slo,
            )
            db_latency_status = db_latency_eval.get("status")

        # Evaluate request rate (surges and cliffs)
        request_rate_eval: dict[str, Any] = {}
        if request_rate > 0 or request_rate_baseline > 0:
            request_rate_eval = self._evaluate_request_rate(
                request_rate=request_rate,
                baseline_mean=request_rate_baseline,
                time_period=time_period,
                latency_status=latency_eval["status"],
                error_status=error_eval["status"],
                slo=slo,
            )

        # Determine overall SLO status (worst of latency, error, and db latency)
        # Note: request_rate doesn't directly contribute to SLO status - it's correlation-based
        slo_status = self._combine_slo_status(
            latency_eval["status"],
            error_eval["status"],
            db_latency_status,
        )
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
        explanation_ctx = ExplanationContext(
            service_name=service_name,
            original_severity=original_severity,
            adjusted_severity=adjusted_severity,
            latency=latency,
            error_rate=error_rate,
            slo=slo,
            slo_status=slo_status,
            is_busy_period=is_busy_period,
            db_latency_eval=db_latency_eval,
            request_rate_eval=request_rate_eval,
        )
        explanation = self._generate_explanation(explanation_ctx)

        return SLOEvaluationResult(
            original_severity=original_severity,
            adjusted_severity=adjusted_severity,
            slo_status=slo_status,
            slo_proximity=slo_proximity,
            operational_impact=operational_impact,
            is_busy_period=is_busy_period,
            latency_evaluation=latency_eval,
            error_rate_evaluation=error_eval,
            database_latency_evaluation=db_latency_eval,
            request_rate_evaluation=request_rate_eval,
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
            status = "approaching_warning"
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
            status = "approaching_warning"
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

    def _evaluate_request_rate(
        self,
        request_rate: float,
        baseline_mean: float,
        time_period: str,
        latency_status: str,
        error_status: str,
        slo: ServiceSLOConfig,
    ) -> dict[str, Any]:
        """
        Evaluate request rate using correlation-based severity.

        This implements the Google SRE / NewRelic / Datadog approach:
        - Traffic surges: informational unless causing other SLO breaches
        - Traffic cliffs: warning by default, high during peak hours

        Args:
            request_rate: Current request rate (requests per second).
            baseline_mean: Training baseline mean request rate.
            time_period: Current time period (business_hours, night_hours, etc.).
            latency_status: Status from latency evaluation (ok/approaching_warning/warning/breached).
            error_status: Status from error rate evaluation (ok/approaching_warning/warning/breached).
            slo: Service SLO configuration with request rate settings.

        Returns:
            Evaluation dict with status, type (surge/cliff/normal), and severity.
        """
        rr_config = slo.request_rate_evaluation

        # If evaluation disabled, return normal status
        if not rr_config.enabled:
            return {
                "status": "ok",
                "type": "normal",
                "value_rps": round(request_rate, 2),
                "baseline_mean_rps": round(baseline_mean, 2) if baseline_mean else None,
                "enabled": False,
            }

        # Check minimum expected traffic for this time period
        min_expected = rr_config.min_expected_rps.get_min_for_period(time_period)

        # Handle edge cases
        if request_rate <= 0 and baseline_mean <= 0:
            return {
                "status": "ok",
                "type": "no_traffic",
                "value_rps": 0.0,
                "baseline_mean_rps": 0.0,
                "min_expected_rps": min_expected,
                "explanation": "No traffic detected",
            }

        # Determine if this is a peak hour (business_hours)
        is_peak_hours = time_period == "business_hours"

        # Calculate ratio if we have baseline
        ratio = request_rate / baseline_mean if baseline_mean > 0 else None

        # Check for surge (traffic > baseline * surge_threshold%)
        surge_threshold = rr_config.surge.threshold_percent / 100.0
        is_surge = ratio is not None and ratio >= surge_threshold

        # Check for cliff (traffic < baseline * cliff_threshold%)
        cliff_threshold = rr_config.cliff.threshold_percent / 100.0
        is_cliff = ratio is not None and ratio <= cliff_threshold

        # Check for below minimum expected traffic
        is_below_minimum = request_rate < min_expected

        # Determine status and severity based on type
        if is_surge:
            # Surge detected - severity depends on correlation with other SLOs
            has_latency_breach = latency_status in ("warning", "breached")
            has_error_breach = error_status in ("warning", "breached")

            if has_error_breach:
                severity = rr_config.surge.with_error_breach_severity
                status = "high" if severity in ("high", "critical") else "warning"
                explanation = (
                    f"Traffic surge ({ratio:.1f}x baseline) correlates with error SLO breach. "
                    f"Likely capacity issue."
                )
            elif has_latency_breach:
                severity = rr_config.surge.with_latency_breach_severity
                status = "warning"
                explanation = (
                    f"Traffic surge ({ratio:.1f}x baseline) correlates with latency degradation. "
                    f"Monitor capacity."
                )
            else:
                severity = rr_config.surge.standalone_severity
                status = "info"
                explanation = (
                    f"Traffic surge ({ratio:.1f}x baseline) without SLO impact. "
                    f"Normal growth or campaign traffic."
                )

            return {
                "status": status,
                "type": "surge",
                "severity": severity,
                "value_rps": round(request_rate, 2),
                "baseline_mean_rps": round(baseline_mean, 2),
                "ratio": round(ratio, 2),
                "threshold_percent": rr_config.surge.threshold_percent,
                "correlated_with_latency": has_latency_breach,
                "correlated_with_errors": has_error_breach,
                "explanation": explanation,
            }

        elif is_cliff:
            # Cliff detected - severity depends on time period and upstream issues
            # Note: For upstream errors, we'd need dependency info - for now, use error_status
            has_upstream_errors = error_status in ("warning", "breached")

            if has_upstream_errors:
                severity = rr_config.cliff.with_upstream_errors_severity
                status = "critical" if severity == "critical" else "high"
                explanation = (
                    f"Traffic cliff ({ratio:.1f}x baseline) with errors - "
                    f"possible service outage or upstream failure."
                )
            elif is_peak_hours:
                severity = rr_config.cliff.peak_hours_severity
                status = "high" if severity == "high" else "warning"
                explanation = (
                    f"Traffic cliff ({ratio:.1f}x baseline) during peak hours - "
                    f"investigate potential incident."
                )
            else:
                severity = rr_config.cliff.standalone_severity
                status = "warning"
                explanation = (
                    f"Traffic cliff ({ratio:.1f}x baseline) - "
                    f"may indicate routing issue or upstream problem."
                )

            return {
                "status": status,
                "type": "cliff",
                "severity": severity,
                "value_rps": round(request_rate, 2),
                "baseline_mean_rps": round(baseline_mean, 2),
                "ratio": round(ratio, 2),
                "threshold_percent": rr_config.cliff.threshold_percent,
                "is_peak_hours": is_peak_hours,
                "correlated_with_errors": has_upstream_errors,
                "explanation": explanation,
            }

        elif is_below_minimum:
            # Below minimum expected but not necessarily a cliff from baseline
            return {
                "status": "info",
                "type": "low_traffic",
                "severity": "informational",
                "value_rps": round(request_rate, 2),
                "baseline_mean_rps": round(baseline_mean, 2) if baseline_mean else None,
                "ratio": round(ratio, 2) if ratio else None,
                "min_expected_rps": min_expected,
                "explanation": (
                    f"Traffic ({request_rate:.1f} rps) below minimum expected "
                    f"({min_expected:.1f} rps) for {time_period}."
                ),
            }

        else:
            # Normal traffic
            return {
                "status": "ok",
                "type": "normal",
                "value_rps": round(request_rate, 2),
                "baseline_mean_rps": round(baseline_mean, 2) if baseline_mean else None,
                "ratio": round(ratio, 2) if ratio else None,
                "min_expected_rps": min_expected,
                "explanation": "Traffic within normal range",
            }

    def _evaluate_database_latency(
        self,
        db_latency_ms: float,
        baseline_mean_ms: float,
        slo: ServiceSLOConfig,
    ) -> dict[str, Any]:
        """
        Evaluate database latency using floor + ratio-based thresholds.

        This implements a hybrid approach:
        1. If db_latency < floor → always OK (noise filtering)
        2. Otherwise, compute ratio = db_latency / baseline_mean
           - ratio < info_threshold    → ok
           - ratio < warning_threshold → info
           - ratio < high_threshold    → warning
           - ratio < critical_threshold → high
           - ratio >= critical_threshold → critical

        This approach ensures:
        - Sub-millisecond changes don't trigger alerts (noise floor)
        - Severity is based on relative change from baseline (ratio-based)
        - Per-service customization via config

        Args:
            db_latency_ms: Current database latency in milliseconds.
            baseline_mean_ms: Training baseline mean latency in milliseconds.
            slo: Service SLO configuration with floor and ratio thresholds.

        Returns:
            Evaluation dict with status, ratio, and threshold details.
        """
        floor_ms = slo.database_latency_floor_ms
        ratios = slo.database_latency_ratios

        # Handle edge cases
        if db_latency_ms <= 0:
            return {
                "status": "ok",
                "value_ms": db_latency_ms,
                "baseline_mean_ms": baseline_mean_ms,
                "ratio": 0.0,
                "below_floor": True,
                "floor_ms": floor_ms,
            }

        # Rule 1: Below noise floor is always OK
        if db_latency_ms < floor_ms:
            return {
                "status": "ok",
                "value_ms": db_latency_ms,
                "baseline_mean_ms": baseline_mean_ms,
                "ratio": 0.0,
                "below_floor": True,
                "floor_ms": floor_ms,
                "explanation": f"Below noise floor ({db_latency_ms:.1f}ms < {floor_ms:.1f}ms)",
            }

        # Handle case where baseline is 0 or very small (avoid division issues)
        if baseline_mean_ms <= 0:
            # No baseline available - can't compute ratio
            # Fall back to absolute threshold approach
            if db_latency_ms < floor_ms:
                status = "ok"
            else:
                # Use floor * ratio thresholds as absolute values
                if db_latency_ms >= floor_ms * ratios.critical:
                    status = "critical"
                elif db_latency_ms >= floor_ms * ratios.high:
                    status = "high"
                elif db_latency_ms >= floor_ms * ratios.warning:
                    status = "warning"
                elif db_latency_ms >= floor_ms * ratios.info:
                    status = "info"
                else:
                    status = "ok"

            return {
                "status": status,
                "value_ms": db_latency_ms,
                "baseline_mean_ms": baseline_mean_ms,
                "ratio": None,
                "below_floor": False,
                "floor_ms": floor_ms,
                "explanation": "No baseline available - using absolute thresholds",
            }

        # Rule 2: Compute ratio and evaluate against thresholds
        ratio = db_latency_ms / baseline_mean_ms

        # Determine status based on ratio thresholds
        # Note: ratios are configured as: info=1.5, warning=2.0, high=3.0, critical=5.0
        if ratio >= ratios.critical:
            status = "critical"
        elif ratio >= ratios.high:
            status = "high"
        elif ratio >= ratios.warning:
            status = "warning"
        elif ratio >= ratios.info:
            status = "info"
        else:
            status = "ok"

        # Build explanation
        if status == "ok":
            explanation = (
                f"DB latency within normal range ({db_latency_ms:.1f}ms, "
                f"{ratio:.1f}x baseline of {baseline_mean_ms:.1f}ms)"
            )
        else:
            explanation = (
                f"DB latency elevated: {db_latency_ms:.1f}ms is {ratio:.1f}x baseline "
                f"({baseline_mean_ms:.1f}ms). Threshold for {status}: {getattr(ratios, status if status != 'high' else 'high'):.1f}x"
            )

        return {
            "status": status,
            "value_ms": round(db_latency_ms, 2),
            "baseline_mean_ms": round(baseline_mean_ms, 2),
            "ratio": round(ratio, 2),
            "below_floor": False,
            "floor_ms": floor_ms,
            "thresholds": {
                "info": ratios.info,
                "warning": ratios.warning,
                "high": ratios.high,
                "critical": ratios.critical,
            },
            "explanation": explanation,
        }

    def _combine_slo_status(
        self,
        latency_status: str,
        error_status: str,
        db_latency_status: str | None = None,
    ) -> str:
        """
        Combine latency, error, and database latency status to get overall status.

        Database latency uses a different status scale (ok/info/warning/high/critical),
        so we map it to the standard SLO scale (ok/approaching_warning/warning/breached).
        """
        status_order = ["ok", "approaching_warning", "warning", "breached"]

        latency_idx = status_order.index(latency_status) if latency_status in status_order else 0
        error_idx = status_order.index(error_status) if error_status in status_order else 0

        # Map database latency status to standard SLO scale
        db_idx = 0
        if db_latency_status:
            db_status_mapping = {
                "ok": 0,
                "info": 1,      # maps to approaching_warning
                "warning": 2,   # maps to warning
                "high": 3,      # maps to breached
                "critical": 3,  # maps to breached
            }
            db_idx = db_status_mapping.get(db_latency_status, 0)

        return status_order[max(latency_idx, error_idx, db_idx)]

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
        elif slo_status == "approaching_warning":
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

        SLO is authoritative for operational severity. The key logic:
        - If SLO breached -> critical (regardless of ML severity)
        - If SLO warning -> high (caps pattern-assigned critical to high)
        - If anomaly but within acceptable -> downgrade to low (if allowed)
        - If no anomaly but SLO issue -> use SLO-based severity
        """
        # SLO breach always results in critical
        if slo_status == "breached":
            return "critical"

        # SLO warning caps severity at high - SLO is authoritative for operational impact
        # If SLO says warning (no critical thresholds breached), we shouldn't alert as critical
        if slo_status == "warning":
            return "high"

        # Anomaly detected but within acceptable thresholds
        if has_anomaly and slo_status in ("ok", "approaching_warning"):
            if self.config.allow_downgrade_to_informational and slo_status == "ok":
                # SLO is ok - all metrics within acceptable thresholds
                # Anomaly is statistically real but operationally not significant
                return "low"
            elif slo_status == "approaching_warning":
                # Approaching warning but not there yet - keep or slight downgrade
                if original_severity == "critical":
                    return "high"
                return original_severity

        # No anomaly - return based on SLO status
        if not has_anomaly:
            if slo_status == "approaching_warning":
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
        """Add SLO context to individual anomalies.

        Also suppresses anomalies when their underlying metric is below the
        operational floor (e.g., database_latency < 1ms is not actionable).
        """
        evaluated_anomalies = {}

        for anomaly_name, anomaly_data in anomalies.items():
            if not isinstance(anomaly_data, dict):
                evaluated_anomalies[anomaly_name] = anomaly_data
                continue

            # Add SLO context to each anomaly
            anomaly_copy = dict(anomaly_data)
            anomaly_lower = anomaly_name.lower()

            # Check if this anomaly is database-related (check first due to overlap with "latency")
            if "database" in anomaly_lower or "db_" in anomaly_lower or "degradation" in anomaly_lower:
                # Support multiple key naming conventions
                db_latency = (
                    metrics.get("database_latency", 0.0)
                    or metrics.get("db_latency_avg", 0.0)
                    or metrics.get("database_latency_avg", 0.0)
                )
                db_baseline = (
                    metrics.get("database_latency_mean", 0.0)
                    or metrics.get("db_latency_mean", 0.0)
                )
                floor_ms = slo.database_latency_floor_ms
                ratios = slo.database_latency_ratios

                # Suppress anomaly if database latency is below noise floor
                # The detection was statistically valid but not operationally significant
                if db_latency < floor_ms:
                    logger.debug(
                        f"Suppressing {anomaly_name}: database_latency {db_latency:.2f}ms "
                        f"below floor {floor_ms}ms"
                    )
                    continue  # Skip this anomaly entirely

                # Compute ratio if we have baseline
                ratio = db_latency / db_baseline if db_baseline > 0 else None

                anomaly_copy["slo_context"] = {
                    "current_value_ms": db_latency,
                    "baseline_mean_ms": db_baseline,
                    "ratio": round(ratio, 2) if ratio else None,
                    "floor_ms": floor_ms,
                    "below_floor": False,  # We already filtered out below-floor cases above
                    "thresholds": {
                        "info": ratios.info,
                        "warning": ratios.warning,
                        "high": ratios.high,
                        "critical": ratios.critical,
                    },
                    "within_acceptable": ratio is not None and ratio < ratios.info,
                    "is_busy_period": is_busy_period,
                }

            # Check if this anomaly is application latency-related
            elif "latency" in anomaly_lower:
                # Support multiple key naming conventions
                latency = (
                    metrics.get("server_latency_avg", 0.0)
                    or metrics.get("application_latency", 0.0)
                    or metrics.get("latency_avg", 0.0)
                )
                anomaly_copy["slo_context"] = {
                    "current_value_ms": latency,
                    "acceptable_threshold_ms": slo.latency_acceptable_ms,
                    "critical_threshold_ms": slo.latency_critical_ms,
                    "within_acceptable": latency <= slo.latency_acceptable_ms,
                    "is_busy_period": is_busy_period,
                }

            # Check if this anomaly is error-related
            elif "error" in anomaly_lower:
                error_rate = metrics.get("error_rate", 0.0)

                # Determine suppression threshold:
                # Use error_rate_floor if set, otherwise use error_rate_acceptable
                suppression_threshold = (
                    slo.error_rate_floor if slo.error_rate_floor > 0
                    else slo.error_rate_acceptable
                )

                # Suppress anomaly if error rate is below the operational threshold
                # The ML detection was statistically valid but not operationally significant
                if error_rate < suppression_threshold:
                    logger.debug(
                        f"Suppressing {anomaly_name}: error_rate {error_rate:.4f} "
                        f"({error_rate * 100:.3f}%) below threshold {suppression_threshold:.4f} "
                        f"({suppression_threshold * 100:.2f}%)"
                    )
                    continue  # Skip this anomaly entirely

                anomaly_copy["slo_context"] = {
                    "current_value": error_rate,
                    "current_value_percent": f"{error_rate * 100:.2f}%",
                    "acceptable_threshold": slo.error_rate_acceptable,
                    "critical_threshold": slo.error_rate_critical,
                    "suppression_threshold": suppression_threshold,
                    "within_acceptable": error_rate <= slo.error_rate_acceptable,
                    "is_busy_period": is_busy_period,
                }

            # Check if this anomaly is traffic/request rate related
            elif any(p in anomaly_lower for p in ("traffic", "surge", "cliff", "request_rate")):
                request_rate = metrics.get("request_rate", 0.0)
                request_rate_baseline = (
                    metrics.get("request_rate_mean", 0.0)
                    or metrics.get("traffic_mean", 0.0)
                )
                rr_config = slo.request_rate_evaluation
                ratio = request_rate / request_rate_baseline if request_rate_baseline > 0 else None

                anomaly_copy["slo_context"] = {
                    "current_value_rps": round(request_rate, 2),
                    "baseline_mean_rps": round(request_rate_baseline, 2) if request_rate_baseline else None,
                    "ratio": round(ratio, 2) if ratio else None,
                    "surge_threshold_percent": rr_config.surge.threshold_percent,
                    "cliff_threshold_percent": rr_config.cliff.threshold_percent,
                    "is_surge": ratio is not None and ratio >= (rr_config.surge.threshold_percent / 100.0),
                    "is_cliff": ratio is not None and ratio <= (rr_config.cliff.threshold_percent / 100.0),
                    "is_busy_period": is_busy_period,
                }

            evaluated_anomalies[anomaly_name] = anomaly_copy

        return evaluated_anomalies

    def _generate_explanation(self, ctx: ExplanationContext) -> str:
        """Generate human-readable explanation of the SLO evaluation.

        Args:
            ctx: ExplanationContext containing all parameters needed for explanation.

        Returns:
            Human-readable explanation string.
        """
        parts = []

        if ctx.original_severity != ctx.adjusted_severity:
            parts.append(
                f"Severity adjusted from {ctx.original_severity} to {ctx.adjusted_severity} "
                f"based on SLO evaluation."
            )

        if ctx.is_busy_period:
            parts.append(
                f"Busy period active - thresholds relaxed by {ctx.slo.busy_period_factor}x."
            )

        if ctx.slo_status == "breached":
            if ctx.latency >= ctx.slo.latency_critical_ms:
                parts.append(
                    f"Latency SLO BREACHED: {ctx.latency:.0f}ms exceeds "
                    f"critical threshold of {ctx.slo.latency_critical_ms:.0f}ms."
                )
            if ctx.error_rate >= ctx.slo.error_rate_critical:
                parts.append(
                    f"Error rate SLO BREACHED: {ctx.error_rate*100:.2f}% exceeds "
                    f"critical threshold of {ctx.slo.error_rate_critical*100:.1f}%."
                )
        elif ctx.slo_status == "warning":
            if ctx.latency >= ctx.slo.latency_warning_ms:
                parts.append(
                    f"Latency approaching SLO: {ctx.latency:.0f}ms "
                    f"(warning at {ctx.slo.latency_warning_ms:.0f}ms, "
                    f"critical at {ctx.slo.latency_critical_ms:.0f}ms)."
                )
            if ctx.error_rate >= ctx.slo.error_rate_warning:
                parts.append(
                    f"Error rate approaching SLO: {ctx.error_rate*100:.2f}% "
                    f"(warning at {ctx.slo.error_rate_warning*100:.1f}%)."
                )
        elif ctx.slo_status == "ok" and ctx.original_severity not in ("none", "informational"):
            parts.append(
                f"Anomaly detected but metrics within acceptable SLO thresholds "
                f"(latency: {ctx.latency:.0f}ms < {ctx.slo.latency_acceptable_ms:.0f}ms, "
                f"errors: {ctx.error_rate*100:.2f}% < {ctx.slo.error_rate_acceptable*100:.1f}%)."
            )

        # Add database latency explanation if relevant
        if ctx.db_latency_eval:
            db_status = ctx.db_latency_eval.get("status", "ok")
            if db_status in ("high", "critical"):
                db_explanation = ctx.db_latency_eval.get("explanation", "")
                if db_explanation:
                    parts.append(f"Database latency issue: {db_explanation}")
            elif db_status == "warning":
                db_value = ctx.db_latency_eval.get("value_ms", 0)
                db_ratio = ctx.db_latency_eval.get("ratio", 0)
                parts.append(
                    f"Database latency elevated: {db_value:.1f}ms ({db_ratio:.1f}x baseline)."
                )
            elif db_status == "ok" and ctx.db_latency_eval.get("below_floor"):
                # Mention that database latency was filtered by noise floor
                db_value = ctx.db_latency_eval.get("value_ms", 0)
                floor = ctx.db_latency_eval.get("floor_ms", 5.0)
                if db_value > 0 and ctx.original_severity not in ("none", "informational"):
                    parts.append(
                        f"Database latency ({db_value:.1f}ms) below noise floor ({floor:.0f}ms) - operationally insignificant."
                    )

        # Add request rate explanation if relevant
        if ctx.request_rate_eval:
            rr_type = ctx.request_rate_eval.get("type", "normal")
            rr_explanation = ctx.request_rate_eval.get("explanation", "")
            if rr_type in ("surge", "cliff") and rr_explanation:
                parts.append(f"Traffic: {rr_explanation}")
            elif rr_type == "low_traffic" and rr_explanation:
                parts.append(rr_explanation)

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
