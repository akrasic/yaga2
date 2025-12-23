"""
SmartboxAnomalyDetector - Enhanced anomaly detection with explainability.

This module provides the main anomaly detection class with:
- Isolation Forest for univariate and multivariate detection
- Pattern-based anomaly detection
- Statistical correlation analysis
- Comprehensive explainability features
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import iqr, trim_mean as scipy_trim_mean
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from smartbox_anomaly.core.constants import MetricName, Thresholds
from smartbox_anomaly.core.exceptions import ModelTrainingError
from smartbox_anomaly.core.logging import get_logger
from smartbox_anomaly.core.types import CascadeAnalysis, DependencyContext
from smartbox_anomaly.detection.interpretations import (
    SeverityContext,
    get_business_impact,
    get_metric_interpretation,
    get_pattern_definition,
    get_recommendations,
)
from smartbox_anomaly.detection.service_config import (
    ServiceParameters,
    get_service_parameters,
)

logger = get_logger(__name__)

# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


@dataclass
class TrainingStatistics:
    """Training statistics for a metric."""

    mean: float
    median: float
    std: float
    min: float
    max: float
    p25: float
    p50: float
    p75: float
    p90: float
    p95: float
    p99: float
    count: int
    coefficient_of_variation: float
    typical_range: tuple[float, float]
    normal_range: tuple[float, float]
    outlier_bounds: tuple[float, float]
    # Zero-normal specific fields
    zero_count: int = 0
    zero_percentage: float = 0.0
    is_zero_dominant: bool = False
    non_zero_mean: float = 0.0
    non_zero_p95: float = 0.0


@dataclass
class FeatureImportance:
    """Feature importance for explainability."""

    variability_score: float
    impact_level: str
    business_criticality: str


@dataclass
class ValidationMetrics:
    """Metrics from validation set evaluation."""

    false_positive_rate: float
    detection_rate: float
    mean_score_normal: float
    mean_score_anomalous: float
    threshold_calibrated: bool = False


@dataclass
class CalibratedThresholds:
    """Calibrated severity thresholds from model score distribution."""

    critical: float = -0.6
    high: float = -0.3
    medium: float = -0.1
    low: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "critical": self.critical,
            "high": self.high,
            "medium": self.medium,
            "low": self.low,
        }

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> CalibratedThresholds:
        """Create from dictionary."""
        return cls(
            critical=data.get("critical", -0.6),
            high=data.get("high", -0.3),
            medium=data.get("medium", -0.1),
            low=data.get("low", 0.0),
        )


@dataclass
class DriftAnalysis:
    """Result of drift detection analysis."""

    has_drift: bool
    drift_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    overall_drift_score: float = 0.0
    recommendation: str = "OK: No significant drift detected."
    # Multivariate drift fields
    multivariate_drift: bool = False
    mahalanobis_distance: float = 0.0
    multivariate_threshold: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "has_drift": self.has_drift,
            "drift_metrics": self.drift_metrics,
            "overall_drift_score": self.overall_drift_score,
            "recommendation": self.recommendation,
            "multivariate_drift": self.multivariate_drift,
            "mahalanobis_distance": self.mahalanobis_distance,
            "multivariate_threshold": self.multivariate_threshold,
        }


@dataclass
class AnomalySignal:
    """Output from Isolation Forest detection - a trigger signal.

    This represents IF detecting something anomalous, before pattern interpretation.
    """

    metric_name: str
    score: float  # IF decision score (negative = anomalous)
    direction: str  # "high" or "low"
    value: float  # Current metric value
    percentile: float  # Position relative to training data (0-100)
    deviation_sigma: float  # Standard deviations from mean


# Pattern priority order for interpretation (most specific/critical first)
PATTERN_PRIORITY: list[str] = [
    # Critical - immediate user impact
    "traffic_surge_failing",
    "partial_outage",
    "fast_rejection",
    "traffic_cliff",
    "reduced_traffic_with_errors",
    # Dependency-aware patterns (check before generic high-severity)
    "upstream_cascade",
    "dependency_chain_degradation",
    # High - significant degradation
    "traffic_surge_degrading",
    "elevated_errors",
    "database_bottleneck",
    "downstream_cascade",
    "internal_bottleneck",
    "isolated_service_issue",  # Confirms not a cascade
    "recent_degradation",
    "fast_failure",
    "partial_fast_fail",
    # Medium - noticeable issues
    "database_degradation",
    "resource_contention",
    "performance_degradation",
    # Low - informational
    "traffic_surge_healthy",
]


class SmartboxAnomalyDetector:
    """Enhanced anomaly detection model with explainability and adaptive tuning.

    This detector uses Isolation Forest for both univariate and multivariate
    anomaly detection, with service-specific parameter tuning and comprehensive
    explainability features.

    Example:
        >>> detector = SmartboxAnomalyDetector("booking")
        >>> detector.train(training_data)
        >>> anomalies = detector.detect(current_metrics)
    """

    CORE_METRICS: list[str] = [
        MetricName.REQUEST_RATE,
        MetricName.APPLICATION_LATENCY,
        MetricName.CLIENT_LATENCY,
        MetricName.DATABASE_LATENCY,
        MetricName.ERROR_RATE,
    ]

    ZERO_NORMAL_METRICS: list[str] = [
        MetricName.CLIENT_LATENCY,
        MetricName.DATABASE_LATENCY,
    ]

    def __init__(self, service_name: str, auto_tune: bool = True) -> None:
        """Initialize the anomaly detector.

        Args:
            service_name: Name of the service to detect anomalies for.
            auto_tune: Whether to enable automatic parameter tuning.
        """
        self.service_name = service_name
        self.auto_tune = auto_tune

        # Model storage
        self.models: dict[str, IsolationForest] = {}
        self.scalers: dict[str, RobustScaler] = {}
        self.thresholds: dict[str, float] = {}
        self.optimal_params: dict[str, ServiceParameters] = {}

        # Feature tracking
        self.feature_columns: list[str] = []
        self.multivariate_feature_names: list[str] = []

        # Explainability data
        self.training_statistics: dict[str, TrainingStatistics] = {}
        self.feature_importance: dict[str, FeatureImportance] = {}
        self.zero_statistics: dict[str, dict[str, Any]] = {}
        self.pattern_thresholds: dict[str, dict[str, float]] = {}

        # ML improvements: calibrated thresholds and correlation analysis
        self.calibrated_thresholds: dict[str, CalibratedThresholds] = {}
        self.correlation_analysis: dict[str, Any] = {}
        self.estimated_contamination: float | None = None
        self.validation_metrics: dict[str, ValidationMetrics] = {}

        # Multivariate drift detection: store training distribution parameters
        self.multivariate_mean: np.ndarray | None = None
        self.multivariate_cov_inv: np.ndarray | None = None  # Inverse covariance for Mahalanobis
        self.multivariate_feature_order: list[str] = []  # Feature order for covariance matrix

        # Model metadata
        self.model_metadata: dict[str, Any] = {}
        self._trained = False

    @property
    def is_trained(self) -> bool:
        """Check if the detector has been trained."""
        return self._trained and len(self.models) > 0

    def train(
        self,
        features_df: pd.DataFrame,
        validation_df: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Train the anomaly detection models with optional validation data.

        Args:
            features_df: DataFrame with training data. Must contain metric columns.
            validation_df: Optional validation data for threshold calibration.
                          If provided, thresholds are calibrated on this held-out data.

        Returns:
            Training results including validation metrics if validation_df provided.

        Raises:
            ModelTrainingError: If training fails.
        """
        if features_df.empty:
            raise ModelTrainingError(
                self.service_name,
                reason="No training data provided",
            )

        logger.info(f"Training anomaly detector for {self.service_name}")
        self.feature_columns = features_df.columns.tolist()

        # Calculate training statistics for explainability (using robust stats)
        self._calculate_training_statistics(features_df)

        # Train univariate models
        trained_univariate = 0
        for metric in self.CORE_METRICS:
            if metric in features_df.columns:
                if self._train_univariate_model(metric, features_df[metric]):
                    trained_univariate += 1

        # Train multivariate model
        available_core = [m for m in self.CORE_METRICS if m in features_df.columns]
        has_multivariate = False
        if len(available_core) >= 3:
            has_multivariate = self._train_multivariate_model(features_df[available_core])

        # Calculate pattern thresholds
        self._calculate_pattern_thresholds(features_df)

        # Calculate feature importance
        self._calculate_feature_importance()

        # Calibrate thresholds on training data (will be overridden if validation_df provided)
        self._calibrate_severity_thresholds(features_df)

        # If validation data provided, calibrate thresholds on held-out data
        validation_results: dict[str, Any] = {}
        if validation_df is not None and not validation_df.empty:
            logger.info(f"Calibrating thresholds on {len(validation_df)} validation samples")
            validation_results = self._calibrate_on_validation(validation_df)

        self._trained = True
        self.model_metadata = {
            "service_name": self.service_name,
            "trained_at": datetime.now().isoformat(),
            "feature_count": len(self.feature_columns),
            "univariate_models": trained_univariate,
            "has_multivariate": has_multivariate,
            "auto_tuned": self.auto_tune,
            "validation_calibrated": validation_df is not None,
            "training_samples": len(features_df),
            "validation_samples": len(validation_df) if validation_df is not None else 0,
        }

        logger.info(
            f"Trained detector for {self.service_name}: "
            f"{trained_univariate} univariate, multivariate={has_multivariate}"
        )

        return {
            "trained_univariate": trained_univariate,
            "has_multivariate": has_multivariate,
            "validation_results": validation_results,
            "calibrated_thresholds": {
                k: v.to_dict() for k, v in self.calibrated_thresholds.items()
            },
        }

    def detect(
        self,
        metrics: dict[str, float],
        timestamp: datetime | None = None,
        dependency_context: DependencyContext | None = None,
        check_drift: bool = False,
    ) -> dict[str, Any]:
        """Detect anomalies using sequential IF → Pattern interpretation pipeline.

        Pipeline:
            1. Phase 1 (IF Detection): Run Isolation Forest to identify anomalous metrics
            2. Phase 2 (Interpretation): Match IF signals against patterns for actionable alerts

        Args:
            metrics: Dictionary of metric name to current value.
            timestamp: Optional timestamp for contextual severity adjustments.
            dependency_context: Optional context about dependency health for cascade detection.
            check_drift: If True, include drift analysis in the result.

        Returns:
            Dictionary containing detected anomalies with metadata, interpretations,
            and actionable recommendations. Includes drift_analysis if check_drift=True.
        """
        if not self.is_trained:
            logger.warning(f"Detector for {self.service_name} is not trained")
            return {"anomalies": {}, "metadata": {"trained": False}}

        detection_timestamp = timestamp or datetime.now()

        # =================================================================
        # Phase 1: IF Detection (Trigger)
        # =================================================================
        # Run Isolation Forest models to identify which metrics are anomalous
        signals = self._detect_if_signals(metrics)

        if not signals:
            # No anomalies detected by IF
            return {
                "anomalies": {},
                "metadata": {
                    "service_name": self.service_name,
                    "detection_timestamp": detection_timestamp.isoformat(),
                    "models_used": list(self.models.keys()),
                    "pipeline": "sequential_if_pattern",
                    "if_signals": 0,
                },
            }

        # =================================================================
        # Phase 2: Pattern Interpretation
        # =================================================================
        # Convert IF signals to interpreted anomaly (pattern match or unknown)
        anomalies = self._interpret_signals(
            signals, metrics, detection_timestamp, dependency_context
        )

        # Add explainability data
        for anomaly_name, anomaly in anomalies.items():
            anomaly["comparison_data"] = self._get_comparison_data(anomaly, metrics)
            anomaly["business_impact"] = self._get_business_impact(anomaly)

            # Contextual severity adjustment
            base_severity = anomaly.get("severity", "medium")
            anomaly_type = anomaly.get("pattern_name", anomaly.get("type", anomaly_name))
            severity_context = self._calculate_contextual_severity(
                anomaly_type, base_severity, metrics, detection_timestamp
            )

            if severity_context.adjusted_severity != severity_context.base_severity:
                anomaly["severity"] = severity_context.adjusted_severity
                anomaly["severity_adjustments"] = severity_context.adjustment_reasons
                anomaly["severity_confidence"] = severity_context.confidence

        result = {
            "anomalies": anomalies,
            "metadata": {
                "service_name": self.service_name,
                "detection_timestamp": detection_timestamp.isoformat(),
                "models_used": list(self.models.keys()),
                "pipeline": "sequential_if_pattern",
                "if_signals": len(signals),
                "features": {
                    "contextual_severity": True,
                    "pattern_interpretation": True,
                    "recommendations": True,
                    "calibrated_thresholds": len(self.calibrated_thresholds) > 0,
                },
            },
        }

        # Add drift analysis if requested
        if check_drift:
            drift_analysis = self.check_drift(metrics)
            result["drift_analysis"] = drift_analysis.to_dict()

        return result

    # =========================================================================
    # Phase 1: IF Detection Methods
    # =========================================================================

    def _detect_if_signals(self, metrics: dict[str, float]) -> list[AnomalySignal]:
        """Phase 1: Run IF models to identify anomalous metrics.

        Returns list of AnomalySignal for metrics flagged as anomalous.
        Does NOT produce final anomalies - just trigger signals.
        """
        signals: list[AnomalySignal] = []

        # Univariate IF detection
        for metric_name in self.CORE_METRICS:
            signal = self._detect_metric_signal(metric_name, metrics)
            if signal:
                signals.append(signal)

        # Multivariate IF detection (adds signal if unusual combination detected)
        mv_signal = self._detect_multivariate_signal(metrics)
        if mv_signal:
            # Multivariate detection enhances existing signals or adds context
            # We don't add a separate signal, but could boost confidence
            pass

        return signals

    def _detect_metric_signal(
        self, metric_name: str, metrics: dict[str, float]
    ) -> AnomalySignal | None:
        """Detect IF signal for a single metric."""
        model_key = f"{metric_name}_isolation"
        scaler_key = f"{metric_name}_scaler"

        if model_key not in self.models or scaler_key not in self.scalers:
            return None

        value = metrics.get(metric_name, 0.0)
        stats = self.training_statistics.get(metric_name)

        # Handle zero-normal metrics
        if metric_name in self.ZERO_NORMAL_METRICS:
            if self.thresholds.get(f"{metric_name}_zero_dominant", False):
                if value > 0:
                    threshold = self.thresholds.get(f"{metric_name}_non_zero_p95", 0.0)
                    if value > threshold * 2:
                        # Significant activation of normally-zero metric
                        return AnomalySignal(
                            metric_name=metric_name,
                            score=-0.3,
                            direction="activated",
                            value=value,
                            percentile=95.0,
                            deviation_sigma=2.0,
                        )
                return None

        try:
            scaler = self.scalers[scaler_key]
            model = self.models[model_key]

            df = pd.DataFrame([[value]], columns=[metric_name])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scaled = scaler.transform(df)

            prediction = model.predict(scaled)[0]
            if prediction == -1:  # Anomaly detected
                score = float(model.decision_function(scaled)[0])

                # Calculate context
                if stats:
                    direction = "high" if value > stats.mean else "low"
                    deviation = (value - stats.mean) / (stats.std + 1e-8)
                    percentile = self._estimate_percentile(value, stats)
                else:
                    direction = "unknown"
                    deviation = 0.0
                    percentile = 50.0

                return AnomalySignal(
                    metric_name=metric_name,
                    score=score,
                    direction=direction,
                    value=value,
                    percentile=percentile,
                    deviation_sigma=deviation,
                )

        except Exception as e:
            logger.debug(f"Error detecting signal for {metric_name}: {e}")

        return None

    def _detect_multivariate_signal(self, metrics: dict[str, float]) -> bool:
        """Detect if multivariate IF flags an unusual combination.

        Returns True if the combination of metrics is anomalous.
        This can boost confidence in pattern matching.
        """
        if "multivariate_detector" not in self.models:
            return False

        try:
            metric_values = []
            available_metrics = []

            for feature in self.multivariate_feature_names:
                if feature in self.CORE_METRICS:
                    available_metrics.append(feature)
                    metric_values.append(metrics.get(feature, 0.0))

            if len(metric_values) < 2:
                return False

            scaler = self.scalers["multivariate_scaler"]
            if scaler.n_features_in_ != len(metric_values):
                return False

            df = pd.DataFrame([metric_values], columns=available_metrics)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scaled = scaler.transform(df)

            model = self.models["multivariate_detector"]
            prediction = model.predict(scaled)[0]

            return prediction == -1  # Anomalous combination

        except Exception as e:
            logger.debug(f"Multivariate detection error: {e}")
            return False

    # =========================================================================
    # Phase 2: Pattern Interpretation Methods
    # =========================================================================

    def _interpret_signals(
        self,
        signals: list[AnomalySignal],
        metrics: dict[str, float],
        timestamp: datetime,
        dependency_context: DependencyContext | None = None,
    ) -> dict[str, Any]:
        """Phase 2: Interpret IF signals using pattern definitions.

        Args:
            signals: Anomaly signals from IF detection
            metrics: Full metrics dict (for ratio calculations)
            timestamp: Detection timestamp
            dependency_context: Optional context about dependency health for cascade detection

        Returns:
            Single interpreted anomaly dict, or unknown anomaly if no pattern matches
        """
        # Convert IF signals to metric levels for pattern matching
        metric_levels = self._signals_to_levels(signals, metrics)

        # Calculate ratios for ratio-based pattern conditions
        ratios = self._calculate_metric_ratios(metrics)

        # Perform cascade analysis if dependency context provided
        cascade_analysis = self._analyze_cascade(signals, metrics, dependency_context)

        # Try to match a named pattern in priority order
        matched_pattern = None
        for pattern_name in PATTERN_PRIORITY:
            pattern_def = get_pattern_definition(pattern_name)
            if pattern_def and self._pattern_matches(
                metric_levels, ratios, pattern_def.conditions, cascade_analysis
            ):
                matched_pattern = pattern_def
                break

        if matched_pattern:
            return self._build_interpreted_anomaly(
                matched_pattern, signals, metrics, timestamp, cascade_analysis
            )
        else:
            return self._build_unknown_anomaly(signals, metrics, timestamp, cascade_analysis)

    def _signals_to_levels(
        self, signals: list[AnomalySignal], metrics: dict[str, float]
    ) -> dict[str, str]:
        """Convert IF signals to metric levels for pattern matching.

        IF signals tell us WHICH metrics are anomalous.
        Levels tell us HOW they're anomalous (high/low/normal).
        """
        levels: dict[str, str] = {}

        # Build set of signaled metrics for quick lookup
        signaled_metrics = {s.metric_name for s in signals}

        # Metrics with IF signals get their level from signal
        for signal in signals:
            if signal.direction == "activated":
                levels[signal.metric_name] = "high"  # Treat activation as high
            elif signal.percentile > 95:
                levels[signal.metric_name] = "very_high"
            elif signal.percentile > 90:
                levels[signal.metric_name] = "high"
            elif signal.percentile < 5:
                levels[signal.metric_name] = "very_low"
            elif signal.percentile < 10:
                levels[signal.metric_name] = "low"
            else:
                # Edge case: IF flagged it but percentile is mid-range
                # Use direction to determine
                levels[signal.metric_name] = signal.direction

        # Metrics WITHOUT IF signals are "normal"
        for metric in self.CORE_METRICS:
            if metric not in levels:
                levels[metric] = "normal"

        return levels

    def _analyze_cascade(
        self,
        signals: list[AnomalySignal],
        metrics: dict[str, float],
        dependency_context: DependencyContext | None,
    ) -> CascadeAnalysis:
        """Analyze whether this anomaly is caused by an upstream cascade.

        Args:
            signals: IF signals indicating anomalous metrics
            metrics: Current metric values
            dependency_context: Context about dependency health

        Returns:
            CascadeAnalysis with root cause identification if cascade detected
        """
        # No cascade possible without dependency context
        if not dependency_context:
            return CascadeAnalysis(
                is_cascade=False,
                root_cause_service=None,
                affected_chain=[],
                cascade_type="none",
                confidence=0.0,
            )

        # Check if we have latency-related signals (typical of cascade effects)
        latency_signals = [
            s for s in signals
            if "latency" in s.metric_name.lower()
        ]

        if not latency_signals:
            # No latency signals - check if all dependencies are healthy
            if dependency_context.all_dependencies_healthy():
                return CascadeAnalysis(
                    is_cascade=False,
                    root_cause_service=None,
                    affected_chain=[],
                    cascade_type="dependencies_healthy",
                    confidence=0.9,
                )
            return CascadeAnalysis(
                is_cascade=False,
                root_cause_service=None,
                affected_chain=[],
                cascade_type="none",
                confidence=0.0,
            )

        # Find root cause in dependency chain
        root_cause, affected_chain = dependency_context.find_root_cause_service(
            self.service_name
        )

        if root_cause:
            root_status = dependency_context.get_status(root_cause)
            # Determine cascade type based on chain length
            if len(affected_chain) >= 2:
                cascade_type = "chain_degraded"
            else:
                cascade_type = "upstream_anomaly"

            return CascadeAnalysis(
                is_cascade=True,
                root_cause_service=root_cause,
                affected_chain=affected_chain,
                cascade_type=cascade_type,
                confidence=0.85 if root_status and root_status.has_anomaly else 0.6,
                root_cause_anomaly_type=root_status.anomaly_type if root_status else None,
                propagation_path=self._build_propagation_path(affected_chain, dependency_context),
            )

        # No cascade found - check if dependencies are healthy
        if dependency_context.all_dependencies_healthy():
            return CascadeAnalysis(
                is_cascade=False,
                root_cause_service=None,
                affected_chain=[],
                cascade_type="dependencies_healthy",
                confidence=0.9,
            )

        return CascadeAnalysis(
            is_cascade=False,
            root_cause_service=None,
            affected_chain=[],
            cascade_type="none",
            confidence=0.0,
        )

    def _build_propagation_path(
        self,
        chain: list[str],
        context: DependencyContext,
    ) -> list[dict[str, Any]]:
        """Build detailed propagation path for cascade visualization."""
        path = []
        for service in chain:
            status = context.get_status(service)
            path.append({
                "service": service,
                "has_anomaly": status.has_anomaly if status else False,
                "anomaly_type": status.anomaly_type if status else None,
                "latency_percentile": status.latency_percentile if status else None,
                "severity": status.severity if status else None,
            })
        return path

    def _build_interpreted_anomaly(
        self,
        pattern: Any,  # PatternDefinition
        signals: list[AnomalySignal],
        metrics: dict[str, float],
        timestamp: datetime,  # noqa: ARG002
        cascade_analysis: CascadeAnalysis | None = None,
    ) -> dict[str, Any]:
        """Build an interpreted anomaly from a matched pattern.

        Combines IF detection evidence with pattern interpretation.
        Output format matches docs/INFERENCE_API_PAYLOAD.md specification.
        """
        # Use worst IF score for overall score
        worst_score = min(s.score for s in signals)

        # Severity from pattern, but can be escalated by IF score
        severity = pattern.severity
        if worst_score < -0.6 and severity != "critical":
            severity = "critical"
        elif worst_score < -0.3 and severity == "low":
            severity = "medium"

        # Calculate confidence based on signal agreement
        signal_count = len(signals)
        high_sev_count = sum(1 for s in signals if s.score < -0.3)
        confidence = min(0.95, 0.5 + (signal_count * 0.15) + (high_sev_count * 0.1))

        # Build format values for message template
        ratios = self._calculate_metric_ratios(metrics)
        format_values = {
            **metrics,
            "client_latency_ratio": ratios.get("client_latency_ratio", 0),
            "db_latency_ratio": ratios.get("db_latency_ratio", 0),
            "expected_rate": self.training_statistics.get(
                MetricName.REQUEST_RATE, TrainingStatistics(
                    mean=0, median=0, std=0, min=0, max=0, p25=0, p50=0, p75=0,
                    p90=0, p95=0, p99=0, count=0, coefficient_of_variation=0,
                    typical_range=(0, 0), normal_range=(0, 0), outlier_bounds=(0, 0)
                )
            ).mean,
            "drop_percent": self._calculate_drop_percent(metrics.get(MetricName.REQUEST_RATE, 0)),
        }

        # Add cascade-specific format values for dependency-aware patterns
        if cascade_analysis and cascade_analysis.is_cascade:
            format_values.update({
                "root_cause_service": cascade_analysis.root_cause_service or "unknown",
                "root_cause_anomaly_type": cascade_analysis.root_cause_anomaly_type or "degradation",
                "affected_chain_length": len(cascade_analysis.affected_chain),
                "affected_services": ", ".join(cascade_analysis.affected_chain[:3]),
            })

        try:
            description = pattern.message_template.format(**format_values)
        except KeyError:
            description = f"{pattern.name}: anomaly detected"

        # Determine root metric from signals
        priority = ["application_latency", "error_rate", "request_rate",
                   "database_latency", "client_latency"]
        root_metric = None
        for metric in priority:
            if any(s.metric_name == metric for s in signals):
                root_metric = metric
                break
        if not root_metric and signals:
            root_metric = signals[0].metric_name

        # Get primary value
        primary_signal = min(signals, key=lambda s: s.score)
        value = primary_signal.value

        # Build detection_signals array (API-compatible format)
        detection_signals = []
        for signal in signals:
            detection_signals.append({
                "method": "isolation_forest",
                "type": "ml_isolation",
                "severity": Thresholds.score_to_severity(signal.score).value,
                "score": signal.score,
                "direction": signal.direction,
                "percentile": signal.percentile,
            })
        # Add pattern signal
        detection_signals.append({
            "method": "named_pattern_matching",
            "type": "multivariate_pattern",
            "severity": severity,
            "score": worst_score,
            "pattern": pattern.name,
        })

        # Format interpretation with cascade values if applicable
        interpretation = pattern.interpretation
        if cascade_analysis and cascade_analysis.is_cascade:
            try:
                interpretation = interpretation.format(**format_values)
            except KeyError:
                pass  # Keep original interpretation if formatting fails

        result = {
            pattern.name: {
                "type": "consolidated",
                "root_metric": root_metric,
                "severity": severity,
                "confidence": round(confidence, 2),
                "score": worst_score,
                "signal_count": len(detection_signals),
                "description": description,
                "interpretation": interpretation,
                "pattern_name": pattern.name,
                "value": value,
                "detection_method": "isolation_forest + pattern_interpretation",
                "detection_signals": detection_signals,
                "recommended_actions": pattern.recommended_actions,
                "contributing_metrics": [s.metric_name for s in signals],
                "metric_values": {m: metrics.get(m, 0) for m in self.CORE_METRICS},
            }
        }

        # Add cascade analysis if present
        if cascade_analysis:
            result[pattern.name]["cascade_analysis"] = cascade_analysis.to_dict()

        return result

    def _build_unknown_anomaly(
        self,
        signals: list[AnomalySignal],
        metrics: dict[str, float],
        timestamp: datetime,  # noqa: ARG002
        cascade_analysis: CascadeAnalysis | None = None,
    ) -> dict[str, Any]:
        """Build an anomaly for IF detections that don't match any pattern.

        This is the fallback - we detected something but don't know what pattern it is.
        Output format matches docs/INFERENCE_API_PAYLOAD.md specification.
        """
        # Primary signal is the most severe
        primary = min(signals, key=lambda s: s.score)
        severity = Thresholds.score_to_severity(primary.score)

        # Calculate confidence (lower for unknown since no pattern match)
        signal_count = len(signals)
        confidence = min(0.7, 0.4 + (signal_count * 0.1))

        # Determine root metric from signals
        priority = ["application_latency", "error_rate", "request_rate",
                   "database_latency", "client_latency"]
        root_metric = None
        for metric in priority:
            if any(s.metric_name == metric for s in signals):
                root_metric = metric
                break
        if not root_metric and signals:
            root_metric = signals[0].metric_name

        # Generate description based on signals
        signal_descriptions = []
        for signal in signals:
            interp = get_metric_interpretation(signal.metric_name, signal.direction)
            if interp:
                try:
                    stats = self.training_statistics.get(signal.metric_name)
                    desc = interp.message_template.format(
                        value=signal.value,
                        percentile=signal.percentile,
                        deviation=abs(signal.deviation_sigma),
                        mean=stats.mean if stats else 0,
                        p90=stats.p90 if stats else 0,
                        p95=stats.p95 if stats else 0,
                    )
                    signal_descriptions.append(desc)
                except (KeyError, AttributeError):
                    signal_descriptions.append(
                        f"{signal.metric_name} {signal.direction}: {signal.value:.2f}"
                    )
            else:
                signal_descriptions.append(
                    f"{signal.metric_name} {signal.direction}: {signal.value:.2f}"
                )

        # Collect possible causes and checks from metric interpretations
        all_causes: list[str] = []
        all_checks: list[str] = []
        for signal in signals:
            interp = get_metric_interpretation(signal.metric_name, signal.direction)
            if interp:
                all_causes.extend(interp.possible_causes[:2])
                all_checks.extend(interp.checks[:2])

        # Build detection_signals array (API-compatible format)
        detection_signals = []
        for signal in signals:
            detection_signals.append({
                "method": "isolation_forest",
                "type": "ml_isolation",
                "severity": Thresholds.score_to_severity(signal.score).value,
                "score": signal.score,
                "direction": signal.direction,
                "percentile": signal.percentile,
            })

        # Generate anomaly name based on root metric
        if root_metric == "application_latency":
            anomaly_name = "latency_anomaly"
        elif root_metric == "error_rate":
            anomaly_name = "error_rate_anomaly"
        elif root_metric == "request_rate":
            anomaly_name = "traffic_anomaly"
        else:
            anomaly_name = f"{root_metric}_anomaly" if root_metric else "unknown_anomaly"

        result = {
            anomaly_name: {
                "type": "ml_isolation" if len(signals) == 1 else "consolidated",
                "root_metric": root_metric,
                "severity": severity.value,
                "confidence": round(confidence, 2),
                "score": primary.score,
                "signal_count": len(detection_signals),
                "description": (
                    f"Unusual behavior detected: {'; '.join(signal_descriptions[:2])}"
                    + (f" (+{len(signal_descriptions) - 2} more)" if len(signal_descriptions) > 2 else "")
                ),
                "interpretation": (
                    "Isolation Forest detected anomalous behavior that doesn't match "
                    "any known pattern. This may be a novel issue requiring investigation. "
                    "Consider adding a new pattern if this recurs."
                ),
                "detection_method": "isolation_forest",
                "value": primary.value,
                "direction": primary.direction,
                "percentile": primary.percentile,
                "deviation_sigma": primary.deviation_sigma,
                "detection_signals": detection_signals,

                # Generic + metric-specific recommendations
                "recommended_actions": [
                    "INVESTIGATE: Review the flagged metrics for unusual values",
                    "CORRELATE: Check for recent deployments or configuration changes",
                    "COMPARE: Review metric trends over the past hour",
                ] + all_checks[:2],

                "possible_causes": all_causes[:5],
                "checks": all_checks[:5],
                "contributing_metrics": [s.metric_name for s in signals],
                "metric_values": {m: metrics.get(m, 0) for m in self.CORE_METRICS},
            }
        }

        # Add cascade analysis if present
        if cascade_analysis:
            result[anomaly_name]["cascade_analysis"] = cascade_analysis.to_dict()

        return result

    # =========================================================================
    # Training Support Methods
    # =========================================================================

    def _clean_metric_data_for_training(
        self, data: pd.Series, metric_name: str
    ) -> tuple[pd.Series, dict[str, Any]]:
        """Clean metric data for IF training - keeps outliers.

        Only removes invalid values (NaN, inf). Outliers are what IF should learn to detect.

        Args:
            data: Raw metric data series.
            metric_name: Name of the metric being cleaned.

        Returns:
            Tuple of (cleaned_data, zero_info) where zero_info contains zero-normal stats.
        """
        # Step 1: Basic cleaning - remove only NaN and infinity
        clean_data = data.dropna().replace([np.inf, -np.inf], np.nan).dropna()

        # Step 2: Calculate zero-normal statistics
        is_zero_normal = metric_name in self.ZERO_NORMAL_METRICS
        zero_count = int((clean_data == 0).sum()) if is_zero_normal else 0
        zero_percentage = zero_count / len(clean_data) if is_zero_normal and len(clean_data) > 0 else 0.0
        non_zero_data = clean_data[clean_data > 0] if is_zero_normal else clean_data

        zero_info = {
            "zero_count": zero_count,
            "zero_percentage": zero_percentage,
            "is_zero_dominant": zero_percentage > 0.5,
            "non_zero_mean": float(non_zero_data.mean()) if len(non_zero_data) > 0 else 0.0,
            "non_zero_p95": float(non_zero_data.quantile(0.95)) if len(non_zero_data) > 0 else 0.0,
        }

        # Step 3: For zero-dominant metrics, use non-zero data only
        if is_zero_normal and zero_percentage > 0.5:
            if len(non_zero_data) >= 50:
                clean_data = non_zero_data
                logger.debug(f"{metric_name}: Using {len(clean_data)} non-zero samples (was {zero_percentage:.1%} zeros)")
            else:
                logger.debug(f"{metric_name}: Zero-dominant with insufficient non-zero data")

        # NOTE: We do NOT remove outliers for IF training - outliers are what IF should detect!
        return clean_data, zero_info

    def _calculate_robust_statistics(
        self, data: pd.Series, metric_name: str
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """Calculate robust statistics for a metric without removing outliers.

        Uses trimmed mean and IQR-based std for robustness against outliers.

        Args:
            data: Raw metric data series.
            metric_name: Name of the metric.

        Returns:
            Tuple of (robust_stats, zero_info).
        """
        # Basic cleaning - only remove NaN/inf
        clean_data = data.dropna().replace([np.inf, -np.inf], np.nan).dropna()

        if len(clean_data) == 0:
            return {}, {}

        # Calculate zero-normal statistics
        is_zero_normal = metric_name in self.ZERO_NORMAL_METRICS
        zero_count = int((clean_data == 0).sum()) if is_zero_normal else 0
        zero_percentage = zero_count / len(clean_data) if is_zero_normal and len(clean_data) > 0 else 0.0
        non_zero_data = clean_data[clean_data > 0] if is_zero_normal else clean_data

        zero_info = {
            "zero_count": zero_count,
            "zero_percentage": zero_percentage,
            "is_zero_dominant": zero_percentage > 0.5,
            "non_zero_mean": float(non_zero_data.mean()) if len(non_zero_data) > 0 else 0.0,
            "non_zero_p95": float(non_zero_data.quantile(0.95)) if len(non_zero_data) > 0 else 0.0,
        }

        # For zero-dominant metrics, calculate stats on non-zero data
        stats_data = non_zero_data if (is_zero_normal and zero_percentage > 0.5 and len(non_zero_data) >= 50) else clean_data

        if len(stats_data) == 0:
            return {}, zero_info

        # Robust statistics using scipy
        # Trimmed mean removes 1% from each tail
        try:
            robust_mean = scipy_trim_mean(stats_data, proportiontocut=0.01)
        except Exception:
            robust_mean = float(stats_data.mean())

        # IQR-based robust std (IQR / 1.349 approximates std for normal distribution)
        data_iqr = iqr(stats_data)
        robust_std = data_iqr / 1.349 if data_iqr > 0 else float(stats_data.std())

        # Standard percentiles (computed on all data including outliers)
        robust_stats = {
            "robust_mean": robust_mean,
            "robust_std": robust_std,
            "mean": float(stats_data.mean()),
            "median": float(stats_data.median()),
            "std": float(stats_data.std()) if len(stats_data) > 1 else 0.0,
            "min": float(stats_data.min()),
            "max": float(stats_data.max()),
            "p25": float(stats_data.quantile(0.25)),
            "p50": float(stats_data.quantile(0.50)),
            "p75": float(stats_data.quantile(0.75)),
            "p90": float(stats_data.quantile(0.90)),
            "p95": float(stats_data.quantile(0.95)),
            "p99": float(stats_data.quantile(0.99)),
            "count": len(stats_data),
        }

        return robust_stats, zero_info

    def _clean_metric_data(
        self, data: pd.Series, metric_name: str
    ) -> tuple[pd.Series, dict[str, Any]]:
        """Clean metric data (backward compatibility - calls training version).

        Args:
            data: Raw metric data series.
            metric_name: Name of the metric being cleaned.

        Returns:
            Tuple of (cleaned_data, zero_info).
        """
        return self._clean_metric_data_for_training(data, metric_name)

    def _calculate_training_statistics(self, features_df: pd.DataFrame) -> None:
        """Calculate training statistics using robust estimators.

        Uses trimmed mean and IQR-based std for robustness against outliers.
        Statistics are computed on all data (including outliers) to accurately
        represent the full distribution the IF model will see.
        """
        for metric in self.CORE_METRICS:
            if metric not in features_df.columns:
                continue

            raw_data = features_df[metric]
            if raw_data.dropna().empty:
                continue

            # Use robust statistics calculation
            robust_stats, zero_info = self._calculate_robust_statistics(raw_data, metric)

            if not robust_stats:
                logger.warning(f"No valid data for {metric} after cleaning")
                continue

            # Use robust mean/std for center and spread estimates
            # but keep actual percentiles for threshold detection
            robust_mean = robust_stats.get("robust_mean", robust_stats["mean"])
            robust_std = robust_stats.get("robust_std", robust_stats["std"])

            stats = TrainingStatistics(
                mean=robust_mean,  # Use robust mean
                median=robust_stats["median"],
                std=robust_std,  # Use robust std
                min=robust_stats["min"],
                max=robust_stats["max"],
                p25=robust_stats["p25"],
                p50=robust_stats["p50"],
                p75=robust_stats["p75"],
                p90=robust_stats["p90"],
                p95=robust_stats["p95"],
                p99=robust_stats["p99"],
                count=robust_stats["count"],
                coefficient_of_variation=robust_std / (robust_mean + 1e-8) if robust_mean else 0.0,
                typical_range=(robust_stats["p25"], robust_stats["p75"]),
                normal_range=(
                    robust_mean - 2 * robust_std,
                    robust_mean + 2 * robust_std,
                ),
                outlier_bounds=(robust_stats["p25"] - 1.5 * (robust_stats["p75"] - robust_stats["p25"]),
                               robust_stats["p75"] + 1.5 * (robust_stats["p75"] - robust_stats["p25"])),
                # Zero-normal info
                zero_count=zero_info.get("zero_count", 0),
                zero_percentage=zero_info.get("zero_percentage", 0.0),
                is_zero_dominant=zero_info.get("is_zero_dominant", False),
                non_zero_mean=zero_info.get("non_zero_mean", 0.0),
                non_zero_p95=zero_info.get("non_zero_p95", 0.0),
            )

            self.training_statistics[metric] = stats

            if metric in self.ZERO_NORMAL_METRICS:
                self.zero_statistics[metric] = zero_info

    def _train_univariate_model(self, metric_name: str, data: pd.Series) -> bool:
        """Train a univariate Isolation Forest model for a metric.

        Uses the same cleaning pipeline as _calculate_training_statistics to ensure
        statistics and model are trained on identical data distributions.
        """
        # Use unified cleaning pipeline (same as statistics calculation)
        clean_data, zero_info = self._clean_metric_data(data, metric_name)

        # Handle zero-dominant metrics that can't be modeled
        if metric_name in self.ZERO_NORMAL_METRICS and zero_info["is_zero_dominant"]:
            if len(clean_data) < 50:
                self.thresholds[f"{metric_name}_zero_dominant"] = True
                self.thresholds[f"{metric_name}_non_zero_p95"] = zero_info["non_zero_p95"]
                logger.debug(f"{metric_name}: Zero-dominant, using threshold detection")
                return False

        clean_df = pd.DataFrame(clean_data.values, columns=[metric_name])

        # Minimum samples for stable IF training (need enough for tree construction)
        # With n_estimators=200 and default subsample, need at least 256+ samples
        min_univariate_samples = 500
        if len(clean_df) < min_univariate_samples:
            logger.warning(
                f"Insufficient data for {metric_name}: {len(clean_df)} points "
                f"(need {min_univariate_samples} for stable IF training)"
            )
            return False

        try:
            # Train scaler first
            scaler = RobustScaler()
            scaled_data = scaler.fit_transform(clean_df)

            if np.std(scaled_data) < 1e-10:
                logger.warning(f"Constant values for {metric_name}, skipping")
                return False

            # Estimate contamination from data if auto_tune enabled
            estimated_contamination = None
            if self.auto_tune and len(clean_df) >= 500:
                estimated_contamination = self._estimate_contamination(
                    clean_df, method="knee"
                )
                logger.debug(
                    f"{metric_name}: Estimated contamination = {estimated_contamination:.3f}"
                )
                # Store for later use
                if self.estimated_contamination is None:
                    self.estimated_contamination = estimated_contamination

            # Get optimal parameters with estimated contamination
            params = get_service_parameters(
                self.service_name,
                data=clean_df if self.auto_tune else None,
                auto_tune=self.auto_tune,
                estimated_contamination=estimated_contamination,
            )

            model = IsolationForest(**params.to_isolation_forest_params())
            model.fit(scaled_data)

            # Store model and metadata
            self.models[f"{metric_name}_isolation"] = model
            self.scalers[f"{metric_name}_scaler"] = scaler
            self.optimal_params[f"{metric_name}_params"] = params

            # Calculate thresholds
            values = clean_df[metric_name].values
            self.thresholds[f"{metric_name}_p95"] = np.percentile(values, 95)
            self.thresholds[f"{metric_name}_p99"] = np.percentile(values, 99)
            self.thresholds[f"{metric_name}_p90"] = np.percentile(values, 90)
            self.thresholds[f"{metric_name}_median"] = np.percentile(values, 50)

            logger.debug(f"Trained {metric_name} model with {len(clean_df)} samples")
            return True

        except Exception as e:
            logger.error(f"Failed to train {metric_name} model: {e}")
            return False

    def _train_multivariate_model(self, core_metrics_df: pd.DataFrame) -> bool:
        """Train multivariate Isolation Forest model."""
        clean_data = core_metrics_df.dropna().replace([np.inf, -np.inf], np.nan).dropna()

        # Multivariate models need more samples due to curse of dimensionality
        # With 5 features, need at least 1000 samples for stable covariance estimation
        min_multivariate_samples = 1000
        if len(clean_data) < min_multivariate_samples:
            logger.warning(
                f"Insufficient data for multivariate: {len(clean_data)} points "
                f"(need {min_multivariate_samples} for stable multivariate IF)"
            )
            return False

        try:
            # Handle constant columns
            constant_cols = clean_data.std() < 1e-10
            if constant_cols.any():
                # Add noise to error_rate if it's constant near zero
                if "error_rate" in clean_data.columns and constant_cols["error_rate"]:
                    if clean_data["error_rate"].mean() < 1e-6:
                        clean_data = clean_data.copy()
                        clean_data["error_rate"] = clean_data["error_rate"] + np.random.normal(
                            0, 1e-6, len(clean_data)
                        )
                        constant_cols = clean_data.std() < 1e-10

                # Remove remaining constant columns
                if constant_cols.any():
                    clean_data = clean_data.loc[:, ~constant_cols]

            if clean_data.shape[1] < 2:
                logger.warning("Insufficient features for multivariate after cleaning")
                return False

            self.multivariate_feature_names = clean_data.columns.tolist()

            # Get parameters
            params = get_service_parameters(
                self.service_name,
                data=clean_data if self.auto_tune else None,
                auto_tune=self.auto_tune,
            )

            # Use higher n_estimators for multivariate
            mv_params = params.to_isolation_forest_params()
            mv_params["n_estimators"] = min(500, params.n_estimators * 2)

            # Train scaler and model
            scaler = RobustScaler()
            scaled_data = scaler.fit_transform(clean_data)

            model = IsolationForest(**mv_params)
            model.fit(scaled_data)

            self.models["multivariate_detector"] = model
            self.scalers["multivariate_scaler"] = scaler
            self.optimal_params["multivariate_params"] = params

            # Analyze feature correlation
            self.correlation_analysis = self._analyze_feature_correlation(clean_data)
            if self.correlation_analysis.get("highly_correlated_pairs"):
                logger.warning(
                    f"Highly correlated features detected: "
                    f"{len(self.correlation_analysis['highly_correlated_pairs'])} pairs"
                )

            # Compute and store multivariate distribution parameters for drift detection
            # Store mean vector and inverse covariance for Mahalanobis distance
            self.multivariate_feature_order = self.multivariate_feature_names.copy()
            self.multivariate_mean = np.array(clean_data.mean())

            # Compute covariance with regularization for numerical stability
            cov_matrix = clean_data.cov().values
            # Add small regularization to diagonal for numerical stability
            regularization = 1e-6 * np.trace(cov_matrix) / len(cov_matrix)
            cov_matrix += regularization * np.eye(len(cov_matrix))

            try:
                self.multivariate_cov_inv = np.linalg.inv(cov_matrix)
                logger.debug(
                    f"Computed inverse covariance matrix for multivariate drift detection "
                    f"({len(self.multivariate_feature_order)} features)"
                )
            except np.linalg.LinAlgError:
                logger.warning("Could not invert covariance matrix - multivariate drift detection disabled")
                self.multivariate_cov_inv = None

            logger.debug(
                f"Trained multivariate model with {len(scaled_data)} samples, "
                f"{len(self.multivariate_feature_names)} features"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to train multivariate model: {e}")
            return False

    # =========================================================================
    # Calibration and Analysis Methods
    # =========================================================================

    def _calibrate_severity_thresholds(self, training_df: pd.DataFrame) -> None:
        """Calibrate severity thresholds from model score distribution.

        Calculates thresholds for critical, high, medium, low severity based on
        actual score percentiles from the trained models.

        Args:
            training_df: Training data used to compute score distributions.
        """
        for metric_name in self.CORE_METRICS:
            model_key = f"{metric_name}_isolation"
            scaler_key = f"{metric_name}_scaler"

            if model_key not in self.models or scaler_key not in self.scalers:
                continue

            if metric_name not in training_df.columns:
                continue

            try:
                # Get scores on training data
                clean_data, _ = self._clean_metric_data_for_training(
                    training_df[metric_name], metric_name
                )
                if len(clean_data) < 50:
                    continue

                clean_df = pd.DataFrame(clean_data.values, columns=[metric_name])
                scaled_data = self.scalers[scaler_key].transform(clean_df)
                scores = self.models[model_key].decision_function(scaled_data)

                # Calibrate thresholds based on score percentiles
                # Bottom percentiles correspond to most anomalous scores
                self.calibrated_thresholds[metric_name] = CalibratedThresholds(
                    critical=float(np.percentile(scores, 0.1)),   # Bottom 0.1%
                    high=float(np.percentile(scores, 1)),         # Bottom 1%
                    medium=float(np.percentile(scores, 5)),       # Bottom 5%
                    low=float(np.percentile(scores, 10)),         # Bottom 10%
                )

                logger.debug(
                    f"Calibrated thresholds for {metric_name}: "
                    f"critical={self.calibrated_thresholds[metric_name].critical:.3f}"
                )

            except Exception as e:
                logger.debug(f"Could not calibrate thresholds for {metric_name}: {e}")

    def _calibrate_on_validation(
        self, validation_df: pd.DataFrame
    ) -> dict[str, Any]:
        """Calibrate thresholds on held-out validation data.

        This provides more accurate thresholds by using data the model hasn't seen.

        Args:
            validation_df: Validation data (should not overlap with training).

        Returns:
            Validation metrics including false positive rate estimates.
        """
        results: dict[str, Any] = {"metrics": {}, "threshold_adjustments": {}}

        for metric_name in self.CORE_METRICS:
            model_key = f"{metric_name}_isolation"
            scaler_key = f"{metric_name}_scaler"

            if model_key not in self.models or scaler_key not in self.scalers:
                continue

            if metric_name not in validation_df.columns:
                continue

            try:
                clean_data, _ = self._clean_metric_data_for_training(
                    validation_df[metric_name], metric_name
                )
                if len(clean_data) < 20:
                    continue

                clean_df = pd.DataFrame(clean_data.values, columns=[metric_name])
                scaled_data = self.scalers[scaler_key].transform(clean_df)

                # Get predictions and scores
                predictions = self.models[model_key].predict(scaled_data)
                scores = self.models[model_key].decision_function(scaled_data)

                # Calculate validation metrics
                anomaly_count = int((predictions == -1).sum())
                false_positive_rate = anomaly_count / len(predictions)

                # Score distribution on validation data
                mean_score = float(np.mean(scores))
                std_score = float(np.std(scores))

                # Re-calibrate thresholds on validation data
                self.calibrated_thresholds[metric_name] = CalibratedThresholds(
                    critical=float(np.percentile(scores, 0.1)),
                    high=float(np.percentile(scores, 1)),
                    medium=float(np.percentile(scores, 5)),
                    low=float(np.percentile(scores, 10)),
                )

                self.validation_metrics[metric_name] = ValidationMetrics(
                    false_positive_rate=false_positive_rate,
                    detection_rate=1.0 - false_positive_rate,  # Assuming validation is normal
                    mean_score_normal=mean_score,
                    mean_score_anomalous=mean_score - 2 * std_score,  # Estimated
                    threshold_calibrated=True,
                )

                results["metrics"][metric_name] = {
                    "validation_samples": len(clean_data),
                    "false_positive_rate": false_positive_rate,
                    "mean_score": mean_score,
                    "calibrated_thresholds": self.calibrated_thresholds[metric_name].to_dict(),
                }

                # Check if FPR is too high and adjust
                if false_positive_rate > 0.10:
                    logger.warning(
                        f"{metric_name}: High FPR ({false_positive_rate:.1%}) on validation"
                    )
                    results["threshold_adjustments"][metric_name] = "high_fpr_warning"

            except Exception as e:
                logger.debug(f"Validation calibration failed for {metric_name}: {e}")

        return results

    def _analyze_feature_correlation(
        self,
        data: pd.DataFrame,
        threshold: float = 0.8,
    ) -> dict[str, Any]:
        """Analyze feature correlation for multivariate model.

        Returns correlation matrix and warnings for highly correlated pairs.
        High correlation can cause multicollinearity issues in the IF model.

        Args:
            data: Feature data to analyze.
            threshold: Correlation threshold above which to warn.

        Returns:
            Dictionary with correlation matrix and warnings.
        """
        if data.empty or len(data.columns) < 2:
            return {"correlation_matrix": {}, "highly_correlated_pairs": [], "feature_count": 0}

        try:
            corr_matrix = data.corr()
            warnings_list: list[dict[str, Any]] = []

            columns = list(data.columns)
            for i, m1 in enumerate(columns):
                for j, m2 in enumerate(columns):
                    if i < j:
                        corr = abs(corr_matrix.loc[m1, m2])
                        if corr > threshold:
                            warnings_list.append({
                                "metric_1": m1,
                                "metric_2": m2,
                                "correlation": float(corr),
                                "recommendation": f"Consider removing {m2} from multivariate model",
                            })

            return {
                "correlation_matrix": corr_matrix.to_dict(),
                "highly_correlated_pairs": warnings_list,
                "feature_count": len(data.columns),
            }

        except Exception as e:
            logger.debug(f"Correlation analysis failed: {e}")
            return {"correlation_matrix": {}, "highly_correlated_pairs": [], "feature_count": 0}

    def _estimate_contamination(
        self,
        data: pd.DataFrame,
        method: str = "knee",
    ) -> float:
        """Estimate contamination using score distribution analysis.

        Args:
            data: Training data to estimate contamination from.
            method: Estimation method - "knee", "gap", or "percentile".

        Returns:
            Estimated contamination rate (0.01 to 0.15).
        """
        if data.empty or len(data) < 100:
            return 0.05  # Default

        try:
            # Fit preliminary IF model with contamination="auto"
            scaler = RobustScaler()
            scaled_data = scaler.fit_transform(data)

            preliminary_model = IsolationForest(
                contamination="auto",
                n_estimators=100,
                random_state=42,
            )
            preliminary_model.fit(scaled_data)

            # Get decision function scores
            scores = preliminary_model.decision_function(scaled_data)
            sorted_scores = np.sort(scores)

            if method == "knee":
                return self._find_knee_contamination(sorted_scores)
            elif method == "gap":
                return self._find_gap_contamination(sorted_scores)
            else:
                # Default percentile method
                return 0.05

        except Exception as e:
            logger.debug(f"Contamination estimation failed: {e}")
            return 0.05

    def _find_knee_contamination(self, sorted_scores: np.ndarray) -> float:
        """Find contamination using knee/elbow detection in score distribution.

        Args:
            sorted_scores: Sorted IF scores (ascending).

        Returns:
            Estimated contamination rate.
        """
        n = len(sorted_scores)
        if n < 100:
            return 0.05

        # Calculate second derivative to find knee point
        # Knee is where the curve changes from steep to flat
        try:
            # Use bottom 20% of scores for knee detection
            test_range = int(n * 0.2)
            indices = np.arange(test_range)
            scores_subset = sorted_scores[:test_range]

            # Calculate first and second derivatives
            first_deriv = np.gradient(scores_subset)
            second_deriv = np.gradient(first_deriv)

            # Find maximum second derivative (knee point)
            knee_idx = np.argmax(second_deriv)

            # Convert to contamination rate
            contamination = (knee_idx + 1) / n

            # Clamp to reasonable range
            return float(np.clip(contamination, 0.01, 0.15))

        except Exception:
            return 0.05

    def _find_gap_contamination(self, sorted_scores: np.ndarray) -> float:
        """Find contamination by detecting largest gap in score distribution.

        Args:
            sorted_scores: Sorted IF scores (ascending).

        Returns:
            Estimated contamination rate.
        """
        n = len(sorted_scores)
        if n < 100:
            return 0.05

        try:
            # Look for gaps in the bottom 20% of scores
            test_range = int(n * 0.2)
            scores_subset = sorted_scores[:test_range]

            # Calculate gaps between consecutive scores
            gaps = np.diff(scores_subset)

            # Find largest gap
            largest_gap_idx = np.argmax(gaps)

            # Points before the gap are likely anomalies
            contamination = (largest_gap_idx + 1) / n

            # Clamp to reasonable range
            return float(np.clip(contamination, 0.01, 0.15))

        except Exception:
            return 0.05

    def check_drift(self, metrics: dict[str, float]) -> DriftAnalysis:
        """Check for distribution drift from training data.

        Performs both univariate (z-score) and multivariate (Mahalanobis distance)
        drift detection. Multivariate drift can detect covariate shift where
        relationships between features change even if marginals stay stable.

        Args:
            metrics: Current metric values to check.

        Returns:
            DriftAnalysis with drift detection results including multivariate analysis.
        """
        drift_metrics: dict[str, dict[str, Any]] = {}
        max_drift = 0.0

        # =====================================================================
        # Univariate Drift Detection (Z-scores)
        # =====================================================================
        for metric_name, current_value in metrics.items():
            if metric_name not in self.training_statistics:
                continue

            stats = self.training_statistics[metric_name]

            # Calculate z-score using robust statistics
            z_score = abs((current_value - stats.mean) / (stats.std + 1e-8))

            if z_score > 3:  # Beyond 3 sigma
                drift_metrics[metric_name] = {
                    "z_score": float(z_score),
                    "training_mean": stats.mean,
                    "training_std": stats.std,
                    "current_value": current_value,
                    "severity": "high" if z_score > 5 else "medium",
                }
                max_drift = max(max_drift, z_score)

        # =====================================================================
        # Multivariate Drift Detection (Mahalanobis Distance)
        # =====================================================================
        multivariate_drift = False
        mahalanobis_dist = 0.0
        mahalanobis_threshold = 0.0

        if (
            self.multivariate_mean is not None
            and self.multivariate_cov_inv is not None
            and len(self.multivariate_feature_order) > 0
        ):
            # Build feature vector in correct order
            feature_vector = []
            features_available = True

            for feature_name in self.multivariate_feature_order:
                if feature_name in metrics:
                    feature_vector.append(metrics[feature_name])
                else:
                    features_available = False
                    break

            if features_available:
                x = np.array(feature_vector)
                diff = x - self.multivariate_mean

                # Mahalanobis distance: sqrt((x-μ)' Σ^(-1) (x-μ))
                mahalanobis_dist = float(np.sqrt(diff @ self.multivariate_cov_inv @ diff))

                # Chi-squared threshold for multivariate normal at 99.9% confidence
                # For p features, chi2(p, 0.999) ≈ p + 3*sqrt(2p) for large p
                p = len(self.multivariate_feature_order)
                # Using scipy would be better, but approximate for now
                # chi2.ppf(0.999, p) approximation
                mahalanobis_threshold = float(p + 3 * np.sqrt(2 * p) + 3)

                if mahalanobis_dist > mahalanobis_threshold:
                    multivariate_drift = True
                    drift_metrics["_multivariate"] = {
                        "mahalanobis_distance": mahalanobis_dist,
                        "threshold": mahalanobis_threshold,
                        "features_used": self.multivariate_feature_order,
                        "severity": "critical" if mahalanobis_dist > mahalanobis_threshold * 1.5 else "high",
                    }

        # =====================================================================
        # Combined Assessment
        # =====================================================================
        has_univariate_drift = len([k for k in drift_metrics if not k.startswith("_")]) > 0
        has_drift = has_univariate_drift or multivariate_drift

        # Build recommendation
        if multivariate_drift and max_drift > 5:
            recommendation = (
                "CRITICAL: Both univariate AND multivariate drift detected. "
                "Feature relationships have changed significantly. Retrain immediately."
            )
        elif multivariate_drift:
            recommendation = (
                f"CRITICAL: Multivariate drift detected (Mahalanobis={mahalanobis_dist:.1f} > {mahalanobis_threshold:.1f}). "
                "Feature relationships have shifted even if individual metrics look normal. Consider retraining."
            )
        elif max_drift > 5:
            recommendation = "CRITICAL: Significant univariate drift detected. Consider retraining."
        elif max_drift > 3:
            recommendation = "WARNING: Moderate univariate drift detected. Monitor closely."
        else:
            recommendation = "OK: No significant drift detected."

        return DriftAnalysis(
            has_drift=has_drift,
            drift_metrics=drift_metrics,
            overall_drift_score=float(max_drift),
            recommendation=recommendation,
            multivariate_drift=multivariate_drift,
            mahalanobis_distance=mahalanobis_dist,
            multivariate_threshold=mahalanobis_threshold,
        )

    # =========================================================================
    # Helper Methods for Pattern Matching
    # =========================================================================

    def _calculate_metric_ratios(self, metrics: dict[str, float]) -> dict[str, float]:
        """Calculate important metric ratios for pattern detection."""
        app_latency = metrics.get(MetricName.APPLICATION_LATENCY, 0)
        client_latency = metrics.get(MetricName.CLIENT_LATENCY, 0)
        db_latency = metrics.get(MetricName.DATABASE_LATENCY, 0)

        return {
            "client_latency_ratio": client_latency / (app_latency + 1e-8) if app_latency > 0 else 0,
            "db_latency_ratio": db_latency / (app_latency + 1e-8) if app_latency > 0 else 0,
        }

    def _calculate_drop_percent(self, current_rate: float) -> float:
        """Calculate percentage drop from expected traffic."""
        stats = self.training_statistics.get(MetricName.REQUEST_RATE)
        if not stats or stats.mean == 0:
            return 0
        return max(0, (stats.mean - current_rate) / stats.mean * 100)

    def _pattern_matches(
        self,
        metric_levels: dict[str, str],
        ratios: dict[str, float],
        conditions: dict[str, str],
        cascade_analysis: CascadeAnalysis | None = None,
    ) -> bool:
        """Check if current metrics match pattern conditions.

        Args:
            metric_levels: Metric name to level mapping (high, low, normal, etc.)
            ratios: Metric ratios for ratio-based conditions
            conditions: Pattern conditions to check
            cascade_analysis: Optional cascade analysis for dependency-aware patterns
        """
        for metric, condition in conditions.items():
            # Handle dependency context conditions
            if metric == "_dependency_context":
                if not cascade_analysis:
                    return False
                if condition == "upstream_anomaly" and cascade_analysis.cascade_type != "upstream_anomaly":
                    return False
                if condition == "chain_degraded" and cascade_analysis.cascade_type != "chain_degraded":
                    return False
                if condition == "dependencies_healthy" and cascade_analysis.cascade_type != "dependencies_healthy":
                    return False
                continue

            # Handle ratio conditions
            if metric.endswith("_ratio"):
                if ">" in condition:
                    threshold = float(condition.split(">")[-1].strip())
                    if ratios.get(metric, 0) <= threshold:
                        return False
                continue

            # Handle "any" condition (always matches)
            if condition == "any":
                continue

            # Handle level conditions
            level = metric_levels.get(metric, "unknown")

            if (condition == "high" and level not in ("high", "very_high")) or (condition == "very_high" and level != "very_high") or (condition == "low" and level not in ("low", "very_low")) or (condition == "very_low" and level != "very_low") or (condition == "normal" and level != "normal"):
                return False

        return True

    # =========================================================================
    # Training Support Methods (continued)
    # =========================================================================

    def _calculate_pattern_thresholds(self, features_df: pd.DataFrame) -> None:
        """Calculate thresholds for pattern detection."""
        required = [MetricName.REQUEST_RATE, MetricName.APPLICATION_LATENCY, MetricName.ERROR_RATE]

        if not all(m in features_df.columns for m in required):
            return

        self.pattern_thresholds = {}
        for metric in required:
            data = features_df[metric].dropna()
            self.pattern_thresholds[metric] = {
                "p90": np.percentile(data, 90),
                "p95": np.percentile(data, 95),
                "p99": np.percentile(data, 99),
                "median": np.percentile(data, 50),
            }

    def _calculate_feature_importance(self) -> None:
        """Calculate feature importance scores."""
        self.feature_importance = {}

        criticality_map = {
            MetricName.ERROR_RATE: "critical - affects user experience",
            MetricName.APPLICATION_LATENCY: "high - impacts user satisfaction",
            MetricName.REQUEST_RATE: "high - indicates service utilization",
            MetricName.DATABASE_LATENCY: "medium - affects backend performance",
            MetricName.CLIENT_LATENCY: "medium - affects service interactions",
        }

        for metric_name, stats in self.training_statistics.items():
            cv = stats.coefficient_of_variation

            if metric_name == MetricName.ERROR_RATE:
                impact = "critical"
            elif ("latency" in metric_name and cv > 0.3) or (metric_name == MetricName.REQUEST_RATE and cv > 0.5):
                impact = "high"
            elif cv > 0.5:
                impact = "medium"
            else:
                impact = "low"

            self.feature_importance[metric_name] = FeatureImportance(
                variability_score=cv,
                impact_level=impact,
                business_criticality=criticality_map.get(metric_name, "low - monitoring metric"),
            )

    def _get_comparison_data(
        self, anomaly: dict[str, Any], metrics: dict[str, float]  # noqa: ARG002
    ) -> dict[str, Any]:
        """Get comparison data for explainability."""
        comparison = {}

        for metric_name in self.CORE_METRICS:
            if metric_name not in self.training_statistics:
                continue

            stats = self.training_statistics[metric_name]
            current = metrics.get(metric_name, 0.0)

            deviation = (current - stats.mean) / (stats.std + 1e-8)

            comparison[metric_name] = {
                "current": current,
                "training_mean": stats.mean,
                "training_std": stats.std,
                "training_p95": stats.p95,
                "deviation_sigma": deviation,
                "percentile_estimate": self._estimate_percentile(current, stats),
            }

        return comparison

    def _estimate_percentile(self, value: float, stats: TrainingStatistics) -> float:
        """Estimate percentile position of a value."""
        if value <= stats.p25:
            return 25 * (value - stats.min) / (stats.p25 - stats.min + 1e-8)
        elif value <= stats.p50:
            return 25 + 25 * (value - stats.p25) / (stats.p50 - stats.p25 + 1e-8)
        elif value <= stats.p75:
            return 50 + 25 * (value - stats.p50) / (stats.p75 - stats.p50 + 1e-8)
        elif value <= stats.p95:
            return 75 + 20 * (value - stats.p75) / (stats.p95 - stats.p75 + 1e-8)
        else:
            return min(100, 95 + 5 * (value - stats.p95) / (stats.p99 - stats.p95 + 1e-8))

    def _get_business_impact(self, anomaly: dict[str, Any]) -> str:
        """Get business impact description for an anomaly."""
        severity = anomaly.get("severity", "medium")
        anomaly_type = anomaly.get("type", "unknown")

        # Use the centralized business impact function
        return get_business_impact(severity, anomaly_type)

    def _calculate_contextual_severity(
        self,
        anomaly_type: str,
        base_severity: str,
        metrics: dict[str, float],
        timestamp: datetime | None = None,
    ) -> SeverityContext:
        """Calculate severity with contextual adjustments."""

        severity_scores = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        severity_score = severity_scores.get(base_severity, 2)
        adjustments = []

        # Time-based adjustment
        if timestamp:
            hour = timestamp.hour
            is_off_hours = hour < 6 or hour > 22
            is_weekend = timestamp.weekday() >= 5

            if is_off_hours or is_weekend:
                # Only downgrade if not user-facing critical patterns
                critical_patterns = {"system_overload", "traffic_surge_failing", "circuit_breaker_open"}
                if anomaly_type not in critical_patterns:
                    severity_score = max(1, severity_score - 1)
                    time_context = "weekend" if is_weekend else f"off-hours ({timestamp.strftime('%H:%M')})"
                    adjustments.append(f"Downgraded: {time_context}")

        # Error rate escalation
        error_rate = metrics.get(MetricName.ERROR_RATE, 0)
        if error_rate > 0.10:
            severity_score = min(4, severity_score + 1)
            adjustments.append(f"Escalated: error rate {error_rate:.1%} > 10%")

        # Traffic impact multiplier
        request_rate = metrics.get(MetricName.REQUEST_RATE, 0)
        req_stats = self.training_statistics.get(MetricName.REQUEST_RATE)
        if req_stats and request_rate > req_stats.p99:
            severity_score = min(4, severity_score + 1)
            adjustments.append(f"Escalated: traffic at p99+ ({request_rate:.1f} req/s)")

        # Map back to severity string
        severity_map = {1: "low", 2: "medium", 3: "high", 4: "critical"}
        adjusted_severity = severity_map.get(severity_score, "medium")

        return SeverityContext(
            base_severity=base_severity,
            adjusted_severity=adjusted_severity,
            adjustment_reasons=adjustments,
            confidence=0.9 if not adjustments else 0.8,
        )

    def _generate_recommendations(
        self,
        anomaly_type: str,
        severity: str,
        metrics: dict[str, float],
        max_count: int = 5,
    ) -> list[str]:
        """Generate prioritized, actionable recommendations for an anomaly."""
        recommendations = []

        # Get pattern-specific recommendations
        pattern_recs = get_recommendations(anomaly_type, severity, max_count)
        recommendations.extend(pattern_recs)

        # Add metric-specific recommendations
        for metric_name in self.CORE_METRICS:
            stats = self.training_statistics.get(metric_name)
            if not stats:
                continue

            value = metrics.get(metric_name, 0)
            if value > stats.p95:
                metric_key = f"{metric_name}_high"
                metric_recs = get_recommendations(metric_key, severity, 2)
                recommendations.extend(metric_recs)

        # Add contextual recommendations based on ratios
        app_latency = metrics.get(MetricName.APPLICATION_LATENCY, 0)
        client_latency = metrics.get(MetricName.CLIENT_LATENCY, 0)
        db_latency = metrics.get(MetricName.DATABASE_LATENCY, 0)

        if app_latency > 0:
            client_ratio = client_latency / app_latency
            db_ratio = db_latency / app_latency

            if client_ratio > 0.6:
                recommendations.append(
                    "FOCUS: External dependency is primary bottleneck - investigate third-party status"
                )
            if db_ratio > 0.7:
                recommendations.append(
                    "FOCUS: Database is primary bottleneck - check slow query logs and connection pool"
                )

        # Deduplicate and prioritize
        priority_order = [
            "IMMEDIATE", "VERIFY", "CHECK", "INVESTIGATE", "CORRELATE",
            "TIMELINE", "ASSESS", "IDENTIFY", "FOCUS", "CONSIDER",
            "PREPARE", "DECIDE", "REVIEW", "MONITOR",
        ]

        seen = set()
        unique_recs = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recs.append(rec)

        def priority_key(rec: str) -> int:
            for i, prefix in enumerate(priority_order):
                if rec.startswith(prefix):
                    return i
            return len(priority_order)

        sorted_recs = sorted(unique_recs, key=priority_key)
        return sorted_recs[:max_count]

    def save_model(self, directory: str, metadata: dict[str, Any] | None = None) -> Path:
        """Save the trained model to disk.

        Args:
            directory: Directory to save the model.
            metadata: Additional metadata to include.

        Returns:
            Path to the saved model directory.
        """
        model_dir = Path(directory) / self.service_name
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model data
        model_data = {
            "service_name": self.service_name,
            "feature_columns": self.feature_columns,
            "multivariate_feature_names": self.multivariate_feature_names,
            "thresholds": self.thresholds,
            "pattern_thresholds": self.pattern_thresholds,
            "training_statistics": {
                k: {
                    "mean": v.mean, "std": v.std, "median": v.median,
                    "min": v.min, "max": v.max,
                    "p25": v.p25, "p50": v.p50, "p75": v.p75,
                    "p90": v.p90, "p95": v.p95, "p99": v.p99,
                }
                for k, v in self.training_statistics.items()
            },
            "zero_statistics": self.zero_statistics,
            "model_metadata": {**self.model_metadata, **(metadata or {})},
            "auto_tune": self.auto_tune,
            # ML improvements: new fields
            "calibrated_thresholds": {
                k: v.to_dict() for k, v in self.calibrated_thresholds.items()
            },
            "correlation_analysis": self.correlation_analysis,
            "estimated_contamination": self.estimated_contamination,
            "validation_metrics": {
                k: {
                    "false_positive_rate": v.false_positive_rate,
                    "detection_rate": v.detection_rate,
                    "mean_score_normal": v.mean_score_normal,
                    "mean_score_anomalous": v.mean_score_anomalous,
                    "threshold_calibrated": v.threshold_calibrated,
                }
                for k, v in self.validation_metrics.items()
            },
            # Multivariate drift detection parameters
            "multivariate_mean": self.multivariate_mean.tolist() if self.multivariate_mean is not None else None,
            "multivariate_cov_inv": self.multivariate_cov_inv.tolist() if self.multivariate_cov_inv is not None else None,
            "multivariate_feature_order": self.multivariate_feature_order,
        }

        with open(model_dir / "model_data.json", "w") as f:
            json.dump(model_data, f, indent=2, default=str)

        # Save sklearn models
        for name, model in self.models.items():
            joblib.dump(model, model_dir / f"{name}.joblib")

        for name, scaler in self.scalers.items():
            joblib.dump(scaler, model_dir / f"{name}.joblib")

        logger.info(f"Saved model for {self.service_name} to {model_dir}")
        return model_dir

    @classmethod
    def load_model(cls, directory: str, service_name: str) -> SmartboxAnomalyDetector:
        """Load a trained model from disk.

        Args:
            directory: Directory containing saved models.
            service_name: Name of the service to load.

        Returns:
            Loaded SmartboxAnomalyDetector instance.
        """
        model_dir = Path(directory) / service_name

        with open(model_dir / "model_data.json") as f:
            model_data = json.load(f)

        detector = cls(
            service_name=model_data["service_name"],
            auto_tune=model_data.get("auto_tune", True),
        )

        detector.feature_columns = model_data["feature_columns"]
        detector.multivariate_feature_names = model_data.get("multivariate_feature_names", [])
        detector.thresholds = model_data["thresholds"]
        detector.pattern_thresholds = model_data.get("pattern_thresholds", {})
        detector.zero_statistics = model_data.get("zero_statistics", {})
        detector.model_metadata = model_data.get("model_metadata", {})

        # Reconstruct training statistics
        for metric, stats_dict in model_data.get("training_statistics", {}).items():
            mean = stats_dict.get("mean", 0)
            std = stats_dict.get("std", 0)
            median = stats_dict.get("median", mean)
            p95 = stats_dict.get("p95", mean + 2 * std)
            # Use saved values or estimate from available data
            detector.training_statistics[metric] = TrainingStatistics(
                mean=mean,
                median=median,
                std=std,
                min=stats_dict.get("min", mean - 3 * std),
                max=stats_dict.get("max", mean + 3 * std),
                p25=stats_dict.get("p25", mean - 0.67 * std),  # ~25th percentile for normal dist
                p50=stats_dict.get("p50", median),
                p75=stats_dict.get("p75", mean + 0.67 * std),  # ~75th percentile for normal dist
                p90=stats_dict.get("p90", mean + 1.28 * std),  # ~90th percentile for normal dist
                p95=p95,
                p99=stats_dict.get("p99", mean + 2.33 * std),  # ~99th percentile for normal dist
                count=0,
                coefficient_of_variation=std / (mean + 1e-8),
                typical_range=(mean - std, mean + std),
                normal_range=(mean - 2 * std, mean + 2 * std),
                outlier_bounds=(mean - 3 * std, mean + 3 * std),
            )

        # Load sklearn models
        for model_file in model_dir.glob("*_isolation.joblib"):
            name = model_file.stem
            detector.models[name] = joblib.load(model_file)

        if (model_dir / "multivariate_detector.joblib").exists():
            detector.models["multivariate_detector"] = joblib.load(
                model_dir / "multivariate_detector.joblib"
            )

        for scaler_file in model_dir.glob("*_scaler.joblib"):
            name = scaler_file.stem
            detector.scalers[name] = joblib.load(scaler_file)

        # Restore ML improvement fields (with backward compatibility defaults)
        for metric, thresholds_dict in model_data.get("calibrated_thresholds", {}).items():
            detector.calibrated_thresholds[metric] = CalibratedThresholds.from_dict(thresholds_dict)

        detector.correlation_analysis = model_data.get("correlation_analysis", {})
        detector.estimated_contamination = model_data.get("estimated_contamination")

        for metric, val_dict in model_data.get("validation_metrics", {}).items():
            detector.validation_metrics[metric] = ValidationMetrics(
                false_positive_rate=val_dict.get("false_positive_rate", 0.0),
                detection_rate=val_dict.get("detection_rate", 1.0),
                mean_score_normal=val_dict.get("mean_score_normal", 0.0),
                mean_score_anomalous=val_dict.get("mean_score_anomalous", -0.5),
                threshold_calibrated=val_dict.get("threshold_calibrated", False),
            )

        # Restore multivariate drift detection parameters
        mv_mean = model_data.get("multivariate_mean")
        detector.multivariate_mean = np.array(mv_mean) if mv_mean is not None else None

        mv_cov_inv = model_data.get("multivariate_cov_inv")
        detector.multivariate_cov_inv = np.array(mv_cov_inv) if mv_cov_inv is not None else None

        detector.multivariate_feature_order = model_data.get("multivariate_feature_order", [])

        detector._trained = True
        logger.info(f"Loaded model for {service_name} from {model_dir}")
        return detector

    # Aliases for backward compatibility
    @classmethod
    def load_model_secure(cls, directory: str, service_name: str) -> SmartboxAnomalyDetector:
        """Alias for load_model (backward compatibility)."""
        return cls.load_model(directory, service_name)

    def save_model_secure(self, directory: str, metadata: dict[str, Any] | None = None) -> Path:
        """Alias for save_model (backward compatibility)."""
        return self.save_model(directory, metadata)

    def detect_anomalies(self, metrics: dict[str, float]) -> dict[str, Any]:
        """Alias for detect method (backward compatibility).

        Returns just the anomalies dict for compatibility with old code.
        """
        result = self.detect(metrics)
        return result.get("anomalies", {})

    def detect_anomalies_with_context(
        self, metrics: dict[str, float], timestamp: datetime | None = None
    ) -> dict[str, Any]:
        """Detect anomalies with full context (backward compatibility)."""
        result = self.detect(metrics)
        anomalies = result.get("anomalies", {})

        return {
            "service": self.service_name,
            "timestamp": (timestamp or datetime.now()).isoformat(),
            "anomalies": list(anomalies.values()),
            "anomaly_count": len(anomalies),
            "overall_severity": self._get_max_severity(anomalies),
            "current_metrics": metrics,
            "explainable": True,
            "model_metadata": self.model_metadata,
        }

    def _get_max_severity(self, anomalies: dict[str, Any]) -> str:
        """Get the maximum severity from anomalies."""
        severity_order = ["low", "medium", "high", "critical"]
        max_severity = "low"
        for anomaly in anomalies.values():
            sev = anomaly.get("severity", "low")
            if severity_order.index(sev) > severity_order.index(max_severity):
                max_severity = sev
        return max_severity


def create_detector(service_name: str, auto_tune: bool = True) -> SmartboxAnomalyDetector:
    """Factory function to create a detector.

    Args:
        service_name: Name of the service.
        auto_tune: Whether to enable automatic parameter tuning.

    Returns:
        New SmartboxAnomalyDetector instance.
    """
    return SmartboxAnomalyDetector(service_name=service_name, auto_tune=auto_tune)
