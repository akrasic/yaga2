"""
Centralized configuration management for the ML anomaly detection pipeline.

This module provides a single source of truth for all configuration values,
supporting:
- JSON configuration file (config.json)
- Environment variable overrides
- Programmatic defaults

Configuration is loaded in priority order:
1. Environment variables (highest priority)
2. JSON config file
3. Dataclass defaults (lowest priority)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

logger = logging.getLogger(__name__)

# =============================================================================
# Default Values (module-level constants for use in from_config methods)
# =============================================================================

DEFAULT_CONTAMINATION_RATES: dict[str, float] = {
    "critical": 0.03,
    "standard": 0.05,
    "micro": 0.08,
    "admin": 0.06,
    "core": 0.04,
    "background": 0.08,
}

# 3-period model validation thresholds (all 7 days combined by time-of-day)
# With more training data per period, we can use stricter thresholds
DEFAULT_VALIDATION_THRESHOLDS: dict[str, float] = {
    "business_hours": 0.22,  # Day hours (8-18) - most predictable
    "evening_hours": 0.28,   # Transition period (18-22)
    "night_hours": 0.40,     # Overnight (22-8) - higher natural variance
}

# Per-period contamination multipliers: low-traffic periods have higher natural variance,
# so we expect more apparent "anomalies" that are actually normal variance.
# This reduces false positives during quiet periods.
# With 3-period model, night hours now include all 7 days of night data.
DEFAULT_PERIOD_CONTAMINATION_MULTIPLIERS: dict[str, float] = {
    "business_hours": 1.0,   # Baseline - most predictable traffic
    "evening_hours": 1.3,    # Moderate traffic reduction
    "night_hours": 1.8,      # Low traffic, higher variance (less extreme than 5-period)
}

# Confidence scores reflect reliability of detection per time period
# With 3-period model, we have more data per period so confidence is generally higher
DEFAULT_CONFIDENCE_SCORES: dict[str, float] = {
    "business_hours": 0.9,   # Highest confidence - most data, most predictable
    "evening_hours": 0.85,   # Good confidence - transition period
    "night_hours": 0.8,      # Still good - more data than old weekend_night
}

# Default config file locations (searched in order)
CONFIG_FILE_PATHS = [
    Path("config.json"),  # Current directory
    Path("./config/config.json"),  # Config subdirectory
    Path.home() / ".smartbox" / "config.json",  # User home
    Path("/etc/smartbox/config.json"),  # System-wide
]


def _load_config_file() -> dict[str, Any]:
    """Load configuration from JSON file.

    Searches for config file in standard locations, or uses
    CONFIG_FILE environment variable if set.

    Returns:
        Dictionary of configuration values, or empty dict if no file found.
    """
    # Check environment variable first
    env_config_path = os.getenv("CONFIG_FILE")
    if env_config_path:
        config_path = Path(env_config_path)
        if config_path.exists():
            with config_path.open() as f:
                return json.load(f)
        else:
            # Log warning but don't fail - fall back to defaults
            logger.warning(f"CONFIG_FILE specified but not found: {env_config_path}")

    # Search standard locations
    for path in CONFIG_FILE_PATHS:
        if path.exists():
            with path.open() as f:
                return json.load(f)

    return {}


def _get_env_or_config(
    env_key: str,
    config_dict: dict[str, Any],
    config_key: str,
    default: Any,
    type_cast: type | None = None
) -> Any:
    """Get value from environment, config file, or default (in priority order).

    Args:
        env_key: Environment variable name
        config_dict: Config dictionary section
        config_key: Key within config dictionary
        default: Default value if not found
        type_cast: Optional type to cast the value to

    Returns:
        Configuration value from highest priority source
    """
    # Priority 1: Environment variable
    env_value = os.getenv(env_key)
    if env_value is not None:
        if type_cast is bool:
            return env_value.lower() in ("true", "1", "yes")
        return type_cast(env_value) if type_cast else env_value

    # Priority 2: Config file
    if config_key in config_dict:
        return config_dict[config_key]

    # Priority 3: Default
    return default


@dataclass(frozen=True)
class VictoriaMetricsConfig:
    """Configuration for VictoriaMetrics client."""

    endpoint: str = "https://otel-metrics.production.smartbox.com"
    timeout_seconds: int = 10
    max_retries: int = 3
    pool_connections: int = 20
    pool_maxsize: int = 20
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout_seconds: int = 300
    retry_backoff_factor: float = 0.3
    retry_status_forcelist: tuple[int, ...] = (500, 502, 503, 504)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> VictoriaMetricsConfig:
        """Create configuration from config dict with environment overrides."""
        vm_config = config.get("victoria_metrics", {})
        return cls(
            endpoint=_get_env_or_config("VM_ENDPOINT", vm_config, "endpoint", cls.endpoint),
            timeout_seconds=_get_env_or_config("VM_TIMEOUT", vm_config, "timeout_seconds", cls.timeout_seconds, int),
            max_retries=_get_env_or_config("VM_MAX_RETRIES", vm_config, "max_retries", cls.max_retries, int),
            pool_connections=vm_config.get("pool_connections", cls.pool_connections),
            pool_maxsize=vm_config.get("pool_maxsize", cls.pool_maxsize),
            circuit_breaker_threshold=vm_config.get("circuit_breaker_threshold", cls.circuit_breaker_threshold),
            circuit_breaker_timeout_seconds=vm_config.get("circuit_breaker_timeout_seconds", cls.circuit_breaker_timeout_seconds),
            retry_backoff_factor=vm_config.get("retry_backoff_factor", cls.retry_backoff_factor),
            retry_status_forcelist=tuple(vm_config.get("retry_status_forcelist", cls.retry_status_forcelist)),
        )

    @classmethod
    def from_env(cls) -> VictoriaMetricsConfig:
        """Create configuration from environment variables (backward compatible)."""
        return cls.from_config(_load_config_file())


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for ML model training and inference."""

    models_directory: str = "./smartbox_models/"
    min_training_samples: int = 500  # Minimum for statistical significance in univariate IF
    min_multivariate_samples: int = 1000  # Higher requirement for cross-metric correlations
    default_contamination: float = 0.05
    default_n_estimators: int = 200
    random_state: int = 42

    # Contamination rates by service category
    contamination_rates: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_CONTAMINATION_RATES)
    )

    # Service-specific contamination overrides
    contamination_by_service: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ModelConfig:
        """Create configuration from config dict with environment overrides."""
        model_config = config.get("model", {})
        return cls(
            models_directory=_get_env_or_config("MODELS_DIR", model_config, "models_directory", cls.models_directory),
            min_training_samples=model_config.get("min_training_samples", cls.min_training_samples),
            min_multivariate_samples=model_config.get("min_multivariate_samples", cls.min_multivariate_samples),
            default_contamination=model_config.get("default_contamination", cls.default_contamination),
            default_n_estimators=model_config.get("default_n_estimators", cls.default_n_estimators),
            random_state=model_config.get("random_state", cls.random_state),
            contamination_rates=model_config.get("contamination_by_category", dict(DEFAULT_CONTAMINATION_RATES)),
            contamination_by_service=model_config.get("contamination_by_service", {}),
        )

    @classmethod
    def from_env(cls) -> ModelConfig:
        """Create configuration from environment variables (backward compatible)."""
        return cls.from_config(_load_config_file())


@dataclass(frozen=True)
class InferenceConfig:
    """Configuration for inference pipeline."""

    alerts_directory: str = "./alerts/"
    max_workers: int = 3
    inter_service_delay_seconds: float = 0.2
    check_drift: bool = False

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> InferenceConfig:
        """Create configuration from config dict with environment overrides."""
        inf_config = config.get("inference", {})
        return cls(
            alerts_directory=_get_env_or_config("ALERTS_DIR", inf_config, "alerts_directory", cls.alerts_directory),
            max_workers=_get_env_or_config("MAX_WORKERS", inf_config, "max_workers", cls.max_workers, int),
            inter_service_delay_seconds=inf_config.get("inter_service_delay_seconds", cls.inter_service_delay_seconds),
            check_drift=inf_config.get("check_drift", cls.check_drift),
        )

    @classmethod
    def from_env(cls) -> InferenceConfig:
        """Create configuration from environment variables (backward compatible)."""
        return cls.from_config(_load_config_file())


@dataclass(frozen=True)
class FingerprintingConfig:
    """Configuration for anomaly fingerprinting."""

    db_path: str = "./anomaly_state.db"
    cleanup_max_age_hours: int = 72
    incident_separation_minutes: int = 30

    # Cycle-based incident lifecycle settings
    confirmation_cycles: int = 2       # Cycles needed to confirm incident (before alerting)
    resolution_grace_cycles: int = 3   # Cycles without detection before closing incident

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> FingerprintingConfig:
        """Create configuration from config dict with environment overrides."""
        fp_config = config.get("fingerprinting", {})
        return cls(
            db_path=_get_env_or_config("FINGERPRINT_DB", fp_config, "db_path", cls.db_path),
            cleanup_max_age_hours=fp_config.get("cleanup_max_age_hours", cls.cleanup_max_age_hours),
            incident_separation_minutes=fp_config.get("incident_separation_minutes", cls.incident_separation_minutes),
            confirmation_cycles=fp_config.get("confirmation_cycles", cls.confirmation_cycles),
            resolution_grace_cycles=fp_config.get("resolution_grace_cycles", cls.resolution_grace_cycles),
        )

    @classmethod
    def from_env(cls) -> FingerprintingConfig:
        """Create configuration from environment variables (backward compatible)."""
        return cls.from_config(_load_config_file())


@dataclass(frozen=True)
class DatabaseLatencyRatios:
    """Ratio thresholds for database latency evaluation relative to baseline."""

    info: float = 1.5       # 1.5x baseline = informational
    warning: float = 2.0    # 2x baseline = warning
    high: float = 3.0       # 3x baseline = high
    critical: float = 5.0   # 5x baseline = critical

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatabaseLatencyRatios:
        """Create from config dict."""
        return cls(
            info=data.get("info", 1.5),
            warning=data.get("warning", 2.0),
            high=data.get("high", 3.0),
            critical=data.get("critical", 5.0),
        )


@dataclass(frozen=True)
class RequestRateSurgeConfig:
    """Configuration for traffic surge evaluation."""

    threshold_percent: float = 200.0  # 200% of baseline = surge
    standalone_severity: str = "informational"  # Surge alone is just info
    with_latency_breach_severity: str = "warning"  # Surge + latency issue
    with_error_breach_severity: str = "high"  # Surge + errors

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RequestRateSurgeConfig:
        """Create from config dict."""
        return cls(
            threshold_percent=data.get("threshold_percent", 200.0),
            standalone_severity=data.get("standalone_severity", "informational"),
            with_latency_breach_severity=data.get("with_latency_breach_severity", "warning"),
            with_error_breach_severity=data.get("with_error_breach_severity", "high"),
        )


@dataclass(frozen=True)
class RequestRateCliffConfig:
    """Configuration for traffic cliff evaluation."""

    threshold_percent: float = 50.0  # Below 50% of baseline = cliff
    standalone_severity: str = "warning"  # Cliff alone is warning
    peak_hours_severity: str = "high"  # Cliff during peak hours
    with_upstream_errors_severity: str = "critical"  # Cliff + upstream errors

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RequestRateCliffConfig:
        """Create from config dict."""
        return cls(
            threshold_percent=data.get("threshold_percent", 50.0),
            standalone_severity=data.get("standalone_severity", "warning"),
            peak_hours_severity=data.get("peak_hours_severity", "high"),
            with_upstream_errors_severity=data.get("with_upstream_errors_severity", "critical"),
        )


@dataclass(frozen=True)
class MinExpectedRpsConfig:
    """Minimum expected traffic by time period."""

    business_hours: float = 5.0
    evening_hours: float = 2.0
    night_hours: float = 0.5
    weekend_day: float = 2.0
    weekend_night: float = 0.2

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MinExpectedRpsConfig:
        """Create from config dict."""
        return cls(
            business_hours=data.get("business_hours", 5.0),
            evening_hours=data.get("evening_hours", 2.0),
            night_hours=data.get("night_hours", 0.5),
            weekend_day=data.get("weekend_day", 2.0),
            weekend_night=data.get("weekend_night", 0.2),
        )

    def get_min_for_period(self, time_period: str) -> float:
        """Get minimum expected RPS for a time period."""
        period_map = {
            "business_hours": self.business_hours,
            "evening_hours": self.evening_hours,
            "night_hours": self.night_hours,
            "weekend_day": self.weekend_day,
            "weekend_night": self.weekend_night,
        }
        return period_map.get(time_period, self.business_hours)


@dataclass(frozen=True)
class RequestRateEvaluationConfig:
    """Configuration for request rate (traffic) evaluation."""

    enabled: bool = True
    surge: RequestRateSurgeConfig = field(default_factory=RequestRateSurgeConfig)
    cliff: RequestRateCliffConfig = field(default_factory=RequestRateCliffConfig)
    min_expected_rps: MinExpectedRpsConfig = field(default_factory=MinExpectedRpsConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RequestRateEvaluationConfig:
        """Create from config dict."""
        surge_data = data.get("surge", {})
        cliff_data = data.get("cliff", {})
        min_rps_data = data.get("min_expected_rps", {})

        return cls(
            enabled=data.get("enabled", True),
            surge=RequestRateSurgeConfig.from_dict(surge_data),
            cliff=RequestRateCliffConfig.from_dict(cliff_data),
            min_expected_rps=MinExpectedRpsConfig.from_dict(min_rps_data),
        )


@dataclass(frozen=True)
class ServiceSLOConfig:
    """SLO thresholds for a single service."""

    # Latency thresholds (milliseconds)
    latency_acceptable_ms: float = 500.0    # Anomaly but operationally fine
    latency_warning_ms: float = 800.0       # Approaching SLO
    latency_critical_ms: float = 1000.0     # SLO breach

    # Error rate thresholds (as decimals, e.g., 0.01 = 1%)
    error_rate_acceptable: float = 0.005    # 0.5% - anomaly but fine
    error_rate_warning: float = 0.01        # 1% - approaching SLO
    error_rate_critical: float = 0.02       # 2% - SLO breach
    error_rate_floor: float = 0.0           # Below this, suppress anomaly entirely (0 = use acceptable)

    # Traffic thresholds (requests per second)
    min_traffic_rps: float = 1.0            # Below this = low traffic, relax alerting

    # Busy period relaxation factor (multiply thresholds by this during busy periods)
    busy_period_factor: float = 1.5

    # Database latency evaluation (ratio-based from baseline)
    database_latency_floor_ms: float = 1.0  # Below this, always OK regardless of ratio
    database_latency_ratios: DatabaseLatencyRatios = field(
        default_factory=DatabaseLatencyRatios
    )

    # Request rate evaluation (surge/cliff with correlation-based severity)
    request_rate_evaluation: RequestRateEvaluationConfig = field(
        default_factory=RequestRateEvaluationConfig
    )


@dataclass
class SLOConfig:
    """Configuration for SLO-aware severity evaluation."""

    enabled: bool = True

    # Default SLO thresholds (used when service-specific not defined)
    defaults: ServiceSLOConfig = field(default_factory=ServiceSLOConfig)

    # Service-specific SLO overrides
    service_slos: dict[str, ServiceSLOConfig] = field(default_factory=dict)

    # Busy period configuration
    busy_periods: list[dict[str, str]] = field(default_factory=list)

    # Severity adjustment settings
    allow_downgrade_to_informational: bool = True  # Allow ML anomaly to become informational
    require_slo_breach_for_critical: bool = True   # Critical only if SLO actually breached

    def get_service_slo(self, service_name: str) -> ServiceSLOConfig:
        """Get SLO config for a service, falling back to defaults."""
        return self.service_slos.get(service_name, self.defaults)

    def is_busy_period(self, timestamp: datetime | None = None) -> bool:
        """Check if current time is within a configured busy period."""
        from datetime import datetime as dt
        if timestamp is None:
            timestamp = dt.now()

        for period in self.busy_periods:
            try:
                start = dt.fromisoformat(period.get('start', ''))
                end = dt.fromisoformat(period.get('end', ''))
                if start <= timestamp <= end:
                    return True
            except (ValueError, TypeError):
                continue
        return False

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> SLOConfig:
        """Create SLO configuration from config dict."""
        slo_config = config.get("slos", {})

        if not slo_config:
            # Return disabled config if no SLOs defined
            return cls(enabled=False)

        # Parse default thresholds
        defaults_dict = slo_config.get("defaults", {})
        default_db_ratios = DatabaseLatencyRatios.from_dict(
            defaults_dict.get("database_latency_ratios", {})
        )
        default_request_rate = RequestRateEvaluationConfig.from_dict(
            defaults_dict.get("request_rate_evaluation", {})
        )
        defaults = ServiceSLOConfig(
            latency_acceptable_ms=defaults_dict.get("latency_acceptable_ms", 500.0),
            latency_warning_ms=defaults_dict.get("latency_warning_ms", 800.0),
            latency_critical_ms=defaults_dict.get("latency_critical_ms", 1000.0),
            error_rate_acceptable=defaults_dict.get("error_rate_acceptable", 0.005),
            error_rate_warning=defaults_dict.get("error_rate_warning", 0.01),
            error_rate_critical=defaults_dict.get("error_rate_critical", 0.02),
            error_rate_floor=defaults_dict.get("error_rate_floor", 0.0),
            min_traffic_rps=defaults_dict.get("min_traffic_rps", 1.0),
            busy_period_factor=defaults_dict.get("busy_period_factor", 1.5),
            database_latency_floor_ms=defaults_dict.get("database_latency_floor_ms", 1.0),
            database_latency_ratios=default_db_ratios,
            request_rate_evaluation=default_request_rate,
        )

        # Parse service-specific SLOs
        service_slos = {}
        for service_name, svc_config in slo_config.get("services", {}).items():
            # Parse database latency ratios for this service (fall back to defaults)
            svc_db_ratios_dict = svc_config.get("database_latency_ratios")
            if svc_db_ratios_dict:
                svc_db_ratios = DatabaseLatencyRatios.from_dict(svc_db_ratios_dict)
            else:
                svc_db_ratios = defaults.database_latency_ratios

            # Parse request rate evaluation for this service (merge with defaults)
            svc_request_rate_dict = svc_config.get("request_rate_evaluation")
            if svc_request_rate_dict:
                # Merge service config with defaults for partial overrides
                merged_surge = {**defaults.request_rate_evaluation.surge.__dict__}
                if "surge" in svc_request_rate_dict:
                    merged_surge.update(svc_request_rate_dict["surge"])

                merged_cliff = {**defaults.request_rate_evaluation.cliff.__dict__}
                if "cliff" in svc_request_rate_dict:
                    merged_cliff.update(svc_request_rate_dict["cliff"])

                merged_min_rps = {**defaults.request_rate_evaluation.min_expected_rps.__dict__}
                if "min_expected_rps" in svc_request_rate_dict:
                    merged_min_rps.update(svc_request_rate_dict["min_expected_rps"])

                svc_request_rate = RequestRateEvaluationConfig(
                    enabled=svc_request_rate_dict.get("enabled", defaults.request_rate_evaluation.enabled),
                    surge=RequestRateSurgeConfig.from_dict(merged_surge),
                    cliff=RequestRateCliffConfig.from_dict(merged_cliff),
                    min_expected_rps=MinExpectedRpsConfig.from_dict(merged_min_rps),
                )
            else:
                svc_request_rate = defaults.request_rate_evaluation

            service_slos[service_name] = ServiceSLOConfig(
                latency_acceptable_ms=svc_config.get("latency_acceptable_ms", defaults.latency_acceptable_ms),
                latency_warning_ms=svc_config.get("latency_warning_ms", defaults.latency_warning_ms),
                latency_critical_ms=svc_config.get("latency_critical_ms", defaults.latency_critical_ms),
                error_rate_acceptable=svc_config.get("error_rate_acceptable", defaults.error_rate_acceptable),
                error_rate_warning=svc_config.get("error_rate_warning", defaults.error_rate_warning),
                error_rate_critical=svc_config.get("error_rate_critical", defaults.error_rate_critical),
                error_rate_floor=svc_config.get("error_rate_floor", defaults.error_rate_floor),
                min_traffic_rps=svc_config.get("min_traffic_rps", defaults.min_traffic_rps),
                busy_period_factor=svc_config.get("busy_period_factor", defaults.busy_period_factor),
                database_latency_floor_ms=svc_config.get(
                    "database_latency_floor_ms", defaults.database_latency_floor_ms
                ),
                database_latency_ratios=svc_db_ratios,
                request_rate_evaluation=svc_request_rate,
            )

        return cls(
            enabled=slo_config.get("enabled", True),
            defaults=defaults,
            service_slos=service_slos,
            busy_periods=slo_config.get("busy_periods", []),
            allow_downgrade_to_informational=slo_config.get("allow_downgrade_to_informational", True),
            require_slo_breach_for_critical=slo_config.get("require_slo_breach_for_critical", True),
        )


@dataclass(frozen=True)
class ExcludedPeriod:
    """A time period to exclude from training data (e.g., holidays, incidents)."""

    start: str  # ISO date string: "2025-12-20"
    end: str    # ISO date string: "2026-01-05"
    reason: str = ""  # Optional description
    model_variant: str = "holiday"  # Which model variant this data belongs to

    def contains(self, timestamp: datetime) -> bool:
        """Check if a timestamp falls within this excluded period."""
        from datetime import datetime as dt
        try:
            start_dt = dt.fromisoformat(self.start)
            end_dt = dt.fromisoformat(self.end)
            # Handle both date-only and datetime strings
            if start_dt.tzinfo is None and timestamp.tzinfo is not None:
                timestamp = timestamp.replace(tzinfo=None)
            return start_dt <= timestamp <= end_dt
        except (ValueError, TypeError):
            return False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExcludedPeriod:
        """Create from config dict."""
        return cls(
            start=data.get("start", ""),
            end=data.get("end", ""),
            reason=data.get("reason", ""),
            model_variant=data.get("model_variant", "holiday"),
        )


@dataclass(frozen=True)
class ModelVariantConfig:
    """Configuration for a model variant (e.g., baseline, holiday)."""

    name: str
    description: str = ""
    weight_factor: float = 1.0  # Weight multiplier for this variant's data

    @classmethod
    def from_dict(cls, data: dict[str, Any], name: str) -> ModelVariantConfig:
        """Create from config dict."""
        return cls(
            name=name,
            description=data.get("description", ""),
            weight_factor=data.get("weight_factor", 1.0),
        )


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for model training pipeline.

    Supports:
    - Extended lookback periods (e.g., 60 days)
    - Excluded periods (e.g., holidays) for filtering
    - Dual-model training (baseline vs holiday variants)
    - Temporal train/validation split
    """

    # Data collection
    lookback_days: int = 60  # Extended from 30 to support dual-model
    min_data_points: int = 2000

    # Temporal validation split
    validation_fraction: float = 0.2

    # Contamination estimation
    contamination_method: str = "knee"  # "knee" or "gap"
    contamination_min_samples: int = 100
    contamination_fallback: float = 0.05

    # Threshold calibration
    threshold_calibration_enabled: bool = True
    threshold_percentiles: dict[str, float] = field(default_factory=lambda: {
        "critical": 0.1,
        "high": 1.0,
        "medium": 5.0,
        "low": 10.0,
    })

    # Drift detection at training time
    drift_detection_enabled: bool = False
    drift_z_score_warning: float = 3.0
    drift_z_score_critical: float = 5.0

    # Excluded periods (holidays, incidents, etc.)
    excluded_periods: tuple[ExcludedPeriod, ...] = field(default_factory=tuple)

    # Model variants configuration
    train_baseline_model: bool = True   # Train on normal (non-excluded) data
    train_holiday_model: bool = True    # Train on excluded period data
    min_excluded_days_for_variant: int = 7  # Minimum days to train holiday variant

    # Holiday data weighting (for future years)
    holiday_data_weight: float = 0.3  # Reduce influence of holiday data in baseline

    def is_in_excluded_period(self, timestamp: datetime) -> tuple[bool, str | None]:
        """Check if timestamp falls within any excluded period.

        Returns:
            Tuple of (is_excluded, variant_name or None)
        """
        for period in self.excluded_periods:
            if period.contains(timestamp):
                return True, period.model_variant
        return False, None

    def get_excluded_period_days(self) -> int:
        """Calculate total days covered by excluded periods."""
        from datetime import datetime as dt
        total_days = 0
        for period in self.excluded_periods:
            try:
                start = dt.fromisoformat(period.start)
                end = dt.fromisoformat(period.end)
                total_days += (end - start).days + 1
            except (ValueError, TypeError):
                continue
        return total_days

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> TrainingConfig:
        """Create configuration from config dict."""
        training_config = config.get("training", {})

        # Parse excluded periods
        excluded_list = training_config.get("excluded_periods", [])
        excluded_periods = tuple(
            ExcludedPeriod.from_dict(p) for p in excluded_list
        )

        # Parse contamination estimation
        contam_config = training_config.get("contamination_estimation", {})

        # Parse threshold calibration
        threshold_config = training_config.get("threshold_calibration", {})

        # Parse drift detection
        drift_config = training_config.get("drift_detection", {})

        # Parse model variants config
        variants_config = training_config.get("model_variants", {})

        return cls(
            lookback_days=training_config.get("lookback_days", 60),
            min_data_points=training_config.get("min_data_points", 2000),
            validation_fraction=training_config.get("validation_fraction", 0.2),
            contamination_method=contam_config.get("method", "knee"),
            contamination_min_samples=contam_config.get("min_samples", 100),
            contamination_fallback=contam_config.get("fallback", 0.05),
            threshold_calibration_enabled=threshold_config.get("enabled", True),
            threshold_percentiles=threshold_config.get("percentiles", {
                "critical": 0.1, "high": 1.0, "medium": 5.0, "low": 10.0
            }),
            drift_detection_enabled=drift_config.get("enabled", False),
            drift_z_score_warning=drift_config.get("z_score_warning", 3.0),
            drift_z_score_critical=drift_config.get("z_score_critical", 5.0),
            excluded_periods=excluded_periods,
            train_baseline_model=variants_config.get("train_baseline", True),
            train_holiday_model=variants_config.get("train_holiday", True),
            min_excluded_days_for_variant=variants_config.get("min_excluded_days", 7),
            holiday_data_weight=variants_config.get("holiday_data_weight", 0.3),
        )


@dataclass(frozen=True)
class ObservabilityConfig:
    """Configuration for observability service integration."""

    base_url: str = "http://localhost:8000"
    anomalies_endpoint: str = "/api/anomalies/batch"
    resolutions_endpoint: str = "/api/incidents/resolve"
    request_timeout_seconds: int = 5
    enabled: bool = True

    @property
    def anomalies_url(self) -> str:
        """Full URL for anomalies endpoint."""
        return f"{self.base_url}{self.anomalies_endpoint}"

    @property
    def resolutions_url(self) -> str:
        """Full URL for resolutions endpoint."""
        return f"{self.base_url}{self.resolutions_endpoint}"

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ObservabilityConfig:
        """Create configuration from config dict with environment overrides."""
        obs_config = config.get("observability_api", {})
        return cls(
            base_url=_get_env_or_config("OBSERVABILITY_URL", obs_config, "base_url", cls.base_url),
            anomalies_endpoint=obs_config.get("anomalies_endpoint", cls.anomalies_endpoint),
            resolutions_endpoint=obs_config.get("resolutions_endpoint", cls.resolutions_endpoint),
            request_timeout_seconds=obs_config.get("request_timeout_seconds", cls.request_timeout_seconds),
            enabled=_get_env_or_config("OBSERVABILITY_ENABLED", obs_config, "enabled", cls.enabled, bool),
        )

    @classmethod
    def from_env(cls) -> ObservabilityConfig:
        """Create configuration from environment variables (backward compatible)."""
        return cls.from_config(_load_config_file())


@dataclass(frozen=True)
class CorrelationConfig:
    """Configuration for alert correlation (grouping related anomalies)."""

    enabled: bool = False  # Disabled by default until fully implemented
    window_seconds: int = 300  # 5 minute correlation window
    primary_selection: str = "highest_confidence"  # How to select primary anomaly
    min_anomalies_to_correlate: int = 2  # Minimum anomalies to trigger correlation

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CorrelationConfig:
        """Create from config dict."""
        return cls(
            enabled=data.get("enabled", False),
            window_seconds=data.get("window_seconds", 300),
            primary_selection=data.get("primary_selection", "highest_confidence"),
            min_anomalies_to_correlate=data.get("min_anomalies_to_correlate", 2),
        )


@dataclass(frozen=True)
class AlertingConfig:
    """Configuration for alert filtering and correlation.

    This controls which anomalies are sent to the Web API and how they are grouped.
    """

    # Severity threshold: only anomalies at or above this level are sent to API
    # Valid values: "low", "medium", "high", "critical"
    # Default "low" means all anomalies are sent (current behavior)
    severity_threshold: str = "low"

    # Whether to log anomalies below threshold (for analytics)
    log_below_threshold: bool = True

    # Log level for below-threshold anomalies
    below_threshold_log_level: str = "INFO"

    # Non-alerting patterns: these patterns are logged but never sent to API
    # regardless of severity (e.g., "healthy" patterns that indicate normal operation)
    # Uses new standardized naming convention: {metric}_{state}_{modifier}
    non_alerting_patterns: frozenset[str] = frozenset({
        "request_rate_surge_healthy",  # System handling load well
    })

    # Alert correlation configuration
    correlation: CorrelationConfig = field(default_factory=CorrelationConfig)

    # Pattern name aliases for backward compatibility (old name -> new name)
    _PATTERN_ALIASES: ClassVar[dict[str, str]] = {
        "traffic_surge_healthy": "request_rate_surge_healthy",
        "traffic_surge_degrading": "request_rate_surge_degrading",
        "traffic_surge_failing": "request_rate_surge_failing",
        "traffic_cliff": "request_rate_cliff",
        "elevated_errors": "error_rate_elevated",
        "fast_rejection": "error_rate_fast_rejection",
        "fast_failure": "error_rate_fast_failure",
        "partial_rejection": "error_rate_partial_rejection",
        "downstream_cascade": "dependency_latency_cascade",
        "internal_bottleneck": "application_latency_bottleneck",
        "database_bottleneck": "database_latency_bottleneck",
        "database_degradation": "database_latency_degraded",
    }

    @classmethod
    def _normalize_pattern_name(cls, pattern_name: str) -> str:
        """Normalize a pattern name using aliases for backward compatibility."""
        return cls._PATTERN_ALIASES.get(pattern_name, pattern_name)

    def should_alert(self, severity: str, pattern_name: str | None = None) -> bool:
        """Determine if an anomaly should be sent to the Web API.

        Args:
            severity: The anomaly severity level.
            pattern_name: Optional pattern name to check against non-alerting patterns.

        Returns:
            True if the anomaly should be alerted, False otherwise.
        """
        # Check non-alerting patterns first (normalize for backward compatibility)
        if pattern_name:
            normalized_name = self._normalize_pattern_name(pattern_name)
            if normalized_name in self.non_alerting_patterns:
                return False

        # Check severity threshold
        severity_order = ["none", "low", "medium", "high", "critical"]

        try:
            severity_idx = severity_order.index(severity.lower())
            threshold_idx = severity_order.index(self.severity_threshold.lower())
            return severity_idx >= threshold_idx
        except ValueError:
            # Unknown severity - default to alerting
            return True

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> AlertingConfig:
        """Create configuration from config dict."""
        alerting_config = config.get("alerting", {})

        # Parse non-alerting patterns (normalize for backward compatibility)
        non_alerting = alerting_config.get("non_alerting_patterns", list(cls.non_alerting_patterns))
        if isinstance(non_alerting, list):
            # Normalize all pattern names in case old names are used in config
            non_alerting = frozenset(cls._normalize_pattern_name(p) for p in non_alerting)
        else:
            non_alerting = frozenset(cls._normalize_pattern_name(p) for p in non_alerting)

        # Parse correlation config
        correlation_dict = alerting_config.get("correlation", {})
        correlation = CorrelationConfig.from_dict(correlation_dict)

        return cls(
            severity_threshold=alerting_config.get("severity_threshold", cls.severity_threshold),
            log_below_threshold=alerting_config.get("log_below_threshold", cls.log_below_threshold),
            below_threshold_log_level=alerting_config.get("below_threshold_log_level", cls.below_threshold_log_level),
            non_alerting_patterns=non_alerting,
            correlation=correlation,
        )


@dataclass(frozen=True)
class ServiceConfig:
    """Service-specific configuration and classification."""

    # Critical services - high traffic, low contamination
    critical_services: frozenset[str] = frozenset({
        "booking", "search", "mobile-api", "shire-api"
    })

    # Standard services - medium traffic
    standard_services: frozenset[str] = frozenset({
        "friday", "gambit", "titan", "r2d2"
    })

    # Micro services - lower traffic, higher contamination tolerance
    micro_services: frozenset[str] = frozenset({"fa5"})

    # Admin services
    admin_services: frozenset[str] = frozenset({
        "m2-fr-adm", "m2-it-adm", "m2-bb-adm"
    })

    # Core services
    core_services: frozenset[str] = frozenset({
        "m2-bb", "m2-fr", "m2-it"
    })

    def get_category(self, service_name: str) -> str:
        """Get the category for a service name."""
        service_lower = service_name.lower()

        if service_lower in self.critical_services:
            return "critical"
        if service_lower in self.standard_services:
            return "standard"
        if service_lower in self.micro_services:
            return "micro"
        if service_lower in self.admin_services:
            return "admin"
        if service_lower in self.core_services:
            return "core"

        # Pattern-based detection for unknown services
        return self._detect_category_from_pattern(service_lower)

    def _detect_category_from_pattern(self, service_name: str) -> str:
        """Detect service category from naming patterns."""
        if any(p in service_name for p in ("api", "gateway", "proxy")):
            return "critical"
        if any(p in service_name for p in ("admin", "adm", "mgmt")):
            return "admin"
        if any(p in service_name for p in ("worker", "job", "task", "queue")):
            return "background"
        if any(p in service_name for p in ("micro", "util", "helper")):
            return "micro"
        if any(p in service_name for p in ("m2-", "core", "platform")):
            return "core"
        return "standard"


@dataclass(frozen=True)
class TimePeriodConfig:
    """Configuration for time-aware anomaly detection."""

    # Time period definitions (hour ranges)
    business_hours: tuple[int, int] = (8, 18)
    evening_hours: tuple[int, int] = (18, 22)
    night_hours: tuple[int, int] = (22, 6)
    weekend_day: tuple[int, int] = (8, 22)
    weekend_night: tuple[int, int] = (22, 8)

    # Minimum samples required per period
    min_samples_weekday: int = 200
    min_samples_weekend: int = 100
    min_samples_micro: int = 50

    # Validation thresholds by period
    default_thresholds: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_VALIDATION_THRESHOLDS)
    )

    # Confidence scores by period
    confidence_scores: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_CONFIDENCE_SCORES)
    )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> TimePeriodConfig:
        """Create configuration from config dict."""
        tp_config = config.get("time_periods", {})

        # Parse time period definitions
        def get_period(name: str, default: tuple[int, int]) -> tuple[int, int]:
            period = tp_config.get(name, {})
            if isinstance(period, dict):
                return (period.get("start", default[0]), period.get("end", default[1]))
            return default

        return cls(
            business_hours=get_period("business_hours", cls.business_hours),
            evening_hours=get_period("evening_hours", cls.evening_hours),
            night_hours=get_period("night_hours", cls.night_hours),
            weekend_day=get_period("weekend_day", cls.weekend_day),
            weekend_night=get_period("weekend_night", cls.weekend_night),
            min_samples_weekday=tp_config.get("min_samples", {}).get("weekday", {}).get("standard", cls.min_samples_weekday),
            min_samples_weekend=tp_config.get("min_samples", {}).get("weekend", {}).get("standard", cls.min_samples_weekend),
            min_samples_micro=tp_config.get("min_samples", {}).get("weekday", {}).get("micro", cls.min_samples_micro),
            default_thresholds=tp_config.get("validation_thresholds", dict(DEFAULT_VALIDATION_THRESHOLDS)),
            confidence_scores=tp_config.get("confidence_scores", dict(DEFAULT_CONFIDENCE_SCORES)),
        )


@dataclass(frozen=True)
class DetectionThresholdConfig:
    """Configuration for anomaly detection thresholds."""

    # Severity score thresholds (Isolation Forest scores)
    severity_critical: float = -0.6
    severity_high: float = -0.3
    severity_medium: float = -0.1

    # Error rate thresholds
    error_rate_critical: float = 0.05
    error_rate_high: float = 0.02
    error_rate_very_high: float = 0.20

    # Latency thresholds (ms)
    latency_critical_ms: float = 2000.0
    latency_high_ms: float = 1000.0

    # Ratio thresholds
    dependency_latency_bottleneck_ratio: float = 0.6
    database_bottleneck_ratio: float = 0.7
    traffic_cliff_ratio: float = 0.3

    # Outlier percentiles
    outlier_lower_percentile: float = 0.001
    outlier_upper_percentile: float = 0.999

    # Validation limits
    max_request_rate: float = 1_000_000.0
    max_latency_ms: float = 300_000.0
    constant_value_threshold: float = 1e-10

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> DetectionThresholdConfig:
        """Create configuration from config dict."""
        dt_config = config.get("detection_thresholds", {})
        severity = dt_config.get("severity_scores", {})
        errors = dt_config.get("error_rates", {})
        latency = dt_config.get("latency_ms", {})
        ratios = dt_config.get("ratios", {})
        outliers = dt_config.get("outlier_percentiles", {})
        validation = dt_config.get("validation", {})

        return cls(
            severity_critical=severity.get("critical", cls.severity_critical),
            severity_high=severity.get("high", cls.severity_high),
            severity_medium=severity.get("medium", cls.severity_medium),
            error_rate_critical=errors.get("critical", cls.error_rate_critical),
            error_rate_high=errors.get("high", cls.error_rate_high),
            error_rate_very_high=errors.get("very_high", cls.error_rate_very_high),
            latency_critical_ms=latency.get("critical", cls.latency_critical_ms),
            latency_high_ms=latency.get("high", cls.latency_high_ms),
            dependency_latency_bottleneck_ratio=ratios.get("dependency_latency_bottleneck", cls.dependency_latency_bottleneck_ratio),
            database_bottleneck_ratio=ratios.get("database_latency_bottleneck", cls.database_bottleneck_ratio),
            traffic_cliff_ratio=ratios.get("traffic_cliff_threshold", cls.traffic_cliff_ratio),
            outlier_lower_percentile=outliers.get("lower", cls.outlier_lower_percentile),
            outlier_upper_percentile=outliers.get("upper", cls.outlier_upper_percentile),
            max_request_rate=validation.get("max_request_rate", cls.max_request_rate),
            max_latency_ms=validation.get("max_latency_ms", cls.max_latency_ms),
            constant_value_threshold=validation.get("constant_value_threshold", cls.constant_value_threshold),
        )


@dataclass(frozen=True)
class EnvoyEnrichmentConfig:
    """Configuration for Envoy edge/ingress metrics enrichment.

    Envoy enrichment adds edge-level context to anomaly alerts by querying
    Mimir for Envoy proxy metrics (request rates by status class, latency
    percentiles, active connections).
    """

    enabled: bool = False
    mimir_endpoint: str = "https://mimir.sbxtest.net/prometheus"
    lookback_minutes: int = 5
    timeout_seconds: int = 10

    # OTel service name to Envoy cluster name mapping
    cluster_mapping: dict[str, str] = field(default_factory=lambda: {
        "booking": "booking",
        "search": "search_k8s",
        "mobile-api": "mobile-api",
        "shire-api": "shireapi_cluster",
        "fa5": "fa5-public",
        "titan": "titan",
        "friday": "friday",
    })

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> EnvoyEnrichmentConfig:
        """Create configuration from config dict."""
        envoy_config = config.get("envoy_enrichment", {})
        return cls(
            enabled=envoy_config.get("enabled", False),
            mimir_endpoint=envoy_config.get("mimir_endpoint", "https://mimir.sbxtest.net/prometheus"),
            lookback_minutes=envoy_config.get("lookback_minutes", 5),
            timeout_seconds=envoy_config.get("timeout_seconds", 10),
            cluster_mapping=envoy_config.get("cluster_mapping", {
                "booking": "booking",
                "search": "search_k8s",
                "mobile-api": "mobile-api",
                "shire-api": "shireapi_cluster",
                "fa5": "fa5-public",
                "titan": "titan",
                "friday": "friday",
            }),
        )


@dataclass
class PipelineConfig:
    """Root configuration aggregating all sub-configurations."""

    victoria_metrics: VictoriaMetricsConfig = field(default_factory=VictoriaMetricsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    fingerprinting: FingerprintingConfig = field(default_factory=FingerprintingConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    service: ServiceConfig = field(default_factory=ServiceConfig)
    time_period: TimePeriodConfig = field(default_factory=TimePeriodConfig)
    detection_thresholds: DetectionThresholdConfig = field(default_factory=DetectionThresholdConfig)
    slo: SLOConfig = field(default_factory=lambda: SLOConfig(enabled=False))
    alerting: AlertingConfig = field(default_factory=AlertingConfig)
    envoy_enrichment: EnvoyEnrichmentConfig = field(default_factory=EnvoyEnrichmentConfig)

    # Track which config file was loaded (if any)
    config_file_path: str | None = None

    @classmethod
    def from_file(cls, file_path: str | Path) -> PipelineConfig:
        """Load configuration from a specific JSON file.

        Args:
            file_path: Path to the JSON configuration file.

        Returns:
            PipelineConfig instance loaded from the file.

        Raises:
            FileNotFoundError: If the config file doesn't exist.
            json.JSONDecodeError: If the file contains invalid JSON.
        """
        path = Path(file_path)
        with path.open() as f:
            config_dict = json.load(f)
        return cls.from_config(config_dict, config_file_path=str(path))

    @classmethod
    def from_config(cls, config: dict[str, Any], config_file_path: str | None = None) -> PipelineConfig:
        """Create full configuration from config dictionary.

        Args:
            config: Configuration dictionary (typically loaded from JSON).
            config_file_path: Optional path to the config file (for tracking).

        Returns:
            PipelineConfig instance.
        """
        return cls(
            victoria_metrics=VictoriaMetricsConfig.from_config(config),
            model=ModelConfig.from_config(config),
            training=TrainingConfig.from_config(config),
            inference=InferenceConfig.from_config(config),
            fingerprinting=FingerprintingConfig.from_config(config),
            observability=ObservabilityConfig.from_config(config),
            service=ServiceConfig(),  # ServiceConfig doesn't change from file yet
            time_period=TimePeriodConfig.from_config(config),
            detection_thresholds=DetectionThresholdConfig.from_config(config),
            slo=SLOConfig.from_config(config),
            alerting=AlertingConfig.from_config(config),
            envoy_enrichment=EnvoyEnrichmentConfig.from_config(config),
            config_file_path=config_file_path,
        )

    @classmethod
    def from_env(cls) -> PipelineConfig:
        """Create full configuration from config file and environment variables.

        Searches for config file in standard locations, then applies
        environment variable overrides.
        """
        config_dict = _load_config_file()

        # Track which file was loaded
        config_path = None
        env_config = os.getenv("CONFIG_FILE")
        if env_config and Path(env_config).exists():
            config_path = env_config
        else:
            for path in CONFIG_FILE_PATHS:
                if path.exists():
                    config_path = str(path)
                    break

        return cls.from_config(config_dict, config_file_path=config_path)

    @classmethod
    def default(cls) -> PipelineConfig:
        """Create configuration with all defaults (no file loading)."""
        return cls()


# Global configuration instance - can be overridden for testing
_config: PipelineConfig | None = None


def get_config() -> PipelineConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = PipelineConfig.from_env()
    return _config


def set_config(config: PipelineConfig) -> None:
    """Set the global configuration instance (useful for testing)."""
    global _config
    _config = config


def reset_config() -> None:
    """Reset configuration to be reloaded on next access."""
    global _config
    _config = None
