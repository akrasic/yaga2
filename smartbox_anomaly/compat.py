"""
Compatibility layer for legacy imports.

This module provides backwards-compatible imports from the old flat structure.
New code should import directly from the appropriate submodules.

Example (old style - still works):
    from smartbox_anomaly.compat import VictoriaMetricsClient, AnomalyFingerprinter

Example (new style - preferred):
    from smartbox_anomaly.metrics import VictoriaMetricsClient
    from smartbox_anomaly.fingerprinting import AnomalyFingerprinter
"""

from __future__ import annotations

import warnings

# API module exports
from smartbox_anomaly.api import (
    API_SCHEMA_VERSION,
    ActionSummary,
    Anomaly,
    AnomalyBatchRequest,
    AnomalyDetectedPayload,
    AnomalyMetadata,
    ComparisonData,
    CurrentMetrics,
    DetectionContext,
    ErrorPayload,
    FeatureContribution,
    FingerprintingMetadata,
    HeartbeatPayload,
    IncidentListResponse,
    IncidentResolvedPayload,
    IncidentSummary,
    IngestionResponse,
    ResolutionBatchRequest,
    ResolutionDetails,
    # Pydantic models use different naming in api module
    Severity,
    create_anomaly_payload,
    create_resolution_payload,
)

# Re-export everything from submodules for backwards compatibility
# Core module exports
from smartbox_anomaly.core import (
    PERIOD_CONFIDENCE_SCORES,
    PERIOD_FALLBACK_MAP,
    TIME_PERIOD_SUFFIXES,
    AlertType,
    # Constants
    AnomalySeverity,
    APIError,
    CircuitBreakerOpenError,
    ConfigurationError,
    DatabaseError,
    DetectionError,
    DetectionMethod,
    EventType,
    FingerprintingConfig,
    FingerprintingError,
    IncidentAction,
    IncidentNotFoundError,
    IncidentStatus,
    InferenceConfig,
    InferenceError,
    LogContext,
    MetricName,
    MetricsCollectionError,
    MetricsError,
    MetricsValidationError,
    ModelConfig,
    ModelError,
    ModelLoadError,
    ModelNotFoundError,
    ModelTrainingError,
    ModelType,
    ObservabilityConfig,
    ObservabilityServiceError,
    PipelineConfig,
    # Exceptions
    PipelineError,
    ServiceCategory,
    ServiceConfig,
    Thresholds,
    TimePeriod,
    TimePeriodConfig,
    TimePeriodError,
    VictoriaMetricsConfig,
    build_full_service_name,
    calculate_duration_minutes,
    # Logging
    configure_logging,
    discover_available_models,
    ensure_directory,
    extract_base_service_names,
    format_duration,
    generate_anomaly_name,
    generate_correlation_id,
    # Utils
    generate_fingerprint_id,
    generate_incident_id,
    # Config
    get_config,
    get_logger,
    get_model_path,
    get_period_type,
    get_time_period,
    log_event,
    merge_dicts,
    now_iso,
    parse_service_model,
    parse_timestamp,
    quiet_mode,
    reset_config,
    safe_get_nested,
    set_config,
    set_log_level,
    timestamp_age_minutes,
    truncate_string,
    verbose_mode,
)

# Detection module exports
from smartbox_anomaly.detection import (
    ServiceParameters,
    SmartboxAnomalyDetector,
    TimeAwareAnomalyDetector,
    create_detector,
    create_time_aware_detector,
    detect_service_category,
    get_service_parameters,
)

# Fingerprinting module exports
from smartbox_anomaly.fingerprinting import (
    SCHEMA_VERSION,
    AnomalyFingerprinter,
    create_fingerprinter,
)

# Metrics module exports
from smartbox_anomaly.metrics import (
    CircuitBreakerState,
    InferenceMetrics,
    ValidationResult,
    VictoriaMetricsClient,
    sanitize_metric_value,
    sanitize_metrics,
    validate_error_rate,
    validate_latency,
    validate_metrics,
    validate_request_rate,
    validate_service_name,
    validate_timestamp,
)


def _deprecation_warning(old_path: str, new_path: str) -> None:
    """Issue a deprecation warning for old import paths."""
    warnings.warn(
        f"Importing from '{old_path}' is deprecated. "
        f"Use '{new_path}' instead.",
        DeprecationWarning,
        stacklevel=3,
    )


__all__ = [
    # Config
    "get_config",
    "set_config",
    "reset_config",
    "PipelineConfig",
    "VictoriaMetricsConfig",
    "ModelConfig",
    "InferenceConfig",
    "FingerprintingConfig",
    "ObservabilityConfig",
    "ServiceConfig",
    "TimePeriodConfig",
    # Constants
    "AnomalySeverity",
    "IncidentStatus",
    "IncidentAction",
    "TimePeriod",
    "ServiceCategory",
    "ModelType",
    "AlertType",
    "DetectionMethod",
    "MetricName",
    "Thresholds",
    "PERIOD_FALLBACK_MAP",
    "PERIOD_CONFIDENCE_SCORES",
    "TIME_PERIOD_SUFFIXES",
    # Exceptions
    "PipelineError",
    "ModelError",
    "ModelLoadError",
    "ModelTrainingError",
    "ModelNotFoundError",
    "MetricsError",
    "MetricsCollectionError",
    "MetricsValidationError",
    "CircuitBreakerOpenError",
    "DetectionError",
    "InferenceError",
    "TimePeriodError",
    "FingerprintingError",
    "DatabaseError",
    "IncidentNotFoundError",
    "ConfigurationError",
    "APIError",
    "ObservabilityServiceError",
    # Logging
    "configure_logging",
    "get_logger",
    "set_log_level",
    "quiet_mode",
    "verbose_mode",
    "LogContext",
    "log_event",
    "EventType",
    # Utils
    "generate_fingerprint_id",
    "generate_incident_id",
    "generate_correlation_id",
    "parse_service_model",
    "build_full_service_name",
    "extract_base_service_names",
    "get_time_period",
    "get_period_type",
    "calculate_duration_minutes",
    "format_duration",
    "generate_anomaly_name",
    "ensure_directory",
    "get_model_path",
    "discover_available_models",
    "safe_get_nested",
    "merge_dicts",
    "truncate_string",
    "now_iso",
    "parse_timestamp",
    "timestamp_age_minutes",
    # Metrics
    "VictoriaMetricsClient",
    "InferenceMetrics",
    "CircuitBreakerState",
    "ValidationResult",
    "validate_metrics",
    "validate_request_rate",
    "validate_latency",
    "validate_error_rate",
    "validate_service_name",
    "validate_timestamp",
    "sanitize_metric_value",
    "sanitize_metrics",
    # Fingerprinting
    "AnomalyFingerprinter",
    "create_fingerprinter",
    "SCHEMA_VERSION",
    # API
    "API_SCHEMA_VERSION",
    "Severity",
    "CurrentMetrics",
    "FeatureContribution",
    "ComparisonData",
    "AnomalyMetadata",
    "Anomaly",
    "ActionSummary",
    "DetectionContext",
    "FingerprintingMetadata",
    "AnomalyDetectedPayload",
    "ResolutionDetails",
    "IncidentResolvedPayload",
    "ErrorPayload",
    "HeartbeatPayload",
    "AnomalyBatchRequest",
    "ResolutionBatchRequest",
    "IngestionResponse",
    "IncidentSummary",
    "IncidentListResponse",
    "create_anomaly_payload",
    "create_resolution_payload",
    # Detection
    "SmartboxAnomalyDetector",
    "create_detector",
    "TimeAwareAnomalyDetector",
    "create_time_aware_detector",
    "ServiceParameters",
    "get_service_parameters",
    "detect_service_category",
]
