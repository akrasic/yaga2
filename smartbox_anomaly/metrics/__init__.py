"""
Metrics module - metric collection and validation.

This module contains:
    - client: VictoriaMetrics client with circuit breaker
    - validation: Metric validation utilities
    - quality: Data quality analysis utilities
"""

from smartbox_anomaly.metrics.client import (
    CircuitBreakerState,
    InferenceMetrics,
    QueryResult,
    VictoriaMetricsClient,
)
from smartbox_anomaly.metrics.quality import (
    DataQualityReport,
    TimeGap,
    analyze_combined_data_quality,
    analyze_data_quality,
    detect_time_gaps,
)
from smartbox_anomaly.metrics.validation import (
    ValidationResult,
    sanitize_metric_value,
    sanitize_metrics,
    validate_error_rate,
    validate_latency,
    validate_metrics,
    validate_request_rate,
    validate_service_name,
    validate_timestamp,
)

__all__ = [
    # Client
    "VictoriaMetricsClient",
    "InferenceMetrics",
    "CircuitBreakerState",
    "QueryResult",
    # Quality analysis
    "DataQualityReport",
    "TimeGap",
    "analyze_data_quality",
    "analyze_combined_data_quality",
    "detect_time_gaps",
    # Validation
    "ValidationResult",
    "validate_metrics",
    "validate_request_rate",
    "validate_latency",
    "validate_error_rate",
    "validate_service_name",
    "validate_timestamp",
    "sanitize_metric_value",
    "sanitize_metrics",
]
