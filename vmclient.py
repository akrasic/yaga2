"""
Backward compatibility stub for vmclient.

This module redirects imports to the new package structure.
New code should import from smartbox_anomaly.metrics instead.
"""

from smartbox_anomaly.core.exceptions import MetricsCollectionError
from smartbox_anomaly.metrics import (
    CircuitBreakerState,
    InferenceMetrics,
    QueryResult,
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

__all__ = [
    "VictoriaMetricsClient",
    "InferenceMetrics",
    "CircuitBreakerState",
    "QueryResult",
    "ValidationResult",
    "MetricsCollectionError",
    "validate_metrics",
    "validate_request_rate",
    "validate_latency",
    "validate_error_rate",
    "validate_service_name",
    "validate_timestamp",
    "sanitize_metric_value",
    "sanitize_metrics",
]
