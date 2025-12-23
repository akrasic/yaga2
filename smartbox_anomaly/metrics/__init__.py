"""
Metrics module - metric collection and validation.

This module contains:
    - client: VictoriaMetrics client with circuit breaker
    - validation: Metric validation utilities
"""

from smartbox_anomaly.metrics.client import (
    CircuitBreakerState,
    InferenceMetrics,
    VictoriaMetricsClient,
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
