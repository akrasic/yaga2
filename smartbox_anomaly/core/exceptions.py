"""
Custom exception hierarchy for the ML anomaly detection pipeline.

This module provides a structured exception hierarchy that enables:
- Specific error handling at different layers
- Rich error context for debugging
- Consistent error messages across the codebase
"""

from __future__ import annotations

from typing import Any


class PipelineError(Exception):
    """Base exception for all pipeline errors.

    All custom exceptions in the pipeline inherit from this class,
    enabling catching all pipeline-related errors with a single except clause.
    """

    def __init__(
        self,
        message: str,
        *,
        context: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        self.message = message
        self.context = context or {}
        self.cause = cause
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format the full error message including context."""
        parts = [self.message]
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            parts.append(f"[{context_str}]")
        if self.cause:
            parts.append(f"Caused by: {self.cause}")
        return " ".join(parts)


# =============================================================================
# Model-Related Exceptions
# =============================================================================


class ModelError(PipelineError):
    """Base exception for model-related errors."""

    pass


class ModelLoadError(ModelError):
    """Raised when a model cannot be loaded.

    Examples:
        - Model file not found
        - Corrupted model data
        - Version mismatch
    """

    def __init__(
        self,
        service_name: str,
        *,
        model_path: str | None = None,
        reason: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        context = {"service": service_name}
        if model_path:
            context["path"] = model_path
        message = f"Failed to load model for {service_name}"
        if reason:
            message = f"{message}: {reason}"
        super().__init__(message, context=context, cause=cause)
        self.service_name = service_name
        self.model_path = model_path


class ModelTrainingError(ModelError):
    """Raised when model training fails.

    Examples:
        - Insufficient training data
        - Invalid feature data
        - Training algorithm failure
    """

    def __init__(
        self,
        service_name: str,
        *,
        reason: str,
        sample_count: int | None = None,
        cause: Exception | None = None,
    ) -> None:
        context: dict[str, Any] = {"service": service_name}
        if sample_count is not None:
            context["samples"] = sample_count
        message = f"Model training failed for {service_name}: {reason}"
        super().__init__(message, context=context, cause=cause)
        self.service_name = service_name
        self.sample_count = sample_count


class ModelNotFoundError(ModelError):
    """Raised when no model exists for a service/period combination."""

    def __init__(
        self,
        service_name: str,
        *,
        time_period: str | None = None,
        available_periods: list[str] | None = None,
    ) -> None:
        context: dict[str, Any] = {"service": service_name}
        if time_period:
            context["period"] = time_period
        if available_periods:
            context["available"] = available_periods
        message = f"No model found for {service_name}"
        if time_period:
            message = f"{message} in period {time_period}"
        super().__init__(message, context=context)
        self.service_name = service_name
        self.time_period = time_period
        self.available_periods = available_periods


# =============================================================================
# Metrics-Related Exceptions
# =============================================================================


class MetricsError(PipelineError):
    """Base exception for metrics-related errors."""

    pass


class MetricsCollectionError(MetricsError):
    """Raised when metrics collection fails.

    Examples:
        - VictoriaMetrics connection failure
        - Query timeout
        - Invalid response format
    """

    def __init__(
        self,
        service_name: str,
        *,
        metric_name: str | None = None,
        reason: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        context = {"service": service_name}
        if metric_name:
            context["metric"] = metric_name
        message = f"Failed to collect metrics for {service_name}"
        if reason:
            message = f"{message}: {reason}"
        super().__init__(message, context=context, cause=cause)
        self.service_name = service_name
        self.metric_name = metric_name


class MetricsValidationError(MetricsError):
    """Raised when collected metrics fail validation.

    Examples:
        - Values out of expected range
        - Missing required metrics
        - Invalid data types
    """

    def __init__(
        self,
        service_name: str,
        *,
        metric_name: str,
        value: Any,
        reason: str,
    ) -> None:
        context = {
            "service": service_name,
            "metric": metric_name,
            "value": value,
        }
        message = f"Invalid {metric_name} for {service_name}: {reason}"
        super().__init__(message, context=context)
        self.service_name = service_name
        self.metric_name = metric_name
        self.value = value


class CircuitBreakerOpenError(MetricsError):
    """Raised when circuit breaker prevents operation."""

    def __init__(
        self,
        service: str = "VictoriaMetrics",
        *,
        failures: int | None = None,
        timeout_remaining: float | None = None,
    ) -> None:
        context: dict[str, Any] = {"service": service}
        if failures is not None:
            context["failures"] = failures
        if timeout_remaining is not None:
            context["timeout_remaining_sec"] = round(timeout_remaining, 1)
        message = f"Circuit breaker open for {service}"
        super().__init__(message, context=context)


# =============================================================================
# Detection-Related Exceptions
# =============================================================================


class DetectionError(PipelineError):
    """Base exception for anomaly detection errors."""

    pass


class InferenceError(DetectionError):
    """Raised when inference fails for a service."""

    def __init__(
        self,
        service_name: str,
        *,
        reason: str,
        cause: Exception | None = None,
    ) -> None:
        context = {"service": service_name}
        message = f"Inference failed for {service_name}: {reason}"
        super().__init__(message, context=context, cause=cause)
        self.service_name = service_name


class TimePeriodError(DetectionError):
    """Raised when time period determination fails."""

    def __init__(
        self,
        *,
        reason: str,
        timestamp: str | None = None,
    ) -> None:
        context = {}
        if timestamp:
            context["timestamp"] = timestamp
        super().__init__(f"Time period error: {reason}", context=context)


# =============================================================================
# Fingerprinting-Related Exceptions
# =============================================================================


class FingerprintingError(PipelineError):
    """Base exception for fingerprinting errors."""

    pass


class DatabaseError(FingerprintingError):
    """Raised when database operations fail."""

    def __init__(
        self,
        operation: str,
        *,
        db_path: str | None = None,
        reason: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        context = {"operation": operation}
        if db_path:
            context["db_path"] = db_path
        message = f"Database {operation} failed"
        if reason:
            message = f"{message}: {reason}"
        super().__init__(message, context=context, cause=cause)


class IncidentNotFoundError(FingerprintingError):
    """Raised when a referenced incident doesn't exist."""

    def __init__(
        self,
        incident_id: str,
        *,
        fingerprint_id: str | None = None,
    ) -> None:
        context = {"incident_id": incident_id}
        if fingerprint_id:
            context["fingerprint_id"] = fingerprint_id
        super().__init__(f"Incident not found: {incident_id}", context=context)
        self.incident_id = incident_id
        self.fingerprint_id = fingerprint_id


# =============================================================================
# Configuration-Related Exceptions
# =============================================================================


class ConfigurationError(PipelineError):
    """Raised when configuration is invalid or missing."""

    def __init__(
        self,
        parameter: str,
        *,
        reason: str,
        value: Any = None,
    ) -> None:
        context = {"parameter": parameter}
        if value is not None:
            context["value"] = value
        super().__init__(f"Configuration error for {parameter}: {reason}", context=context)
        self.parameter = parameter


# =============================================================================
# API-Related Exceptions
# =============================================================================


class APIError(PipelineError):
    """Base exception for external API errors."""

    pass


class ObservabilityServiceError(APIError):
    """Raised when observability service communication fails."""

    def __init__(
        self,
        endpoint: str,
        *,
        status_code: int | None = None,
        reason: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        context: dict[str, Any] = {"endpoint": endpoint}
        if status_code is not None:
            context["status_code"] = status_code
        message = f"Observability service error at {endpoint}"
        if reason:
            message = f"{message}: {reason}"
        super().__init__(message, context=context, cause=cause)
        self.endpoint = endpoint
        self.status_code = status_code
