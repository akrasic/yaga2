"""
Input validation utilities for the ML anomaly detection pipeline.

This module provides consistent validation across the codebase,
ensuring data quality before processing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from smartbox_anomaly.core.constants import MetricName, Thresholds


@dataclass
class ValidationResult:
    """Result of a validation operation."""

    is_valid: bool
    errors: list[str]
    warnings: list[str]

    @classmethod
    def success(cls, warnings: list[str] | None = None) -> ValidationResult:
        """Create a successful validation result."""
        return cls(is_valid=True, errors=[], warnings=warnings or [])

    @classmethod
    def failure(cls, errors: list[str], warnings: list[str] | None = None) -> ValidationResult:
        """Create a failed validation result."""
        return cls(is_valid=False, errors=errors, warnings=warnings or [])

    def merge(self, other: ValidationResult) -> ValidationResult:
        """Merge two validation results."""
        return ValidationResult(
            is_valid=self.is_valid and other.is_valid,
            errors=self.errors + other.errors,
            warnings=self.warnings + other.warnings,
        )


def validate_service_name(service_name: str) -> ValidationResult:
    """Validate a service name.

    Args:
        service_name: The service name to validate.

    Returns:
        ValidationResult indicating success or failure.
    """
    errors = []
    warnings = []

    if not service_name:
        errors.append("Service name cannot be empty")
    elif not isinstance(service_name, str):
        errors.append(f"Service name must be a string, got {type(service_name).__name__}")
    else:
        # Check for valid characters
        if not all(c.isalnum() or c in "-_" for c in service_name):
            warnings.append(f"Service name '{service_name}' contains unusual characters")

        # Check reasonable length
        if len(service_name) > 100:
            errors.append(f"Service name too long: {len(service_name)} characters")

    if errors:
        return ValidationResult.failure(errors, warnings)
    return ValidationResult.success(warnings)


def validate_metrics(
    metrics: dict[str, Any],
    service_name: str | None = None,
) -> ValidationResult:
    """Validate metrics dictionary.

    Args:
        metrics: Dictionary of metric names to values.
        service_name: Optional service name for context.

    Returns:
        ValidationResult indicating success or failure.
    """
    errors = []
    warnings = []
    context = f" for {service_name}" if service_name else ""

    if not isinstance(metrics, dict):
        return ValidationResult.failure([f"Metrics must be a dictionary{context}"])

    # Validate request_rate
    request_rate = metrics.get(MetricName.REQUEST_RATE)
    if request_rate is not None:
        result = validate_request_rate(request_rate)
        if not result.is_valid:
            errors.extend([f"{e}{context}" for e in result.errors])
        warnings.extend(result.warnings)

    # Validate latencies
    for latency_name in MetricName.latency_metrics():
        latency = metrics.get(latency_name)
        if latency is not None:
            result = validate_latency(latency, latency_name)
            if not result.is_valid:
                errors.extend([f"{e}{context}" for e in result.errors])
            warnings.extend(result.warnings)

    # Validate error_rate
    error_rate = metrics.get(MetricName.ERROR_RATE)
    if error_rate is not None:
        result = validate_error_rate(error_rate)
        if not result.is_valid:
            errors.extend([f"{e}{context}" for e in result.errors])
        warnings.extend(result.warnings)

    if errors:
        return ValidationResult.failure(errors, warnings)
    return ValidationResult.success(warnings)


def validate_request_rate(value: Any) -> ValidationResult:
    """Validate a request rate value.

    Args:
        value: The request rate value to validate.

    Returns:
        ValidationResult indicating success or failure.
    """
    errors = []
    warnings = []

    try:
        rate = float(value)
        if rate < 0:
            errors.append(f"Request rate cannot be negative: {rate}")
        elif rate > Thresholds.MAX_REQUEST_RATE:
            errors.append(
                f"Request rate exceeds maximum ({Thresholds.MAX_REQUEST_RATE}): {rate}"
            )
        elif rate == 0:
            warnings.append("Request rate is zero - service may be idle")
    except (TypeError, ValueError):
        errors.append(f"Request rate must be numeric: {value}")

    if errors:
        return ValidationResult.failure(errors, warnings)
    return ValidationResult.success(warnings)


def validate_latency(value: Any, metric_name: str = "latency") -> ValidationResult:
    """Validate a latency value.

    Args:
        value: The latency value to validate (in milliseconds).
        metric_name: Name of the specific latency metric.

    Returns:
        ValidationResult indicating success or failure.
    """
    errors = []
    warnings = []

    try:
        latency = float(value)
        if latency < 0:
            errors.append(f"{metric_name} cannot be negative: {latency}")
        elif latency > Thresholds.MAX_LATENCY_MS:
            errors.append(
                f"{metric_name} exceeds maximum ({Thresholds.MAX_LATENCY_MS}ms): {latency}"
            )
        elif latency > 10000:  # > 10 seconds
            warnings.append(f"{metric_name} is very high: {latency}ms")
    except (TypeError, ValueError):
        errors.append(f"{metric_name} must be numeric: {value}")

    if errors:
        return ValidationResult.failure(errors, warnings)
    return ValidationResult.success(warnings)


def validate_error_rate(value: Any) -> ValidationResult:
    """Validate an error rate value.

    Args:
        value: The error rate to validate (0.0 to 1.0).

    Returns:
        ValidationResult indicating success or failure.
    """
    errors = []
    warnings = []

    try:
        rate = float(value)
        if rate < 0:
            errors.append(f"Error rate cannot be negative: {rate}")
        elif rate > Thresholds.MAX_ERROR_RATE:
            errors.append(f"Error rate exceeds 100%: {rate}")
        elif rate > 0.5:
            warnings.append(f"Error rate is critically high: {rate * 100:.1f}%")
        elif rate > 0.1:
            warnings.append(f"Error rate is elevated: {rate * 100:.1f}%")
    except (TypeError, ValueError):
        errors.append(f"Error rate must be numeric: {value}")

    if errors:
        return ValidationResult.failure(errors, warnings)
    return ValidationResult.success(warnings)


def validate_timestamp(value: Any) -> ValidationResult:
    """Validate a timestamp value.

    Args:
        value: The timestamp to validate.

    Returns:
        ValidationResult indicating success or failure.
    """
    errors = []
    warnings = []

    if value is None:
        errors.append("Timestamp cannot be None")
    elif isinstance(value, datetime):
        # Check for future timestamps
        if value > datetime.now():
            warnings.append(f"Timestamp is in the future: {value}")
        # Check for very old timestamps (> 1 week)
        if value < datetime.now() - timedelta(days=7):
            warnings.append(f"Timestamp is more than a week old: {value}")
    elif isinstance(value, str):
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            errors.append(f"Invalid timestamp format: {value}")
    else:
        errors.append(f"Timestamp must be datetime or ISO string: {type(value).__name__}")

    if errors:
        return ValidationResult.failure(errors, warnings)
    return ValidationResult.success(warnings)


def validate_anomaly_result(result: dict[str, Any]) -> ValidationResult:
    """Validate an anomaly detection result.

    Args:
        result: The anomaly result dictionary to validate.

    Returns:
        ValidationResult indicating success or failure.
    """
    errors = []
    warnings = []

    if not isinstance(result, dict):
        return ValidationResult.failure(["Anomaly result must be a dictionary"])

    # Check for required fields
    if "service" not in result and "service_name" not in result:
        errors.append("Anomaly result missing service identifier")

    # Validate anomalies if present
    anomalies = result.get("anomalies")
    if anomalies is not None:
        if isinstance(anomalies, dict):
            for _name, data in anomalies.items():
                if isinstance(data, dict):
                    severity = data.get("severity")
                    if severity and severity not in ("low", "medium", "high", "critical"):
                        warnings.append(f"Unknown severity level: {severity}")
        elif isinstance(anomalies, list):
            for i, anomaly in enumerate(anomalies):
                if not isinstance(anomaly, dict):
                    errors.append(f"Anomaly at index {i} is not a dictionary")

    if errors:
        return ValidationResult.failure(errors, warnings)
    return ValidationResult.success(warnings)


def sanitize_metric_value(value: Any, default: float = 0.0) -> float:
    """Sanitize a metric value for safe processing.

    Args:
        value: The value to sanitize.
        default: Default value if sanitization fails.

    Returns:
        Sanitized float value.
    """
    if value is None:
        return default

    try:
        result = float(value)
        # Handle infinity and NaN
        if result != result:  # NaN check
            return default
        if result == float("inf") or result == float("-inf"):
            return default
        return result
    except (TypeError, ValueError):
        return default


def sanitize_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    """Sanitize all metrics in a dictionary.

    Args:
        metrics: Dictionary of metric names to values.

    Returns:
        Dictionary with sanitized float values.
    """
    return {
        key: sanitize_metric_value(value)
        for key, value in metrics.items()
        if key in MetricName.core_metrics()
    }
