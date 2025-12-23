"""
Tests for validation utilities.
"""

from __future__ import annotations

from smartbox_anomaly.core import MetricName
from smartbox_anomaly.metrics import (
    ValidationResult,
    sanitize_metric_value,
    sanitize_metrics,
    validate_error_rate,
    validate_latency,
    validate_metrics,
    validate_request_rate,
    validate_service_name,
)


class TestValidationResult:
    """Tests for ValidationResult class."""

    def test_success_creation(self):
        """Test creating a successful result."""
        result = ValidationResult.success()
        assert result.is_valid
        assert result.errors == []
        assert result.warnings == []

    def test_success_with_warnings(self):
        """Test success with warnings."""
        result = ValidationResult.success(warnings=["Warning 1"])
        assert result.is_valid
        assert "Warning 1" in result.warnings

    def test_failure_creation(self):
        """Test creating a failed result."""
        result = ValidationResult.failure(["Error 1"])
        assert not result.is_valid
        assert "Error 1" in result.errors

    def test_merge_results(self):
        """Test merging validation results."""
        result1 = ValidationResult.success(warnings=["Warn 1"])
        result2 = ValidationResult.failure(["Error 1"], warnings=["Warn 2"])

        merged = result1.merge(result2)
        assert not merged.is_valid
        assert "Error 1" in merged.errors
        assert "Warn 1" in merged.warnings
        assert "Warn 2" in merged.warnings


class TestValidateServiceName:
    """Tests for service name validation."""

    def test_valid_service_name(self):
        """Test valid service names."""
        assert validate_service_name("booking").is_valid
        assert validate_service_name("mobile-api").is_valid
        assert validate_service_name("m2_fr").is_valid

    def test_empty_service_name(self):
        """Test empty service name."""
        result = validate_service_name("")
        assert not result.is_valid
        assert any("empty" in e.lower() for e in result.errors)

    def test_long_service_name(self):
        """Test overly long service name."""
        result = validate_service_name("a" * 101)
        assert not result.is_valid
        assert any("long" in e.lower() for e in result.errors)


class TestValidateMetrics:
    """Tests for metrics validation."""

    def test_valid_metrics(self):
        """Test valid metrics dictionary."""
        metrics = {
            MetricName.REQUEST_RATE: 100.0,
            MetricName.APPLICATION_LATENCY: 50.0,
            MetricName.ERROR_RATE: 0.01,
        }
        result = validate_metrics(metrics)
        assert result.is_valid

    def test_invalid_request_rate(self):
        """Test invalid request rate."""
        metrics = {MetricName.REQUEST_RATE: -100.0}
        result = validate_metrics(metrics)
        assert not result.is_valid

    def test_invalid_error_rate(self):
        """Test invalid error rate."""
        metrics = {MetricName.ERROR_RATE: 1.5}  # > 100%
        result = validate_metrics(metrics)
        assert not result.is_valid

    def test_non_dict_input(self):
        """Test that non-dict input fails."""
        result = validate_metrics("not a dict")
        assert not result.is_valid


class TestValidateRequestRate:
    """Tests for request rate validation."""

    def test_valid_rates(self):
        """Test valid request rates."""
        assert validate_request_rate(0.0).is_valid
        assert validate_request_rate(100.0).is_valid
        assert validate_request_rate(1000.0).is_valid

    def test_negative_rate(self):
        """Test negative rate fails."""
        result = validate_request_rate(-1.0)
        assert not result.is_valid

    def test_excessive_rate(self):
        """Test excessive rate fails."""
        result = validate_request_rate(2_000_000.0)
        assert not result.is_valid

    def test_zero_rate_warning(self):
        """Test zero rate produces warning."""
        result = validate_request_rate(0.0)
        assert result.is_valid
        assert len(result.warnings) > 0


class TestValidateLatency:
    """Tests for latency validation."""

    def test_valid_latency(self):
        """Test valid latency values."""
        assert validate_latency(0.0).is_valid
        assert validate_latency(50.0).is_valid
        assert validate_latency(1000.0).is_valid

    def test_negative_latency(self):
        """Test negative latency fails."""
        result = validate_latency(-10.0)
        assert not result.is_valid

    def test_excessive_latency(self):
        """Test excessive latency fails."""
        result = validate_latency(500_000.0)
        assert not result.is_valid

    def test_high_latency_warning(self):
        """Test high latency produces warning."""
        result = validate_latency(15000.0)
        assert result.is_valid
        assert len(result.warnings) > 0


class TestValidateErrorRate:
    """Tests for error rate validation."""

    def test_valid_error_rate(self):
        """Test valid error rates."""
        assert validate_error_rate(0.0).is_valid
        assert validate_error_rate(0.05).is_valid
        assert validate_error_rate(1.0).is_valid

    def test_negative_error_rate(self):
        """Test negative error rate fails."""
        result = validate_error_rate(-0.1)
        assert not result.is_valid

    def test_excessive_error_rate(self):
        """Test error rate > 100% fails."""
        result = validate_error_rate(1.5)
        assert not result.is_valid

    def test_high_error_rate_warning(self):
        """Test high error rate produces warning."""
        result = validate_error_rate(0.6)
        assert result.is_valid
        assert len(result.warnings) > 0


class TestSanitizeMetricValue:
    """Tests for metric value sanitization."""

    def test_valid_value(self):
        """Test valid values pass through."""
        assert sanitize_metric_value(100.0) == 100.0
        assert sanitize_metric_value(0.0) == 0.0

    def test_none_value(self):
        """Test None returns default."""
        assert sanitize_metric_value(None) == 0.0
        assert sanitize_metric_value(None, default=50.0) == 50.0

    def test_nan_value(self):
        """Test NaN returns default."""
        assert sanitize_metric_value(float("nan")) == 0.0

    def test_infinity_value(self):
        """Test infinity returns default."""
        assert sanitize_metric_value(float("inf")) == 0.0
        assert sanitize_metric_value(float("-inf")) == 0.0

    def test_string_value(self):
        """Test non-numeric value returns default."""
        assert sanitize_metric_value("not a number") == 0.0


class TestSanitizeMetrics:
    """Tests for metrics dictionary sanitization."""

    def test_valid_metrics(self):
        """Test valid metrics pass through."""
        metrics = {
            MetricName.REQUEST_RATE: 100.0,
            MetricName.ERROR_RATE: 0.05,
        }
        result = sanitize_metrics(metrics)
        assert result[MetricName.REQUEST_RATE] == 100.0
        assert result[MetricName.ERROR_RATE] == 0.05

    def test_filters_non_core_metrics(self):
        """Test that non-core metrics are filtered."""
        metrics = {
            MetricName.REQUEST_RATE: 100.0,
            "custom_metric": 50.0,
        }
        result = sanitize_metrics(metrics)
        assert MetricName.REQUEST_RATE in result
        assert "custom_metric" not in result

    def test_sanitizes_invalid_values(self):
        """Test that invalid values are sanitized."""
        metrics = {
            MetricName.REQUEST_RATE: None,
            MetricName.ERROR_RATE: float("nan"),
        }
        result = sanitize_metrics(metrics)
        assert result[MetricName.REQUEST_RATE] == 0.0
        assert result[MetricName.ERROR_RATE] == 0.0
