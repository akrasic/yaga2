"""
Tests for constants and enumerations.
"""

from __future__ import annotations

from smartbox_anomaly.core import (
    PERIOD_FALLBACK_MAP,
    TIME_PERIOD_SUFFIXES,
    AnomalySeverity,
    MetricName,
    Thresholds,
    TimePeriod,
)


class TestAnomalySeverity:
    """Tests for AnomalySeverity enum."""

    def test_severity_values(self):
        """Test that severity values are correct strings."""
        assert AnomalySeverity.LOW.value == "low"
        assert AnomalySeverity.MEDIUM.value == "medium"
        assert AnomalySeverity.HIGH.value == "high"
        assert AnomalySeverity.CRITICAL.value == "critical"

    def test_from_string_valid(self):
        """Test converting valid strings to severity."""
        assert AnomalySeverity.from_string("low") == AnomalySeverity.LOW
        assert AnomalySeverity.from_string("HIGH") == AnomalySeverity.HIGH
        assert AnomalySeverity.from_string("Critical") == AnomalySeverity.CRITICAL

    def test_from_string_invalid(self):
        """Test that invalid strings default to MEDIUM."""
        assert AnomalySeverity.from_string("invalid") == AnomalySeverity.MEDIUM
        assert AnomalySeverity.from_string("") == AnomalySeverity.MEDIUM
        assert AnomalySeverity.from_string(None) == AnomalySeverity.MEDIUM

    def test_severity_comparison(self):
        """Test severity comparison operators."""
        assert AnomalySeverity.LOW < AnomalySeverity.MEDIUM
        assert AnomalySeverity.MEDIUM < AnomalySeverity.HIGH
        assert AnomalySeverity.HIGH < AnomalySeverity.CRITICAL
        assert AnomalySeverity.CRITICAL > AnomalySeverity.LOW
        assert AnomalySeverity.MEDIUM >= AnomalySeverity.MEDIUM
        assert AnomalySeverity.HIGH <= AnomalySeverity.HIGH

    def test_numeric_values(self):
        """Test numeric value property."""
        assert AnomalySeverity.LOW.numeric_value == 1
        assert AnomalySeverity.MEDIUM.numeric_value == 2
        assert AnomalySeverity.HIGH.numeric_value == 3
        assert AnomalySeverity.CRITICAL.numeric_value == 4


class TestTimePeriod:
    """Tests for TimePeriod enum."""

    def test_all_periods(self):
        """Test that all_periods returns all values."""
        periods = TimePeriod.all_periods()
        assert len(periods) == 5
        assert TimePeriod.BUSINESS_HOURS in periods
        assert TimePeriod.WEEKEND_NIGHT in periods

    def test_is_weekend(self):
        """Test weekend detection."""
        assert not TimePeriod.BUSINESS_HOURS.is_weekend
        assert not TimePeriod.EVENING_HOURS.is_weekend
        assert not TimePeriod.NIGHT_HOURS.is_weekend
        assert TimePeriod.WEEKEND_DAY.is_weekend
        assert TimePeriod.WEEKEND_NIGHT.is_weekend

    def test_is_night(self):
        """Test night period detection."""
        assert not TimePeriod.BUSINESS_HOURS.is_night
        assert not TimePeriod.EVENING_HOURS.is_night
        assert TimePeriod.NIGHT_HOURS.is_night
        assert not TimePeriod.WEEKEND_DAY.is_night
        assert TimePeriod.WEEKEND_NIGHT.is_night


class TestMetricName:
    """Tests for MetricName constants."""

    def test_core_metrics(self):
        """Test core metrics list."""
        core = MetricName.core_metrics()
        assert MetricName.REQUEST_RATE in core
        assert MetricName.APPLICATION_LATENCY in core
        assert MetricName.ERROR_RATE in core
        assert len(core) == 5

    def test_zero_normal_metrics(self):
        """Test zero-normal metrics list."""
        zero_normal = MetricName.zero_normal_metrics()
        assert MetricName.CLIENT_LATENCY in zero_normal
        assert MetricName.DATABASE_LATENCY in zero_normal
        assert MetricName.REQUEST_RATE not in zero_normal

    def test_latency_metrics(self):
        """Test latency metrics list."""
        latency = MetricName.latency_metrics()
        assert len(latency) == 3
        assert MetricName.APPLICATION_LATENCY in latency
        assert MetricName.CLIENT_LATENCY in latency
        assert MetricName.DATABASE_LATENCY in latency


class TestThresholds:
    """Tests for Thresholds class."""

    def test_score_to_severity_critical(self):
        """Test critical severity mapping."""
        assert Thresholds.score_to_severity(-0.7) == AnomalySeverity.CRITICAL
        assert Thresholds.score_to_severity(-0.9) == AnomalySeverity.CRITICAL

    def test_score_to_severity_high(self):
        """Test high severity mapping."""
        assert Thresholds.score_to_severity(-0.4) == AnomalySeverity.HIGH
        assert Thresholds.score_to_severity(-0.5) == AnomalySeverity.HIGH

    def test_score_to_severity_medium(self):
        """Test medium severity mapping."""
        assert Thresholds.score_to_severity(-0.2) == AnomalySeverity.MEDIUM
        assert Thresholds.score_to_severity(-0.15) == AnomalySeverity.MEDIUM

    def test_score_to_severity_low(self):
        """Test low severity mapping."""
        assert Thresholds.score_to_severity(0.0) == AnomalySeverity.LOW
        assert Thresholds.score_to_severity(-0.05) == AnomalySeverity.LOW


class TestPeriodFallbackMap:
    """Tests for period fallback configuration."""

    def test_all_periods_have_fallbacks(self):
        """Test that all periods have fallback configurations."""
        for period in TimePeriod.all_periods():
            assert period.value in PERIOD_FALLBACK_MAP
            assert len(PERIOD_FALLBACK_MAP[period.value]) > 0

    def test_fallbacks_are_valid_periods(self):
        """Test that all fallbacks are valid period values."""
        valid_periods = {p.value for p in TimePeriod}
        for fallbacks in PERIOD_FALLBACK_MAP.values():
            for fallback in fallbacks:
                assert fallback in valid_periods


class TestTimePeriodSuffixes:
    """Tests for time period suffix configuration."""

    def test_suffixes_include_all_periods(self):
        """Test that suffixes include all current periods."""
        for period in TimePeriod:
            assert f"_{period.value}" in TIME_PERIOD_SUFFIXES

    def test_legacy_weekend_suffix(self):
        """Test that legacy weekend suffix is included."""
        assert "_weekend" in TIME_PERIOD_SUFFIXES
