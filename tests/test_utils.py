"""
Tests for utility functions.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from smartbox_anomaly.core import (
    TimePeriod,
    calculate_duration_minutes,
    extract_base_service_names,
    format_duration,
    generate_anomaly_name,
    generate_correlation_id,
    generate_fingerprint_id,
    generate_incident_id,
    get_period_type,
    get_time_period,
    merge_dicts,
    now_iso,
    parse_service_model,
    parse_timestamp,
    safe_get_nested,
    timestamp_age_minutes,
    truncate_string,
)


class TestIDGeneration:
    """Tests for ID generation functions."""

    def test_generate_fingerprint_id_deterministic(self):
        """Test fingerprint ID is deterministic."""
        fp1 = generate_fingerprint_id("booking", "high_latency")
        fp2 = generate_fingerprint_id("booking", "high_latency")
        assert fp1 == fp2
        assert fp1.startswith("anomaly_")

    def test_generate_fingerprint_id_unique_for_different_inputs(self):
        """Test different inputs produce different fingerprints."""
        fp1 = generate_fingerprint_id("booking", "high_latency")
        fp2 = generate_fingerprint_id("search", "high_latency")
        fp3 = generate_fingerprint_id("booking", "high_errors")
        assert fp1 != fp2
        assert fp1 != fp3

    def test_generate_incident_id_unique(self):
        """Test incident IDs are unique."""
        ids = [generate_incident_id() for _ in range(100)]
        assert len(set(ids)) == 100
        assert all(id.startswith("incident_") for id in ids)

    def test_generate_correlation_id(self):
        """Test correlation ID generation."""
        corr_id = generate_correlation_id()
        assert corr_id.startswith("corr_")
        assert len(corr_id) > 10


class TestServiceNameParsing:
    """Tests for service name parsing functions."""

    def test_parse_service_model_with_period(self):
        """Test parsing service name with time period."""
        assert parse_service_model("booking_evening_hours") == ("booking", "evening_hours")
        assert parse_service_model("fa5_business_hours") == ("fa5", "business_hours")
        assert parse_service_model("mobile-api_weekend_night") == ("mobile-api", "weekend_night")

    def test_parse_service_model_without_period(self):
        """Test parsing service name without known period."""
        service, period = parse_service_model("booking_unknown")
        assert service == "booking"
        assert period == "unknown"

    def test_parse_service_model_simple_name(self):
        """Test parsing simple service name."""
        service, period = parse_service_model("booking")
        assert service == "booking"
        assert period == "unknown"

    def test_extract_base_service_names(self):
        """Test extracting base service names from list."""
        names = [
            "booking_evening_hours",
            "booking_night_hours",
            "search_business_hours",
            "fa5_weekend_day",
        ]
        bases = extract_base_service_names(names)
        assert bases == ["booking", "fa5", "search"]


class TestTimePeriod:
    """Tests for time period functions."""

    def test_get_time_period_business_hours(self):
        """Test business hours detection."""
        # Wednesday at 10am
        dt = datetime(2024, 1, 10, 10, 0)
        assert get_time_period(dt) == TimePeriod.BUSINESS_HOURS

    def test_get_time_period_evening_hours(self):
        """Test evening hours detection."""
        # Wednesday at 7pm
        dt = datetime(2024, 1, 10, 19, 0)
        assert get_time_period(dt) == TimePeriod.EVENING_HOURS

    def test_get_time_period_night_hours(self):
        """Test night hours detection."""
        # Wednesday at 2am
        dt = datetime(2024, 1, 10, 2, 0)
        assert get_time_period(dt) == TimePeriod.NIGHT_HOURS
        # Wednesday at 11pm
        dt = datetime(2024, 1, 10, 23, 0)
        assert get_time_period(dt) == TimePeriod.NIGHT_HOURS

    def test_get_time_period_weekend_day(self):
        """Test weekend day detection."""
        # Saturday at 2pm
        dt = datetime(2024, 1, 13, 14, 0)
        assert get_time_period(dt) == TimePeriod.WEEKEND_DAY

    def test_get_time_period_weekend_night(self):
        """Test weekend night detection."""
        # Saturday at 11pm
        dt = datetime(2024, 1, 13, 23, 0)
        assert get_time_period(dt) == TimePeriod.WEEKEND_NIGHT

    def test_get_period_type(self):
        """Test period type classification."""
        assert get_period_type("business_hours") == "peak_activity"
        assert get_period_type("night_hours") == "minimal_activity"
        assert get_period_type(TimePeriod.WEEKEND_DAY) == "weekend_moderate_activity"


class TestDurationCalculations:
    """Tests for duration calculation functions."""

    def test_calculate_duration_minutes(self):
        """Test duration calculation."""
        start = datetime.now() - timedelta(hours=2, minutes=30)
        end = datetime.now()
        duration = calculate_duration_minutes(start, end)
        assert 149 <= duration <= 151  # ~150 minutes

    def test_calculate_duration_minutes_from_string(self):
        """Test duration calculation with ISO string."""
        start = (datetime.now() - timedelta(hours=1)).isoformat()
        duration = calculate_duration_minutes(start)
        assert 59 <= duration <= 61

    def test_format_duration_minutes(self):
        """Test formatting duration in minutes."""
        assert format_duration(45) == "45m"

    def test_format_duration_hours(self):
        """Test formatting duration in hours."""
        assert format_duration(90) == "1h 30m"
        assert format_duration(120) == "2h"

    def test_format_duration_days(self):
        """Test formatting duration in days."""
        assert format_duration(1500) == "1d 1h"
        assert format_duration(2880) == "2d"


class TestAnomalyNameGeneration:
    """Tests for anomaly name generation."""

    def test_generate_anomaly_name_multivariate(self):
        """Test multivariate anomaly name."""
        data = {"type": "multivariate", "detection_method": "isolation_forest"}
        name = generate_anomaly_name(data)
        assert name == "multivariate_isolation_forest"

    def test_generate_anomaly_name_threshold(self):
        """Test threshold anomaly name."""
        data = {"type": "threshold", "detection_method": "static"}
        name = generate_anomaly_name(data)
        assert name == "threshold_static"

    def test_generate_anomaly_name_fallback(self):
        """Test fallback anomaly name."""
        data = {"type": "custom"}
        name = generate_anomaly_name(data, index=5)
        assert name == "anomaly_5_custom"

    def test_generate_anomaly_name_empty(self):
        """Test empty data handling."""
        name = generate_anomaly_name({}, index=0)
        assert name == "anomaly_0_unknown"

    def test_generate_anomaly_name_dependency_latency_from_description(self):
        """Test dependency latency naming from description."""
        data = {
            "type": "ml_isolation",
            "direction": "high",
            "description": "External dependency slow: 191ms (p90: 68ms)",
        }
        name = generate_anomaly_name(data)
        assert name == "dependency_latency_high"

    def test_generate_anomaly_name_dependency_latency_from_root_metric(self):
        """Test dependency latency naming from root_metric."""
        data = {
            "type": "ml_isolation",
            "direction": "high",
            "root_metric": "dependency_latency",
            "description": "Some generic description",
        }
        name = generate_anomaly_name(data)
        assert name == "dependency_latency_high"

    def test_generate_anomaly_name_application_latency_fast_responses(self):
        """Test application latency naming for fast responses."""
        data = {
            "type": "ml_isolation",
            "direction": "low",
            "description": "Unusually fast responses: 46ms (normally 72ms)",
        }
        name = generate_anomaly_name(data)
        assert name == "application_latency_low"

    def test_generate_anomaly_name_database_latency(self):
        """Test database latency naming."""
        data = {
            "type": "ml_isolation",
            "direction": "high",
            "root_metric": "database_latency",
        }
        name = generate_anomaly_name(data)
        assert name == "database_latency_high"

    def test_generate_anomaly_name_pattern_takes_precedence(self):
        """Test that pattern_name takes precedence over other naming."""
        data = {
            "type": "ml_isolation",
            "direction": "high",
            "pattern_name": "traffic_surge_failing",
            "description": "External dependency slow",
        }
        name = generate_anomaly_name(data)
        assert name == "traffic_surge_failing"

    def test_generate_anomaly_name_anomaly_key_highest_priority(self):
        """Test that _anomaly_key (preserved from detector) takes highest priority."""
        data = {
            "type": "ml_isolation",
            "direction": "high",
            "_anomaly_key": "dependency_latency_high",
            "pattern_name": "some_pattern",  # Should be ignored
            "root_metric": "application_latency",  # Should be ignored
            "description": "Some description",
        }
        name = generate_anomaly_name(data)
        assert name == "dependency_latency_high"

    def test_generate_anomaly_name_ignores_generic_anomaly_key(self):
        """Test that generic _anomaly_key values (anomaly_*, metric_*) are skipped."""
        data = {
            "type": "ml_isolation",
            "direction": "high",
            "_anomaly_key": "anomaly_0_consolidated",  # Generic, should skip
            "root_metric": "dependency_latency",
        }
        name = generate_anomaly_name(data)
        assert name == "dependency_latency_high"

    def test_generate_anomaly_name_ignores_metric_key(self):
        """Test that _anomaly_key starting with metric_ is skipped."""
        data = {
            "type": "ml_isolation",
            "direction": "low",
            "_anomaly_key": "metric_low",  # Generic, should skip
            "root_metric": "database_latency",
        }
        name = generate_anomaly_name(data)
        assert name == "database_latency_low"

    def test_generate_anomaly_name_comparison_data_fallback(self):
        """Test using comparison_data to find most anomalous metric."""
        data = {
            "type": "ml_isolation",
            "direction": "high",
            "comparison_data": {
                "application_latency": {"deviation_sigma": 0.5},
                "dependency_latency": {"deviation_sigma": 3.2},  # Most anomalous
                "error_rate": {"deviation_sigma": 0.1},
            },
        }
        name = generate_anomaly_name(data)
        assert name == "dependency_latency_high"

    def test_generate_anomaly_name_comparison_data_ignores_low_deviation(self):
        """Test that comparison_data with low deviations falls through."""
        data = {
            "type": "ml_isolation",
            "direction": "high",
            "description": "External dependency slow",
            "comparison_data": {
                "application_latency": {"deviation_sigma": 0.3},  # Not significant
                "error_rate": {"deviation_sigma": 0.5},  # Not significant
            },
        }
        name = generate_anomaly_name(data)
        # Falls through to description inference
        assert name == "dependency_latency_high"

    def test_generate_anomaly_name_consolidated_type(self):
        """Test naming for consolidated anomaly type."""
        data = {
            "type": "consolidated",
            "direction": "high",
            "root_metric": "dependency_latency",
        }
        name = generate_anomaly_name(data)
        assert name == "dependency_latency_high"

    def test_generate_anomaly_name_consolidated_with_comparison_data(self):
        """Test consolidated type falls back to comparison_data."""
        data = {
            "type": "consolidated",
            "direction": "low",
            "comparison_data": {
                "database_latency": {"deviation_sigma": 2.5},
                "application_latency": {"deviation_sigma": 0.8},
            },
        }
        name = generate_anomaly_name(data)
        assert name == "database_latency_low"

    def test_generate_anomaly_name_consolidated_fallback(self):
        """Test consolidated type fallback when no root_metric or comparison_data."""
        data = {
            "type": "consolidated",
            "direction": "high",
        }
        name = generate_anomaly_name(data)
        assert name == "consolidated_high"

    def test_generate_anomaly_name_error_rate_from_root_metric(self):
        """Test error_rate naming from root_metric."""
        data = {
            "type": "ml_isolation",
            "direction": "high",
            "root_metric": "error_rate",
        }
        name = generate_anomaly_name(data)
        assert name == "error_rate_high"

    def test_generate_anomaly_name_request_rate_from_root_metric(self):
        """Test request_rate naming from root_metric."""
        data = {
            "type": "ml_isolation",
            "direction": "low",
            "root_metric": "request_rate",
        }
        name = generate_anomaly_name(data)
        assert name == "request_rate_low"


class TestDataUtilities:
    """Tests for data utility functions."""

    def test_safe_get_nested(self):
        """Test safe nested dictionary access."""
        data = {"a": {"b": {"c": 42}}}
        assert safe_get_nested(data, "a", "b", "c") == 42
        assert safe_get_nested(data, "a", "x", "c", default=-1) == -1
        assert safe_get_nested(data, "x", default=None) is None

    def test_merge_dicts(self):
        """Test dictionary merging."""
        result = merge_dicts({"a": 1}, {"b": 2}, {"a": 3})
        assert result == {"a": 3, "b": 2}

    def test_merge_dicts_empty(self):
        """Test merging with empty/None dicts."""
        result = merge_dicts({"a": 1}, None, {"b": 2})
        assert result == {"a": 1, "b": 2}

    def test_truncate_string(self):
        """Test string truncation."""
        assert truncate_string("hello", 10) == "hello"
        assert truncate_string("hello world", 8) == "hello..."
        assert truncate_string("hello", 5) == "hello"


class TestTimestampUtilities:
    """Tests for timestamp utility functions."""

    def test_now_iso(self):
        """Test ISO timestamp generation."""
        ts = now_iso()
        # Should be parseable
        parsed = datetime.fromisoformat(ts)
        assert parsed is not None

    def test_parse_timestamp_datetime(self):
        """Test parsing datetime object."""
        dt = datetime.now()
        assert parse_timestamp(dt) == dt

    def test_parse_timestamp_string(self):
        """Test parsing ISO string."""
        dt = datetime.now()
        parsed = parse_timestamp(dt.isoformat())
        assert parsed is not None

    def test_parse_timestamp_none(self):
        """Test parsing None."""
        assert parse_timestamp(None) is None

    def test_parse_timestamp_invalid(self):
        """Test parsing invalid string."""
        assert parse_timestamp("not a date") is None

    def test_timestamp_age_minutes(self):
        """Test calculating timestamp age."""
        ts = datetime.now() - timedelta(minutes=30)
        age = timestamp_age_minutes(ts)
        assert 29 <= age <= 31
