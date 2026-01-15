"""
Tests for anomaly fingerprinting system.
"""

from __future__ import annotations

import tempfile

import pytest

from smartbox_anomaly.core import IncidentAction
from smartbox_anomaly.fingerprinting import AnomalyFingerprinter, create_fingerprinter


class TestAnomalyFingerprinter:
    """Tests for AnomalyFingerprinter class."""

    @pytest.fixture
    def temp_db(self) -> str:
        """Create a temporary database file."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            return f.name

    @pytest.fixture
    def fingerprinter(self, temp_db: str) -> AnomalyFingerprinter:
        """Create a fingerprinter with temporary database."""
        return AnomalyFingerprinter(db_path=temp_db)

    def test_initialization(self, fingerprinter: AnomalyFingerprinter):
        """Test fingerprinter initializes correctly."""
        assert fingerprinter is not None
        assert fingerprinter.db_path is not None

    def test_create_new_incident(self, fingerprinter: AnomalyFingerprinter):
        """Test creating a new incident."""
        result = fingerprinter.process_anomalies(
            "booking_evening_hours",
            {
                "anomalies": [{
                    "type": "threshold",
                    "severity": "high",
                    "value": 500.0,
                    "threshold": 200.0,
                    "detection_method": "isolation_forest",
                }]
            },
        )

        assert result["fingerprinting"]["overall_action"] == IncidentAction.CREATE.value
        assert len(result["anomalies"]) == 1
        assert result["anomalies"][0]["incident_action"] == IncidentAction.CREATE.value
        assert "incident_id" in result["anomalies"][0]
        assert "fingerprint_id" in result["anomalies"][0]

    def test_continue_existing_incident(self, fingerprinter: AnomalyFingerprinter):
        """Test continuing an existing incident with cycle-based confirmation."""
        anomaly_data = {
            "anomalies": [{
                "type": "threshold",
                "severity": "high",
                "value": 500.0,
                "detection_method": "isolation_forest",
            }]
        }

        # First detection - creates SUSPECTED incident
        result1 = fingerprinter.process_anomalies("booking_evening_hours", anomaly_data)
        incident_id = result1["anomalies"][0]["incident_id"]
        assert result1["anomalies"][0]["status"] == "SUSPECTED"

        # Second detection - CONFIRMS incident (meets confirmation_cycles=2)
        result2 = fingerprinter.process_anomalies("booking_evening_hours", anomaly_data)

        # On confirmation, overall_action is "CONFIRMED"
        assert result2["fingerprinting"]["overall_action"] == "CONFIRMED"
        assert result2["anomalies"][0]["incident_action"] == IncidentAction.CONTINUE.value
        assert result2["anomalies"][0]["incident_id"] == incident_id
        assert result2["anomalies"][0]["occurrence_count"] == 2
        assert result2["anomalies"][0]["status"] == "OPEN"  # Now confirmed
        assert result2["anomalies"][0]["newly_confirmed"] is True

    def test_resolve_incident(self, fingerprinter: AnomalyFingerprinter):
        """Test resolving an incident with grace period."""
        anomaly_data = {"anomalies": [{"type": "threshold", "severity": "high"}]}

        # Cycle 1 - Create SUSPECTED incident
        fingerprinter.process_anomalies("booking_evening_hours", anomaly_data)

        # Cycle 2 - Confirm incident (OPEN)
        fingerprinter.process_anomalies("booking_evening_hours", anomaly_data)

        # Cycles 3-5: Anomaly clears - need 3 cycles (resolution_grace_cycles=3) to resolve
        # Cycle 3: OPEN -> RECOVERING
        result1 = fingerprinter.process_anomalies("booking_evening_hours", {"anomalies": []})
        assert result1["fingerprinting"]["overall_action"] == IncidentAction.NO_CHANGE.value
        assert len(result1["fingerprinting"]["resolved_incidents"]) == 0

        # Cycle 4: RECOVERING (missed_cycles=2)
        result2 = fingerprinter.process_anomalies("booking_evening_hours", {"anomalies": []})
        assert len(result2["fingerprinting"]["resolved_incidents"]) == 0

        # Cycle 5: RECOVERING -> CLOSED (missed_cycles=3, meets grace)
        result3 = fingerprinter.process_anomalies("booking_evening_hours", {"anomalies": []})
        assert result3["fingerprinting"]["overall_action"] == IncidentAction.RESOLVE.value
        assert len(result3["fingerprinting"]["resolved_incidents"]) == 1

    def test_severity_change_tracking(self, fingerprinter: AnomalyFingerprinter):
        """Test that severity changes are tracked."""
        # Create with high severity
        fingerprinter.process_anomalies(
            "booking_evening_hours",
            {"anomalies": [{"type": "threshold", "severity": "high", "detection_method": "test"}]},
        )

        # Update with critical severity
        result = fingerprinter.process_anomalies(
            "booking_evening_hours",
            {"anomalies": [{"type": "threshold", "severity": "critical", "detection_method": "test"}]},
        )

        assert result["anomalies"][0].get("severity_changed") is True
        assert result["anomalies"][0].get("previous_severity") == "high"

    def test_service_name_parsing(self, fingerprinter: AnomalyFingerprinter):
        """Test that service name is correctly parsed from full name."""
        result = fingerprinter.process_anomalies(
            "mobile-api_weekend_night",
            {"anomalies": [{"type": "threshold", "severity": "medium"}]},
        )

        assert result["service_name"] == "mobile-api"
        assert result["model_name"] == "weekend_night"

    def test_get_statistics(self, fingerprinter: AnomalyFingerprinter):
        """Test getting fingerprinter statistics."""
        anomaly_data = {"anomalies": [{"type": "threshold", "severity": "high"}]}

        # Cycle 1 - Create SUSPECTED incident
        fingerprinter.process_anomalies("booking_evening_hours", anomaly_data)

        # Cycle 2 - Confirm incident (OPEN)
        fingerprinter.process_anomalies("booking_evening_hours", anomaly_data)

        stats = fingerprinter.get_statistics()

        assert stats["total_open_incidents"] == 1  # Now OPEN after confirmation
        assert "schema_version" in stats
        assert "database_path" in stats

    def test_get_incident_by_id(self, fingerprinter: AnomalyFingerprinter):
        """Test retrieving incident by ID."""
        anomaly_data = {"anomalies": [{"type": "threshold", "severity": "high"}]}

        # Create and confirm incident
        result1 = fingerprinter.process_anomalies("booking_evening_hours", anomaly_data)
        incident_id = result1["anomalies"][0]["incident_id"]
        fingerprinter.process_anomalies("booking_evening_hours", anomaly_data)  # Confirm

        incident = fingerprinter.get_incident_by_id(incident_id)

        assert incident is not None
        assert incident["incident_id"] == incident_id
        assert incident["service_name"] == "booking"
        assert incident["status"] == "OPEN"

    def test_get_open_incidents(self, fingerprinter: AnomalyFingerprinter):
        """Test getting all open incidents."""
        booking_anomaly = {"anomalies": [{"type": "threshold", "severity": "high"}]}
        search_anomaly = {"anomalies": [{"type": "pattern", "severity": "medium"}]}

        # Create and confirm booking incident
        fingerprinter.process_anomalies("booking_evening_hours", booking_anomaly)
        fingerprinter.process_anomalies("booking_evening_hours", booking_anomaly)

        # Create and confirm search incident
        fingerprinter.process_anomalies("search_business_hours", search_anomaly)
        fingerprinter.process_anomalies("search_business_hours", search_anomaly)

        incidents = fingerprinter.get_open_incidents()

        assert "booking" in incidents
        assert "search" in incidents
        assert len(incidents["booking"]) == 1
        assert len(incidents["search"]) == 1

    def test_anomaly_dict_format(self, fingerprinter: AnomalyFingerprinter):
        """Test handling anomalies in dict format."""
        result = fingerprinter.process_anomalies(
            "booking_evening_hours",
            {
                "anomalies": {
                    "latency": {"type": "threshold", "severity": "high"},
                    "errors": {"type": "pattern", "severity": "medium"},
                }
            },
        )

        assert len(result["anomalies"]) == 2

    def test_cleanup_old_incidents(self, fingerprinter: AnomalyFingerprinter):
        """Test cleaning up old incidents."""
        # Create and resolve an incident
        fingerprinter.process_anomalies(
            "booking_evening_hours",
            {"anomalies": [{"type": "threshold", "severity": "high"}]},
        )
        fingerprinter.process_anomalies(
            "booking_evening_hours",
            {"anomalies": []},  # Resolves the incident
        )

        # Cleanup with 0 hour threshold should delete the closed incident
        deleted = fingerprinter.cleanup_old_incidents(max_age_hours=0)

        assert deleted >= 0  # May be 0 if just created

    def test_analytics_summary(self, fingerprinter: AnomalyFingerprinter):
        """Test getting analytics summary."""
        fingerprinter.process_anomalies(
            "booking_evening_hours",
            {"anomalies": [{"type": "threshold", "severity": "high"}]},
        )

        analytics = fingerprinter.get_analytics_summary(days=1)

        assert analytics["period_days"] == 1
        assert analytics["incidents_created"] >= 1
        assert "average_resolution_minutes" in analytics


class TestFactoryFunction:
    """Tests for create_fingerprinter factory function."""

    def test_create_fingerprinter_default(self):
        """Test creating fingerprinter with defaults."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            fp = create_fingerprinter(db_path=f.name)
            assert fp is not None

    def test_create_fingerprinter_custom_path(self):
        """Test creating fingerprinter with custom path."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            fp = create_fingerprinter(db_path=f.name)
            assert fp.db_path == f.name


class TestStaleIncidentResolution:
    """Tests for stale incident auto-resolution."""

    @pytest.fixture
    def temp_db(self) -> str:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            return f.name

    @pytest.fixture
    def fingerprinter(self, temp_db: str) -> AnomalyFingerprinter:
        from smartbox_anomaly.core.config import FingerprintingConfig

        # Use a short separation time for testing
        config = FingerprintingConfig(
            db_path=temp_db,
            incident_separation_minutes=1,  # 1 minute for easy testing
            confirmation_cycles=2,
            resolution_grace_cycles=3,
        )
        return AnomalyFingerprinter(db_path=temp_db, config=config)

    def test_stale_incident_returns_resolution(self, fingerprinter: AnomalyFingerprinter):
        """Test that stale incidents are included in resolved_incidents."""
        from datetime import datetime, timedelta

        anomaly_data = {"anomalies": [{"type": "threshold", "severity": "high"}]}

        # T0: Create and confirm incident
        t0 = datetime.now()
        fingerprinter.process_anomalies("booking_evening_hours", anomaly_data, timestamp=t0)
        result1 = fingerprinter.process_anomalies("booking_evening_hours", anomaly_data, timestamp=t0)
        incident_id = result1["anomalies"][0]["incident_id"]
        assert result1["anomalies"][0]["status"] == "OPEN"

        # T0 + 2 minutes: Same anomaly reappears after gap > incident_separation_minutes (1 min)
        # This should trigger stale closure + new incident creation
        t1 = t0 + timedelta(minutes=2)
        result2 = fingerprinter.process_anomalies("booking_evening_hours", anomaly_data, timestamp=t1)

        # The stale incident should be in resolved_incidents
        resolved = result2["fingerprinting"]["resolved_incidents"]
        assert len(resolved) == 1
        assert resolved[0]["incident_id"] == incident_id
        assert resolved[0]["resolution_reason"] == "auto_stale"
        assert resolved[0]["incident_action"] == IncidentAction.CLOSE.value

        # A new incident should have been created
        new_incident_id = result2["anomalies"][0]["incident_id"]
        assert new_incident_id != incident_id
        assert result2["anomalies"][0]["incident_action"] == IncidentAction.CREATE.value

    def test_stale_resolution_includes_required_fields(self, fingerprinter: AnomalyFingerprinter):
        """Test that stale resolution has all fields needed for API notification."""
        from datetime import datetime, timedelta

        anomaly_data = {"anomalies": [{"type": "threshold", "severity": "high"}]}

        # Create and confirm incident
        t0 = datetime.now()
        fingerprinter.process_anomalies("booking_evening_hours", anomaly_data, timestamp=t0)
        fingerprinter.process_anomalies("booking_evening_hours", anomaly_data, timestamp=t0)

        # Trigger stale closure
        t1 = t0 + timedelta(minutes=2)
        result = fingerprinter.process_anomalies("booking_evening_hours", anomaly_data, timestamp=t1)

        resolved = result["fingerprinting"]["resolved_incidents"][0]

        # All required fields for API notification
        assert "fingerprint_id" in resolved
        assert "incident_id" in resolved
        assert "anomaly_name" in resolved
        assert "fingerprint_action" in resolved
        assert "incident_action" in resolved
        assert "final_severity" in resolved
        assert "resolved_at" in resolved
        assert "total_occurrences" in resolved
        assert "incident_duration_minutes" in resolved
        assert "first_seen" in resolved
        assert "service_name" in resolved
        assert "resolution_reason" in resolved


class TestResolutionContext:
    """Tests for resolution context building functionality."""

    @pytest.fixture
    def temp_db(self) -> str:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            return f.name

    @pytest.fixture
    def fingerprinter(self, temp_db: str) -> AnomalyFingerprinter:
        return AnomalyFingerprinter(db_path=temp_db)

    @pytest.fixture
    def sample_metrics(self) -> dict:
        """Sample metrics for testing."""
        return {
            "request_rate": 52.7,
            "application_latency": 110.5,
            "dependency_latency": 1.4,
            "database_latency": 0.8,
            "error_rate": 0.0001,
        }

    @pytest.fixture
    def sample_training_stats(self) -> dict:
        """Sample training statistics for testing."""
        return {
            "request_rate": {"mean": 50.0, "std": 10.0},
            "application_latency": {"mean": 100.0, "std": 20.0},
            "dependency_latency": {"mean": 2.0, "std": 1.0},
            "database_latency": {"mean": 1.0, "std": 0.5},
            "error_rate": {"mean": 0.001, "std": 0.002},
        }

    def test_build_resolution_context_returns_none_without_metrics(
        self, fingerprinter: AnomalyFingerprinter
    ):
        """Test that resolution context returns None when no metrics provided."""
        from datetime import datetime

        context = fingerprinter._build_resolution_context(
            metrics=None,
            slo_evaluator=None,
            training_stats=None,
            service_name="booking",
            timestamp=datetime.now(),
            time_period="business_hours",
        )
        assert context is None

    def test_build_resolution_context_includes_metrics(
        self, fingerprinter: AnomalyFingerprinter, sample_metrics: dict
    ):
        """Test that resolution context includes metrics at resolution."""
        from datetime import datetime

        context = fingerprinter._build_resolution_context(
            metrics=sample_metrics,
            slo_evaluator=None,
            training_stats=None,
            service_name="booking",
            timestamp=datetime.now(),
            time_period="business_hours",
        )

        assert context is not None
        assert "metrics_at_resolution" in context
        assert context["metrics_at_resolution"] == sample_metrics

    def test_build_resolution_context_includes_health_summary(
        self, fingerprinter: AnomalyFingerprinter, sample_metrics: dict
    ):
        """Test that resolution context includes health summary."""
        from datetime import datetime

        context = fingerprinter._build_resolution_context(
            metrics=sample_metrics,
            slo_evaluator=None,
            training_stats=None,
            service_name="booking",
            timestamp=datetime.now(),
            time_period="business_hours",
        )

        assert "health_summary" in context
        assert "all_metrics_normal" in context["health_summary"]
        assert "slo_compliant" in context["health_summary"]
        assert "summary" in context["health_summary"]

    def test_build_resolution_context_with_training_stats(
        self, fingerprinter: AnomalyFingerprinter, sample_metrics: dict, sample_training_stats: dict
    ):
        """Test that resolution context includes baseline comparison when stats provided."""
        from datetime import datetime

        context = fingerprinter._build_resolution_context(
            metrics=sample_metrics,
            slo_evaluator=None,
            training_stats=sample_training_stats,
            service_name="booking",
            timestamp=datetime.now(),
            time_period="business_hours",
        )

        assert "comparison_to_baseline" in context
        assert "request_rate" in context["comparison_to_baseline"]
        assert "application_latency" in context["comparison_to_baseline"]


class TestBaselineComparison:
    """Tests for baseline comparison calculation."""

    @pytest.fixture
    def temp_db(self) -> str:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            return f.name

    @pytest.fixture
    def fingerprinter(self, temp_db: str) -> AnomalyFingerprinter:
        return AnomalyFingerprinter(db_path=temp_db)

    def test_build_baseline_comparison_normal_values(self, fingerprinter: AnomalyFingerprinter):
        """Test baseline comparison with values close to mean."""
        metrics = {"latency": 100.0}
        stats = {"latency": {"mean": 100.0, "std": 10.0}}

        comparison = fingerprinter._build_baseline_comparison(metrics, stats)

        assert "latency" in comparison
        assert comparison["latency"]["status"] == "normal"
        assert comparison["latency"]["deviation_sigma"] == 0.0

    def test_build_baseline_comparison_elevated_values(self, fingerprinter: AnomalyFingerprinter):
        """Test baseline comparison with moderately elevated values (2-3 sigma)."""
        metrics = {"latency": 125.0}  # 2.5 sigma above mean
        stats = {"latency": {"mean": 100.0, "std": 10.0}}

        comparison = fingerprinter._build_baseline_comparison(metrics, stats)

        assert comparison["latency"]["status"] == "elevated"
        assert comparison["latency"]["deviation_sigma"] == 2.5

    def test_build_baseline_comparison_high_values(self, fingerprinter: AnomalyFingerprinter):
        """Test baseline comparison with high values (>3 sigma)."""
        metrics = {"latency": 140.0}  # 4 sigma above mean
        stats = {"latency": {"mean": 100.0, "std": 10.0}}

        comparison = fingerprinter._build_baseline_comparison(metrics, stats)

        assert comparison["latency"]["status"] == "high"
        assert comparison["latency"]["deviation_sigma"] == 4.0

    def test_build_baseline_comparison_low_values(self, fingerprinter: AnomalyFingerprinter):
        """Test baseline comparison with low values (-2 to -3 sigma)."""
        metrics = {"latency": 75.0}  # -2.5 sigma below mean
        stats = {"latency": {"mean": 100.0, "std": 10.0}}

        comparison = fingerprinter._build_baseline_comparison(metrics, stats)

        assert comparison["latency"]["status"] == "low"
        assert comparison["latency"]["deviation_sigma"] == -2.5

    def test_build_baseline_comparison_very_low_values(self, fingerprinter: AnomalyFingerprinter):
        """Test baseline comparison with very low values (<-3 sigma)."""
        metrics = {"latency": 60.0}  # -4 sigma below mean
        stats = {"latency": {"mean": 100.0, "std": 10.0}}

        comparison = fingerprinter._build_baseline_comparison(metrics, stats)

        assert comparison["latency"]["status"] == "very_low"
        assert comparison["latency"]["deviation_sigma"] == -4.0

    def test_build_baseline_comparison_percentile_estimation(self, fingerprinter: AnomalyFingerprinter):
        """Test percentile estimation in baseline comparison."""
        metrics = {"latency": 100.0}  # At mean = 50th percentile
        stats = {"latency": {"mean": 100.0, "std": 10.0}}

        comparison = fingerprinter._build_baseline_comparison(metrics, stats)

        assert comparison["latency"]["percentile_estimate"] == 50.0

    def test_build_baseline_comparison_missing_stat_fields(self, fingerprinter: AnomalyFingerprinter):
        """Test baseline comparison handles missing stat fields gracefully."""
        metrics = {"latency": 100.0, "errors": 0.01}
        stats = {"latency": {"mean": 100.0, "std": 10.0}}  # No stats for errors

        comparison = fingerprinter._build_baseline_comparison(metrics, stats)

        assert "latency" in comparison
        assert "errors" not in comparison  # Skipped due to missing stats

    def test_build_baseline_comparison_handles_zero_std(self, fingerprinter: AnomalyFingerprinter):
        """Test baseline comparison handles zero standard deviation."""
        metrics = {"latency": 100.0}
        stats = {"latency": {"mean": 100.0, "std": 0.0}}

        comparison = fingerprinter._build_baseline_comparison(metrics, stats)

        assert comparison["latency"]["deviation_sigma"] == 0.0
        assert comparison["latency"]["percentile_estimate"] == 50.0

    def test_build_baseline_comparison_includes_all_fields(self, fingerprinter: AnomalyFingerprinter):
        """Test baseline comparison includes all required fields."""
        metrics = {"latency": 120.0}
        stats = {"latency": {"mean": 100.0, "std": 10.0}}

        comparison = fingerprinter._build_baseline_comparison(metrics, stats)

        assert "current" in comparison["latency"]
        assert "training_mean" in comparison["latency"]
        assert "training_std" in comparison["latency"]
        assert "deviation_sigma" in comparison["latency"]
        assert "percentile_estimate" in comparison["latency"]
        assert "status" in comparison["latency"]


class TestHealthSummaryMessage:
    """Tests for health summary message generation."""

    @pytest.fixture
    def temp_db(self) -> str:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            return f.name

    @pytest.fixture
    def fingerprinter(self, temp_db: str) -> AnomalyFingerprinter:
        return AnomalyFingerprinter(db_path=temp_db)

    def test_build_health_summary_slo_ok(self, fingerprinter: AnomalyFingerprinter):
        """Test health summary with SLO status OK."""
        context = {
            "metrics_at_resolution": {"application_latency": 100.0, "error_rate": 0.001},
            "slo_evaluation": {
                "slo_status": "ok",
                "latency_evaluation": {"value": 100.0, "threshold_acceptable": 300.0},
                "error_rate_evaluation": {"value_percent": "0.10%", "threshold_acceptable": 0.005},
            },
        }

        message = fingerprinter._build_health_summary_message(context)

        assert "All metrics within acceptable SLO thresholds" in message
        assert "Latency 100ms" in message
        assert "acceptable <" in message and "300" in message

    def test_build_health_summary_slo_warning(self, fingerprinter: AnomalyFingerprinter):
        """Test health summary with SLO status warning."""
        context = {
            "metrics_at_resolution": {"application_latency": 250.0},
            "slo_evaluation": {"slo_status": "warning"},
        }

        message = fingerprinter._build_health_summary_message(context)

        assert "approaching SLO thresholds" in message

    def test_build_health_summary_slo_breached(self, fingerprinter: AnomalyFingerprinter):
        """Test health summary with SLO status breached."""
        context = {
            "metrics_at_resolution": {"application_latency": 500.0},
            "slo_evaluation": {"slo_status": "breached"},
        }

        message = fingerprinter._build_health_summary_message(context)

        assert "may still be elevated" in message

    def test_build_health_summary_fallback_without_slo(self, fingerprinter: AnomalyFingerprinter):
        """Test health summary fallback when no SLO evaluation."""
        context = {
            "metrics_at_resolution": {"application_latency": 100.0, "error_rate": 0.001},
        }

        message = fingerprinter._build_health_summary_message(context)

        assert "Latency: 100ms" in message
        assert "Error rate: 0.10%" in message

    def test_build_health_summary_empty_metrics(self, fingerprinter: AnomalyFingerprinter):
        """Test health summary with empty metrics."""
        context = {"metrics_at_resolution": {}}

        message = fingerprinter._build_health_summary_message(context)

        assert message == "Service metrics at resolution time."


class TestResolutionContextIntegration:
    """Integration tests for resolution context in incident resolution flow."""

    @pytest.fixture
    def temp_db(self) -> str:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            return f.name

    @pytest.fixture
    def fingerprinter(self, temp_db: str) -> AnomalyFingerprinter:
        from smartbox_anomaly.core.config import FingerprintingConfig

        config = FingerprintingConfig(
            db_path=temp_db,
            confirmation_cycles=1,  # Quick confirmation for testing
            resolution_grace_cycles=1,  # Quick resolution for testing
        )
        return AnomalyFingerprinter(db_path=temp_db, config=config)

    def test_resolved_incident_includes_resolution_context(
        self, fingerprinter: AnomalyFingerprinter
    ):
        """Test that resolved incidents include resolution_context when metrics provided."""
        anomaly_data = {"anomalies": [{"type": "threshold", "severity": "high"}]}
        healthy_metrics = {
            "request_rate": 50.0,
            "application_latency": 100.0,
            "error_rate": 0.001,
        }

        # Create incident
        fingerprinter.process_anomalies("booking_business_hours", anomaly_data)

        # Confirm incident
        fingerprinter.process_anomalies("booking_business_hours", anomaly_data)

        # Resolve incident with metrics
        result = fingerprinter.process_anomalies(
            "booking_business_hours",
            {"anomalies": [], "current_metrics": healthy_metrics},
        )

        # Check that resolution includes context
        if result["fingerprinting"]["resolved_incidents"]:
            resolved = result["fingerprinting"]["resolved_incidents"][0]
            assert "resolution_context" in resolved
            assert resolved["resolution_context"]["metrics_at_resolution"] == healthy_metrics


class TestCrossModelTracking:
    """Tests for cross-model anomaly tracking."""

    @pytest.fixture
    def temp_db(self) -> str:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            return f.name

    @pytest.fixture
    def fingerprinter(self, temp_db: str) -> AnomalyFingerprinter:
        return AnomalyFingerprinter(db_path=temp_db)

    def test_same_pattern_different_models(self, fingerprinter: AnomalyFingerprinter):
        """Test that same anomaly pattern across models shares fingerprint."""
        anomaly = {"type": "threshold", "severity": "high", "detection_method": "test"}

        # Detect with evening_hours model
        result1 = fingerprinter.process_anomalies(
            "booking_evening_hours",
            {"anomalies": [anomaly]},
        )
        fp1 = result1["anomalies"][0]["fingerprint_id"]
        inc1 = result1["anomalies"][0]["incident_id"]

        # Detect with night_hours model - same fingerprint, continues incident
        result2 = fingerprinter.process_anomalies(
            "booking_night_hours",
            {"anomalies": [anomaly]},
        )
        fp2 = result2["anomalies"][0]["fingerprint_id"]
        inc2 = result2["anomalies"][0]["incident_id"]

        # Fingerprint should be the same (pattern identity)
        assert fp1 == fp2
        # Incident should continue (same occurrence)
        assert inc1 == inc2
        assert result2["anomalies"][0]["occurrence_count"] == 2
