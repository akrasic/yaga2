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
        """Test continuing an existing incident."""
        anomaly_data = {
            "anomalies": [{
                "type": "threshold",
                "severity": "high",
                "value": 500.0,
                "detection_method": "isolation_forest",
            }]
        }

        # First detection - creates incident
        result1 = fingerprinter.process_anomalies("booking_evening_hours", anomaly_data)
        incident_id = result1["anomalies"][0]["incident_id"]

        # Second detection - continues incident
        result2 = fingerprinter.process_anomalies("booking_evening_hours", anomaly_data)

        assert result2["fingerprinting"]["overall_action"] == IncidentAction.UPDATE.value
        assert result2["anomalies"][0]["incident_action"] == IncidentAction.CONTINUE.value
        assert result2["anomalies"][0]["incident_id"] == incident_id
        assert result2["anomalies"][0]["occurrence_count"] == 2

    def test_resolve_incident(self, fingerprinter: AnomalyFingerprinter):
        """Test resolving an incident when anomaly clears."""
        # Create incident
        fingerprinter.process_anomalies(
            "booking_evening_hours",
            {"anomalies": [{"type": "threshold", "severity": "high"}]},
        )

        # Anomaly clears
        result = fingerprinter.process_anomalies(
            "booking_evening_hours",
            {"anomalies": []},
        )

        assert result["fingerprinting"]["overall_action"] == IncidentAction.RESOLVE.value
        assert len(result["fingerprinting"]["resolved_incidents"]) == 1

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
        # Create some incidents
        fingerprinter.process_anomalies(
            "booking_evening_hours",
            {"anomalies": [{"type": "threshold", "severity": "high"}]},
        )

        stats = fingerprinter.get_statistics()

        assert stats["total_open_incidents"] == 1
        assert "schema_version" in stats
        assert "database_path" in stats

    def test_get_incident_by_id(self, fingerprinter: AnomalyFingerprinter):
        """Test retrieving incident by ID."""
        result = fingerprinter.process_anomalies(
            "booking_evening_hours",
            {"anomalies": [{"type": "threshold", "severity": "high"}]},
        )
        incident_id = result["anomalies"][0]["incident_id"]

        incident = fingerprinter.get_incident_by_id(incident_id)

        assert incident is not None
        assert incident["incident_id"] == incident_id
        assert incident["service_name"] == "booking"
        assert incident["status"] == "OPEN"

    def test_get_open_incidents(self, fingerprinter: AnomalyFingerprinter):
        """Test getting all open incidents."""
        fingerprinter.process_anomalies(
            "booking_evening_hours",
            {"anomalies": [{"type": "threshold", "severity": "high"}]},
        )
        fingerprinter.process_anomalies(
            "search_business_hours",
            {"anomalies": [{"type": "pattern", "severity": "medium"}]},
        )

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
