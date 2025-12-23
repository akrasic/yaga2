"""
Tests for API Pydantic models.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from smartbox_anomaly.api import (
    API_SCHEMA_VERSION,
    ActionSummary,
    AlertType,
    Anomaly,
    AnomalyBatchRequest,
    AnomalyDetectedPayload,
    AnomalyMetadata,
    ComparisonData,
    CurrentMetrics,
    DetectionContext,
    DetectionMethod,
    DetectionSignal,
    ErrorPayload,
    FeatureContribution,
    FingerprintingMetadata,
    HeartbeatPayload,
    IncidentAction,
    IncidentListResponse,
    IncidentResolvedPayload,
    IncidentSummary,
    IngestionResponse,
    ModelType,
    PayloadMetadata,
    ResolutionBatchRequest,
    ResolutionDetails,
    Severity,
    create_anomaly_payload,
    create_resolution_payload,
)


class TestEnums:
    """Tests for API enumerations."""

    def test_severity_values(self):
        """Test severity enum values."""
        assert Severity.NONE.value == "none"
        assert Severity.LOW.value == "low"
        assert Severity.MEDIUM.value == "medium"
        assert Severity.HIGH.value == "high"
        assert Severity.CRITICAL.value == "critical"

    def test_severity_max(self):
        """Test severity max_severity method."""
        severities = [Severity.LOW, Severity.HIGH, Severity.MEDIUM]
        assert Severity.max_severity(severities) == Severity.HIGH
        assert Severity.max_severity([]) == Severity.NONE

    def test_severity_from_score(self):
        """Test severity from isolation forest score."""
        assert Severity.from_score(-0.7) == Severity.CRITICAL
        assert Severity.from_score(-0.4) == Severity.HIGH
        assert Severity.from_score(-0.2) == Severity.MEDIUM
        assert Severity.from_score(0.0) == Severity.LOW

    def test_incident_action_values(self):
        """Test incident action enum values."""
        assert IncidentAction.CREATE.value == "CREATE"
        assert IncidentAction.CONTINUE.value == "CONTINUE"
        assert IncidentAction.RESOLVE.value == "RESOLVE"
        assert IncidentAction.CLOSE.value == "CLOSE"

    def test_detection_method_values(self):
        """Test detection method enum values."""
        assert DetectionMethod.ISOLATION_FOREST.value == "isolation_forest"
        assert DetectionMethod.MULTIVARIATE.value == "multivariate"
        assert DetectionMethod.THRESHOLD.value == "threshold"

    def test_model_type_values(self):
        """Test model type enum values."""
        assert ModelType.TIME_AWARE.value == "time_aware"
        assert ModelType.STANDARD_ML.value == "standard_ml"


class TestCurrentMetrics:
    """Tests for CurrentMetrics model."""

    def test_valid_metrics(self):
        """Test valid metrics creation."""
        metrics = CurrentMetrics(
            request_rate=100.0,
            application_latency=50.0,
            error_rate=0.01,
        )
        assert metrics.request_rate == 100.0
        assert metrics.application_latency == 50.0
        assert metrics.error_rate == 0.01

    def test_optional_metrics(self):
        """Test that all metrics are optional."""
        metrics = CurrentMetrics()
        assert metrics.request_rate is None
        assert metrics.application_latency is None

    def test_extra_metrics_allowed(self):
        """Test that extra metrics are allowed."""
        metrics = CurrentMetrics(
            request_rate=100.0,
            custom_metric=42.0,
        )
        assert metrics.request_rate == 100.0

    def test_validation_error_rate_bounds(self):
        """Test error rate must be between 0 and 1."""
        with pytest.raises(ValidationError):
            CurrentMetrics(error_rate=1.5)  # > 1.0

        with pytest.raises(ValidationError):
            CurrentMetrics(error_rate=-0.1)  # < 0

    def test_validation_non_negative(self):
        """Test latencies must be non-negative."""
        with pytest.raises(ValidationError):
            CurrentMetrics(application_latency=-10.0)


class TestFeatureContribution:
    """Tests for FeatureContribution model."""

    def test_valid_contribution(self):
        """Test valid feature contribution."""
        fc = FeatureContribution(
            feature="request_rate",
            contribution=0.35,
            direction="elevated",
            percentile=95.0,
        )
        assert fc.feature == "request_rate"
        assert fc.contribution == 0.35
        assert fc.direction == "elevated"

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            FeatureContribution(contribution=0.5)  # Missing feature


class TestComparisonData:
    """Tests for ComparisonData model."""

    def test_valid_comparison(self):
        """Test valid comparison data."""
        cd = ComparisonData(
            training_mean=100.0,
            training_std=20.0,
            current_value=150.0,
            deviation_sigma=2.5,
        )
        assert cd.training_mean == 100.0
        assert cd.deviation_sigma == 2.5

    def test_all_optional(self):
        """Test all fields are optional."""
        cd = ComparisonData()
        assert cd.training_mean is None


class TestAnomalyMetadata:
    """Tests for AnomalyMetadata model."""

    def test_fingerprinting_fields(self):
        """Test fingerprinting metadata fields."""
        meta = AnomalyMetadata(
            fingerprint_id="anomaly_abc123",
            incident_id="incident_xyz789",
            occurrence_count=5,
            first_seen="2024-01-15T10:00:00",
        )
        assert meta.fingerprint_id == "anomaly_abc123"
        assert meta.occurrence_count == 5

    def test_time_awareness_fields(self):
        """Test time-awareness metadata fields."""
        meta = AnomalyMetadata(
            time_period="business_hours",
            time_confidence=0.95,
            lazy_loaded=True,
        )
        assert meta.time_period == "business_hours"
        assert meta.time_confidence == 0.95

    def test_explainability_fields(self):
        """Test explainability metadata fields."""
        fc = FeatureContribution(feature="latency", contribution=0.5)
        meta = AnomalyMetadata(
            feature_contributions=[fc],
            business_impact="High user impact expected",
        )
        assert len(meta.feature_contributions) == 1
        assert meta.business_impact == "High user impact expected"


class TestDetectionSignal:
    """Tests for DetectionSignal model."""

    def test_valid_signal(self):
        """Test valid detection signal creation."""
        signal = DetectionSignal(
            method="isolation_forest",
            type="ml_isolation",
            severity=Severity.HIGH,
            score=-0.45,
            direction="high",
            percentile=95.0,
        )
        assert signal.method == "isolation_forest"
        assert signal.severity == Severity.HIGH
        assert signal.score == -0.45

    def test_pattern_signal(self):
        """Test pattern-based detection signal."""
        signal = DetectionSignal(
            method="named_pattern_matching",
            type="multivariate_pattern",
            severity=Severity.HIGH,
            score=-0.5,
            pattern="recent_degradation",
        )
        assert signal.pattern == "recent_degradation"


class TestAnomaly:
    """Tests for Anomaly model."""

    def test_valid_anomaly(self):
        """Test valid anomaly creation."""
        anomaly = Anomaly(
            type="consolidated",
            severity=Severity.HIGH,
            confidence=0.85,
            score=-0.45,
            description="Unusual metric combination detected",
            root_metric="application_latency",
        )
        assert anomaly.type == "consolidated"
        assert anomaly.severity == Severity.HIGH
        assert anomaly.confidence == 0.85
        assert anomaly.score == -0.45

    def test_confidence_clamping(self):
        """Test confidence is clamped to [0, 1]."""
        # Confidence above 1.0 is clamped to 1.0
        anomaly = Anomaly(
            type="test",
            severity=Severity.MEDIUM,
            confidence=1.5,
            score=-0.1,
            description="Test",
        )
        assert anomaly.confidence == 1.0

        # Confidence below 0.0 is clamped to 0.0
        anomaly2 = Anomaly(
            type="test",
            severity=Severity.MEDIUM,
            confidence=-0.5,
            score=-0.1,
            description="Test",
        )
        assert anomaly2.confidence == 0.0

    def test_with_detection_signals(self):
        """Test anomaly with detection signals."""
        signals = [
            DetectionSignal(
                method="isolation_forest",
                type="ml_isolation",
                severity=Severity.LOW,
                score=-0.05,
                direction="high",
            ),
            DetectionSignal(
                method="named_pattern_matching",
                type="multivariate_pattern",
                severity=Severity.HIGH,
                score=-0.5,
                pattern="recent_degradation",
            ),
        ]
        anomaly = Anomaly(
            type="consolidated",
            severity=Severity.HIGH,
            confidence=0.8,
            score=-0.5,
            description="Recent degradation detected",
            signal_count=2,
            detection_signals=signals,
        )
        assert len(anomaly.detection_signals) == 2
        assert anomaly.signal_count == 2

    def test_optional_fields(self):
        """Test optional fields."""
        anomaly = Anomaly(
            type="ml_isolation",
            severity=Severity.LOW,
            confidence=0.7,
            score=-0.1,
            description="Test anomaly",
            value=150.0,
            root_metric="application_latency",
            pattern_name="latency_high",
            fingerprint_id="anomaly_abc123",
            incident_id="incident_xyz789",
        )
        assert anomaly.value == 150.0
        assert anomaly.fingerprint_id == "anomaly_abc123"


class TestFingerprintingMetadata:
    """Tests for FingerprintingMetadata model."""

    def test_valid_fingerprinting(self):
        """Test valid fingerprinting metadata."""
        fp = FingerprintingMetadata(
            service_name="booking",
            model_name="business_hours",
            timestamp="2024-01-15T10:30:00",
            action_summary=ActionSummary(
                incident_creates=1,
                incident_continues=2,
                incident_closes=0,
            ),
            overall_action=IncidentAction.CREATE,
            detection_context=DetectionContext(
                model_used="booking_business_hours",
                inference_timestamp="2024-01-15T10:30:00",
            ),
        )
        assert fp.service_name == "booking"
        assert fp.action_summary.incident_creates == 1
        assert fp.overall_action == IncidentAction.CREATE


class TestAnomalyDetectedPayload:
    """Tests for AnomalyDetectedPayload model."""

    @pytest.fixture
    def valid_anomaly(self) -> Anomaly:
        """Create a valid anomaly for testing."""
        return Anomaly(
            type="consolidated",
            severity=Severity.HIGH,
            confidence=0.85,
            score=-0.45,
            description="Test anomaly",
            root_metric="application_latency",
        )

    def test_valid_payload(self, valid_anomaly):
        """Test valid payload creation."""
        payload = AnomalyDetectedPayload(
            alert_type=AlertType.ANOMALY_DETECTED,
            service_name="booking",
            timestamp="2024-01-15T10:30:00",
            time_period="business_hours",
            model_name="business_hours",
            overall_severity=Severity.HIGH,
            anomaly_count=1,
            current_metrics=CurrentMetrics(request_rate=100.0),
            anomalies={"recent_degradation": valid_anomaly},
        )
        assert payload.service_name == "booking"
        assert payload.overall_severity == Severity.HIGH
        assert len(payload.anomalies) == 1
        assert "recent_degradation" in payload.anomalies

    def test_anomaly_count_auto_correction(self, valid_anomaly):
        """Test anomaly_count is auto-corrected to match actual count."""
        payload = AnomalyDetectedPayload(
            alert_type=AlertType.ANOMALY_DETECTED,
            service_name="booking",
            timestamp="2024-01-15T10:30:00",
            time_period="business_hours",
            model_name="business_hours",
            overall_severity=Severity.HIGH,
            anomaly_count=5,  # Wrong count
            current_metrics=CurrentMetrics(),
            anomalies={"test": valid_anomaly},  # Only 1 anomaly
        )
        assert payload.anomaly_count == 1  # Auto-corrected

    def test_no_anomaly_payload(self):
        """Test empty anomalies creates no_anomaly payload."""
        payload = AnomalyDetectedPayload(
            alert_type=AlertType.ANOMALY_DETECTED,  # Will be corrected
            service_name="booking",
            timestamp="2024-01-15T10:30:00",
            time_period="business_hours",
            model_name="business_hours",
            overall_severity=Severity.HIGH,  # Will be corrected
            anomaly_count=0,
            current_metrics=CurrentMetrics(),
            anomalies={},  # Empty
        )
        assert payload.alert_type == AlertType.NO_ANOMALY
        assert payload.overall_severity == Severity.NONE
        assert payload.anomaly_count == 0

    def test_service_name_required(self, valid_anomaly):
        """Test service name is required."""
        with pytest.raises(ValidationError):
            AnomalyDetectedPayload(
                alert_type=AlertType.ANOMALY_DETECTED,
                service_name="",  # Empty
                timestamp="2024-01-15T10:30:00",
                time_period="business_hours",
                model_name="business_hours",
                overall_severity=Severity.HIGH,
                anomaly_count=1,
                current_metrics=CurrentMetrics(),
                anomalies={"test": valid_anomaly},
            )


class TestResolutionDetails:
    """Tests for ResolutionDetails model."""

    def test_valid_resolution(self):
        """Test valid resolution details."""
        details = ResolutionDetails(
            final_severity=Severity.HIGH,
            total_occurrences=10,
            incident_duration_minutes=45,
            first_seen="2024-01-15T09:45:00",
        )
        assert details.final_severity == Severity.HIGH
        assert details.total_occurrences == 10
        assert details.incident_duration_minutes == 45


class TestIncidentResolvedPayload:
    """Tests for IncidentResolvedPayload model."""

    def test_valid_resolution_payload(self):
        """Test valid resolution payload."""
        payload = IncidentResolvedPayload(
            service_name="booking",
            timestamp="2024-01-15T10:30:00",
            incident_id="incident_abc123",
            fingerprint_id="anomaly_xyz789",
            anomaly_name="multivariate_isolation_forest",
            resolution_details=ResolutionDetails(
                final_severity=Severity.HIGH,
                total_occurrences=5,
                incident_duration_minutes=30,
                first_seen="2024-01-15T10:00:00",
            ),
        )
        assert payload.incident_id == "incident_abc123"
        assert payload.resolution_details.total_occurrences == 5


class TestErrorPayload:
    """Tests for ErrorPayload model."""

    def test_valid_error_payload(self):
        """Test valid error payload."""
        payload = ErrorPayload(
            service="booking",
            timestamp="2024-01-15T10:30:00",
            error_message="Model not found",
            error_code="MODEL_NOT_FOUND",
        )
        assert payload.error_message == "Model not found"


class TestHeartbeatPayload:
    """Tests for HeartbeatPayload model."""

    def test_valid_heartbeat(self):
        """Test valid heartbeat payload."""
        payload = HeartbeatPayload(
            timestamp="2024-01-15T10:30:00",
            status="healthy",
            services_evaluated=15,
            anomalies_detected=3,
            incidents_resolved=1,
            inference_duration_ms=250.5,
        )
        assert payload.status == "healthy"
        assert payload.services_evaluated == 15


class TestBatchRequests:
    """Tests for batch request models."""

    def test_anomaly_batch_request(self):
        """Test anomaly batch request."""
        anomaly = Anomaly(
            type="ml_isolation",
            severity=Severity.MEDIUM,
            confidence=0.8,
            score=-0.15,
            description="Test",
        )
        alert = AnomalyDetectedPayload(
            alert_type=AlertType.ANOMALY_DETECTED,
            service_name="test",
            timestamp="2024-01-15T10:30:00",
            time_period="business_hours",
            model_name="business_hours",
            overall_severity=Severity.MEDIUM,
            anomaly_count=1,
            current_metrics=CurrentMetrics(),
            anomalies={"test_anomaly": anomaly},
        )
        batch = AnomalyBatchRequest(alerts=[alert])
        assert len(batch.alerts) == 1
        assert batch.schema_version == API_SCHEMA_VERSION

    def test_resolution_batch_request(self):
        """Test resolution batch request."""
        resolution = IncidentResolvedPayload(
            service_name="test",
            timestamp="2024-01-15T10:30:00",
            incident_id="inc_123",
            fingerprint_id="fp_456",
            anomaly_name="test_anomaly",
            resolution_details=ResolutionDetails(
                final_severity=Severity.LOW,
                total_occurrences=1,
                incident_duration_minutes=5,
                first_seen="2024-01-15T10:25:00",
            ),
        )
        batch = ResolutionBatchRequest(resolutions=[resolution])
        assert len(batch.resolutions) == 1


class TestResponseModels:
    """Tests for response models."""

    def test_ingestion_response(self):
        """Test ingestion response model."""
        response = IngestionResponse(
            success=True,
            processed_count=5,
            failed_count=1,
            errors=["Failed to process item 3"],
            timestamp="2024-01-15T10:30:00",
        )
        assert response.success is True
        assert response.processed_count == 5

    def test_incident_summary(self):
        """Test incident summary model."""
        summary = IncidentSummary(
            incident_id="inc_123",
            fingerprint_id="fp_456",
            service_name="booking",
            anomaly_name="high_latency",
            severity=Severity.HIGH,
            status="OPEN",
            first_seen="2024-01-15T10:00:00",
            last_updated="2024-01-15T10:30:00",
            occurrence_count=5,
            duration_minutes=30,
        )
        assert summary.status == "OPEN"

    def test_incident_list_response(self):
        """Test incident list response model."""
        response = IncidentListResponse(
            incidents=[],
            total_count=0,
            open_count=0,
            closed_count=0,
        )
        assert response.total_count == 0


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_anomaly_payload(self):
        """Test anomaly payload factory function."""
        anomalies = {
            "latency_high": {
                "type": "ml_isolation",
                "severity": "high",
                "confidence": 0.9,
                "score": -0.45,
                "description": "High latency detected",
                "root_metric": "application_latency",
            }
        }
        metrics = {"request_rate": 100.0, "application_latency": 500.0}

        payload = create_anomaly_payload(
            service_name="booking",
            anomalies=anomalies,
            metrics=metrics,
            time_period="business_hours",
        )

        assert payload.service_name == "booking"
        assert payload.overall_severity == Severity.HIGH
        assert len(payload.anomalies) == 1
        assert "latency_high" in payload.anomalies

    def test_create_anomaly_payload_with_timestamp(self):
        """Test anomaly payload with custom timestamp."""
        ts = datetime(2024, 1, 15, 10, 30, 0)
        payload = create_anomaly_payload(
            service_name="test",
            anomalies={"test": {"type": "test", "severity": "low", "confidence": 0.5, "score": -0.1, "description": "Test"}},
            metrics={},
            time_period="business_hours",
            timestamp=ts,
        )
        assert "2024-01-15" in payload.timestamp

    def test_create_resolution_payload(self):
        """Test resolution payload factory function."""
        payload = create_resolution_payload(
            service_name="booking",
            incident_id="inc_123",
            fingerprint_id="fp_456",
            anomaly_name="test_anomaly",
            final_severity="high",
            total_occurrences=5,
            duration_minutes=30,
            first_seen="2024-01-15T10:00:00",
        )

        assert payload.service_name == "booking"
        assert payload.incident_id == "inc_123"
        assert payload.resolution_details.total_occurrences == 5


class TestSchemaVersion:
    """Tests for schema version."""

    def test_schema_version_format(self):
        """Test schema version follows semver."""
        parts = API_SCHEMA_VERSION.split(".")
        assert len(parts) == 3
        assert all(part.isdigit() for part in parts)


class TestModelSerialization:
    """Tests for model serialization."""

    def test_anomaly_to_json(self):
        """Test anomaly serialization to JSON."""
        anomaly = Anomaly(
            type="ml_isolation",
            severity=Severity.HIGH,
            confidence=0.9,
            score=-0.45,
            description="Test anomaly",
        )
        json_data = anomaly.model_dump_json()
        assert "ml_isolation" in json_data
        assert "high" in json_data

    def test_payload_to_dict(self):
        """Test payload serialization to dict."""
        anomaly = Anomaly(
            type="ml_isolation",
            severity=Severity.MEDIUM,
            confidence=0.8,
            score=-0.15,
            description="Test",
        )
        payload = AnomalyDetectedPayload(
            alert_type=AlertType.ANOMALY_DETECTED,
            service_name="test",
            timestamp="2024-01-15T10:30:00",
            time_period="business_hours",
            model_name="business_hours",
            overall_severity=Severity.MEDIUM,
            anomaly_count=1,
            current_metrics=CurrentMetrics(request_rate=100.0),
            anomalies={"test_anomaly": anomaly},
        )
        data = payload.model_dump()
        assert data["service_name"] == "test"
        assert data["current_metrics"]["request_rate"] == 100.0
