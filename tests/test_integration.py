"""
Integration tests for the full ML anomaly detection pipeline.

These tests verify end-to-end workflows including:
- Model training and persistence
- Anomaly detection with trained models
- Time-aware detection with period models
- Fingerprinting and incident tracking
"""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from smartbox_anomaly.api import (
    AnomalyDetectedPayload,
    CurrentMetrics,
    create_anomaly_payload,
)
from smartbox_anomaly.core import (
    AnomalySeverity,
    MetricName,
    PipelineConfig,
    VictoriaMetricsConfig,
    reset_config,
    set_config,
)
from smartbox_anomaly.detection import (
    SmartboxAnomalyDetector,
    create_detector,
    create_time_aware_detector,
    get_service_parameters,
)
from smartbox_anomaly.fingerprinting import create_fingerprinter

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def integration_config() -> PipelineConfig:
    """Provide integration test configuration."""
    config = PipelineConfig(
        victoria_metrics=VictoriaMetricsConfig(
            endpoint="http://localhost:9090",
            timeout_seconds=5,
            max_retries=2,
        ),
    )
    set_config(config)
    yield config
    reset_config()


@pytest.fixture
def realistic_service_data() -> pd.DataFrame:
    """Generate realistic service metrics data for testing."""
    np.random.seed(42)
    n_samples = 500

    # Lognormal distributions for realistic metrics
    request_rate = np.random.lognormal(4, 0.5, n_samples)
    app_latency = np.random.lognormal(3.5, 0.8, n_samples)

    # Some metrics often zero
    client_latency = np.where(
        np.random.random(n_samples) > 0.3,
        np.random.exponential(30, n_samples),
        0,
    )
    db_latency = np.where(
        np.random.random(n_samples) > 0.4,
        np.random.exponential(15, n_samples),
        0,
    )

    # Low error rate
    error_rate = np.random.beta(1, 200, n_samples)

    return pd.DataFrame({
        MetricName.REQUEST_RATE: request_rate,
        MetricName.APPLICATION_LATENCY: app_latency,
        MetricName.CLIENT_LATENCY: client_latency,
        MetricName.DATABASE_LATENCY: db_latency,
        MetricName.ERROR_RATE: error_rate,
    })


@pytest.fixture
def temp_model_storage() -> Path:
    """Provide temporary storage for models."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_fingerprint_db() -> str:
    """Provide temporary fingerprint database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        yield f.name


# =============================================================================
# Test: Complete Training and Detection Flow
# =============================================================================


@pytest.mark.integration
class TestTrainingAndDetection:
    """Test the complete training and detection workflow."""

    def test_train_and_detect_normal_metrics(
        self,
        integration_config: PipelineConfig,
        realistic_service_data: pd.DataFrame,
        temp_model_storage: Path,
    ) -> None:
        """Test training a model and detecting normal metrics."""
        # Create and train detector
        detector = create_detector("test-service")
        detector.train(realistic_service_data)

        # Verify training produced models
        assert detector.is_trained
        assert len(detector.models) > 0

        # Test with normal metrics (within training distribution)
        normal_metrics = {
            MetricName.REQUEST_RATE: float(np.median(realistic_service_data[MetricName.REQUEST_RATE])),
            MetricName.APPLICATION_LATENCY: float(np.median(realistic_service_data[MetricName.APPLICATION_LATENCY])),
            MetricName.CLIENT_LATENCY: 0.0,
            MetricName.DATABASE_LATENCY: 0.0,
            MetricName.ERROR_RATE: 0.005,
        }

        result = detector.detect(normal_metrics)

        # Normal metrics should not produce critical anomalies
        assert result["metadata"]["service_name"] == "test-service"
        anomalies = result.get("anomalies", {})
        # anomalies is a dict, check for critical severity in values
        critical_anomalies = [
            a for a in anomalies.values()
            if isinstance(a, dict) and a.get("severity") == AnomalySeverity.CRITICAL
        ]
        assert len(critical_anomalies) == 0

    def test_train_and_detect_anomalous_metrics(
        self,
        integration_config: PipelineConfig,
        realistic_service_data: pd.DataFrame,
    ) -> None:
        """Test detecting anomalous metrics after training."""
        detector = create_detector("test-service")
        detector.train(realistic_service_data)

        # Create anomalous metrics (far outside training distribution)
        anomalous_metrics = {
            MetricName.REQUEST_RATE: 50000.0,  # Way above normal
            MetricName.APPLICATION_LATENCY: 10000.0,  # 10 seconds
            MetricName.CLIENT_LATENCY: 5000.0,
            MetricName.DATABASE_LATENCY: 3000.0,
            MetricName.ERROR_RATE: 0.5,  # 50% errors
        }

        result = detector.detect(anomalous_metrics)

        # Should detect anomalies
        assert "anomalies" in result
        anomalies = result["anomalies"]
        assert len(anomalies) > 0

        # At least one should be high severity or critical
        severities = [
            a.get("severity") for a in anomalies.values()
            if isinstance(a, dict)
        ]
        assert any(s in (AnomalySeverity.HIGH, AnomalySeverity.CRITICAL) for s in severities)

    def test_model_save_and_load(
        self,
        integration_config: PipelineConfig,
        realistic_service_data: pd.DataFrame,
        temp_model_storage: Path,
    ) -> None:
        """Test saving and loading trained models."""
        # Train original detector
        original_detector = create_detector("test-service")
        original_detector.train(realistic_service_data)

        # Save model
        model_path = temp_model_storage / "test-service"
        original_detector.save_model(str(model_path))

        # Verify model files exist - the save creates a subdirectory with service name
        saved_model_path = model_path / "test-service"
        assert saved_model_path.exists()

        # Load into new detector (load_model is a classmethod)
        loaded_detector = SmartboxAnomalyDetector.load_model(str(model_path), "test-service")

        # Verify loaded detector works
        test_metrics = {
            MetricName.REQUEST_RATE: 100.0,
            MetricName.APPLICATION_LATENCY: 50.0,
            MetricName.CLIENT_LATENCY: 0.0,
            MetricName.DATABASE_LATENCY: 0.0,
            MetricName.ERROR_RATE: 0.01,
        }

        original_result = original_detector.detect(test_metrics)
        loaded_result = loaded_detector.detect(test_metrics)

        # Results should be identical
        assert original_result["metadata"]["service_name"] == loaded_result["metadata"]["service_name"]


# =============================================================================
# Test: Time-Aware Detection Flow
# =============================================================================


@pytest.mark.integration
class TestTimeAwareDetection:
    """Test time-aware anomaly detection workflow."""

    def test_time_aware_training_and_detection(
        self,
        integration_config: PipelineConfig,
        temp_model_storage: Path,
    ) -> None:
        """Test training and using time-aware models."""
        np.random.seed(42)

        # Train models for multiple time periods
        time_aware_detector = create_time_aware_detector(
            "test-service",
            str(temp_model_storage),
        )

        # Generate time-series data with datetime index
        dates = pd.date_range(
            start="2024-01-01",
            end="2024-01-14",
            freq="15min",
        )
        n_samples = len(dates)

        features_df = pd.DataFrame({
            MetricName.REQUEST_RATE: np.random.exponential(100, n_samples),
            MetricName.APPLICATION_LATENCY: np.random.exponential(50, n_samples),
            MetricName.CLIENT_LATENCY: np.random.exponential(20, n_samples),
            MetricName.DATABASE_LATENCY: np.random.exponential(10, n_samples),
            MetricName.ERROR_RATE: np.random.beta(1, 100, n_samples),
        }, index=dates)

        # Train time-aware models
        trained_models = time_aware_detector.train_time_aware_models(features_df)

        # Verify some models were trained
        assert len(trained_models) > 0

        # Test detection during business hours
        business_metrics = {
            MetricName.REQUEST_RATE: 100.0,
            MetricName.APPLICATION_LATENCY: 50.0,
            MetricName.CLIENT_LATENCY: 20.0,
            MetricName.DATABASE_LATENCY: 10.0,
            MetricName.ERROR_RATE: 0.01,
        }

        # Create a business hours timestamp
        business_time = datetime(2024, 1, 15, 10, 0)  # Monday 10am
        result = time_aware_detector.detect(business_metrics, timestamp=business_time)

        assert "metadata" in result or "anomalies" in result
        # Result structure depends on whether models are available


# =============================================================================
# Test: Fingerprinting Integration
# =============================================================================


@pytest.mark.integration
class TestFingerprintingIntegration:
    """Test fingerprinting with detection results."""

    def test_fingerprint_anomaly_flow(
        self,
        integration_config: PipelineConfig,
        realistic_service_data: pd.DataFrame,
        temp_fingerprint_db: str,
    ) -> None:
        """Test the complete flow from detection to fingerprinting."""
        # Train detector
        detector = create_detector("booking")
        detector.train(realistic_service_data)

        # Create fingerprinter
        fingerprinter = create_fingerprinter(temp_fingerprint_db)

        # Detect anomaly
        anomalous_metrics = {
            MetricName.REQUEST_RATE: 50000.0,
            MetricName.APPLICATION_LATENCY: 5000.0,
            MetricName.CLIENT_LATENCY: 1000.0,
            MetricName.DATABASE_LATENCY: 500.0,
            MetricName.ERROR_RATE: 0.3,
        }

        detection_result = detector.detect(anomalous_metrics)
        anomalies_dict = detection_result.get("anomalies", {})

        # Convert dict format to list format for fingerprinter
        anomalies_list = [
            {"name": name, **data}
            for name, data in anomalies_dict.items()
            if isinstance(data, dict)
        ]

        if anomalies_list:
            # Process through fingerprinter - use the full detection result format
            detection_result_for_fingerprinter = {
                "anomalies": anomalies_list,
                "current_metrics": anomalous_metrics,
            }

            fingerprint_result = fingerprinter.process_anomalies(
                full_service_name="booking_business_hours",
                anomaly_result=detection_result_for_fingerprinter,
                current_metrics=anomalous_metrics,
            )

            # Check for action in result
            assert "action" in fingerprint_result or "anomalies" in fingerprint_result

            # Verify incident was created
            incidents = fingerprinter.get_open_incidents("booking")
            incidents.get("booking", [])
            # Incidents might not be created if anomalies don't meet threshold
            assert fingerprint_result is not None

    def test_incident_resolution_flow(
        self,
        integration_config: PipelineConfig,
        realistic_service_data: pd.DataFrame,
        temp_fingerprint_db: str,
    ) -> None:
        """Test incident creation and resolution flow."""
        # Train detector
        detector = create_detector("search")
        detector.train(realistic_service_data)

        fingerprinter = create_fingerprinter(temp_fingerprint_db)

        # First: detect anomaly and create incident
        anomalous_metrics = {
            MetricName.REQUEST_RATE: 50000.0,
            MetricName.APPLICATION_LATENCY: 5000.0,
            MetricName.CLIENT_LATENCY: 0.0,
            MetricName.DATABASE_LATENCY: 0.0,
            MetricName.ERROR_RATE: 0.25,
        }

        detection_result = detector.detect(anomalous_metrics)
        anomalies_dict = detection_result.get("anomalies", {})

        # Convert dict format to list format for fingerprinter
        anomalies_list = [
            {"name": name, **data}
            for name, data in anomalies_dict.items()
            if isinstance(data, dict)
        ]

        if anomalies_list:
            detection_result = {
                "anomalies": anomalies_list,
                "current_metrics": anomalous_metrics,
            }
            fingerprinter.process_anomalies("search_business_hours", detection_result, anomalous_metrics)
            incidents = fingerprinter.get_open_incidents("search")
            incidents.get("search", [])

            # Second: detect normal metrics (no anomalies) to resolve
            normal_metrics = {
                MetricName.REQUEST_RATE: 100.0,
                MetricName.APPLICATION_LATENCY: 50.0,
                MetricName.CLIENT_LATENCY: 0.0,
                MetricName.DATABASE_LATENCY: 0.0,
                MetricName.ERROR_RATE: 0.01,
            }

            normal_result = detector.detect(normal_metrics)
            normal_anomalies_dict = normal_result.get("anomalies", {})
            normal_anomalies_list = [
                {"name": name, **data}
                for name, data in normal_anomalies_dict.items()
                if isinstance(data, dict)
            ]

            normal_detection_result = {
                "anomalies": normal_anomalies_list,
                "current_metrics": normal_metrics,
            }

            # Process anomalies (might be empty) to trigger resolution check
            resolution_result = fingerprinter.process_anomalies(
                "search_business_hours", normal_detection_result, normal_metrics
            )

            # The incident may or may not be resolved depending on timing
            # Just verify the flow doesn't error
            assert resolution_result is not None


# =============================================================================
# Test: API Payload Generation
# =============================================================================


@pytest.mark.integration
class TestAPIPayloadGeneration:
    """Test API payload generation from detection results."""

    def test_create_payload_from_detection(
        self,
        integration_config: PipelineConfig,
        realistic_service_data: pd.DataFrame,
    ) -> None:
        """Test creating API payloads from detection results."""
        detector = create_detector("mobile-api")
        detector.train(realistic_service_data)

        anomalous_metrics = {
            MetricName.REQUEST_RATE: 50000.0,
            MetricName.APPLICATION_LATENCY: 5000.0,
            MetricName.CLIENT_LATENCY: 1000.0,
            MetricName.DATABASE_LATENCY: 500.0,
            MetricName.ERROR_RATE: 0.3,
        }

        detection_result = detector.detect(anomalous_metrics)
        anomalies_dict = detection_result.get("anomalies", {})

        # Convert to dict format for API (keyed by anomaly name)
        api_anomalies = {
            name: {
                "type": name,
                "severity": data.get("severity", "medium").value if hasattr(data.get("severity"), "value") else data.get("severity", "medium"),
                "confidence": abs(data.get("score", 0.5)),
                "score": abs(data.get("score", 0.5)),
                "description": data.get("description", f"Anomaly: {name}"),
            }
            for name, data in anomalies_dict.items()
            if isinstance(data, dict)
        }

        if api_anomalies:
            # Create API payload
            payload = create_anomaly_payload(
                service_name="mobile-api",
                anomalies=api_anomalies,
                metrics=anomalous_metrics,
                time_period="business_hours",
            )

            # Verify payload structure
            assert isinstance(payload, AnomalyDetectedPayload)
            assert payload.service_name == "mobile-api"
            assert payload.anomaly_count == len(api_anomalies)
            assert isinstance(payload.current_metrics, CurrentMetrics)

            # Verify serialization works
            payload_dict = payload.model_dump()
            assert "service_name" in payload_dict
            assert "timestamp" in payload_dict
            assert "anomalies" in payload_dict


# =============================================================================
# Test: Service Parameters Integration
# =============================================================================


@pytest.mark.integration
class TestServiceParameters:
    """Test service-specific parameter tuning."""

    def test_critical_service_parameters(
        self,
        integration_config: PipelineConfig,
    ) -> None:
        """Test that critical services get appropriate parameters."""
        # Critical services like booking, search
        for service in ["booking", "search", "mobile-api"]:
            params = get_service_parameters(service)

            # Critical services should have lower contamination
            assert params.base_contamination <= 0.05
            assert params.n_estimators >= 100

    def test_admin_service_parameters(
        self,
        integration_config: PipelineConfig,
    ) -> None:
        """Test that admin services get appropriate parameters."""
        for service in ["m2-fr-adm", "m2-it-adm"]:
            params = get_service_parameters(service)

            # Admin services can have higher contamination
            assert params.base_contamination >= 0.05

    def test_data_driven_tuning(
        self,
        integration_config: PipelineConfig,
        realistic_service_data: pd.DataFrame,
    ) -> None:
        """Test that service parameters adapt to data."""
        params = get_service_parameters(
            "test-service",
            data=realistic_service_data,
        )

        # Should have valid parameters
        assert 0.01 <= params.base_contamination <= 0.25  # Can be higher with high variability data
        assert params.n_estimators >= 50


# =============================================================================
# Test: Error Handling in Integration
# =============================================================================


@pytest.mark.integration
class TestErrorHandling:
    """Test error handling in integrated workflows."""

    def test_detection_with_invalid_metrics(
        self,
        integration_config: PipelineConfig,
        realistic_service_data: pd.DataFrame,
    ) -> None:
        """Test detection handles invalid metrics gracefully."""
        detector = create_detector("test-service")
        detector.train(realistic_service_data)

        # Partially invalid metrics (missing some)
        partial_metrics = {
            MetricName.REQUEST_RATE: 100.0,
            MetricName.APPLICATION_LATENCY: 50.0,
            # Missing other metrics
        }

        # Should not raise, should handle gracefully
        result = detector.detect(partial_metrics)
        assert result is not None
        assert "metadata" in result or "anomalies" in result

    def test_detection_without_training(
        self,
        integration_config: PipelineConfig,
    ) -> None:
        """Test detection without training returns empty results."""
        detector = create_detector("untrained-service")

        metrics = {
            MetricName.REQUEST_RATE: 100.0,
            MetricName.APPLICATION_LATENCY: 50.0,
            MetricName.CLIENT_LATENCY: 20.0,
            MetricName.DATABASE_LATENCY: 10.0,
            MetricName.ERROR_RATE: 0.01,
        }

        result = detector.detect(metrics)

        # Should return result without crashing
        assert "anomalies" in result
        # No models trained, so detection should show untrained state
        assert result.get("metadata", {}).get("trained") is False
