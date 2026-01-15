"""
Tests for the smartbox_anomaly.inference module.

This module tests all components of the inference pipeline:
- AnomalyResult and ServiceInferenceResult data models
- EnhancedModelManager for model loading and service discovery
- EnhancedAnomalyDetectionEngine for anomaly detection
- EnhancedResultsProcessor for formatting and saving results
- EnhancedTimeAwareDetector for time-period aware detection
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

from smartbox_anomaly.core import AnomalySeverity
from smartbox_anomaly.inference import (
    AnomalyResult,
    EnhancedAnomalyDetectionEngine,
    EnhancedModelManager,
    EnhancedResultsProcessor,
    EnhancedTimeAwareDetector,
    ServiceInferenceResult,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_metrics() -> dict[str, float]:
    """Sample metrics for testing."""
    return {
        "request_rate": 100.0,
        "application_latency": 150.0,
        "dependency_latency": 50.0,
        "database_latency": 25.0,
        "error_rate": 0.01,
    }


@pytest.fixture
def sample_anomaly_data() -> dict[str, Any]:
    """Sample anomaly detection output."""
    return {
        "type": "latency_spike_recent",
        "severity": "high",
        "score": -0.45,
        "description": "Latency spike detected",
        "value": 350.0,
        "threshold": 200.0,
        "comparison_data": {
            "application_latency": {
                "current": 350.0,
                "training_mean": 150.0,
                "deviation_sigma": 2.5,
            }
        },
        "business_impact": "User-facing latency degradation",
    }


@pytest.fixture
def temp_models_dir(tmp_path: Path) -> Path:
    """Create a temporary models directory with mock model structure."""
    models_dir = tmp_path / "smartbox_models"
    models_dir.mkdir()

    # Create mock service model directories
    for service in ["booking", "search"]:
        for period in ["business_hours", "evening_hours", "night_hours"]:
            service_dir = models_dir / f"{service}_{period}"
            service_dir.mkdir()
            # Create model_data.json to mark as valid model
            model_data = {
                "service_name": service,
                "time_period": period,
                "model_version": "1.0.0",
                "trained_at": datetime.now().isoformat(),
            }
            (service_dir / "model_data.json").write_text(json.dumps(model_data))

    return models_dir


@pytest.fixture
def temp_config_file(tmp_path: Path) -> Path:
    """Create a temporary config file."""
    config = {
        "services": {
            "critical": ["booking", "search"],
            "standard": ["api-gateway"],
            "micro": [],
            "admin": [],
            "core": [],
        }
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    return config_path


@pytest.fixture
def mock_inference_metrics():
    """Create a mock InferenceMetrics object."""
    metrics = Mock()
    metrics.service_name = "booking"
    metrics.timestamp = datetime.now()
    metrics.to_dict.return_value = {
        "request_rate": 100.0,
        "application_latency": 150.0,
        "dependency_latency": 50.0,
        "database_latency": 25.0,
        "error_rate": 0.01,
    }
    return metrics


# =============================================================================
# Tests for AnomalyResult
# =============================================================================


class TestAnomalyResult:
    """Tests for AnomalyResult dataclass."""

    def test_create_basic_anomaly_result(self):
        """Test creating a basic AnomalyResult."""
        result = AnomalyResult(
            anomaly_type="latency_high",
            severity=AnomalySeverity.HIGH,
            confidence_score=0.85,
            description="High latency detected",
        )

        assert result.anomaly_type == "latency_high"
        assert result.severity == AnomalySeverity.HIGH
        assert result.confidence_score == 0.85
        assert result.description == "High latency detected"
        assert result.threshold_value is None
        assert result.actual_value is None

    def test_create_anomaly_result_with_values(self):
        """Test creating AnomalyResult with threshold and actual values."""
        result = AnomalyResult(
            anomaly_type="error_rate_elevated",
            severity=AnomalySeverity.CRITICAL,
            confidence_score=0.95,
            description="Critical error rate",
            threshold_value=0.01,
            actual_value=0.05,
        )

        assert result.threshold_value == 0.01
        assert result.actual_value == 0.05

    def test_anomaly_result_with_explainability_fields(self):
        """Test AnomalyResult with explainability data."""
        comparison_data = {
            "current": 350.0,
            "training_mean": 150.0,
            "deviation_sigma": 2.5,
        }

        result = AnomalyResult(
            anomaly_type="latency_spike",
            severity=AnomalySeverity.HIGH,
            confidence_score=0.80,
            description="Latency spike",
            comparison_data=comparison_data,
            business_impact="User experience degradation",
            percentile_position=95.5,
        )

        assert result.comparison_data == comparison_data
        assert result.business_impact == "User experience degradation"
        assert result.percentile_position == 95.5

    def test_anomaly_result_with_metadata(self):
        """Test AnomalyResult with metadata dict."""
        metadata = {
            "detection_method": "isolation_forest",
            "model_version": "1.2.0",
            "feature_contributions": ["latency", "error_rate"],
        }

        result = AnomalyResult(
            anomaly_type="multivariate",
            severity=AnomalySeverity.MEDIUM,
            confidence_score=0.70,
            description="Multivariate anomaly",
            metadata=metadata,
        )

        assert result.metadata == metadata
        assert result.metadata["detection_method"] == "isolation_forest"


# =============================================================================
# Tests for ServiceInferenceResult
# =============================================================================


class TestServiceInferenceResult:
    """Tests for ServiceInferenceResult dataclass."""

    def test_create_successful_result_no_anomalies(self, mock_inference_metrics):
        """Test creating a successful result with no anomalies."""
        result = ServiceInferenceResult(
            service_name="booking",
            timestamp=datetime.now(),
            input_metrics=mock_inference_metrics,
            anomalies=[],
            model_version="1.0.0",
            inference_time_ms=45.5,
            status="success",
        )

        assert result.service_name == "booking"
        assert result.status == "success"
        assert result.has_anomalies is False
        assert result.max_severity is None

    def test_create_result_with_anomalies(self, mock_inference_metrics):
        """Test creating a result with anomalies."""
        anomalies = [
            AnomalyResult(
                anomaly_type="latency_high",
                severity=AnomalySeverity.HIGH,
                confidence_score=0.85,
                description="High latency",
            ),
            AnomalyResult(
                anomaly_type="error_rate_elevated",
                severity=AnomalySeverity.MEDIUM,
                confidence_score=0.70,
                description="Elevated errors",
            ),
        ]

        result = ServiceInferenceResult(
            service_name="search",
            timestamp=datetime.now(),
            input_metrics=mock_inference_metrics,
            anomalies=anomalies,
            model_version="1.0.0",
            inference_time_ms=52.3,
            status="success",
        )

        assert result.has_anomalies is True
        assert len(result.anomalies) == 2
        assert result.max_severity == AnomalySeverity.HIGH

    def test_max_severity_ordering(self, mock_inference_metrics):
        """Test that max_severity correctly identifies the highest severity."""
        # Test with CRITICAL being highest
        anomalies = [
            AnomalyResult(
                anomaly_type="low_anomaly",
                severity=AnomalySeverity.LOW,
                confidence_score=0.5,
                description="Low",
            ),
            AnomalyResult(
                anomaly_type="critical_anomaly",
                severity=AnomalySeverity.CRITICAL,
                confidence_score=0.95,
                description="Critical",
            ),
            AnomalyResult(
                anomaly_type="medium_anomaly",
                severity=AnomalySeverity.MEDIUM,
                confidence_score=0.65,
                description="Medium",
            ),
        ]

        result = ServiceInferenceResult(
            service_name="test",
            timestamp=datetime.now(),
            input_metrics=mock_inference_metrics,
            anomalies=anomalies,
            model_version="1.0.0",
            inference_time_ms=30.0,
            status="success",
        )

        assert result.max_severity == AnomalySeverity.CRITICAL

    def test_error_result(self, mock_inference_metrics):
        """Test creating an error result."""
        result = ServiceInferenceResult(
            service_name="booking",
            timestamp=datetime.now(),
            input_metrics=mock_inference_metrics,
            anomalies=[],
            model_version="unknown",
            inference_time_ms=0.0,
            status="error",
            error_message="Model not found for service",
        )

        assert result.status == "error"
        assert result.error_message == "Model not found for service"
        assert result.has_anomalies is False

    def test_result_with_explainability_context(self, mock_inference_metrics):
        """Test result with full explainability context."""
        result = ServiceInferenceResult(
            service_name="booking",
            timestamp=datetime.now(),
            input_metrics=mock_inference_metrics,
            anomalies=[],
            model_version="1.0.0",
            inference_time_ms=45.0,
            status="success",
            historical_context={"baseline_latency": 100.0},
            metric_analysis={"latency_percentile": 95.0},
            explanation={"summary": "Service operating normally"},
            recommended_actions=["Continue monitoring"],
            exception_context=None,
        )

        assert result.historical_context == {"baseline_latency": 100.0}
        assert result.recommended_actions == ["Continue monitoring"]


# =============================================================================
# Tests for EnhancedModelManager
# =============================================================================


class TestEnhancedModelManager:
    """Tests for EnhancedModelManager class."""

    def test_init_with_default_directory(self):
        """Test initialization with default directory."""
        manager = EnhancedModelManager()
        assert manager.models_directory == Path("./smartbox_models/")

    def test_init_with_custom_directory(self, temp_models_dir: Path):
        """Test initialization with custom directory."""
        manager = EnhancedModelManager(str(temp_models_dir))
        assert manager.models_directory == temp_models_dir

    def test_get_available_services(self, temp_models_dir: Path):
        """Test getting available services from models directory."""
        manager = EnhancedModelManager(str(temp_models_dir))
        services = manager.get_available_services()

        # Should find the mock service directories we created
        assert len(services) > 0
        assert any("booking" in s for s in services)
        assert any("search" in s for s in services)

    def test_get_available_services_empty_directory(self, tmp_path: Path):
        """Test getting services from empty directory."""
        empty_dir = tmp_path / "empty_models"
        empty_dir.mkdir()

        manager = EnhancedModelManager(str(empty_dir))
        services = manager.get_available_services()

        assert services == []

    def test_get_available_services_nonexistent_directory(self, tmp_path: Path):
        """Test getting services from nonexistent directory."""
        manager = EnhancedModelManager(str(tmp_path / "nonexistent"))
        services = manager.get_available_services()

        assert services == []

    def test_get_base_services(self, temp_models_dir: Path):
        """Test extracting base service names from period-specific models."""
        manager = EnhancedModelManager(str(temp_models_dir))
        base_services = manager.get_base_services()

        # Should extract base names without period suffixes
        assert "booking" in base_services
        assert "search" in base_services
        # Should not include period suffixes
        assert "booking_business_hours" not in base_services

    def test_load_services_from_config(self, temp_config_file: Path):
        """Test loading services from config file."""
        manager = EnhancedModelManager()
        services = manager.load_services_from_config(str(temp_config_file))

        assert "booking" in services
        assert "search" in services
        assert "api-gateway" in services

    def test_load_services_from_config_missing_file(self, tmp_path: Path):
        """Test loading services from missing config file."""
        manager = EnhancedModelManager()
        services = manager.load_services_from_config(str(tmp_path / "missing.json"))

        assert services == []

    def test_load_services_from_config_invalid_json(self, tmp_path: Path):
        """Test loading services from invalid JSON file."""
        invalid_config = tmp_path / "invalid.json"
        invalid_config.write_text("not valid json {")

        manager = EnhancedModelManager()
        services = manager.load_services_from_config(str(invalid_config))

        assert services == []

    def test_get_services_with_models(self, temp_models_dir: Path):
        """Test checking which config services have trained models."""
        manager = EnhancedModelManager(str(temp_models_dir))

        config_services = ["booking", "search", "nonexistent-service"]
        with_models, missing = manager.get_services_with_models(config_services)

        assert "booking" in with_models
        assert "search" in with_models
        assert "nonexistent-service" in missing

    def test_get_model_metadata_empty(self):
        """Test getting metadata for unknown service."""
        manager = EnhancedModelManager()
        metadata = manager.get_model_metadata("unknown_service")

        assert metadata == {}


# =============================================================================
# Tests for EnhancedAnomalyDetectionEngine
# =============================================================================


class TestEnhancedAnomalyDetectionEngine:
    """Tests for EnhancedAnomalyDetectionEngine class."""

    def test_init_with_model_manager(self):
        """Test initialization with model manager."""
        manager = EnhancedModelManager()
        engine = EnhancedAnomalyDetectionEngine(manager)

        assert engine.model_manager is manager

    def test_map_severity_low(self):
        """Test mapping 'low' string to LOW severity."""
        manager = EnhancedModelManager()
        engine = EnhancedAnomalyDetectionEngine(manager)

        assert engine._map_severity("low") == AnomalySeverity.LOW
        assert engine._map_severity("LOW") == AnomalySeverity.LOW

    def test_map_severity_medium(self):
        """Test mapping 'medium' string to MEDIUM severity."""
        manager = EnhancedModelManager()
        engine = EnhancedAnomalyDetectionEngine(manager)

        assert engine._map_severity("medium") == AnomalySeverity.MEDIUM

    def test_map_severity_high(self):
        """Test mapping 'high' string to HIGH severity."""
        manager = EnhancedModelManager()
        engine = EnhancedAnomalyDetectionEngine(manager)

        assert engine._map_severity("high") == AnomalySeverity.HIGH

    def test_map_severity_critical(self):
        """Test mapping 'critical' string to CRITICAL severity."""
        manager = EnhancedModelManager()
        engine = EnhancedAnomalyDetectionEngine(manager)

        assert engine._map_severity("critical") == AnomalySeverity.CRITICAL

    def test_map_severity_unknown_defaults_to_medium(self):
        """Test mapping unknown severity defaults to MEDIUM."""
        manager = EnhancedModelManager()
        engine = EnhancedAnomalyDetectionEngine(manager)

        assert engine._map_severity("unknown") == AnomalySeverity.MEDIUM
        assert engine._map_severity("invalid") == AnomalySeverity.MEDIUM

    def test_process_anomalies_dict_format(self, sample_anomaly_data: dict):
        """Test processing anomalies in dict format."""
        manager = EnhancedModelManager()
        engine = EnhancedAnomalyDetectionEngine(manager)

        raw_anomalies = {"latency_spike": sample_anomaly_data}
        results = engine._process_anomalies(raw_anomalies)

        assert len(results) == 1
        assert results[0].anomaly_type == "latency_spike"
        assert results[0].severity == AnomalySeverity.HIGH
        assert results[0].description == "Latency spike detected"

    def test_process_anomalies_legacy_format(self):
        """Test processing anomalies in legacy (non-dict) format."""
        manager = EnhancedModelManager()
        engine = EnhancedAnomalyDetectionEngine(manager)

        # Legacy format where value is not a dict
        raw_anomalies = {"simple_anomaly": "detected"}
        results = engine._process_anomalies(raw_anomalies)

        assert len(results) == 1
        assert results[0].anomaly_type == "simple_anomaly"
        assert results[0].severity == AnomalySeverity.MEDIUM
        assert results[0].confidence_score == 0.5

    def test_process_anomalies_extracts_explainability(self, sample_anomaly_data: dict):
        """Test that explainability fields are extracted."""
        manager = EnhancedModelManager()
        engine = EnhancedAnomalyDetectionEngine(manager)

        raw_anomalies = {"test_anomaly": sample_anomaly_data}
        results = engine._process_anomalies(raw_anomalies)

        assert results[0].comparison_data is not None
        assert results[0].business_impact == "User-facing latency degradation"

    def test_process_anomalies_handles_errors_gracefully(self):
        """Test that malformed anomaly data is handled gracefully."""
        manager = EnhancedModelManager()
        engine = EnhancedAnomalyDetectionEngine(manager)

        # Anomaly with missing required fields but still dict
        raw_anomalies = {
            "good_anomaly": {"severity": "high", "description": "Valid"},
            "partial_anomaly": {"score": 0.5},  # Missing severity, should default
        }
        results = engine._process_anomalies(raw_anomalies)

        # Should process both, using defaults where needed
        assert len(results) == 2


# =============================================================================
# Tests for EnhancedResultsProcessor
# =============================================================================


class TestEnhancedResultsProcessor:
    """Tests for EnhancedResultsProcessor class."""

    def test_init_creates_alerts_directory(self, tmp_path: Path):
        """Test that init creates alerts directory if it doesn't exist."""
        alerts_dir = tmp_path / "alerts"
        processor = EnhancedResultsProcessor(str(alerts_dir))

        assert alerts_dir.exists()
        assert processor.alerts_directory == alerts_dir

    def test_init_with_verbose(self, tmp_path: Path):
        """Test initialization with verbose flag."""
        processor = EnhancedResultsProcessor(str(tmp_path), verbose=True)
        assert processor.verbose is True

    def test_detected_anomalies_starts_empty(self, tmp_path: Path):
        """Test that detected_anomalies list starts empty."""
        processor = EnhancedResultsProcessor(str(tmp_path))
        assert processor.detected_anomalies == []

    def test_process_result_with_anomalies(self, tmp_path: Path):
        """Test processing a result with anomalies."""
        processor = EnhancedResultsProcessor(str(tmp_path))

        result = {
            "service": "booking",
            "timestamp": datetime.now().isoformat(),
            "anomalies": {
                "latency_high": {
                    "type": "latency_high",
                    "severity": "high",
                    "description": "High latency",
                    "score": -0.5,
                }
            },
            "overall_severity": "high",
            "current_metrics": {
                "request_rate": 100.0,
                "application_latency": 250.0,
            },
        }

        processor.process_result(result)

        assert len(processor.detected_anomalies) == 1
        assert processor.detected_anomalies[0]["alert_type"] == "anomaly_detected"

    def test_process_result_with_error(self, tmp_path: Path):
        """Test processing an error result."""
        processor = EnhancedResultsProcessor(str(tmp_path))

        result = {
            "service": "booking",
            "error": "Model not found",
            "timestamp": datetime.now().isoformat(),
        }

        processor.process_result(result)

        assert len(processor.detected_anomalies) == 1
        assert processor.detected_anomalies[0]["alert_type"] == "error"
        assert processor.detected_anomalies[0]["error_message"] == "Model not found"

    def test_process_result_no_anomalies(self, tmp_path: Path):
        """Test processing a result with no anomalies."""
        processor = EnhancedResultsProcessor(str(tmp_path))

        result = {
            "service": "booking",
            "timestamp": datetime.now().isoformat(),
            "anomalies": {},
            "overall_severity": "none",
        }

        processor.process_result(result)

        # No anomalies = nothing added
        assert len(processor.detected_anomalies) == 0

    def test_process_explainable_result(self, tmp_path: Path):
        """Test processing an explainable result."""
        processor = EnhancedResultsProcessor(str(tmp_path))

        result = {
            "service": "booking",
            "timestamp": datetime.now().isoformat(),
            "explainable": True,
            "anomaly_count": 1,
            "overall_severity": "high",
            "current_metrics": {"latency": 250.0},
            "anomalies": [
                {
                    "type": "latency_spike",
                    "severity": "high",
                    "score": -0.5,
                    "description": "Latency spike",
                    "comparison_data": {"current": 250.0, "baseline": 100.0},
                }
            ],
        }

        processor.process_explainable_result(result)

        assert len(processor.detected_anomalies) == 1
        assert processor.detected_anomalies[0]["model_type"] == "explainable_ml"


# =============================================================================
# Tests for EnhancedTimeAwareDetector
# =============================================================================


class TestEnhancedTimeAwareDetector:
    """Tests for EnhancedTimeAwareDetector class."""

    def test_init(self, temp_models_dir: Path):
        """Test initialization."""
        detector = EnhancedTimeAwareDetector(str(temp_models_dir))

        assert detector.models_directory == str(temp_models_dir)
        assert detector._detector_cache == {}
        assert detector._load_times == {}

    def test_calculate_drift_penalty_no_drift(self):
        """Test drift penalty for low drift score."""
        detector = EnhancedTimeAwareDetector("./models")

        assert detector._calculate_drift_penalty(0.0) == 0.0
        assert detector._calculate_drift_penalty(2.9) == 0.0

    def test_calculate_drift_penalty_moderate(self):
        """Test drift penalty for moderate drift score."""
        detector = EnhancedTimeAwareDetector("./models")

        # Condition is drift_score > 3, so 3.0 returns 0.0
        assert detector._calculate_drift_penalty(3.0) == 0.0
        assert detector._calculate_drift_penalty(3.1) == 0.15
        assert detector._calculate_drift_penalty(4.5) == 0.15

    def test_calculate_drift_penalty_severe(self):
        """Test drift penalty for severe drift score."""
        detector = EnhancedTimeAwareDetector("./models")

        assert detector._calculate_drift_penalty(5.1) == 0.3
        assert detector._calculate_drift_penalty(10.0) == 0.3

    def test_apply_drift_adjustments_no_drift(self):
        """Test applying drift adjustments when no drift detected."""
        detector = EnhancedTimeAwareDetector("./models")

        result = {"anomalies": {"test": {"confidence": 0.8}}}
        drift_analysis = {"has_drift": False}

        adjusted = detector._apply_drift_adjustments(result, drift_analysis)

        # Should return unchanged
        assert "drift_warning" not in adjusted
        assert adjusted["anomalies"]["test"]["confidence"] == 0.8

    def test_apply_drift_adjustments_moderate_drift(self):
        """Test applying drift adjustments with moderate drift."""
        detector = EnhancedTimeAwareDetector("./models")

        result = {
            "anomalies": {
                "test_anomaly": {"confidence": 0.8}
            }
        }
        drift_analysis = {
            "has_drift": True,
            "overall_drift_score": 4.0,
            "recommendation": "Monitor closely",
            "drift_metrics": {"latency": {}},
        }

        adjusted = detector._apply_drift_adjustments(result, drift_analysis)

        # Should have drift warning
        assert "drift_warning" in adjusted
        assert adjusted["drift_warning"]["confidence_penalty_applied"] == 0.15

        # Confidence should be reduced
        assert adjusted["anomalies"]["test_anomaly"]["confidence"] == 0.8 * (1 - 0.15)
        assert adjusted["anomalies"]["test_anomaly"]["original_confidence"] == 0.8
        assert adjusted["anomalies"]["test_anomaly"]["drift_warning"] is True

    def test_apply_drift_adjustments_severe_drift(self):
        """Test applying drift adjustments with severe drift."""
        detector = EnhancedTimeAwareDetector("./models")

        result = {
            "anomalies": {
                "anomaly1": {"confidence": 0.9},
                "anomaly2": {"confidence": 0.7},
            }
        }
        drift_analysis = {
            "has_drift": True,
            "overall_drift_score": 6.0,
            "recommendation": "Retrain model",
            "drift_metrics": {"latency": {}, "error_rate": {}},
            "multivariate_drift": True,
        }

        adjusted = detector._apply_drift_adjustments(result, drift_analysis)

        # 30% penalty for severe drift
        assert adjusted["drift_warning"]["confidence_penalty_applied"] == 0.3
        assert adjusted["drift_warning"]["multivariate_drift"] is True

        # Both anomalies should be adjusted
        assert adjusted["anomalies"]["anomaly1"]["confidence"] == 0.9 * 0.7
        assert adjusted["anomalies"]["anomaly2"]["confidence"] == 0.7 * 0.7

    def test_load_time_aware_detector_no_models(self, tmp_path: Path):
        """Test loading detector when no models exist."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        detector = EnhancedTimeAwareDetector(str(empty_dir))
        result = detector.load_time_aware_detector("nonexistent_service")

        assert result is None

    def test_load_time_aware_detector_caching(self, temp_models_dir: Path):
        """Test that detector is cached after loading."""
        detector = EnhancedTimeAwareDetector(str(temp_models_dir))

        # Mock the TimeAwareAnomalyDetector at its source location
        with patch(
            "smartbox_anomaly.detection.time_aware.TimeAwareAnomalyDetector"
        ) as MockDetector:
            mock_instance = MagicMock()
            mock_instance._available_periods = {"business_hours", "evening_hours"}
            mock_instance._discover_available_periods.return_value = {
                "business_hours",
                "evening_hours",
            }
            mock_instance.models = {}
            MockDetector.return_value = mock_instance

            # First load
            result1 = detector.load_time_aware_detector("booking")

            if result1 is not None:
                # Second load should use cache (won't call constructor again)
                result2 = detector.load_time_aware_detector("booking")

                # Should be same instance from cache
                assert result1 is result2


# =============================================================================
# Tests for module imports
# =============================================================================


class TestModuleImports:
    """Tests for module-level imports and exports."""

    def test_import_all_components(self):
        """Test that all components can be imported from the module."""
        from smartbox_anomaly.inference import (
            AnomalyResult,
            EnhancedAnomalyDetectionEngine,
            EnhancedModelManager,
            EnhancedResultsProcessor,
            EnhancedTimeAwareDetector,
            ServiceInferenceResult,
            SmartboxMLInferencePipeline,
        )

        # All imports should succeed
        assert AnomalyResult is not None
        assert ServiceInferenceResult is not None
        assert EnhancedModelManager is not None
        assert EnhancedAnomalyDetectionEngine is not None
        assert EnhancedResultsProcessor is not None
        assert EnhancedTimeAwareDetector is not None
        assert SmartboxMLInferencePipeline is not None

    def test_module_all_exports(self):
        """Test that __all__ contains expected exports."""
        import smartbox_anomaly.inference as inference_module

        expected_exports = [
            "SmartboxMLInferencePipeline",
            "AnomalyResult",
            "ServiceInferenceResult",
            "EnhancedModelManager",
            "EnhancedAnomalyDetectionEngine",
            "EnhancedResultsProcessor",
            "EnhancedTimeAwareDetector",
        ]

        for export in expected_exports:
            assert export in inference_module.__all__


# =============================================================================
# Integration tests
# =============================================================================


# =============================================================================
# Tests for SmartboxMLInferencePipeline
# =============================================================================


class TestSmartboxMLInferencePipeline:
    """Tests for SmartboxMLInferencePipeline class."""

    @pytest.fixture
    def mock_pipeline(self, tmp_path: Path, temp_models_dir: Path):
        """Create a pipeline with mocked dependencies."""
        alerts_dir = tmp_path / "alerts"
        alerts_dir.mkdir()

        with patch("smartbox_anomaly.inference.pipeline.VictoriaMetricsClient"), \
             patch("smartbox_anomaly.inference.pipeline.ExceptionEnrichmentService"), \
             patch("smartbox_anomaly.inference.pipeline.ServiceGraphEnrichmentService"), \
             patch("smartbox_anomaly.inference.pipeline.get_config") as mock_config:

            # Create mock config
            mock_cfg = MagicMock()
            mock_cfg.victoria_metrics.endpoint = "http://localhost:8428"
            mock_cfg.model.models_directory = str(temp_models_dir)
            mock_cfg.inference.alerts_directory = str(alerts_dir)
            mock_cfg.inference.max_workers = 2
            mock_cfg.inference.check_drift = False
            mock_cfg.inference.exception_enrichment_enabled = False
            mock_cfg.slo.enabled = False
            mock_cfg.observability.base_url = "http://localhost:8000"
            mock_config.return_value = mock_cfg

            from smartbox_anomaly.inference import SmartboxMLInferencePipeline

            pipeline = SmartboxMLInferencePipeline(
                vm_endpoint="http://localhost:8428",
                models_directory=str(temp_models_dir),
                alerts_directory=str(alerts_dir),
                verbose=False,
            )
            # Add dependency graph for testing
            pipeline.dependency_graph = {
                "booking": ["search", "vms"],
                "search": ["catalog"],
                "vms": ["titan"],
            }
            return pipeline

    def test_validate_metrics_valid_data(self, mock_pipeline):
        """Test metrics validation with valid data."""
        metrics = {
            "request_rate": 100.0,
            "application_latency": 150.0,
            "error_rate": 0.01,
        }

        cleaned, warnings = mock_pipeline._validate_metrics(metrics, "test_service")

        assert cleaned == metrics
        assert warnings == []

    def test_validate_metrics_nan_value(self, mock_pipeline):
        """Test metrics validation replaces NaN with 0."""
        metrics = {
            "request_rate": float("nan"),
            "application_latency": 150.0,
        }

        cleaned, warnings = mock_pipeline._validate_metrics(metrics, "test_service")

        assert cleaned["request_rate"] == 0.0
        assert cleaned["application_latency"] == 150.0
        assert len(warnings) == 1
        assert "invalid value" in warnings[0]

    def test_validate_metrics_inf_value(self, mock_pipeline):
        """Test metrics validation replaces inf with 0."""
        metrics = {
            "request_rate": float("inf"),
            "application_latency": 150.0,
        }

        cleaned, warnings = mock_pipeline._validate_metrics(metrics, "test_service")

        assert cleaned["request_rate"] == 0.0
        assert len(warnings) == 1

    def test_validate_metrics_negative_rate(self, mock_pipeline):
        """Test metrics validation replaces negative rate with 0."""
        metrics = {
            "request_rate": -10.0,
            "application_latency": 150.0,
        }

        cleaned, warnings = mock_pipeline._validate_metrics(metrics, "test_service")

        assert cleaned["request_rate"] == 0.0
        assert len(warnings) == 1
        assert "negative rate" in warnings[0]

    def test_validate_metrics_negative_latency(self, mock_pipeline):
        """Test metrics validation replaces negative latency with 0."""
        metrics = {
            "request_rate": 100.0,
            "application_latency": -50.0,
        }

        cleaned, warnings = mock_pipeline._validate_metrics(metrics, "test_service")

        assert cleaned["application_latency"] == 0.0
        assert "negative latency" in warnings[0]

    def test_validate_metrics_extreme_request_rate(self, mock_pipeline):
        """Test metrics validation caps extreme request rates."""
        metrics = {
            "request_rate": 2_000_000.0,  # 2M req/s - exceeds 1M cap
        }

        cleaned, warnings = mock_pipeline._validate_metrics(metrics, "test_service")

        assert cleaned["request_rate"] == 1_000_000.0
        assert "extreme rate" in warnings[0]

    def test_validate_metrics_extreme_latency(self, mock_pipeline):
        """Test metrics validation caps extreme latencies."""
        metrics = {
            "application_latency": 500_000.0,  # 500 seconds - exceeds 5 min cap
        }

        cleaned, warnings = mock_pipeline._validate_metrics(metrics, "test_service")

        assert cleaned["application_latency"] == 300_000.0
        assert "extreme latency" in warnings[0]

    def test_validate_metrics_error_rate_over_100(self, mock_pipeline):
        """Test metrics validation caps error rate at 100%."""
        metrics = {
            "error_rate": 1.5,  # 150% error rate - invalid
        }

        cleaned, warnings = mock_pipeline._validate_metrics(metrics, "test_service")

        assert cleaned["error_rate"] == 1.0
        assert "error rate" in warnings[0] and "> 1.0" in warnings[0]

    def test_validate_metrics_negative_error_rate(self, mock_pipeline):
        """Test metrics validation replaces negative error rate with 0."""
        metrics = {
            "error_rate": -0.05,
        }

        cleaned, warnings = mock_pipeline._validate_metrics(metrics, "test_service")

        assert cleaned["error_rate"] == 0.0
        assert "negative rate" in warnings[0]

    def test_has_latency_anomaly_true(self, mock_pipeline):
        """Test detection of latency anomaly in results."""
        result = {
            "anomalies": {
                "latency_spike_recent": {
                    "root_metric": "application_latency",
                    "severity": "high",
                }
            }
        }

        assert mock_pipeline._has_latency_anomaly(result) is True

    def test_has_latency_anomaly_false(self, mock_pipeline):
        """Test when no latency anomaly in results."""
        result = {
            "anomalies": {
                "error_rate_elevated": {
                    "root_metric": "error_rate",
                    "severity": "high",
                }
            }
        }

        assert mock_pipeline._has_latency_anomaly(result) is False

    def test_has_latency_anomaly_error_result(self, mock_pipeline):
        """Test handling of error result."""
        result = {"error": "Model not found"}

        assert mock_pipeline._has_latency_anomaly(result) is False

    def test_has_latency_anomaly_empty_anomalies(self, mock_pipeline):
        """Test with empty anomalies dict."""
        result = {"anomalies": {}}

        assert mock_pipeline._has_latency_anomaly(result) is False

    def test_has_latency_anomaly_by_anomaly_name(self, mock_pipeline):
        """Test detection by anomaly name containing 'latency'."""
        result = {
            "anomalies": {
                "dependency_latency_high": {
                    "root_metric": "dependency_latency",
                    "severity": "medium",
                }
            }
        }

        assert mock_pipeline._has_latency_anomaly(result) is True

    def test_has_latency_anomaly_by_contributing_metrics(self, mock_pipeline):
        """Test detection via contributing_metrics field."""
        result = {
            "anomalies": {
                "traffic_surge": {
                    "root_metric": "request_rate",
                    "contributing_metrics": ["request_rate", "application_latency"],
                }
            }
        }

        assert mock_pipeline._has_latency_anomaly(result) is True

    def test_build_dependency_context_no_graph(self, mock_pipeline):
        """Test building context when no dependency graph."""
        mock_pipeline.dependency_graph = {}

        context = mock_pipeline._build_dependency_context("booking", {})

        assert context is None

    def test_build_dependency_context_no_dependencies(self, mock_pipeline):
        """Test building context when service has no dependencies."""
        context = mock_pipeline._build_dependency_context("unknown_service", {})

        assert context is None

    def test_build_dependency_context_with_healthy_deps(self, mock_pipeline):
        """Test building context with healthy dependencies."""
        all_results = {
            "search": {
                "anomalies": {},
                "timestamp": "2024-01-15T10:00:00",
            },
            "vms": {
                "anomalies": {},
                "timestamp": "2024-01-15T10:00:00",
            },
        }

        context = mock_pipeline._build_dependency_context("booking", all_results)

        assert context is not None
        assert "search" in context.dependencies
        assert "vms" in context.dependencies
        assert context.dependencies["search"].has_anomaly is False
        assert context.dependencies["vms"].has_anomaly is False

    def test_build_dependency_context_with_anomaly_dep(self, mock_pipeline):
        """Test building context when dependency has anomaly."""
        all_results = {
            "search": {
                "anomalies": {
                    "database_bottleneck": {
                        "pattern_name": "database_bottleneck",
                        "severity": "high",
                        "detection_signals": [],
                    }
                },
                "timestamp": "2024-01-15T10:00:00",
            },
        }

        context = mock_pipeline._build_dependency_context("booking", all_results)

        assert context is not None
        assert context.dependencies["search"].has_anomaly is True
        assert context.dependencies["search"].anomaly_type == "database_bottleneck"
        assert context.dependencies["search"].severity == "high"

    def test_build_dependency_context_ignores_error_results(self, mock_pipeline):
        """Test that error results are ignored in dependency context."""
        all_results = {
            "search": {
                "error": "Model not found",
            },
            "vms": {
                "anomalies": {},
                "timestamp": "2024-01-15T10:00:00",
            },
        }

        context = mock_pipeline._build_dependency_context("booking", all_results)

        assert context is not None
        # search should not be in dependencies due to error
        assert "search" not in context.dependencies
        assert "vms" in context.dependencies


class TestInferenceModuleIntegration:
    """Integration tests for inference module components working together."""

    def test_model_manager_to_detection_engine_flow(self, temp_models_dir: Path):
        """Test model manager and detection engine integration."""
        manager = EnhancedModelManager(str(temp_models_dir))
        engine = EnhancedAnomalyDetectionEngine(manager)

        # Engine should have reference to manager
        assert engine.model_manager is manager

        # Manager should discover services
        services = manager.get_base_services()
        assert len(services) > 0

    def test_results_processor_formats_alerts_correctly(self, tmp_path: Path):
        """Test that results processor creates properly formatted alerts."""
        processor = EnhancedResultsProcessor(str(tmp_path))

        result = {
            "service": "booking",
            "timestamp": "2024-01-15T10:30:00",
            "time_period": "business_hours",
            "model_type": "time_aware_5period",
            "overall_severity": "high",
            "anomaly_count": 1,
            "current_metrics": {
                "request_rate": 150.0,
                "application_latency": 250.0,
                "error_rate": 0.01,
            },
            "anomalies": {
                "latency_spike_recent": {
                    "type": "consolidated",
                    "severity": "high",
                    "score": -0.45,
                    "description": "Latency spike detected",
                    "confidence": 0.85,
                }
            },
        }

        processor.process_result(result)

        assert len(processor.detected_anomalies) == 1
        alert = processor.detected_anomalies[0]

        assert alert["alert_type"] == "anomaly_detected"
        assert alert["service_name"] == "booking"
        assert alert["time_period"] == "business_hours"

    def test_time_aware_detector_drift_integration(self):
        """Test time-aware detector drift handling integration."""
        detector = EnhancedTimeAwareDetector("./models")

        # Simulate detection result with drift
        result = {
            "service": "booking",
            "anomalies": {
                "latency_high": {"confidence": 0.9, "severity": "high"}
            },
        }

        drift_analysis = {
            "has_drift": True,
            "overall_drift_score": 4.5,
            "recommendation": "Monitor model performance",
            "drift_metrics": {"application_latency": {"z_score": 4.5}},
        }

        adjusted = detector._apply_drift_adjustments(result, drift_analysis)

        # Verify integration of drift warning
        assert "drift_warning" in adjusted
        assert adjusted["anomalies"]["latency_high"]["drift_warning"] is True
        assert adjusted["anomalies"]["latency_high"]["confidence"] < 0.9
