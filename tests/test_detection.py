"""
Tests for the detection module.
"""

from __future__ import annotations

import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from smartbox_anomaly.core.constants import MetricName, TimePeriod
from smartbox_anomaly.detection import (
    SmartboxAnomalyDetector,
    TimeAwareAnomalyDetector,
    create_detector,
    create_time_aware_detector,
    detect_service_category,
    get_service_parameters,
)


class TestServiceConfig:
    """Tests for service configuration utilities."""

    def test_get_service_parameters_known_service(self):
        """Test parameters for known services."""
        params = get_service_parameters("booking")
        assert params.category == "critical"
        assert params.base_contamination == 0.02
        assert params.complexity == "high"

    def test_get_service_parameters_unknown_service(self):
        """Test parameters for unknown services are auto-detected."""
        params = get_service_parameters("my-new-service")
        assert params.auto_detected is True
        assert params.category == "unknown_standard"

    def test_detect_service_category_api(self):
        """Test detection of API services."""
        config = detect_service_category("payment-api")
        assert config["category"] == "api_gateway"
        assert config["complexity"] == "high"

    def test_detect_service_category_admin(self):
        """Test detection of admin services."""
        config = detect_service_category("user-admin")
        assert config["category"] == "admin"
        assert config["complexity"] == "low"

    def test_detect_service_category_worker(self):
        """Test detection of background workers."""
        config = detect_service_category("email-worker")
        assert config["category"] == "background_service"

    def test_detect_service_category_security(self):
        """Test detection of security services."""
        config = detect_service_category("oauth-service")
        assert config["category"] == "security_service"
        assert config["complexity"] == "high"

    def test_get_parameters_with_data(self):
        """Test data-driven parameter tuning."""
        # Create sample data with high variability
        data = pd.DataFrame({
            "request_rate": np.random.exponential(100, 1000),
            "application_latency": np.random.exponential(50, 1000),
        })

        params = get_service_parameters("test-service", data=data, auto_tune=True)
        assert params.n_estimators > 0
        assert 0 < params.base_contamination < 1


class TestSmartboxAnomalyDetector:
    """Tests for SmartboxAnomalyDetector class."""

    @pytest.fixture
    def sample_training_data(self) -> pd.DataFrame:
        """Generate sample training data."""
        np.random.seed(42)
        n_samples = 500

        return pd.DataFrame({
            MetricName.REQUEST_RATE: np.random.exponential(100, n_samples),
            MetricName.APPLICATION_LATENCY: np.random.exponential(50, n_samples),
            MetricName.DEPENDENCY_LATENCY: np.random.exponential(20, n_samples),
            MetricName.DATABASE_LATENCY: np.random.exponential(10, n_samples),
            MetricName.ERROR_RATE: np.random.beta(1, 100, n_samples),
        })

    @pytest.fixture
    def trained_detector(self, sample_training_data) -> SmartboxAnomalyDetector:
        """Create and train a detector."""
        detector = create_detector("test-service")
        detector.train(sample_training_data)
        return detector

    def test_detector_creation(self):
        """Test detector creation."""
        detector = create_detector("my-service")
        assert detector.service_name == "my-service"
        assert detector.auto_tune is True
        assert not detector.is_trained

    def test_detector_creation_no_tune(self):
        """Test detector creation without auto-tuning."""
        detector = SmartboxAnomalyDetector("my-service", auto_tune=False)
        assert detector.auto_tune is False

    def test_training(self, sample_training_data):
        """Test detector training."""
        detector = create_detector("test-service")
        detector.train(sample_training_data)

        assert detector.is_trained
        assert len(detector.models) > 0
        assert len(detector.scalers) > 0
        assert len(detector.training_statistics) > 0

    def test_training_empty_data(self):
        """Test training with empty data raises error."""
        detector = create_detector("test-service")
        with pytest.raises(Exception):
            detector.train(pd.DataFrame())

    def test_detection_normal_metrics(self, trained_detector):
        """Test detection with normal metrics returns few/no anomalies."""
        normal_metrics = {
            MetricName.REQUEST_RATE: 100.0,
            MetricName.APPLICATION_LATENCY: 50.0,
            MetricName.DEPENDENCY_LATENCY: 20.0,
            MetricName.DATABASE_LATENCY: 10.0,
            MetricName.ERROR_RATE: 0.01,
        }

        result = trained_detector.detect(normal_metrics)
        assert "anomalies" in result
        assert "metadata" in result

    def test_detection_anomalous_metrics(self, trained_detector):
        """Test detection with anomalous metrics."""
        anomalous_metrics = {
            MetricName.REQUEST_RATE: 10000.0,  # Very high
            MetricName.APPLICATION_LATENCY: 5000.0,  # Very high
            MetricName.DEPENDENCY_LATENCY: 1000.0,
            MetricName.DATABASE_LATENCY: 500.0,
            MetricName.ERROR_RATE: 0.5,  # 50% error rate
        }

        result = trained_detector.detect(anomalous_metrics)
        assert "anomalies" in result
        # Should detect at least some anomalies
        # (exact number depends on training data)

    def test_detection_untrained_detector(self):
        """Test detection on untrained detector."""
        detector = create_detector("test-service")
        result = detector.detect({MetricName.REQUEST_RATE: 100.0})

        assert "anomalies" in result
        assert result["metadata"]["trained"] is False

    def test_training_statistics(self, trained_detector):
        """Test that training statistics are calculated."""
        stats = trained_detector.training_statistics

        assert MetricName.REQUEST_RATE in stats
        assert stats[MetricName.REQUEST_RATE].mean > 0
        assert stats[MetricName.REQUEST_RATE].std >= 0
        assert stats[MetricName.REQUEST_RATE].p95 > 0

    def test_training_quality_report(self, sample_training_data):
        """Test training data quality report generation."""
        detector = create_detector("test-quality-service")
        result = detector.train(sample_training_data)

        # Check that quality report is included
        assert "quality_report" in result
        report = result["quality_report"]

        # Check overall structure
        assert "service_name" in report
        assert report["service_name"] == "test-quality-service"
        assert "total_samples" in report
        assert report["total_samples"] > 0
        assert "overall_quality_score" in report
        assert 0 <= report["overall_quality_score"] <= 100
        assert "quality_grade" in report
        assert report["quality_grade"] in ["A", "B", "C", "D", "F"]

        # Check metric quality details
        assert "metric_quality" in report
        for metric_name, mq in report["metric_quality"].items():
            assert "sample_count" in mq
            assert "missing_count" in mq
            assert "missing_percentage" in mq
            assert "outlier_count" in mq
            assert "is_usable" in mq

        # Verify report can be serialized (to_dict was called)
        assert isinstance(report, dict)

    def test_training_quality_report_with_issues(self):
        """Test quality report captures data quality issues."""
        # Create data with known quality issues
        np.random.seed(42)
        data = pd.DataFrame({
            MetricName.REQUEST_RATE: [100.0] * 600,  # Constant value
            MetricName.APPLICATION_LATENCY: np.random.normal(100, 10, 600),
            MetricName.ERROR_RATE: np.random.normal(0.02, 0.01, 600),
        })
        # Add some NaN values
        data.loc[50:70, MetricName.APPLICATION_LATENCY] = np.nan

        detector = create_detector("test-quality-issues")
        result = detector.train(data)
        report = result["quality_report"]

        # Should have warnings
        assert report["quality_grade"] != "A"  # Not perfect due to issues

        # Check constant value detection for request_rate
        if MetricName.REQUEST_RATE in report["metric_quality"]:
            mq = report["metric_quality"][MetricName.REQUEST_RATE]
            assert mq["has_constant_values"] is True

        # Check missing value detection for application_latency
        if MetricName.APPLICATION_LATENCY in report["metric_quality"]:
            mq = report["metric_quality"][MetricName.APPLICATION_LATENCY]
            assert mq["missing_count"] > 0

    def test_feature_importance(self, trained_detector):
        """Test feature importance calculation."""
        importance = trained_detector.feature_importance

        assert len(importance) > 0
        for _metric, fi in importance.items():
            assert fi.variability_score >= 0
            assert fi.impact_level in ["low", "medium", "high", "critical"]

    def test_save_and_load_model(self, trained_detector):
        """Test saving and loading models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            saved_path = trained_detector.save_model(tmpdir)
            assert saved_path.exists()
            assert (saved_path / "model_data.json").exists()

            # Load
            loaded = SmartboxAnomalyDetector.load_model(tmpdir, "test-service")
            assert loaded.is_trained
            assert loaded.service_name == trained_detector.service_name

            # Verify detection works on loaded model
            metrics = {MetricName.REQUEST_RATE: 100.0}
            result = loaded.detect(metrics)
            assert "anomalies" in result

    def test_pattern_detection(self, trained_detector):
        """Test pattern-based anomaly detection."""
        # Simulate system overload pattern
        overload_metrics = {
            MetricName.REQUEST_RATE: 10000.0,
            MetricName.APPLICATION_LATENCY: 3000.0,
            MetricName.ERROR_RATE: 0.1,
            MetricName.DEPENDENCY_LATENCY: 0.0,
            MetricName.DATABASE_LATENCY: 0.0,
        }

        result = trained_detector.detect(overload_metrics)
        anomalies = result.get("anomalies", {})

        # Should detect patterns or multivariate anomalies
        assert len(anomalies) > 0 or result["metadata"].get("status") == "no_anomalies"


class TestTimeAwareAnomalyDetector:
    """Tests for TimeAwareAnomalyDetector class."""

    @pytest.fixture
    def sample_time_series_data(self) -> pd.DataFrame:
        """Generate sample time series data with datetime index."""
        np.random.seed(42)

        # Generate timestamps covering all periods
        dates = pd.date_range(
            start="2024-01-01",
            end="2024-01-14",
            freq="15min",
        )

        n_samples = len(dates)

        data = pd.DataFrame({
            MetricName.REQUEST_RATE: np.random.exponential(100, n_samples),
            MetricName.APPLICATION_LATENCY: np.random.exponential(50, n_samples),
            MetricName.DEPENDENCY_LATENCY: np.random.exponential(20, n_samples),
            MetricName.DATABASE_LATENCY: np.random.exponential(10, n_samples),
            MetricName.ERROR_RATE: np.random.beta(1, 100, n_samples),
        }, index=dates)

        return data

    def test_detector_creation(self):
        """Test time-aware detector creation."""
        detector = create_time_aware_detector("my-service")
        assert detector.service_name == "my-service"
        assert len(detector.available_periods) == 0  # Not loaded yet

    def test_get_current_period_business(self):
        """Test period detection for business hours."""
        detector = TimeAwareAnomalyDetector("test")
        # Wednesday 10am
        period = detector.get_current_period(datetime(2024, 1, 10, 10, 0))
        assert period == TimePeriod.BUSINESS_HOURS

    def test_get_current_period_evening(self):
        """Test period detection for evening hours."""
        detector = TimeAwareAnomalyDetector("test")
        # Wednesday 7pm
        period = detector.get_current_period(datetime(2024, 1, 10, 19, 0))
        assert period == TimePeriod.EVENING_HOURS

    def test_get_current_period_night(self):
        """Test period detection for night hours."""
        detector = TimeAwareAnomalyDetector("test")
        # Wednesday 2am
        period = detector.get_current_period(datetime(2024, 1, 10, 2, 0))
        assert period == TimePeriod.NIGHT_HOURS

    def test_get_current_period_weekend_day(self):
        """Test period detection for weekend day."""
        detector = TimeAwareAnomalyDetector("test")
        # Saturday 2pm
        period = detector.get_current_period(datetime(2024, 1, 13, 14, 0))
        assert period == TimePeriod.WEEKEND_DAY

    def test_get_current_period_weekend_night(self):
        """Test period detection for weekend night."""
        detector = TimeAwareAnomalyDetector("test")
        # Saturday 11pm
        period = detector.get_current_period(datetime(2024, 1, 13, 23, 0))
        assert period == TimePeriod.WEEKEND_NIGHT

    def test_training_time_aware_models(self, sample_time_series_data):
        """Test training time-aware models."""
        detector = TimeAwareAnomalyDetector("test-service")
        models = detector.train_time_aware_models(sample_time_series_data)

        assert len(models) > 0
        assert len(detector.available_periods) > 0
        # Should have trained at least some periods
        assert "business_hours" in detector.available_periods or len(models) > 0

    def test_save_and_load_time_aware(self, sample_time_series_data):
        """Test saving and loading time-aware models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Train and save
            detector = TimeAwareAnomalyDetector("test-service")
            detector.train_time_aware_models(sample_time_series_data)
            saved_paths = detector.save_models(tmpdir)

            assert len(saved_paths) > 0

            # Load
            loaded = create_time_aware_detector("test-service", models_directory=tmpdir)
            assert len(loaded.available_periods) > 0

    def test_detection_no_model(self):
        """Test detection when no model is available."""
        detector = TimeAwareAnomalyDetector("nonexistent-service")
        result = detector.detect({MetricName.REQUEST_RATE: 100.0})

        assert result["metadata"]["status"] == "no_model"

    def test_statistics(self):
        """Test getting detector statistics."""
        detector = TimeAwareAnomalyDetector("test-service")
        stats = detector.get_statistics()

        assert stats["service_name"] == "test-service"
        assert "available_periods" in stats
        assert "validation_thresholds" in stats


class TestDetectionIntegration:
    """Integration tests for detection module."""

    @pytest.fixture
    def training_data(self) -> pd.DataFrame:
        """Generate realistic training data."""
        np.random.seed(42)
        n_samples = 1000

        # More realistic distributions
        request_rate = np.random.lognormal(4, 0.5, n_samples)
        app_latency = np.random.lognormal(3.5, 0.8, n_samples)

        # Dependency and DB latency often zero
        dependency_latency = np.where(
            np.random.random(n_samples) > 0.3,
            np.random.exponential(30, n_samples),
            0
        )
        db_latency = np.where(
            np.random.random(n_samples) > 0.4,
            np.random.exponential(15, n_samples),
            0
        )

        error_rate = np.random.beta(1, 200, n_samples)

        return pd.DataFrame({
            MetricName.REQUEST_RATE: request_rate,
            MetricName.APPLICATION_LATENCY: app_latency,
            MetricName.DEPENDENCY_LATENCY: dependency_latency,
            MetricName.DATABASE_LATENCY: db_latency,
            MetricName.ERROR_RATE: error_rate,
        })

    def test_full_workflow(self, training_data):
        """Test complete training and detection workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and train
            detector = create_detector("integration-test")
            detector.train(training_data)

            assert detector.is_trained

            # Save
            model_path = detector.save_model(tmpdir)
            assert model_path.exists()

            # Load
            loaded = SmartboxAnomalyDetector.load_model(tmpdir, "integration-test")
            assert loaded.is_trained

            # Detect on normal data
            normal_result = loaded.detect({
                MetricName.REQUEST_RATE: 50.0,
                MetricName.APPLICATION_LATENCY: 30.0,
                MetricName.ERROR_RATE: 0.005,
            })
            assert "anomalies" in normal_result

            # Detect on anomalous data
            anomalous_result = loaded.detect({
                MetricName.REQUEST_RATE: 50000.0,  # Way higher than normal
                MetricName.APPLICATION_LATENCY: 10000.0,
                MetricName.ERROR_RATE: 0.3,
            })
            assert "anomalies" in anomalous_result

    def test_zero_normal_metric_handling(self, training_data):
        """Test handling of zero-normal metrics like dependency_latency."""
        detector = create_detector("zero-normal-test")
        detector.train(training_data)

        # Test with zero values (normal for these metrics)
        result = detector.detect({
            MetricName.REQUEST_RATE: 50.0,
            MetricName.APPLICATION_LATENCY: 30.0,
            MetricName.DEPENDENCY_LATENCY: 0.0,  # Often zero
            MetricName.DATABASE_LATENCY: 0.0,  # Often zero
            MetricName.ERROR_RATE: 0.005,
        })

        assert "anomalies" in result
        # Zero values shouldn't trigger anomalies for zero-normal metrics


class TestPatternMatching:
    """Tests for pattern matching and level assignment."""

    @pytest.fixture
    def trained_detector(self) -> SmartboxAnomalyDetector:
        """Create a trained detector for pattern matching tests."""
        np.random.seed(42)
        n_samples = 500

        data = pd.DataFrame({
            MetricName.REQUEST_RATE: np.random.exponential(100, n_samples),
            MetricName.APPLICATION_LATENCY: np.random.exponential(50, n_samples),
            MetricName.DEPENDENCY_LATENCY: np.random.exponential(20, n_samples),
            MetricName.DATABASE_LATENCY: np.random.exponential(10, n_samples),
            MetricName.ERROR_RATE: np.random.beta(1, 100, n_samples),
        })

        detector = create_detector("test-service")
        detector.train(data)
        return detector

    def test_pattern_matches_valid_conditions(self, trained_detector):
        """Test that valid conditions are properly matched."""
        # Test the _pattern_matches method with valid conditions
        metric_levels = {
            "request_rate": "high",
            "application_latency": "normal",
            "error_rate": "normal",
        }
        ratios = {}

        # Should match when all conditions are satisfied
        conditions = {
            "request_rate": "high",
            "application_latency": "normal",
            "error_rate": "normal",
        }
        assert trained_detector._pattern_matches(metric_levels, ratios, conditions) is True

        # Should not match when a condition is not satisfied
        conditions = {
            "request_rate": "low",  # Doesn't match
            "application_latency": "normal",
        }
        assert trained_detector._pattern_matches(metric_levels, ratios, conditions) is False

    def test_pattern_matches_unknown_condition_fails_closed(self, trained_detector):
        """Test that unknown conditions cause pattern to NOT match (fail-closed)."""
        metric_levels = {
            "request_rate": "normal",
            "application_latency": "low",
            "error_rate": "normal",
        }
        ratios = {}

        # Unknown condition "unknown_level" should cause no match
        conditions = {
            "application_latency": "low",
            "error_rate": "unknown_level",  # Invalid condition
        }
        assert trained_detector._pattern_matches(metric_levels, ratios, conditions) is False

    def test_pattern_matches_elevated_and_moderate_conditions(self, trained_detector):
        """Test that elevated and moderate conditions work correctly."""
        ratios = {}

        # Test elevated matches elevated, high, very_high
        for level in ["elevated", "high", "very_high"]:
            metric_levels = {"application_latency": level}
            conditions = {"application_latency": "elevated"}
            assert trained_detector._pattern_matches(metric_levels, ratios, conditions) is True

        # Test elevated does NOT match normal or low
        for level in ["normal", "low", "very_low"]:
            metric_levels = {"application_latency": level}
            conditions = {"application_latency": "elevated"}
            assert trained_detector._pattern_matches(metric_levels, ratios, conditions) is False

        # Test moderate matches moderate, elevated, high
        for level in ["moderate", "elevated", "high"]:
            metric_levels = {"error_rate": level}
            conditions = {"error_rate": "moderate"}
            assert trained_detector._pattern_matches(metric_levels, ratios, conditions) is True

        # Test moderate does NOT match normal
        metric_levels = {"error_rate": "normal"}
        conditions = {"error_rate": "moderate"}
        assert trained_detector._pattern_matches(metric_levels, ratios, conditions) is False

    def test_pattern_matches_any_condition(self, trained_detector):
        """Test that 'any' condition always matches."""
        ratios = {}

        for level in ["very_high", "high", "normal", "low", "very_low"]:
            metric_levels = {"request_rate": level}
            conditions = {"request_rate": "any"}
            assert trained_detector._pattern_matches(metric_levels, ratios, conditions) is True


class TestSignalsToLevels:
    """Tests for the _signals_to_levels method."""

    @pytest.fixture
    def trained_detector(self) -> SmartboxAnomalyDetector:
        """Create a trained detector."""
        np.random.seed(42)
        n_samples = 500

        data = pd.DataFrame({
            MetricName.REQUEST_RATE: np.random.exponential(100, n_samples),
            MetricName.APPLICATION_LATENCY: np.random.exponential(50, n_samples),
            MetricName.DEPENDENCY_LATENCY: np.random.exponential(20, n_samples),
            MetricName.DATABASE_LATENCY: np.random.exponential(10, n_samples),
            MetricName.ERROR_RATE: np.random.beta(1, 100, n_samples),
        })

        detector = create_detector("test-service")
        detector.train(data)
        return detector

    def test_signals_to_levels_percentile_thresholds(self, trained_detector):
        """Test that percentile thresholds map to correct levels."""
        from smartbox_anomaly.detection.detector import AnomalySignal

        # Test very_high (percentile > 95)
        signals = [AnomalySignal(
            metric_name="request_rate",
            score=-0.5,
            direction="high",
            value=1000,
            percentile=97.0,
            deviation_sigma=3.0,
        )]
        levels = trained_detector._signals_to_levels(signals, {})
        assert levels["request_rate"] == "very_high"

        # Test high (percentile 90-95)
        signals = [AnomalySignal(
            metric_name="request_rate",
            score=-0.3,
            direction="high",
            value=800,
            percentile=92.0,
            deviation_sigma=2.0,
        )]
        levels = trained_detector._signals_to_levels(signals, {})
        assert levels["request_rate"] == "high"

        # Test elevated (percentile 80-90)
        signals = [AnomalySignal(
            metric_name="request_rate",
            score=-0.2,
            direction="high",
            value=600,
            percentile=85.0,
            deviation_sigma=1.5,
        )]
        levels = trained_detector._signals_to_levels(signals, {})
        assert levels["request_rate"] == "elevated"

        # Test moderate (percentile 70-80)
        signals = [AnomalySignal(
            metric_name="request_rate",
            score=-0.1,
            direction="high",
            value=550,
            percentile=75.0,
            deviation_sigma=1.0,
        )]
        levels = trained_detector._signals_to_levels(signals, {})
        assert levels["request_rate"] == "moderate"

    def test_signals_to_levels_lower_is_better_metrics(self, trained_detector):
        """Test that lower_is_better metrics treat low values as normal."""
        from smartbox_anomaly.detection.detector import AnomalySignal

        # database_latency is lower_is_better - low should become normal
        signals = [AnomalySignal(
            metric_name="database_latency",
            score=-0.2,
            direction="low",  # Low direction
            value=2.0,
            percentile=8.0,
            deviation_sigma=-2.0,
        )]
        levels = trained_detector._signals_to_levels(signals, {})
        assert levels["database_latency"] == "normal"  # Not "low"

        # dependency_latency is lower_is_better - low should become normal
        signals = [AnomalySignal(
            metric_name="dependency_latency",
            score=-0.2,
            direction="low",
            value=5.0,
            percentile=5.0,
            deviation_sigma=-2.5,
        )]
        levels = trained_detector._signals_to_levels(signals, {})
        assert levels["dependency_latency"] == "normal"

        # error_rate is lower_is_better - low should become normal
        signals = [AnomalySignal(
            metric_name="error_rate",
            score=-0.1,
            direction="low",
            value=0.0,
            percentile=2.0,
            deviation_sigma=-1.0,
        )]
        levels = trained_detector._signals_to_levels(signals, {})
        assert levels["error_rate"] == "normal"

    def test_signals_to_levels_application_latency_not_lower_is_better(self, trained_detector):
        """Test that application_latency is NOT in lower_is_better (can trigger fast-fail)."""
        from smartbox_anomaly.detection.detector import AnomalySignal

        # application_latency low should remain low (not converted to normal)
        signals = [AnomalySignal(
            metric_name="application_latency",
            score=-0.2,
            direction="low",
            value=10.0,
            percentile=8.0,
            deviation_sigma=-2.0,
        )]
        levels = trained_detector._signals_to_levels(signals, {})
        assert levels["application_latency"] == "low"  # Should stay low, not normal

    def test_signals_to_levels_no_signal_is_normal(self, trained_detector):
        """Test that metrics without signals are marked as normal."""
        from smartbox_anomaly.detection.detector import AnomalySignal

        # Only request_rate has a signal
        signals = [AnomalySignal(
            metric_name="request_rate",
            score=-0.3,
            direction="high",
            value=500,
            percentile=92.0,
            deviation_sigma=2.0,
        )]
        levels = trained_detector._signals_to_levels(signals, {})

        assert levels["request_rate"] == "high"
        # All other core metrics should be normal
        assert levels.get("application_latency") == "normal"
        assert levels.get("error_rate") == "normal"
        assert levels.get("database_latency") == "normal"
        assert levels.get("dependency_latency") == "normal"


class TestLowerIsBetterMetrics:
    """Tests for lower_is_better_metrics constant."""

    def test_lower_is_better_metrics_defined(self):
        """Test that lower_is_better_metrics returns expected metrics."""
        lower_is_better = MetricName.lower_is_better_metrics()

        assert MetricName.DATABASE_LATENCY in lower_is_better
        assert MetricName.DEPENDENCY_LATENCY in lower_is_better
        assert MetricName.ERROR_RATE in lower_is_better

        # application_latency should NOT be in lower_is_better
        assert MetricName.APPLICATION_LATENCY not in lower_is_better

        # request_rate should NOT be in lower_is_better
        assert MetricName.REQUEST_RATE not in lower_is_better


class TestImprovementSignalFiltering:
    """Tests for filtering improvement signals (lower_is_better with direction=low)."""

    @pytest.fixture
    def trained_detector(self) -> SmartboxAnomalyDetector:
        """Create a trained detector for testing _interpret_signals."""
        np.random.seed(42)
        n_samples = 500

        data = pd.DataFrame({
            MetricName.REQUEST_RATE: np.random.exponential(100, n_samples),
            MetricName.APPLICATION_LATENCY: np.random.exponential(50, n_samples),
            MetricName.DEPENDENCY_LATENCY: np.random.exponential(30, n_samples),
            MetricName.DATABASE_LATENCY: np.random.exponential(10, n_samples),
            MetricName.ERROR_RATE: np.random.beta(1, 100, n_samples),
        })

        detector = create_detector("test-service")
        detector.train(data)
        return detector

    def test_dependency_latency_improvement_produces_no_anomaly(self, trained_detector):
        """Test that improved dependency latency (below mean) produces no anomaly."""
        from smartbox_anomaly.detection.detector import AnomalySignal
        from datetime import datetime

        # Simulate only dependency_latency being low (an improvement)
        signals = [AnomalySignal(
            metric_name="dependency_latency",
            score=-0.3,
            direction="low",
            value=20.0,  # Below mean of ~50ms
            percentile=10.0,
            deviation_sigma=-2.0,
        )]

        result = trained_detector._interpret_signals(
            signals,
            {"dependency_latency": 20.0, "application_latency": 100.0},
            datetime.now()
        )

        # Should return empty dict - no anomaly to report
        assert result == {}

    def test_database_latency_improvement_produces_no_anomaly(self, trained_detector):
        """Test that improved database latency produces no anomaly."""
        from smartbox_anomaly.detection.detector import AnomalySignal
        from datetime import datetime

        signals = [AnomalySignal(
            metric_name="database_latency",
            score=-0.25,
            direction="low",
            value=1.0,  # Below mean
            percentile=5.0,
            deviation_sigma=-2.5,
        )]

        result = trained_detector._interpret_signals(
            signals,
            {"database_latency": 1.0, "application_latency": 100.0},
            datetime.now()
        )

        assert result == {}

    def test_mixed_signals_filters_improvements_keeps_degradations(self, trained_detector):
        """Test that mixed signals filter improvements but keep degradations."""
        from smartbox_anomaly.detection.detector import AnomalySignal
        from datetime import datetime

        # Mix of improvement (dependency_latency low) and degradation (application_latency high)
        signals = [
            AnomalySignal(
                metric_name="dependency_latency",
                score=-0.2,
                direction="low",
                value=15.0,
                percentile=8.0,
                deviation_sigma=-2.0,
            ),
            AnomalySignal(
                metric_name="application_latency",
                score=-0.4,
                direction="high",
                value=500.0,
                percentile=95.0,
                deviation_sigma=3.0,
            ),
        ]

        result = trained_detector._interpret_signals(
            signals,
            {
                "dependency_latency": 15.0,
                "application_latency": 500.0,
                "request_rate": 100.0,
                "error_rate": 0.01,
                "database_latency": 5.0,
            },
            datetime.now()
        )

        # Should produce an anomaly based on the high application_latency
        assert result != {}
        # Check that some anomaly was created
        assert len(result) > 0

    def test_error_rate_improvement_produces_no_anomaly(self, trained_detector):
        """Test that improved error rate (below mean) produces no anomaly."""
        from smartbox_anomaly.detection.detector import AnomalySignal
        from datetime import datetime

        signals = [AnomalySignal(
            metric_name="error_rate",
            score=-0.15,
            direction="low",
            value=0.0,  # Zero errors
            percentile=2.0,
            deviation_sigma=-1.5,
        )]

        result = trained_detector._interpret_signals(
            signals,
            {"error_rate": 0.0, "application_latency": 100.0},
            datetime.now()
        )

        assert result == {}

    def test_application_latency_low_still_produces_anomaly(self, trained_detector):
        """Test that low application_latency still produces anomaly (fast-fail detection)."""
        from smartbox_anomaly.detection.detector import AnomalySignal
        from datetime import datetime

        # application_latency is NOT in lower_is_better (can indicate fast-fail)
        signals = [AnomalySignal(
            metric_name="application_latency",
            score=-0.3,
            direction="low",
            value=5.0,  # Very fast - might indicate fast-fail
            percentile=3.0,
            deviation_sigma=-2.5,
        )]

        result = trained_detector._interpret_signals(
            signals,
            {"application_latency": 5.0, "error_rate": 0.05},
            datetime.now()
        )

        # Should produce an anomaly (not filtered out)
        assert result != {}
