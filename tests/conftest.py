"""
Pytest configuration and shared fixtures.

This module provides reusable fixtures for testing the smartbox_anomaly package.
"""

from __future__ import annotations

import tempfile
from collections.abc import Generator
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from smartbox_anomaly.core import (
    MetricName,
    PipelineConfig,
    TimePeriod,
    VictoriaMetricsConfig,
    reset_config,
    set_config,
)
from smartbox_anomaly.metrics import InferenceMetrics

# =============================================================================
# Configuration Fixtures
# =============================================================================


@pytest.fixture
def test_config() -> Generator[PipelineConfig, None, None]:
    """Provide a test configuration."""
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
def mock_vm_endpoint() -> str:
    """Provide a mock VictoriaMetrics endpoint."""
    return "http://mock-victoria-metrics:9090"


@pytest.fixture
def temp_model_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory for model storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_db_path() -> Generator[str, None, None]:
    """Provide a temporary database path for fingerprinting."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        yield f.name


# =============================================================================
# Data Fixtures - Basic Metrics
# =============================================================================


@pytest.fixture
def sample_metrics() -> InferenceMetrics:
    """Provide sample inference metrics."""
    return InferenceMetrics(
        service_name="test-service",
        timestamp=datetime.now(),
        request_rate=100.0,
        application_latency=50.0,
        client_latency=30.0,
        database_latency=20.0,
        error_rate=0.01,
    )


@pytest.fixture
def sample_metrics_dict() -> dict[str, float]:
    """Provide sample metrics as dictionary."""
    return {
        MetricName.REQUEST_RATE: 100.0,
        MetricName.APPLICATION_LATENCY: 50.0,
        MetricName.CLIENT_LATENCY: 30.0,
        MetricName.DATABASE_LATENCY: 20.0,
        MetricName.ERROR_RATE: 0.01,
    }


@pytest.fixture
def anomalous_metrics_dict() -> dict[str, float]:
    """Provide anomalous metrics as dictionary."""
    return {
        MetricName.REQUEST_RATE: 10000.0,  # Very high
        MetricName.APPLICATION_LATENCY: 5000.0,  # Very high
        MetricName.CLIENT_LATENCY: 1000.0,
        MetricName.DATABASE_LATENCY: 500.0,
        MetricName.ERROR_RATE: 0.3,  # 30% error rate
    }


# =============================================================================
# Data Fixtures - Training Data
# =============================================================================


@pytest.fixture
def sample_training_data() -> pd.DataFrame:
    """Provide sample training data for ML models."""
    np.random.seed(42)
    n_samples = 500

    return pd.DataFrame({
        MetricName.REQUEST_RATE: np.random.exponential(100, n_samples),
        MetricName.APPLICATION_LATENCY: np.random.exponential(50, n_samples),
        MetricName.CLIENT_LATENCY: np.random.exponential(20, n_samples),
        MetricName.DATABASE_LATENCY: np.random.exponential(10, n_samples),
        MetricName.ERROR_RATE: np.random.beta(1, 100, n_samples),
    })


@pytest.fixture
def time_series_training_data() -> pd.DataFrame:
    """Provide time-series training data with datetime index."""
    np.random.seed(42)

    dates = pd.date_range(
        start="2024-01-01",
        end="2024-01-14",
        freq="15min",
    )
    n_samples = len(dates)

    return pd.DataFrame({
        MetricName.REQUEST_RATE: np.random.exponential(100, n_samples),
        MetricName.APPLICATION_LATENCY: np.random.exponential(50, n_samples),
        MetricName.CLIENT_LATENCY: np.random.exponential(20, n_samples),
        MetricName.DATABASE_LATENCY: np.random.exponential(10, n_samples),
        MetricName.ERROR_RATE: np.random.beta(1, 100, n_samples),
    }, index=dates)


@pytest.fixture
def realistic_training_data() -> pd.DataFrame:
    """Provide realistic training data with zero-normal metrics."""
    np.random.seed(42)
    n_samples = 1000

    # Lognormal distributions for traffic metrics
    request_rate = np.random.lognormal(4, 0.5, n_samples)
    app_latency = np.random.lognormal(3.5, 0.8, n_samples)

    # Client and DB latency often zero
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

    # Beta distribution for error rate (typically low)
    error_rate = np.random.beta(1, 200, n_samples)

    return pd.DataFrame({
        MetricName.REQUEST_RATE: request_rate,
        MetricName.APPLICATION_LATENCY: app_latency,
        MetricName.CLIENT_LATENCY: client_latency,
        MetricName.DATABASE_LATENCY: db_latency,
        MetricName.ERROR_RATE: error_rate,
    })


# =============================================================================
# Data Fixtures - Anomaly Results
# =============================================================================


@pytest.fixture
def sample_anomaly_result() -> dict[str, Any]:
    """Provide a sample anomaly detection result."""
    return {
        "service": "test-service",
        "timestamp": datetime.now().isoformat(),
        "anomalies": {
            "high_latency": {
                "type": "threshold",
                "severity": "high",
                "value": 500.0,
                "threshold": 200.0,
                "score": -0.4,
                "description": "Application latency exceeded threshold",
                "detection_method": "isolation_forest",
            },
        },
        "current_metrics": {
            MetricName.REQUEST_RATE: 100.0,
            MetricName.APPLICATION_LATENCY: 500.0,
            MetricName.ERROR_RATE: 0.02,
        },
    }


@pytest.fixture
def empty_anomaly_result() -> dict[str, Any]:
    """Provide an anomaly result with no anomalies."""
    return {
        "service": "test-service",
        "timestamp": datetime.now().isoformat(),
        "anomalies": {},
        "current_metrics": {
            MetricName.REQUEST_RATE: 100.0,
            MetricName.APPLICATION_LATENCY: 50.0,
            MetricName.ERROR_RATE: 0.01,
        },
    }


@pytest.fixture
def multiple_anomalies_result() -> dict[str, Any]:
    """Provide a result with multiple anomalies."""
    return {
        "service": "test-service",
        "timestamp": datetime.now().isoformat(),
        "anomalies": [
            {
                "type": "multivariate",
                "severity": "critical",
                "score": -0.7,
                "description": "Multiple metrics showing unusual patterns",
                "detection_method": "isolation_forest",
            },
            {
                "type": "threshold",
                "severity": "high",
                "value": 5000.0,
                "threshold": 200.0,
                "description": "Application latency exceeded threshold",
                "detection_method": "threshold",
            },
        ],
        "current_metrics": {
            MetricName.REQUEST_RATE: 10000.0,
            MetricName.APPLICATION_LATENCY: 5000.0,
            MetricName.ERROR_RATE: 0.25,
        },
    }


# =============================================================================
# Mock Fixtures
# =============================================================================


@pytest.fixture
def mock_session() -> MagicMock:
    """Provide a mock requests session."""
    session = MagicMock()
    session.get.return_value.status_code = 200
    session.get.return_value.json.return_value = {
        "status": "success",
        "data": {
            "result": [
                {
                    "metric": {"service_name": "test-service"},
                    "value": [1234567890, "100.0"],
                }
            ]
        },
    }
    return session


@pytest.fixture
def mock_vm_response() -> dict[str, Any]:
    """Provide a mock VictoriaMetrics response."""
    return {
        "status": "success",
        "data": {
            "result": [
                {
                    "metric": {"service_name": "test-service"},
                    "value": [1234567890, "100.0"],
                }
            ]
        },
    }


# =============================================================================
# Service Fixtures
# =============================================================================


@pytest.fixture
def critical_services() -> list[str]:
    """Provide list of critical service names."""
    return ["booking", "search", "mobile-api", "shire-api"]


@pytest.fixture
def admin_services() -> list[str]:
    """Provide list of admin service names."""
    return ["m2-fr-adm", "m2-it-adm", "m2-bb-adm"]


@pytest.fixture
def all_time_periods() -> list[TimePeriod]:
    """Provide all time periods."""
    return list(TimePeriod)


# =============================================================================
# Helper Functions
# =============================================================================


def create_mock_metrics_response(value: float) -> dict[str, Any]:
    """Create a mock VictoriaMetrics response."""
    return {
        "status": "success",
        "data": {
            "result": [
                {
                    "metric": {"service_name": "test-service"},
                    "value": [1234567890, str(value)],
                }
            ]
        },
    }


def create_test_anomaly(
    anomaly_type: str = "test_anomaly",
    severity: str = "medium",
    value: float = 100.0,
    threshold: float = 50.0,
) -> dict[str, Any]:
    """Create a test anomaly dictionary."""
    return {
        "type": anomaly_type,
        "severity": severity,
        "value": value,
        "threshold": threshold,
        "score": -0.3,
        "description": f"Test anomaly: {anomaly_type}",
        "detection_method": "test",
    }


def create_datetime_for_period(period: TimePeriod) -> datetime:
    """Create a datetime that falls within the specified time period."""
    base_date = datetime(2024, 1, 15)  # Monday

    period_times = {
        TimePeriod.BUSINESS_HOURS: datetime(2024, 1, 15, 10, 0),  # Monday 10am
        TimePeriod.EVENING_HOURS: datetime(2024, 1, 15, 19, 0),  # Monday 7pm
        TimePeriod.NIGHT_HOURS: datetime(2024, 1, 15, 2, 0),  # Monday 2am
        TimePeriod.WEEKEND_DAY: datetime(2024, 1, 13, 14, 0),  # Saturday 2pm
        TimePeriod.WEEKEND_NIGHT: datetime(2024, 1, 13, 23, 0),  # Saturday 11pm
    }
    return period_times.get(period, base_date)


def generate_training_data_for_period(
    period: TimePeriod,
    n_samples: int = 200,
) -> pd.DataFrame:
    """Generate training data for a specific time period."""
    np.random.seed(42)

    # Vary characteristics based on period
    multipliers = {
        TimePeriod.BUSINESS_HOURS: {"rate": 1.0, "latency": 1.0},
        TimePeriod.EVENING_HOURS: {"rate": 0.7, "latency": 1.1},
        TimePeriod.NIGHT_HOURS: {"rate": 0.3, "latency": 0.8},
        TimePeriod.WEEKEND_DAY: {"rate": 0.5, "latency": 1.0},
        TimePeriod.WEEKEND_NIGHT: {"rate": 0.2, "latency": 0.9},
    }

    mult = multipliers.get(period, {"rate": 1.0, "latency": 1.0})

    return pd.DataFrame({
        MetricName.REQUEST_RATE: np.random.exponential(100 * mult["rate"], n_samples),
        MetricName.APPLICATION_LATENCY: np.random.exponential(50 * mult["latency"], n_samples),
        MetricName.CLIENT_LATENCY: np.random.exponential(20, n_samples),
        MetricName.DATABASE_LATENCY: np.random.exponential(10, n_samples),
        MetricName.ERROR_RATE: np.random.beta(1, 100, n_samples),
    })
