"""
Tests for VictoriaMetrics client.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from smartbox_anomaly.core import (
    CircuitBreakerOpenError,
    VictoriaMetricsConfig,
)
from smartbox_anomaly.metrics import (
    CircuitBreakerState,
    InferenceMetrics,
    VictoriaMetricsClient,
)


class TestInferenceMetrics:
    """Tests for InferenceMetrics dataclass."""

    def test_creation(self):
        """Test creating inference metrics."""
        metrics = InferenceMetrics(
            service_name="test",
            timestamp=datetime.now(),
            request_rate=100.0,
            application_latency=50.0,
            error_rate=0.01,
        )
        assert metrics.service_name == "test"
        assert metrics.request_rate == 100.0

    def test_to_dict(self):
        """Test converting to dictionary."""
        metrics = InferenceMetrics(
            service_name="test",
            timestamp=datetime.now(),
            request_rate=100.0,
            application_latency=50.0,
        )
        result = metrics.to_dict()
        assert result["request_rate"] == 100.0
        assert result["application_latency"] == 50.0
        assert result["dependency_latency"] == 0.0  # Default for None

    def test_validation_valid(self):
        """Test validation of valid metrics."""
        metrics = InferenceMetrics(
            service_name="test",
            timestamp=datetime.now(),
            request_rate=100.0,
            application_latency=50.0,
            error_rate=0.01,
        )
        assert metrics.is_valid()

    def test_validation_invalid_negative(self):
        """Test validation catches negative values."""
        metrics = InferenceMetrics(
            service_name="test",
            timestamp=datetime.now(),
            request_rate=-100.0,
        )
        result = metrics.validate()
        assert not result.is_valid


class TestCircuitBreakerState:
    """Tests for CircuitBreakerState."""

    def test_initial_state(self):
        """Test initial state is closed."""
        cb = CircuitBreakerState()
        assert not cb.is_open()
        assert cb.failure_count == 0

    def test_record_success(self):
        """Test success resets state."""
        cb = CircuitBreakerState()
        cb.failure_count = 3
        cb.record_success()
        assert cb.failure_count == 0
        assert cb.last_failure_time is None

    def test_record_failure(self):
        """Test failure increments count."""
        cb = CircuitBreakerState()
        cb.record_failure()
        assert cb.failure_count == 1
        assert cb.last_failure_time is not None

    def test_circuit_opens_after_threshold(self):
        """Test circuit opens after threshold failures."""
        cb = CircuitBreakerState(threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.is_open()

    def test_circuit_stays_closed_below_threshold(self):
        """Test circuit stays closed below threshold."""
        cb = CircuitBreakerState(threshold=5)
        for _ in range(4):
            cb.record_failure()
        assert not cb.is_open()

    def test_time_until_reset(self):
        """Test time until reset calculation."""
        cb = CircuitBreakerState(threshold=3, timeout_seconds=300)
        for _ in range(3):
            cb.record_failure()

        remaining = cb.time_until_reset()
        assert remaining is not None
        assert 0 < remaining <= 300


class TestVictoriaMetricsClient:
    """Tests for VictoriaMetricsClient."""

    @pytest.fixture
    def client(self):
        """Create a client with mock config."""
        config = VictoriaMetricsConfig(
            endpoint="http://test:9090",
            timeout_seconds=5,
            max_retries=2,
        )
        return VictoriaMetricsClient(config=config)

    def test_initialization(self, client):
        """Test client initialization."""
        assert client.endpoint == "http://test:9090"
        assert not client.is_circuit_open()

    def test_circuit_breaker_integration(self, client):
        """Test circuit breaker is used correctly."""
        # Force circuit breaker open
        for _ in range(5):
            client._circuit_breaker.record_failure()

        assert client.is_circuit_open()

        with pytest.raises(CircuitBreakerOpenError):
            client.collect_service_metrics("test")

    def test_query_templates_format(self, client):
        """Test query templates are properly formatted."""
        for _metric, query in client.QUERIES.items():
            assert isinstance(query, str)
            assert len(query) > 0

    @pytest.mark.integration
    def test_collect_service_metrics_requires_network(self, client):
        """Test metrics collection (requires network, skip in CI)."""
        # This test would require actual VictoriaMetrics connection
        # Mark as integration test to skip in normal test runs
        pytest.skip("Requires network connection to VictoriaMetrics")

    def test_context_manager(self):
        """Test client works as context manager."""
        config = VictoriaMetricsConfig(endpoint="http://test:9090")
        with VictoriaMetricsClient(config=config) as client:
            assert client is not None
        # Session should be closed after context exit

    def test_health_check_when_circuit_open(self, client):
        """Test health check fails when circuit is open."""
        for _ in range(5):
            client._circuit_breaker.record_failure()

        # health_check returns (bool, dict), is_healthy returns just bool
        healthy, details = client.health_check()
        assert not healthy
        assert details["circuit_breaker_open"] is True
        assert not client.is_healthy()


class TestVictoriaMetricsClientQueries:
    """Tests for query construction."""

    def test_query_templates_defined(self):
        """Test that all query templates are defined."""
        assert len(VictoriaMetricsClient.QUERIES) == 5
        assert "request_rate" in VictoriaMetricsClient.QUERIES
        assert "application_latency" in VictoriaMetricsClient.QUERIES
        assert "error_rate" in VictoriaMetricsClient.QUERIES
