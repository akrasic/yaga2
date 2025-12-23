"""
Tests for configuration module.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from smartbox_anomaly.core import (
    PipelineConfig,
    ServiceConfig,
    TimePeriodConfig,
    VictoriaMetricsConfig,
    get_config,
    reset_config,
    set_config,
)


class TestVictoriaMetricsConfig:
    """Tests for VictoriaMetrics configuration."""

    def test_default_values(self):
        """Test default configuration values."""
        config = VictoriaMetricsConfig()
        assert config.timeout_seconds == 10
        assert config.max_retries == 3
        assert config.pool_connections == 20
        assert config.circuit_breaker_threshold == 5

    def test_from_env(self):
        """Test loading from environment variables."""
        with patch.dict(os.environ, {
            "VM_ENDPOINT": "http://custom:9090",
            "VM_TIMEOUT": "20",
        }):
            config = VictoriaMetricsConfig.from_env()
            assert config.endpoint == "http://custom:9090"
            assert config.timeout_seconds == 20

    def test_immutability(self):
        """Test that config is immutable (frozen)."""
        config = VictoriaMetricsConfig()
        with pytest.raises(Exception):  # FrozenInstanceError
            config.timeout_seconds = 999


class TestServiceConfig:
    """Tests for service classification configuration."""

    def test_get_category_known_service(self):
        """Test category lookup for known services."""
        config = ServiceConfig()
        assert config.get_category("booking") == "critical"
        assert config.get_category("fa5") == "micro"
        assert config.get_category("m2-bb") == "core"

    def test_get_category_pattern_detection(self):
        """Test pattern-based category detection."""
        config = ServiceConfig()
        assert config.get_category("payment-api") == "critical"
        assert config.get_category("admin-console") == "admin"
        assert config.get_category("email-worker") == "background"

    def test_get_category_unknown(self):
        """Test fallback for unknown services."""
        config = ServiceConfig()
        category = config.get_category("completely-unknown-service")
        assert category == "standard"


class TestTimePeriodConfig:
    """Tests for time period configuration."""

    def test_default_hour_ranges(self):
        """Test default hour ranges."""
        config = TimePeriodConfig()
        assert config.business_hours == (8, 18)
        assert config.evening_hours == (18, 22)
        assert config.night_hours == (22, 6)

    def test_default_thresholds(self):
        """Test default validation thresholds."""
        config = TimePeriodConfig()
        assert "business_hours" in config.default_thresholds
        assert "weekend_night" in config.default_thresholds


class TestPipelineConfig:
    """Tests for root pipeline configuration."""

    def test_default_creation(self):
        """Test creating config with defaults."""
        config = PipelineConfig.default()
        assert config.victoria_metrics is not None
        assert config.model is not None
        assert config.inference is not None

    def test_from_env(self):
        """Test creating config from environment."""
        with patch.dict(os.environ, {
            "VM_ENDPOINT": "http://test:9090",
            "MODELS_DIR": "/custom/models",
        }):
            config = PipelineConfig.from_env()
            assert config.victoria_metrics.endpoint == "http://test:9090"
            assert config.model.models_directory == "/custom/models"


class TestGlobalConfig:
    """Tests for global configuration management."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def teardown_method(self):
        """Reset config after each test."""
        reset_config()

    def test_get_config_creates_default(self):
        """Test that get_config creates default if none set."""
        config = get_config()
        assert config is not None
        assert isinstance(config, PipelineConfig)

    def test_set_config(self):
        """Test setting a custom configuration."""
        custom = PipelineConfig(
            victoria_metrics=VictoriaMetricsConfig(endpoint="http://custom:9090")
        )
        set_config(custom)

        retrieved = get_config()
        assert retrieved.victoria_metrics.endpoint == "http://custom:9090"

    def test_reset_config(self):
        """Test resetting configuration."""
        custom = PipelineConfig(
            victoria_metrics=VictoriaMetricsConfig(endpoint="http://custom:9090")
        )
        set_config(custom)
        reset_config()

        # Should create new default
        config = get_config()
        assert config.victoria_metrics.endpoint != "http://custom:9090"

    def test_config_singleton_behavior(self):
        """Test that get_config returns same instance."""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2
