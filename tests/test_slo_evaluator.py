"""
Tests for SLO evaluator module.

This module tests the SLO-aware severity evaluation, including:
- Application latency evaluation
- Error rate evaluation
- Database latency evaluation (floor + ratio-based thresholds)
- Severity adjustment logic
- Explanation generation
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from smartbox_anomaly.core.config import (
    DatabaseLatencyRatios,
    ServiceSLOConfig,
    SLOConfig,
)
from smartbox_anomaly.slo.evaluator import SLOEvaluator, SLOEvaluationResult


@pytest.fixture
def sample_config() -> dict:
    """Load the actual config.json for testing."""
    with open("config.json", "r") as f:
        return json.load(f)


@pytest.fixture
def slo_config(sample_config: dict) -> SLOConfig:
    """Create SLO config from sample config."""
    return SLOConfig.from_config(sample_config)


@pytest.fixture
def evaluator(slo_config: SLOConfig) -> SLOEvaluator:
    """Create SLO evaluator instance."""
    return SLOEvaluator(slo_config)


class TestDatabaseLatencyRatios:
    """Tests for DatabaseLatencyRatios dataclass."""

    def test_default_values(self):
        """Test default ratio values."""
        ratios = DatabaseLatencyRatios()
        assert ratios.info == 1.5
        assert ratios.warning == 2.0
        assert ratios.high == 3.0
        assert ratios.critical == 5.0

    def test_from_dict(self):
        """Test creating from dictionary."""
        ratios = DatabaseLatencyRatios.from_dict({
            "info": 1.3,
            "warning": 1.5,
            "high": 2.0,
            "critical": 3.0,
        })
        assert ratios.info == 1.3
        assert ratios.warning == 1.5
        assert ratios.high == 2.0
        assert ratios.critical == 3.0

    def test_from_dict_partial(self):
        """Test creating from partial dictionary uses defaults."""
        ratios = DatabaseLatencyRatios.from_dict({"info": 1.2})
        assert ratios.info == 1.2
        assert ratios.warning == 2.0  # Default
        assert ratios.high == 3.0     # Default
        assert ratios.critical == 5.0  # Default

    def test_immutability(self):
        """Test that ratios are immutable."""
        ratios = DatabaseLatencyRatios()
        with pytest.raises(Exception):  # FrozenInstanceError
            ratios.info = 999


class TestServiceSLOConfigDatabaseLatency:
    """Tests for database latency fields in ServiceSLOConfig."""

    def test_default_database_latency_settings(self):
        """Test default database latency floor and ratios."""
        slo = ServiceSLOConfig()
        assert slo.database_latency_floor_ms == 1.0
        assert slo.database_latency_ratios.info == 1.5
        assert slo.database_latency_ratios.warning == 2.0
        assert slo.database_latency_ratios.high == 3.0
        assert slo.database_latency_ratios.critical == 5.0

    def test_custom_database_latency_settings(self):
        """Test custom database latency settings."""
        ratios = DatabaseLatencyRatios(info=1.3, warning=1.5, high=2.0, critical=3.0)
        slo = ServiceSLOConfig(
            database_latency_floor_ms=2.0,
            database_latency_ratios=ratios,
        )
        assert slo.database_latency_floor_ms == 2.0
        assert slo.database_latency_ratios.critical == 3.0


class TestSLOConfigDatabaseLatencyParsing:
    """Tests for parsing database latency settings from config."""

    def test_defaults_loaded(self, slo_config: SLOConfig):
        """Test that default database latency settings are loaded."""
        default_slo = slo_config.get_service_slo("unknown_service")
        assert default_slo.database_latency_floor_ms == 1.0
        assert default_slo.database_latency_ratios.info == 1.5

    def test_booking_service_override(self, slo_config: SLOConfig):
        """Test booking service floor override."""
        booking_slo = slo_config.get_service_slo("booking")
        assert booking_slo.database_latency_floor_ms == 0.5
        # Should use defaults for ratios since not overridden
        assert booking_slo.database_latency_ratios.info == 1.5

    def test_search_service_full_override(self, slo_config: SLOConfig):
        """Test search service with custom floor and ratios."""
        search_slo = slo_config.get_service_slo("search")
        assert search_slo.database_latency_floor_ms == 0.3
        assert search_slo.database_latency_ratios.info == 1.3
        assert search_slo.database_latency_ratios.warning == 1.5
        assert search_slo.database_latency_ratios.high == 2.0
        assert search_slo.database_latency_ratios.critical == 3.0

    def test_vms_service_high_floor(self, slo_config: SLOConfig):
        """Test VMS service with higher floor for slower DB."""
        vms_slo = slo_config.get_service_slo("vms")
        assert vms_slo.database_latency_floor_ms == 5.0


class TestDatabaseLatencyEvaluation:
    """Tests for _evaluate_database_latency method."""

    def test_below_floor_always_ok(self, evaluator: SLOEvaluator):
        """Test that latency below floor is always OK."""
        slo = ServiceSLOConfig(database_latency_floor_ms=5.0)

        result = evaluator._evaluate_database_latency(
            db_latency_ms=2.0,
            baseline_mean_ms=1.0,  # Even though 2x baseline
            slo=slo,
        )

        assert result["status"] == "ok"
        assert result["below_floor"] is True
        assert "Below noise floor" in result.get("explanation", "")

    def test_zero_latency_ok(self, evaluator: SLOEvaluator):
        """Test zero latency returns OK."""
        slo = ServiceSLOConfig()
        result = evaluator._evaluate_database_latency(
            db_latency_ms=0.0,
            baseline_mean_ms=5.0,
            slo=slo,
        )
        assert result["status"] == "ok"
        assert result["below_floor"] is True

    def test_ratio_below_info_is_ok(self, evaluator: SLOEvaluator):
        """Test ratio below info threshold is OK."""
        slo = ServiceSLOConfig(database_latency_floor_ms=5.0)

        result = evaluator._evaluate_database_latency(
            db_latency_ms=7.0,    # Above floor
            baseline_mean_ms=5.0,  # 1.4x ratio (below 1.5)
            slo=slo,
        )

        assert result["status"] == "ok"
        assert result["ratio"] == 1.4
        assert result["below_floor"] is False

    def test_ratio_at_info_threshold(self, evaluator: SLOEvaluator):
        """Test ratio at info threshold returns info."""
        slo = ServiceSLOConfig(database_latency_floor_ms=5.0)

        result = evaluator._evaluate_database_latency(
            db_latency_ms=7.5,    # Above floor
            baseline_mean_ms=5.0,  # 1.5x ratio (exactly at info)
            slo=slo,
        )

        assert result["status"] == "info"
        assert result["ratio"] == 1.5

    def test_ratio_at_warning_threshold(self, evaluator: SLOEvaluator):
        """Test ratio at warning threshold returns warning."""
        slo = ServiceSLOConfig(database_latency_floor_ms=5.0)

        result = evaluator._evaluate_database_latency(
            db_latency_ms=10.0,
            baseline_mean_ms=5.0,  # 2.0x ratio
            slo=slo,
        )

        assert result["status"] == "warning"
        assert result["ratio"] == 2.0

    def test_ratio_at_high_threshold(self, evaluator: SLOEvaluator):
        """Test ratio at high threshold returns high."""
        slo = ServiceSLOConfig(database_latency_floor_ms=5.0)

        result = evaluator._evaluate_database_latency(
            db_latency_ms=15.0,
            baseline_mean_ms=5.0,  # 3.0x ratio
            slo=slo,
        )

        assert result["status"] == "high"
        assert result["ratio"] == 3.0

    def test_ratio_at_critical_threshold(self, evaluator: SLOEvaluator):
        """Test ratio at critical threshold returns critical."""
        slo = ServiceSLOConfig(database_latency_floor_ms=5.0)

        result = evaluator._evaluate_database_latency(
            db_latency_ms=25.0,
            baseline_mean_ms=5.0,  # 5.0x ratio
            slo=slo,
        )

        assert result["status"] == "critical"
        assert result["ratio"] == 5.0

    def test_ratio_above_critical(self, evaluator: SLOEvaluator):
        """Test ratio above critical threshold returns critical."""
        slo = ServiceSLOConfig(database_latency_floor_ms=5.0)

        result = evaluator._evaluate_database_latency(
            db_latency_ms=50.0,
            baseline_mean_ms=5.0,  # 10x ratio
            slo=slo,
        )

        assert result["status"] == "critical"
        assert result["ratio"] == 10.0

    def test_no_baseline_fallback(self, evaluator: SLOEvaluator):
        """Test fallback to absolute thresholds when no baseline."""
        slo = ServiceSLOConfig(database_latency_floor_ms=5.0)

        result = evaluator._evaluate_database_latency(
            db_latency_ms=10.0,
            baseline_mean_ms=0.0,  # No baseline
            slo=slo,
        )

        assert result["ratio"] is None
        assert "No baseline available" in result.get("explanation", "")

    def test_custom_thresholds_search(self, evaluator: SLOEvaluator):
        """Test search service with stricter custom thresholds."""
        # Search has: floor=2ms, ratios: 1.3/1.5/2.0/3.0
        ratios = DatabaseLatencyRatios(info=1.3, warning=1.5, high=2.0, critical=3.0)
        slo = ServiceSLOConfig(
            database_latency_floor_ms=2.0,
            database_latency_ratios=ratios,
        )

        # 3x baseline should be critical for search (but only high for default)
        result = evaluator._evaluate_database_latency(
            db_latency_ms=6.0,
            baseline_mean_ms=2.0,  # 3.0x ratio
            slo=slo,
        )

        assert result["status"] == "critical"

    def test_includes_thresholds_in_result(self, evaluator: SLOEvaluator):
        """Test that result includes threshold information."""
        slo = ServiceSLOConfig()

        result = evaluator._evaluate_database_latency(
            db_latency_ms=10.0,
            baseline_mean_ms=5.0,
            slo=slo,
        )

        assert "thresholds" in result
        assert result["thresholds"]["info"] == 1.5
        assert result["thresholds"]["warning"] == 2.0
        assert result["thresholds"]["high"] == 3.0
        assert result["thresholds"]["critical"] == 5.0


class TestSLOEvaluatorIntegration:
    """Integration tests for database latency in full evaluation flow."""

    def test_db_latency_below_floor_suppresses_anomaly(self, evaluator: SLOEvaluator):
        """Test that DB latency below floor suppresses the anomaly entirely.

        When the database_latency is below the noise floor (e.g., 0.5ms < 1ms),
        the database_degradation anomaly should be filtered out completely
        because it's not operationally significant.
        """
        result = evaluator.evaluate_result({
            "service": "titan",
            "overall_severity": "medium",
            "alert_type": "anomaly_detected",
            "metrics": {
                "server_latency_avg": 50.0,
                "error_rate": 0.001,
                "request_rate": 100.0,
                "db_latency_avg": 0.5,   # Below 1ms floor
                "db_latency_mean": 0.3,
            },
            "anomalies": {"database_degradation": {"severity": "medium"}},
        })

        slo_eval = result.get("slo_evaluation", {})
        db_eval = slo_eval.get("database_latency_evaluation", {})

        # DB evaluation should show below floor
        assert db_eval["status"] == "ok"
        assert db_eval["below_floor"] is True
        assert slo_eval["slo_status"] == "ok"

        # The anomaly should be completely filtered out
        assert len(result.get("anomalies", {})) == 0
        assert result.get("anomaly_count") == 0
        assert result.get("alert_type") == "no_anomaly"
        assert result.get("overall_severity") == "none"

        # Explanation should mention suppression
        assert "suppressed" in slo_eval.get("explanation", "").lower()

    def test_db_latency_at_floor_not_suppressed(self, evaluator: SLOEvaluator):
        """Test that DB latency exactly at floor is NOT suppressed.

        The floor is exclusive - values >= floor should be evaluated normally.
        """
        result = evaluator.evaluate_result({
            "service": "titan",
            "overall_severity": "medium",
            "alert_type": "anomaly_detected",
            "metrics": {
                "server_latency_avg": 50.0,
                "error_rate": 0.001,
                "request_rate": 100.0,
                "db_latency_avg": 1.0,   # Exactly at 1ms floor
                "db_latency_mean": 0.5,
            },
            "anomalies": {"database_degradation": {"severity": "medium"}},
        })

        # Anomaly should NOT be suppressed when at floor
        assert len(result.get("anomalies", {})) == 1
        assert "database_degradation" in result.get("anomalies", {})
        assert result.get("alert_type") == "anomaly_detected"

    def test_db_latency_above_floor_not_suppressed(self, evaluator: SLOEvaluator):
        """Test that DB latency above floor is NOT suppressed."""
        result = evaluator.evaluate_result({
            "service": "titan",
            "overall_severity": "medium",
            "alert_type": "anomaly_detected",
            "metrics": {
                "server_latency_avg": 50.0,
                "error_rate": 0.001,
                "request_rate": 100.0,
                "db_latency_avg": 5.0,   # Well above 1ms floor
                "db_latency_mean": 2.0,
            },
            "anomalies": {"database_degradation": {"severity": "medium"}},
        })

        # Anomaly should NOT be suppressed
        assert len(result.get("anomalies", {})) == 1
        assert "database_degradation" in result.get("anomalies", {})
        assert result.get("alert_type") == "anomaly_detected"

    def test_mixed_anomalies_partial_suppression(self, evaluator: SLOEvaluator):
        """Test that only below-floor anomalies are suppressed, others remain.

        When we have multiple anomalies and only one is database-related with
        below-floor latency, only that one should be suppressed.
        """
        result = evaluator.evaluate_result({
            "service": "titan",
            "overall_severity": "high",
            "alert_type": "anomaly_detected",
            "metrics": {
                "server_latency_avg": 800.0,  # High latency - above warning threshold
                "error_rate": 0.001,
                "request_rate": 100.0,
                "db_latency_avg": 0.5,   # Below floor - should be suppressed
                "db_latency_mean": 0.3,
            },
            "anomalies": {
                "database_degradation": {"severity": "medium"},  # Should be suppressed
                "latency_elevated": {"severity": "high"},        # Should remain
            },
        })

        # Only latency anomaly should remain
        anomalies = result.get("anomalies", {})
        assert len(anomalies) == 1
        assert "latency_elevated" in anomalies
        assert "database_degradation" not in anomalies
        assert result.get("alert_type") == "anomaly_detected"
        assert result.get("anomaly_count") == 1

    def test_db_latency_critical_breaches_slo(self, evaluator: SLOEvaluator):
        """Test that critical DB latency breaches SLO."""
        result = evaluator.evaluate_result({
            "service": "titan",
            "overall_severity": "medium",
            "metrics": {
                "server_latency_avg": 50.0,
                "error_rate": 0.001,
                "request_rate": 100.0,
                "db_latency_avg": 25.0,   # 5x baseline
                "db_latency_mean": 5.0,
            },
            "anomalies": {"database_degradation": {"severity": "medium"}},
        })

        slo_eval = result.get("slo_evaluation", {})
        db_eval = slo_eval.get("database_latency_evaluation", {})

        assert db_eval["status"] == "critical"
        assert slo_eval["slo_status"] == "breached"

    def test_db_latency_warning_affects_overall_status(self, evaluator: SLOEvaluator):
        """Test that warning DB latency affects overall SLO status."""
        result = evaluator.evaluate_result({
            "service": "titan",
            "overall_severity": "low",
            "metrics": {
                "server_latency_avg": 50.0,
                "error_rate": 0.001,
                "request_rate": 100.0,
                "db_latency_avg": 10.0,   # 2x baseline = warning
                "db_latency_mean": 5.0,
            },
            "anomalies": {"database_degradation": {"severity": "low"}},
        })

        slo_eval = result.get("slo_evaluation", {})

        assert slo_eval["slo_status"] == "warning"

    def test_anomaly_context_includes_db_info(self, evaluator: SLOEvaluator):
        """Test that database anomalies get SLO context."""
        result = evaluator.evaluate_result({
            "service": "titan",
            "overall_severity": "medium",
            "metrics": {
                "server_latency_avg": 50.0,
                "error_rate": 0.001,
                "request_rate": 100.0,
                "db_latency_avg": 10.0,
                "db_latency_mean": 5.0,
            },
            "anomalies": {"database_degradation": {"severity": "medium"}},
        })

        anomalies = result.get("anomalies", {})
        db_anomaly = anomalies.get("database_degradation", {})
        slo_context = db_anomaly.get("slo_context", {})

        assert slo_context["current_value_ms"] == 10.0
        assert slo_context["baseline_mean_ms"] == 5.0
        assert slo_context["ratio"] == 2.0
        assert slo_context["floor_ms"] == 1.0  # Default floor is now 1ms
        assert "thresholds" in slo_context

    def test_explanation_mentions_db_latency_issue(self, evaluator: SLOEvaluator):
        """Test that explanation includes DB latency when elevated."""
        result = evaluator.evaluate_result({
            "service": "titan",
            "overall_severity": "high",
            "metrics": {
                "server_latency_avg": 50.0,
                "error_rate": 0.001,
                "request_rate": 100.0,
                "db_latency_avg": 25.0,
                "db_latency_mean": 5.0,
            },
            "anomalies": {"database_degradation": {"severity": "high"}},
        })

        explanation = result.get("slo_evaluation", {}).get("explanation", "")
        assert "Database latency" in explanation or "DB latency" in explanation

    def test_explanation_mentions_noise_floor(self, evaluator: SLOEvaluator):
        """Test that explanation mentions noise floor when applicable."""
        result = evaluator.evaluate_result({
            "service": "titan",
            "overall_severity": "medium",
            "metrics": {
                "server_latency_avg": 50.0,
                "error_rate": 0.001,
                "request_rate": 100.0,
                "db_latency_avg": 0.5,   # Below 1ms floor
                "db_latency_mean": 0.3,
            },
            "anomalies": {"database_degradation": {"severity": "medium"}},
        })

        explanation = result.get("slo_evaluation", {}).get("explanation", "")
        assert "noise floor" in explanation.lower()

    def test_no_db_metrics_no_db_evaluation(self, evaluator: SLOEvaluator):
        """Test that missing DB metrics results in no DB evaluation."""
        result = evaluator.evaluate_result({
            "service": "titan",
            "overall_severity": "medium",
            "metrics": {
                "server_latency_avg": 50.0,
                "error_rate": 0.001,
                "request_rate": 100.0,
                # No db_latency_avg
            },
            "anomalies": {"elevated_latency": {"severity": "medium"}},
        })

        db_eval = result.get("slo_evaluation", {}).get("database_latency_evaluation", {})
        assert db_eval == {}


class TestCombineSLOStatus:
    """Tests for _combine_slo_status method with database latency."""

    def test_ok_all_around(self, evaluator: SLOEvaluator):
        """Test all OK returns OK."""
        status = evaluator._combine_slo_status("ok", "ok", "ok")
        assert status == "ok"

    def test_db_info_maps_to_elevated(self, evaluator: SLOEvaluator):
        """Test DB info status maps to elevated."""
        status = evaluator._combine_slo_status("ok", "ok", "info")
        assert status == "elevated"

    def test_db_warning_maps_to_warning(self, evaluator: SLOEvaluator):
        """Test DB warning status maps to warning."""
        status = evaluator._combine_slo_status("ok", "ok", "warning")
        assert status == "warning"

    def test_db_high_maps_to_breached(self, evaluator: SLOEvaluator):
        """Test DB high status maps to breached."""
        status = evaluator._combine_slo_status("ok", "ok", "high")
        assert status == "breached"

    def test_db_critical_maps_to_breached(self, evaluator: SLOEvaluator):
        """Test DB critical status maps to breached."""
        status = evaluator._combine_slo_status("ok", "ok", "critical")
        assert status == "breached"

    def test_worst_status_wins(self, evaluator: SLOEvaluator):
        """Test that worst status among all three wins."""
        # DB critical should override latency warning
        status = evaluator._combine_slo_status("warning", "ok", "critical")
        assert status == "breached"

        # Error breached should override DB info
        status = evaluator._combine_slo_status("ok", "breached", "info")
        assert status == "breached"

    def test_none_db_status_ignored(self, evaluator: SLOEvaluator):
        """Test that None DB status is ignored."""
        status = evaluator._combine_slo_status("warning", "ok", None)
        assert status == "warning"


class TestBusyPeriodDatabaseLatency:
    """Tests for busy period factor applied to database latency."""

    def test_busy_period_relaxes_floor(self, evaluator: SLOEvaluator):
        """Test that busy period relaxes database latency floor."""
        slo = ServiceSLOConfig(
            database_latency_floor_ms=5.0,
            busy_period_factor=1.5,
        )

        relaxed_slo = evaluator._apply_busy_period_factor(slo, is_busy=True)

        assert relaxed_slo.database_latency_floor_ms == 7.5  # 5.0 * 1.5

    def test_busy_period_preserves_ratios(self, evaluator: SLOEvaluator):
        """Test that busy period preserves ratio thresholds."""
        ratios = DatabaseLatencyRatios(info=1.3, warning=1.5, high=2.0, critical=3.0)
        slo = ServiceSLOConfig(
            database_latency_floor_ms=5.0,
            database_latency_ratios=ratios,
            busy_period_factor=1.5,
        )

        relaxed_slo = evaluator._apply_busy_period_factor(slo, is_busy=True)

        # Ratios should not change
        assert relaxed_slo.database_latency_ratios.info == 1.3
        assert relaxed_slo.database_latency_ratios.critical == 3.0

    def test_not_busy_no_change(self, evaluator: SLOEvaluator):
        """Test that non-busy period returns original SLO."""
        slo = ServiceSLOConfig(database_latency_floor_ms=5.0)

        result = evaluator._apply_busy_period_factor(slo, is_busy=False)

        assert result is slo


class TestSLOEvaluationResultSerialization:
    """Tests for SLOEvaluationResult serialization."""

    def test_to_dict_includes_db_eval_when_present(self):
        """Test that to_dict includes database_latency_evaluation when present."""
        result = SLOEvaluationResult(
            original_severity="medium",
            adjusted_severity="low",
            slo_status="ok",
            slo_proximity=0.5,
            operational_impact="informational",
            database_latency_evaluation={
                "status": "ok",
                "below_floor": True,
                "value_ms": 2.0,
            },
        )

        d = result.to_dict()
        assert "database_latency_evaluation" in d
        assert d["database_latency_evaluation"]["status"] == "ok"

    def test_to_dict_excludes_empty_db_eval(self):
        """Test that to_dict excludes empty database_latency_evaluation."""
        result = SLOEvaluationResult(
            original_severity="medium",
            adjusted_severity="medium",
            slo_status="ok",
            slo_proximity=0.5,
            operational_impact="none",
            database_latency_evaluation={},  # Empty
        )

        d = result.to_dict()
        assert "database_latency_evaluation" not in d

    def test_to_dict_includes_request_rate_eval_when_present(self):
        """Test that to_dict includes request_rate_evaluation when present."""
        result = SLOEvaluationResult(
            original_severity="medium",
            adjusted_severity="low",
            slo_status="ok",
            slo_proximity=0.5,
            operational_impact="informational",
            request_rate_evaluation={
                "status": "info",
                "type": "surge",
                "ratio": 2.5,
            },
        )

        d = result.to_dict()
        assert "request_rate_evaluation" in d
        assert d["request_rate_evaluation"]["type"] == "surge"

    def test_to_dict_excludes_empty_request_rate_eval(self):
        """Test that to_dict excludes empty request_rate_evaluation."""
        result = SLOEvaluationResult(
            original_severity="medium",
            adjusted_severity="medium",
            slo_status="ok",
            slo_proximity=0.5,
            operational_impact="none",
            request_rate_evaluation={},  # Empty
        )

        d = result.to_dict()
        assert "request_rate_evaluation" not in d


class TestRequestRateEvaluation:
    """Tests for request rate (traffic surge/cliff) evaluation."""

    def test_surge_standalone_informational(self, evaluator: SLOEvaluator):
        """Test that a surge without SLO issues is informational."""
        result = evaluator.evaluate_result({
            "service": "search",
            "overall_severity": "medium",
            "time_period": "business_hours",
            "metrics": {
                "server_latency_avg": 100.0,  # Well within SLO
                "error_rate": 0.001,           # Well within SLO
                "request_rate": 250.0,         # 2.5x baseline = surge
                "request_rate_mean": 100.0,    # Baseline
            },
            "anomalies": {"traffic_surge": {"severity": "medium"}},
        })

        rr_eval = result.get("slo_evaluation", {}).get("request_rate_evaluation", {})

        assert rr_eval.get("type") == "surge"
        assert rr_eval.get("severity") == "informational"
        assert rr_eval.get("status") == "info"
        assert rr_eval.get("ratio") == 2.5
        assert rr_eval.get("correlated_with_latency") is False
        assert rr_eval.get("correlated_with_errors") is False

    def test_surge_with_latency_breach(self, evaluator: SLOEvaluator):
        """Test that a surge with latency SLO breach escalates to warning."""
        result = evaluator.evaluate_result({
            "service": "search",
            "overall_severity": "high",
            "time_period": "business_hours",
            "metrics": {
                "server_latency_avg": 400.0,   # Above warning threshold (350ms)
                "error_rate": 0.001,
                "request_rate": 250.0,
                "request_rate_mean": 100.0,
            },
            "anomalies": {"traffic_surge": {"severity": "high"}},
        })

        rr_eval = result.get("slo_evaluation", {}).get("request_rate_evaluation", {})

        assert rr_eval.get("type") == "surge"
        assert rr_eval.get("severity") == "warning"
        assert rr_eval.get("correlated_with_latency") is True
        assert "latency" in rr_eval.get("explanation", "").lower()

    def test_surge_with_error_breach(self, evaluator: SLOEvaluator):
        """Test that a surge with error SLO breach escalates to high."""
        result = evaluator.evaluate_result({
            "service": "search",
            "overall_severity": "high",
            "time_period": "business_hours",
            "metrics": {
                "server_latency_avg": 100.0,
                "error_rate": 0.015,           # Above warning threshold (1%)
                "request_rate": 250.0,
                "request_rate_mean": 100.0,
            },
            "anomalies": {"traffic_surge": {"severity": "high"}},
        })

        rr_eval = result.get("slo_evaluation", {}).get("request_rate_evaluation", {})

        assert rr_eval.get("type") == "surge"
        assert rr_eval.get("severity") == "high"
        assert rr_eval.get("correlated_with_errors") is True
        assert "error" in rr_eval.get("explanation", "").lower()

    def test_cliff_standalone_warning(self, evaluator: SLOEvaluator):
        """Test that a cliff during off-peak hours is warning."""
        result = evaluator.evaluate_result({
            "service": "search",
            "overall_severity": "medium",
            "time_period": "night_hours",  # Off-peak
            "metrics": {
                "server_latency_avg": 100.0,
                "error_rate": 0.001,
                "request_rate": 30.0,          # 30% of baseline = cliff
                "request_rate_mean": 100.0,
            },
            "anomalies": {"traffic_cliff": {"severity": "medium"}},
        })

        rr_eval = result.get("slo_evaluation", {}).get("request_rate_evaluation", {})

        assert rr_eval.get("type") == "cliff"
        assert rr_eval.get("severity") == "warning"
        assert rr_eval.get("is_peak_hours") is False

    def test_cliff_peak_hours_high(self, evaluator: SLOEvaluator):
        """Test that a cliff during peak hours escalates to high."""
        result = evaluator.evaluate_result({
            "service": "search",
            "overall_severity": "medium",
            "time_period": "business_hours",  # Peak hours
            "metrics": {
                "server_latency_avg": 100.0,
                "error_rate": 0.001,
                "request_rate": 30.0,
                "request_rate_mean": 100.0,
            },
            "anomalies": {"traffic_cliff": {"severity": "medium"}},
        })

        rr_eval = result.get("slo_evaluation", {}).get("request_rate_evaluation", {})

        assert rr_eval.get("type") == "cliff"
        assert rr_eval.get("severity") == "high"
        assert rr_eval.get("is_peak_hours") is True

    def test_cliff_with_errors_critical(self, evaluator: SLOEvaluator):
        """Test that a cliff with errors escalates to critical."""
        result = evaluator.evaluate_result({
            "service": "search",
            "overall_severity": "high",
            "time_period": "night_hours",
            "metrics": {
                "server_latency_avg": 100.0,
                "error_rate": 0.015,           # Above warning threshold
                "request_rate": 30.0,
                "request_rate_mean": 100.0,
            },
            "anomalies": {"traffic_cliff": {"severity": "high"}},
        })

        rr_eval = result.get("slo_evaluation", {}).get("request_rate_evaluation", {})

        assert rr_eval.get("type") == "cliff"
        assert rr_eval.get("severity") == "critical"
        assert "outage" in rr_eval.get("explanation", "").lower() or "failure" in rr_eval.get("explanation", "").lower()

    def test_below_minimum_expected_traffic(self, evaluator: SLOEvaluator):
        """Test detection of below minimum expected traffic."""
        result = evaluator.evaluate_result({
            "service": "titan",
            "overall_severity": "low",
            "time_period": "business_hours",
            "metrics": {
                "server_latency_avg": 100.0,
                "error_rate": 0.001,
                "request_rate": 2.0,           # Below 5.0 rps minimum
                "request_rate_mean": 3.0,      # Not a cliff (67% of baseline)
            },
            "anomalies": {},
        })

        rr_eval = result.get("slo_evaluation", {}).get("request_rate_evaluation", {})

        assert rr_eval.get("type") == "low_traffic"
        assert rr_eval.get("severity") == "informational"
        assert rr_eval.get("min_expected_rps") == 5.0

    def test_normal_traffic(self, evaluator: SLOEvaluator):
        """Test that normal traffic returns OK status."""
        result = evaluator.evaluate_result({
            "service": "search",
            "overall_severity": "none",
            "time_period": "business_hours",
            "metrics": {
                "server_latency_avg": 100.0,
                "error_rate": 0.001,
                "request_rate": 100.0,         # Same as baseline
                "request_rate_mean": 100.0,
            },
            "anomalies": {},
        })

        rr_eval = result.get("slo_evaluation", {}).get("request_rate_evaluation", {})

        assert rr_eval.get("type") == "normal"
        assert rr_eval.get("status") == "ok"
        assert rr_eval.get("ratio") == 1.0

    def test_no_baseline_no_surge_cliff(self, evaluator: SLOEvaluator):
        """Test that without baseline, surge/cliff detection uses minimum expected."""
        result = evaluator.evaluate_result({
            "service": "titan",
            "overall_severity": "none",
            "time_period": "business_hours",
            "metrics": {
                "server_latency_avg": 100.0,
                "error_rate": 0.001,
                "request_rate": 50.0,          # Good traffic but no baseline
                # No request_rate_mean
            },
            "anomalies": {},
        })

        rr_eval = result.get("slo_evaluation", {}).get("request_rate_evaluation", {})

        # Without baseline, ratio should be None and traffic should be normal
        assert rr_eval.get("ratio") is None
        assert rr_eval.get("type") == "normal"

    def test_explanation_includes_traffic_info(self, evaluator: SLOEvaluator):
        """Test that explanation includes traffic surge/cliff info."""
        result = evaluator.evaluate_result({
            "service": "search",
            "overall_severity": "medium",
            "time_period": "business_hours",
            "metrics": {
                "server_latency_avg": 100.0,
                "error_rate": 0.001,
                "request_rate": 250.0,
                "request_rate_mean": 100.0,
            },
            "anomalies": {"traffic_surge": {"severity": "medium"}},
        })

        explanation = result.get("slo_evaluation", {}).get("explanation", "")
        assert "Traffic" in explanation or "traffic" in explanation

    def test_anomaly_context_includes_traffic_info(self, evaluator: SLOEvaluator):
        """Test that traffic anomalies get SLO context."""
        result = evaluator.evaluate_result({
            "service": "search",
            "overall_severity": "medium",
            "time_period": "business_hours",
            "metrics": {
                "server_latency_avg": 100.0,
                "error_rate": 0.001,
                "request_rate": 250.0,
                "request_rate_mean": 100.0,
            },
            "anomalies": {"traffic_surge": {"severity": "medium"}},
        })

        anomalies = result.get("anomalies", {})
        traffic_anomaly = anomalies.get("traffic_surge", {})
        slo_context = traffic_anomaly.get("slo_context", {})

        assert slo_context.get("current_value_rps") == 250.0
        assert slo_context.get("baseline_mean_rps") == 100.0
        assert slo_context.get("ratio") == 2.5
        assert slo_context.get("is_surge") is True
        assert slo_context.get("is_cliff") is False

    def test_booking_service_cliff_stricter(self, evaluator: SLOEvaluator):
        """Test that booking service has stricter cliff settings."""
        # According to config.json, booking has standalone_severity=high for cliff
        result = evaluator.evaluate_result({
            "service": "booking",
            "overall_severity": "medium",
            "time_period": "night_hours",  # Off-peak
            "metrics": {
                "server_latency_avg": 100.0,
                "error_rate": 0.001,
                "request_rate": 30.0,
                "request_rate_mean": 100.0,
            },
            "anomalies": {"traffic_cliff": {"severity": "medium"}},
        })

        rr_eval = result.get("slo_evaluation", {}).get("request_rate_evaluation", {})

        # Booking has stricter cliff settings (standalone=high vs default warning)
        assert rr_eval.get("type") == "cliff"
        assert rr_eval.get("severity") == "high"

    def test_time_period_affects_min_expected(self, evaluator: SLOEvaluator):
        """Test that different time periods have different minimum expected traffic."""
        # Night hours should have lower minimum expected (0.5 rps default)
        result = evaluator.evaluate_result({
            "service": "titan",
            "overall_severity": "none",
            "time_period": "night_hours",
            "metrics": {
                "server_latency_avg": 100.0,
                "error_rate": 0.001,
                "request_rate": 0.3,           # Below night minimum but that's OK
                "request_rate_mean": 0.4,      # Not a cliff
            },
            "anomalies": {},
        })

        rr_eval = result.get("slo_evaluation", {}).get("request_rate_evaluation", {})

        # Night hours has min_expected=0.5, so 0.3 is below minimum
        assert rr_eval.get("min_expected_rps") == 0.5
        assert rr_eval.get("type") == "low_traffic"


class TestSLOSeverityPropagation:
    """Tests for SLO severity propagation to API payload.

    These tests verify that when SLO evaluation adjusts severity,
    the adjustment is properly reflected in the root overall_severity
    field of the API payload.
    """

    def test_slo_adjusted_severity_in_result(self, evaluator: SLOEvaluator):
        """Test that SLO evaluator updates result's overall_severity."""
        # Original severity is critical, but metrics are within SLO
        result = evaluator.evaluate_result({
            "service": "booking",
            "overall_severity": "critical",
            "time_period": "evening_hours",
            "metrics": {
                "server_latency_avg": 134.0,   # Within acceptable (< 300ms)
                "error_rate": 0.0011,          # Within acceptable (< 0.2%)
                "request_rate": 8.9,
                "request_rate_mean": 31.25,
            },
            "anomalies": {
                "partial_outage": {
                    "severity": "critical",
                    "root_metric": "error_rate",
                    "value": 0.0011,
                }
            },
        })

        # The result's overall_severity should be updated to low (SLO ok = low)
        assert result.get("overall_severity") == "low"
        assert result.get("slo_evaluation", {}).get("severity_changed") is True
        assert result.get("slo_evaluation", {}).get("original_severity") == "critical"
        assert result.get("slo_evaluation", {}).get("adjusted_severity") == "low"

    def test_slo_severity_not_changed_when_slo_breached(self, evaluator: SLOEvaluator):
        """Test that severity is NOT adjusted when SLO is breached."""
        result = evaluator.evaluate_result({
            "service": "booking",
            "overall_severity": "critical",
            "time_period": "business_hours",
            "metrics": {
                "server_latency_avg": 600.0,   # Above critical (500ms)
                "error_rate": 0.015,           # Above warning (1%)
                "request_rate": 50.0,
                "request_rate_mean": 50.0,
            },
            "anomalies": {
                "latency_spike": {
                    "severity": "critical",
                    "root_metric": "application_latency",
                }
            },
        })

        # Severity should NOT be adjusted - SLO is breached
        assert result.get("overall_severity") == "critical"
        assert result.get("slo_evaluation", {}).get("severity_changed") is False

    def test_severity_downgrade_from_high_to_low(self, evaluator: SLOEvaluator):
        """Test severity can be downgraded from high to low when SLO is ok."""
        result = evaluator.evaluate_result({
            "service": "search",
            "overall_severity": "high",
            "time_period": "business_hours",
            "metrics": {
                "server_latency_avg": 50.0,    # Well within acceptable
                "error_rate": 0.0001,          # Minimal errors
                "request_rate": 100.0,
                "request_rate_mean": 100.0,
            },
            "anomalies": {
                "statistical_anomaly": {
                    "severity": "high",
                    "type": "ml_isolation",
                }
            },
        })

        # Should be downgraded to low since SLO is ok (all metrics within acceptable)
        assert result.get("overall_severity") == "low"
        assert result.get("slo_evaluation", {}).get("severity_changed") is True
        assert result.get("slo_evaluation", {}).get("original_severity") == "high"
        assert result.get("slo_evaluation", {}).get("adjusted_severity") == "low"
