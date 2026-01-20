"""
Service-specific configuration for anomaly detection.

This module provides service-aware parameter tuning based on
service category, traffic patterns, and operational requirements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from smartbox_anomaly.core.config import DEFAULT_PERIOD_CONTAMINATION_MULTIPLIERS
from smartbox_anomaly.core.logging import get_logger

if TYPE_CHECKING:
    import pandas as pd

logger = get_logger(__name__)

ServiceCategory = Literal[
    "critical", "standard", "micro", "admin", "core",
    "api_gateway", "background_service", "data_service",
    "security_service", "observability", "legacy_or_internal",
    "development", "unknown_standard"
]

Complexity = Literal["high", "medium", "low"]


@dataclass
class ServiceParameters:
    """Optimal parameters for a service based on its characteristics."""

    base_contamination: float
    complexity: Complexity
    category: ServiceCategory
    n_estimators: int
    max_samples: float | str
    bootstrap: bool
    adjustment_reason: str
    auto_detected: bool

    def to_isolation_forest_params(self) -> dict[str, Any]:
        """Get parameters for IsolationForest constructor."""
        return {
            "contamination": self.base_contamination,
            "n_estimators": self.n_estimators,
            "max_samples": self.max_samples,
            "bootstrap": self.bootstrap,
            "random_state": 42,
            "n_jobs": -1,
        }


# Default expected metrics (minimum all services should have)
DEFAULT_EXPECTED_METRICS: list[str] = [
    "request_rate",
    "application_latency",
    "error_rate",
]

# Known service configurations
# expected_metrics: metrics this service is expected to have (affects quality scoring)
# - Services with dependencies should have dependency_latency
# - Services with databases should have database_latency
# - Origin services (no downstream calls) may not have dependency_latency
KNOWN_SERVICE_PARAMS: dict[str, dict[str, Any]] = {
    # High-traffic, critical services
    "booking": {
        "base_contamination": 0.04,
        "complexity": "high",
        "category": "critical",
        "expected_metrics": ["request_rate", "application_latency", "dependency_latency", "error_rate"],
    },
    "search": {
        "base_contamination": 0.04,
        "complexity": "high",
        "category": "critical",
        "expected_metrics": ["request_rate", "application_latency", "dependency_latency", "error_rate"],
    },
    "mobile-api": {
        "base_contamination": 0.03,
        "complexity": "high",
        "category": "critical",
        "expected_metrics": ["request_rate", "application_latency", "dependency_latency", "error_rate"],
    },
    "shire-api": {
        "base_contamination": 0.03,
        "complexity": "high",
        "category": "critical",
        "expected_metrics": ["request_rate", "application_latency", "dependency_latency", "database_latency", "error_rate"],
    },
    # Medium-traffic services
    "vms": {
        "base_contamination": 0.05,
        "complexity": "medium",
        "category": "critical",
        "expected_metrics": ["request_rate", "application_latency", "database_latency", "error_rate"],
    },
    "friday": {
        "base_contamination": 0.05,
        "complexity": "medium",
        "category": "standard",
        "expected_metrics": ["request_rate", "application_latency", "dependency_latency", "error_rate"],
    },
    "gambit": {
        "base_contamination": 0.05,
        "complexity": "medium",
        "category": "standard",
        "expected_metrics": ["request_rate", "application_latency", "dependency_latency", "error_rate"],
    },
    "titan": {
        "base_contamination": 0.05,
        "complexity": "medium",
        "category": "standard",
        "expected_metrics": ["request_rate", "application_latency", "dependency_latency", "error_rate"],
    },
    "r2d2": {
        "base_contamination": 0.05,
        "complexity": "medium",
        "category": "standard",
        "expected_metrics": ["request_rate", "application_latency", "dependency_latency", "database_latency", "error_rate"],
    },
    "catalog": {
        "base_contamination": 0.05,
        "complexity": "medium",
        "category": "standard",
        "expected_metrics": ["request_rate", "application_latency", "dependency_latency", "error_rate"],
    },
    "tc14": {
        "base_contamination": 0.05,
        "complexity": "medium",
        "category": "standard",
        "expected_metrics": ["request_rate", "application_latency", "dependency_latency", "error_rate"],
    },
    # Micro-services
    "fa5": {
        "base_contamination": 0.08,
        "complexity": "low",
        "category": "micro",
        "expected_metrics": ["request_rate", "application_latency", "dependency_latency", "database_latency", "error_rate"],
    },
    # Admin services (have both dependency and database access)
    "m2-fr-adm": {
        "base_contamination": 0.06,
        "complexity": "low",
        "category": "admin",
        "expected_metrics": ["request_rate", "application_latency", "dependency_latency", "database_latency", "error_rate"],
    },
    "m2-it-adm": {
        "base_contamination": 0.06,
        "complexity": "low",
        "category": "admin",
        "expected_metrics": ["request_rate", "application_latency", "dependency_latency", "database_latency", "error_rate"],
    },
    "m2-bb-adm": {
        "base_contamination": 0.06,
        "complexity": "low",
        "category": "admin",
        "expected_metrics": ["request_rate", "application_latency", "dependency_latency", "database_latency", "error_rate"],
    },
    "m2-bn-adm": {
        "base_contamination": 0.06,
        "complexity": "low",
        "category": "admin",
        "expected_metrics": ["request_rate", "application_latency", "dependency_latency", "database_latency", "error_rate"],
    },
    "m2-cb-adm": {
        "base_contamination": 0.06,
        "complexity": "low",
        "category": "admin",
        "expected_metrics": ["request_rate", "application_latency", "dependency_latency", "database_latency", "error_rate"],
    },
    "m2-ch-adm": {
        "base_contamination": 0.06,
        "complexity": "low",
        "category": "admin",
        "expected_metrics": ["request_rate", "application_latency", "dependency_latency", "database_latency", "error_rate"],
    },
    "m2-es-adm": {
        "base_contamination": 0.06,
        "complexity": "low",
        "category": "admin",
        "expected_metrics": ["request_rate", "application_latency", "dependency_latency", "database_latency", "error_rate"],
    },
    "m2-ez-adm": {
        "base_contamination": 0.06,
        "complexity": "low",
        "category": "admin",
        "expected_metrics": ["request_rate", "application_latency", "dependency_latency", "database_latency", "error_rate"],
    },
    "m2-lv-adm": {
        "base_contamination": 0.06,
        "complexity": "low",
        "category": "admin",
        "expected_metrics": ["request_rate", "application_latency", "dependency_latency", "database_latency", "error_rate"],
    },
    "m2-df-adm": {
        "base_contamination": 0.06,
        "complexity": "low",
        "category": "admin",
        "expected_metrics": ["request_rate", "application_latency", "dependency_latency", "database_latency", "error_rate"],
    },
    "m2-dk-adm": {
        "base_contamination": 0.06,
        "complexity": "low",
        "category": "admin",
        "expected_metrics": ["request_rate", "application_latency", "dependency_latency", "database_latency", "error_rate"],
    },
    "m2-se-adm": {
        "base_contamination": 0.06,
        "complexity": "low",
        "category": "admin",
        "expected_metrics": ["request_rate", "application_latency", "dependency_latency", "database_latency", "error_rate"],
    },
    # Core services (have both dependency and database access)
    "m2-bb": {
        "base_contamination": 0.04,
        "complexity": "medium",
        "category": "core",
        "expected_metrics": ["request_rate", "application_latency", "dependency_latency", "database_latency", "error_rate"],
    },
    "m2-bn": {
        "base_contamination": 0.04,
        "complexity": "medium",
        "category": "core",
        "expected_metrics": ["request_rate", "application_latency", "dependency_latency", "database_latency", "error_rate"],
    },
    "m2-cb": {
        "base_contamination": 0.04,
        "complexity": "medium",
        "category": "core",
        "expected_metrics": ["request_rate", "application_latency", "dependency_latency", "database_latency", "error_rate"],
    },
    "m2-ch": {
        "base_contamination": 0.04,
        "complexity": "medium",
        "category": "core",
        "expected_metrics": ["request_rate", "application_latency", "dependency_latency", "database_latency", "error_rate"],
    },
    "m2-ez": {
        "base_contamination": 0.04,
        "complexity": "medium",
        "category": "core",
        "expected_metrics": ["request_rate", "application_latency", "dependency_latency", "database_latency", "error_rate"],
    },
    "m2-fr": {
        "base_contamination": 0.04,
        "complexity": "medium",
        "category": "core",
        "expected_metrics": ["request_rate", "application_latency", "dependency_latency", "database_latency", "error_rate"],
    },
    "m2-it": {
        "base_contamination": 0.04,
        "complexity": "medium",
        "category": "core",
        "expected_metrics": ["request_rate", "application_latency", "dependency_latency", "database_latency", "error_rate"],
    },
    "m2-es": {
        "base_contamination": 0.04,
        "complexity": "medium",
        "category": "core",
        "expected_metrics": ["request_rate", "application_latency", "dependency_latency", "database_latency", "error_rate"],
    },
    "m2-df": {
        "base_contamination": 0.04,
        "complexity": "medium",
        "category": "core",
        "expected_metrics": ["request_rate", "application_latency", "dependency_latency", "database_latency", "error_rate"],
    },
    "m2-dk": {
        "base_contamination": 0.04,
        "complexity": "medium",
        "category": "core",
        "expected_metrics": ["request_rate", "application_latency", "dependency_latency", "database_latency", "error_rate"],
    },
    "m2-ds": {
        "base_contamination": 0.04,
        "complexity": "medium",
        "category": "core",
        "expected_metrics": ["request_rate", "application_latency", "dependency_latency", "database_latency", "error_rate"],
    },
    "m2-se": {
        "base_contamination": 0.04,
        "complexity": "medium",
        "category": "core",
        "expected_metrics": ["request_rate", "application_latency", "dependency_latency", "database_latency", "error_rate"],
    },
    "m2-lv": {
        "base_contamination": 0.04,
        "complexity": "medium",
        "category": "core",
        "expected_metrics": ["request_rate", "application_latency", "dependency_latency", "database_latency", "error_rate"],
    },
}


def get_expected_metrics(service_name: str) -> list[str]:
    """Get expected metrics for a service for quality scoring.

    Services are only penalized for missing metrics that they are expected to have.
    Origin services (no downstream dependencies) are not expected to have dependency_latency.
    Services without databases are not expected to have database_latency.

    Args:
        service_name: Name of the service.

    Returns:
        List of metric names this service is expected to have.
    """
    if service_name in KNOWN_SERVICE_PARAMS:
        config = KNOWN_SERVICE_PARAMS[service_name]
        if "expected_metrics" in config:
            return config["expected_metrics"]

    # Fall back to default expected metrics for unknown services
    return DEFAULT_EXPECTED_METRICS.copy()


def detect_service_category(service_name: str) -> dict[str, Any]:
    """Detect service category from naming patterns.

    Args:
        service_name: Name of the service.

    Returns:
        Configuration dictionary with category, complexity, and contamination.
    """
    service_lower = service_name.lower()

    patterns = [
        (["api", "gateway", "proxy"], {"base_contamination": 0.03, "complexity": "high", "category": "api_gateway"}),
        (["admin", "adm", "management", "mgmt"], {"base_contamination": 0.06, "complexity": "low", "category": "admin"}),
        (["worker", "job", "task", "queue", "processor"], {"base_contamination": 0.08, "complexity": "low", "category": "background_service"}),
        (["micro", "util", "helper", "tool"], {"base_contamination": 0.07, "complexity": "low", "category": "micro"}),
        (["db", "database", "storage", "cache"], {"base_contamination": 0.04, "complexity": "medium", "category": "data_service"}),
        (["auth", "login", "security", "oauth"], {"base_contamination": 0.02, "complexity": "high", "category": "security_service"}),
        (["monitor", "metric", "log", "trace"], {"base_contamination": 0.05, "complexity": "medium", "category": "observability"}),
        (["test", "staging", "dev"], {"base_contamination": 0.10, "complexity": "low", "category": "development"}),
    ]

    for keywords, config in patterns:
        if any(kw in service_lower for kw in keywords):
            return config

    # Short names like 'fa5', 'r2d2'
    if len(service_lower) <= 3:
        return {"base_contamination": 0.07, "complexity": "low", "category": "legacy_or_internal"}

    # Conservative default
    return {"base_contamination": 0.05, "complexity": "medium", "category": "unknown_standard"}


def get_service_parameters(
    service_name: str,
    data: pd.DataFrame | None = None,
    auto_tune: bool = True,
    estimated_contamination: float | None = None,
    period: str | None = None,
) -> ServiceParameters:
    """Get optimal parameters for a service.

    Args:
        service_name: Name of the service.
        data: Optional training data for data-driven adjustments.
        auto_tune: Whether to enable automatic parameter tuning.
        estimated_contamination: Optional data-driven contamination estimate from
                                knee/gap detection. If provided, this takes precedence
                                over variability-based adjustments.
        period: Optional time period name (e.g., "business_hours", "night_hours").
               When provided, contamination is multiplied by period-specific factor
               to account for higher variance during low-traffic periods.

    Returns:
        ServiceParameters with optimal configuration.
    """
    # Check known services first
    if service_name in KNOWN_SERVICE_PARAMS:
        base_config = KNOWN_SERVICE_PARAMS[service_name]
        auto_detected = False
        logger.debug(f"Using known configuration for {service_name} ({base_config['category']})")
    else:
        base_config = detect_service_category(service_name)
        auto_detected = True
        logger.debug(f"Auto-detected pattern for {service_name}: {base_config['category']}")

    base_contamination = base_config["base_contamination"]
    complexity = base_config["complexity"]
    category = base_config["category"]
    adjustment_reason = "default"

    # Use estimated contamination if provided (data-driven, takes precedence)
    if estimated_contamination is not None and auto_tune:
        # Blend estimated contamination with category bounds for safety
        # Don't let estimated go below category minimum or above category maximum
        category_min = base_contamination * 0.5  # 50% of base as floor
        category_max = min(0.15, base_contamination * 3.0)  # 3x base or 15% as ceiling

        contamination = max(category_min, min(category_max, estimated_contamination))
        adjustment_reason = f"estimated_knee={estimated_contamination:.3f}, bounded to [{category_min:.3f}, {category_max:.3f}]"
        logger.debug(
            f"{service_name}: Using estimated contamination {estimated_contamination:.3f} "
            f"→ bounded to {contamination:.3f}"
        )
    # Fall back to variability-based adjustment
    elif data is not None and auto_tune and len(data) > 0:
        data_size = len(data)
        variability = data.std().mean()

        # Adjust contamination based on variability
        if variability > 3.0:
            contamination = min(0.20, base_contamination * 2.0)
            adjustment_reason = "high variability"
        elif variability > 2.0:
            contamination = min(0.15, base_contamination * 1.5)
            adjustment_reason = "moderate variability"
        elif variability < 0.5:
            contamination = max(0.01, base_contamination * 0.7)
            adjustment_reason = "low variability"
        else:
            contamination = base_contamination
            adjustment_reason = "normal variability"
    else:
        contamination = base_contamination

    # Data-driven n_estimators and other params
    if data is not None and auto_tune and len(data) > 0:
        data_size = len(data)

        # Calculate n_estimators based on complexity and data size
        n_estimators = _calculate_n_estimators(complexity, data_size)

        # Adjust max_samples for large datasets
        max_samples: float | str = min(1.0, 4000 / data_size) if data_size > 5000 else "auto"

        # Use bootstrap for large datasets
        bootstrap = data_size > 10000
    else:
        data_size = 0
        n_estimators = 200
        max_samples = "auto"
        bootstrap = False

    # Apply period-specific contamination multiplier for low-traffic periods
    # This reduces false positives during night/weekend when variance is naturally higher
    if period is not None:
        period_multiplier = DEFAULT_PERIOD_CONTAMINATION_MULTIPLIERS.get(period, 1.0)
        if period_multiplier != 1.0:
            original_contamination = contamination
            contamination = min(0.25, contamination * period_multiplier)  # Cap at 25%
            adjustment_reason = f"{adjustment_reason}, period={period} (x{period_multiplier:.1f})"
            logger.debug(
                f"{service_name}: Period {period} multiplier {period_multiplier:.1f}x "
                f"adjusted contamination {original_contamination:.3f} → {contamination:.3f}"
            )

    return ServiceParameters(
        base_contamination=round(contamination, 3),
        complexity=complexity,
        category=category,
        n_estimators=n_estimators,
        max_samples=max_samples,
        bootstrap=bootstrap,
        adjustment_reason=adjustment_reason,
        auto_detected=auto_detected,
    )


def _calculate_n_estimators(complexity: Complexity, data_size: int) -> int:
    """Calculate optimal n_estimators based on complexity and data size."""
    if complexity == "high":
        if data_size > 10000:
            return 400
        if data_size > 5000:
            return 300
        return 250
    elif complexity == "medium":
        if data_size > 8000:
            return 250
        if data_size > 3000:
            return 200
        return 150
    else:  # low
        if data_size > 5000:
            return 150
        return 100


def get_validation_thresholds(service_name: str) -> dict[str, float]:
    """Get service-specific validation thresholds for time periods.

    Uses 3 time-of-day buckets (all 7 days combined). This provides more
    training data per period, leading to more stable models and more
    meaningful validation thresholds.

    Args:
        service_name: Name of the service.

    Returns:
        Dictionary mapping time period to validation threshold.
    """
    service_lower = service_name.lower()

    # With 3-period model (all 7 days combined), we can use stricter thresholds
    # since we have more training data per period
    if any(p in service_lower for p in ["booking", "search", "mobile-api", "shire-api"]):
        # Critical services: strictest thresholds
        return {
            "business_hours": 0.20,
            "evening_hours": 0.25,
            "night_hours": 0.35,
        }
    elif any(p in service_lower for p in ["adm", "admin", "management"]):
        # Admin services: moderate thresholds
        return {
            "business_hours": 0.25,
            "evening_hours": 0.30,
            "night_hours": 0.40,
        }
    elif any(p in service_lower for p in ["fa5", "micro", "internal", "util", "worker", "job", "task"]):
        # Micro/worker services: most lenient due to naturally high variance
        return {
            "business_hours": 0.30,
            "evening_hours": 0.40,
            "night_hours": 0.50,
        }
    elif any(p in service_lower for p in ["m2-", "core", "platform"]):
        # Core platform services: balanced thresholds
        return {
            "business_hours": 0.22,
            "evening_hours": 0.28,
            "night_hours": 0.38,
        }
    else:
        # Default thresholds
        return {
            "business_hours": 0.22,
            "evening_hours": 0.28,
            "night_hours": 0.40,
        }


def get_min_samples_for_period(service_name: str, period: str) -> int:
    """Get minimum training samples required for a time period.

    With 3-period model (all 7 days combined), we have more data per period,
    so we can use consistent minimum requirements across all periods.

    Args:
        service_name: Name of the service.
        period: Time period name.

    Returns:
        Minimum number of samples required.
    """
    service_lower = service_name.lower()

    # Determine service type
    if any(p in service_lower for p in ["booking", "search", "mobile-api", "shire-api"]):
        service_type = "critical"
    elif any(p in service_lower for p in ["adm", "admin"]):
        service_type = "admin"
    elif any(p in service_lower for p in ["fa5", "micro", "util"]):
        service_type = "micro"
    else:
        service_type = "standard"

    # With 3-period model, all periods have similar data volume
    # Night hours may have slightly less data but still sufficient
    return {"micro": 100, "admin": 150, "critical": 200, "standard": 200}.get(service_type, 200)
