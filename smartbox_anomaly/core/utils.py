"""
Utility functions and helpers for the ML anomaly detection pipeline.

This module provides common operations used across multiple components,
reducing code duplication and centralizing shared logic.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from smartbox_anomaly.core.constants import TIME_PERIOD_SUFFIXES, TimePeriod

# =============================================================================
# ID Generation
# =============================================================================


def generate_fingerprint_id(service_name: str, anomaly_name: str) -> str:
    """Generate deterministic fingerprint ID for anomaly pattern.

    The fingerprint is model-agnostic to enable cross-temporal tracking.
    Same anomaly pattern detected by different models = same fingerprint.

    Args:
        service_name: Base service name (without time period suffix).
        anomaly_name: Name/type of the anomaly.

    Returns:
        Deterministic fingerprint ID string.
    """
    content = f"{service_name}_{anomaly_name}"
    hash_obj = hashlib.sha256(content.encode())
    return f"anomaly_{hash_obj.hexdigest()[:12]}"


def generate_incident_id() -> str:
    """Generate unique incident ID for a new occurrence.

    Returns:
        Unique incident ID string.
    """
    return f"incident_{uuid.uuid4().hex[:12]}"


def generate_correlation_id() -> str:
    """Generate unique correlation ID for request tracing.

    Returns:
        Unique correlation ID string.
    """
    return f"corr_{uuid.uuid4().hex[:16]}"


# =============================================================================
# Service Name Parsing
# =============================================================================


def parse_service_model(full_service_name: str) -> tuple[str, str]:
    """Parse full service name into service and model components.

    Examples:
        >>> parse_service_model("booking_evening_hours")
        ("booking", "evening_hours")
        >>> parse_service_model("fa5_business_hours")
        ("fa5", "business_hours")
        >>> parse_service_model("mobile-api_weekend_night")
        ("mobile-api", "weekend_night")

    Args:
        full_service_name: Full service name with potential time period suffix.

    Returns:
        Tuple of (base_service_name, time_period).
    """
    for suffix in TIME_PERIOD_SUFFIXES:
        if full_service_name.endswith(suffix):
            service_name = full_service_name[: -len(suffix)]
            period = suffix.lstrip("_")
            return service_name, period

    # Fallback for unexpected formats
    parts = full_service_name.rsplit("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]

    return full_service_name, "unknown"


def build_full_service_name(service_name: str, time_period: str | TimePeriod) -> str:
    """Build full service name from base name and time period.

    Args:
        service_name: Base service name.
        time_period: Time period (string or TimePeriod enum).

    Returns:
        Full service name with time period suffix.
    """
    period = time_period.value if isinstance(time_period, TimePeriod) else time_period
    return f"{service_name}_{period}"


def extract_base_service_names(service_names: list[str]) -> list[str]:
    """Extract unique base service names from a list of full names.

    Args:
        service_names: List of service names (may include time period suffixes).

    Returns:
        Sorted list of unique base service names.
    """
    base_names = set()
    for name in service_names:
        base_name, _ = parse_service_model(name)
        base_names.add(base_name)
    return sorted(base_names)


# =============================================================================
# Time Period Detection
# =============================================================================


def get_time_period(timestamp: datetime) -> TimePeriod:
    """Determine which time period a timestamp falls into.

    Args:
        timestamp: The datetime to classify.

    Returns:
        TimePeriod enum value.
    """
    hour = timestamp.hour
    day_of_week = timestamp.weekday()
    is_weekend = day_of_week >= 5

    if is_weekend:
        if 8 <= hour < 22:
            return TimePeriod.WEEKEND_DAY
        else:
            return TimePeriod.WEEKEND_NIGHT
    elif 8 <= hour < 18:
        return TimePeriod.BUSINESS_HOURS
    elif hour >= 22 or hour < 6:
        return TimePeriod.NIGHT_HOURS
    else:
        return TimePeriod.EVENING_HOURS


def get_period_type(period: str | TimePeriod) -> str:
    """Get the activity type classification for a time period.

    Args:
        period: Time period string or enum.

    Returns:
        Activity type description.
    """
    period_value = period.value if isinstance(period, TimePeriod) else period
    return {
        "business_hours": "peak_activity",
        "night_hours": "minimal_activity",
        "evening_hours": "transition_activity",
        "weekend_day": "weekend_moderate_activity",
        "weekend_night": "weekend_minimal_activity",
    }.get(period_value, "unknown_activity")


# =============================================================================
# Duration Calculations
# =============================================================================


def calculate_duration_minutes(
    start_time: str | datetime,
    end_time: str | datetime | None = None,
) -> int:
    """Calculate duration in minutes between two timestamps.

    Args:
        start_time: Start timestamp (datetime or ISO string).
        end_time: End timestamp (datetime, ISO string, or None for now).

    Returns:
        Duration in minutes (0 if calculation fails).
    """
    try:
        # Parse start time
        if isinstance(start_time, str):
            start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
        else:
            start_dt = start_time

        # Parse end time
        if end_time is None:
            end_dt = datetime.now()
        elif isinstance(end_time, str):
            end_dt = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
        else:
            end_dt = end_time

        # Remove timezone info for comparison
        start_dt = start_dt.replace(tzinfo=None)
        end_dt = end_dt.replace(tzinfo=None)

        duration = end_dt - start_dt
        return max(0, int(duration.total_seconds() / 60))
    except Exception:
        return 0


def format_duration(minutes: int) -> str:
    """Format duration in minutes to human-readable string.

    Args:
        minutes: Duration in minutes.

    Returns:
        Human-readable duration string.
    """
    if minutes < 60:
        return f"{minutes}m"
    elif minutes < 1440:  # Less than 24 hours
        hours = minutes // 60
        mins = minutes % 60
        return f"{hours}h {mins}m" if mins > 0 else f"{hours}h"
    else:
        days = minutes // 1440
        remaining = minutes % 1440
        hours = remaining // 60
        return f"{days}d {hours}h" if hours > 0 else f"{days}d"


# =============================================================================
# Anomaly Name Generation
# =============================================================================


def generate_anomaly_name(anomaly_data: dict[str, Any], index: int = 0) -> str:  # noqa: PLR0911
    """Generate consistent, descriptive anomaly name from anomaly data.

    Naming convention:
    - For univariate (single metric): "{metric}_{direction}" e.g., "latency_high"
    - For named patterns: "{pattern_name}" e.g., "recent_degradation"
    - For multivariate (unnamed): "multivariate_{detection_method}"
    - For correlations: "correlation_{type}" e.g., "correlation_database_bottleneck"

    Args:
        anomaly_data: Dictionary containing anomaly information.
        index: Fallback index for naming.

    Returns:
        Descriptive anomaly name.
    """
    try:
        anomaly_type = anomaly_data.get("type", "unknown")
        detection_method = anomaly_data.get("detection_method", "unknown")

        # Named patterns get their pattern name (most descriptive)
        if anomaly_data.get("pattern_name"):
            return str(anomaly_data["pattern_name"])

        # Univariate ML detection: use metric + direction
        if anomaly_type == "ml_isolation":
            # Try to extract metric name from the key or description
            direction = anomaly_data.get("direction", "anomaly")
            # Look for metric name in contributing_metrics or infer from description
            description = anomaly_data.get("description", "")
            if "latency" in description.lower():
                return f"latency_{direction}"
            elif "error" in description.lower():
                return f"error_rate_{direction}"
            elif "traffic" in description.lower() or "request" in description.lower():
                return f"request_rate_{direction}"
            return f"metric_{direction}"

        # Threshold detection (zero-normal metrics)
        if anomaly_type == "threshold":
            # Prefer direction if available, otherwise use detection_method
            direction = anomaly_data.get("direction")
            if direction:
                return f"threshold_{direction}"
            return f"threshold_{detection_method}"

        # Correlation detection: use the correlation type
        if anomaly_type == "correlation":
            # The key in anomalies dict often contains the correlation type
            return f"correlation_{detection_method}"

        # Multivariate without pattern name
        if anomaly_type in ("multivariate", "multivariate_pattern"):
            return f"multivariate_{detection_method}"

        # Pattern detection
        if anomaly_type == "pattern":
            return f"service_pattern_{index}"

        # Fallback
        return f"anomaly_{index}_{anomaly_type}"
    except Exception:
        return f"anomaly_{index}_unknown"


# =============================================================================
# Path Utilities
# =============================================================================


def ensure_directory(path: str | Path) -> Path:
    """Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path.

    Returns:
        Path object for the directory.
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_model_path(
    models_directory: str | Path,
    service_name: str,
    time_period: str | TimePeriod | None = None,
) -> Path:
    """Get the path for a service's model directory.

    Args:
        models_directory: Base models directory.
        service_name: Service name.
        time_period: Optional time period.

    Returns:
        Path to the model directory.
    """
    base_path = Path(models_directory)
    if time_period:
        period = time_period.value if isinstance(time_period, TimePeriod) else time_period
        return base_path / f"{service_name}_{period}"
    return base_path / service_name


def discover_available_models(models_directory: str | Path) -> dict[str, list[str]]:
    """Discover available models in the models directory.

    Args:
        models_directory: Path to models directory.

    Returns:
        Dictionary mapping base service names to available time periods.
    """
    models_path = Path(models_directory)
    if not models_path.exists():
        return {}

    available: dict[str, list[str]] = {}

    for model_dir in models_path.iterdir():
        if not model_dir.is_dir():
            continue

        # Check for model file
        model_file = model_dir / "model_data.json"
        if not model_file.exists():
            continue

        # Parse service and period
        base_name, period = parse_service_model(model_dir.name)

        if base_name not in available:
            available[base_name] = []
        if period != "unknown":
            available[base_name].append(period)

    return available


# =============================================================================
# Data Utilities
# =============================================================================


def safe_get_nested(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Safely get a nested value from a dictionary.

    Args:
        data: Source dictionary.
        *keys: Key path to traverse.
        default: Default value if path not found.

    Returns:
        Value at path or default.
    """
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


def merge_dicts(*dicts: dict[str, Any]) -> dict[str, Any]:
    """Merge multiple dictionaries, later values override earlier.

    Args:
        *dicts: Dictionaries to merge.

    Returns:
        Merged dictionary.
    """
    result = {}
    for d in dicts:
        if d:
            result.update(d)
    return result


def truncate_string(s: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate a string to a maximum length.

    Args:
        s: String to truncate.
        max_length: Maximum length.
        suffix: Suffix to add if truncated.

    Returns:
        Truncated string.
    """
    if len(s) <= max_length:
        return s
    return s[: max_length - len(suffix)] + suffix


# =============================================================================
# Timestamp Utilities
# =============================================================================


def now_iso() -> str:
    """Get current timestamp as ISO string.

    Returns:
        Current timestamp in ISO format.
    """
    return datetime.now().isoformat()


def parse_timestamp(value: str | datetime | None) -> datetime | None:
    """Parse a timestamp from various formats.

    Args:
        value: Timestamp string, datetime, or None.

    Returns:
        Parsed datetime or None.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def timestamp_age_minutes(timestamp: str | datetime) -> int:
    """Calculate the age of a timestamp in minutes.

    Args:
        timestamp: Timestamp to check.

    Returns:
        Age in minutes.
    """
    return calculate_duration_minutes(timestamp, datetime.now())
