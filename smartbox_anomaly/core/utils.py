"""
Utility functions and helpers for the ML anomaly detection pipeline.

This module provides common operations used across multiple components,
reducing code duplication and centralizing shared logic.
"""

from __future__ import annotations

import hashlib
import math
import uuid
from dataclasses import dataclass
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
    except (ValueError, TypeError, AttributeError, OverflowError):
        # ValueError: invalid ISO format
        # TypeError: None or wrong type passed
        # AttributeError: object doesn't have expected methods
        # OverflowError: datetime calculation overflow
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

    Naming priority:
    1. `_anomaly_key`: Preserved name from detector (added during normalization)
    2. `pattern_name`: Named pattern from pattern matching
    3. `root_metric` + direction: For univariate ML detection
    4. `comparison_data`: Find most anomalous metric as fallback
    5. Description-based inference: Last resort heuristics

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
        direction = anomaly_data.get("direction", "anomaly")

        # 1. Check for preserved anomaly key (highest priority)
        # This is set by _normalize_anomalies() when converting dict to list
        anomaly_key = anomaly_data.get("_anomaly_key", "")
        if anomaly_key and not anomaly_key.startswith(("anomaly_", "metric_")):
            # Valid preserved name - use it directly
            return str(anomaly_key)

        # 2. Named patterns get their pattern name (most descriptive)
        if anomaly_data.get("pattern_name"):
            return str(anomaly_data["pattern_name"])

        # 3. Univariate ML detection: use metric + direction
        if anomaly_type == "ml_isolation":
            # Check for root_metric first (most accurate source)
            root_metric = anomaly_data.get("root_metric", "").lower()
            if root_metric:
                metric_name = _metric_to_name(root_metric)
                if metric_name:
                    return f"{metric_name}_{direction}"

            # 4. Fallback to comparison_data to find most anomalous metric
            comparison_data = anomaly_data.get("comparison_data", {})
            if comparison_data:
                most_anomalous = _find_most_anomalous_metric(comparison_data)
                if most_anomalous:
                    return f"{most_anomalous}_{direction}"

            # 5. Last resort: description-based inference
            description = anomaly_data.get("description", "").lower()
            metric_name = _infer_metric_from_description(description)
            if metric_name:
                return f"{metric_name}_{direction}"

            return f"metric_{direction}"

        # Consolidated anomaly type
        if anomaly_type == "consolidated":
            # Check root_metric
            root_metric = anomaly_data.get("root_metric", "").lower()
            if root_metric:
                metric_name = _metric_to_name(root_metric)
                if metric_name:
                    return f"{metric_name}_{direction}"

            # Check comparison_data
            comparison_data = anomaly_data.get("comparison_data", {})
            if comparison_data:
                most_anomalous = _find_most_anomalous_metric(comparison_data)
                if most_anomalous:
                    return f"{most_anomalous}_{direction}"

            return f"consolidated_{direction}"

        # Threshold detection (zero-normal metrics)
        if anomaly_type == "threshold":
            # Prefer direction if available, otherwise use detection_method
            if direction and direction != "anomaly":
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
    except (KeyError, TypeError, AttributeError):
        # KeyError: missing dictionary key
        # TypeError: None value or wrong type
        # AttributeError: object doesn't have expected attribute
        return f"anomaly_{index}_unknown"


def _metric_to_name(metric: str) -> str | None:
    """Convert a metric identifier to a standard name.

    Args:
        metric: Metric identifier (e.g., "dependency_latency", "error_rate").

    Returns:
        Standard metric name or None if not recognized.
    """
    metric = metric.lower()
    if "dependency" in metric:
        return "dependency_latency"
    elif "database" in metric or metric == "db_latency":
        return "database_latency"
    elif "application" in metric or metric == "app_latency":
        return "application_latency"
    elif "error" in metric:
        return "error_rate"
    elif "request" in metric or "traffic" in metric:
        return "request_rate"
    return None


def _infer_metric_from_description(description: str) -> str | None:
    """Infer metric name from anomaly description.

    Args:
        description: Anomaly description text (lowercase).

    Returns:
        Inferred metric name or None.
    """
    # Dependency latency: "External dependency slow"
    if "dependency" in description or "external" in description:
        return "dependency_latency"
    # Database latency
    elif "database" in description or "db " in description:
        return "database_latency"
    # Application latency: "fast responses", "response time", general "latency"
    elif "latency" in description or "response" in description:
        return "application_latency"
    elif "error" in description:
        return "error_rate"
    elif "traffic" in description or "request" in description:
        return "request_rate"
    return None


def _find_most_anomalous_metric(comparison_data: dict[str, Any]) -> str | None:
    """Find the most anomalous metric from comparison data.

    Uses deviation_sigma (standard deviations from mean) to determine
    which metric has the most significant deviation.

    Args:
        comparison_data: Dictionary of metric comparisons with deviation_sigma.

    Returns:
        Name of most anomalous metric, or None if no significant deviation.
    """
    if not comparison_data:
        return None

    max_deviation = 0.0
    most_anomalous = None

    for metric_name, stats in comparison_data.items():
        if not isinstance(stats, dict):
            continue

        deviation = abs(stats.get("deviation_sigma", 0.0))
        if deviation > max_deviation:
            max_deviation = deviation
            most_anomalous = metric_name

    # Only return if there's a significant deviation (> 1 sigma)
    if max_deviation > 1.0 and most_anomalous:
        return _metric_to_name(most_anomalous) or most_anomalous

    return None


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


# =============================================================================
# Metric Validation Utilities
# =============================================================================


@dataclass
class ValidationResult:
    """Result of metric validation."""

    value: float
    is_valid: bool
    was_corrected: bool
    original_value: float | None
    warning: str | None


def validate_numeric(
    value: float | int | None,
    *,
    default: float = 0.0,
    min_value: float | None = None,
    max_value: float | None = None,
    field_name: str = "value",
) -> ValidationResult:
    """Validate and sanitize a numeric value.

    Handles NaN, Inf, None, and out-of-bounds values.

    Args:
        value: The value to validate.
        default: Default value for invalid inputs.
        min_value: Minimum allowed value (None = no minimum).
        max_value: Maximum allowed value (None = no maximum).
        field_name: Name of the field for warning messages.

    Returns:
        ValidationResult with sanitized value and metadata.
    """
    original = value
    warning = None
    was_corrected = False

    # Handle None
    if value is None:
        return ValidationResult(
            value=default,
            is_valid=False,
            was_corrected=True,
            original_value=None,
            warning=f"{field_name}: None value, using default {default}",
        )

    # Handle NaN and Inf
    if math.isnan(value) or math.isinf(value):
        return ValidationResult(
            value=default,
            is_valid=False,
            was_corrected=True,
            original_value=float(value) if not math.isnan(value) else None,
            warning=f"{field_name}: {'NaN' if math.isnan(value) else 'Inf'} value, using default {default}",
        )

    result = float(value)

    # Apply bounds
    if min_value is not None and result < min_value:
        warning = f"{field_name}: value {result} < {min_value}, capping at {min_value}"
        result = min_value
        was_corrected = True

    if max_value is not None and result > max_value:
        warning = f"{field_name}: value {result} > {max_value}, capping at {max_value}"
        result = max_value
        was_corrected = True

    return ValidationResult(
        value=result,
        is_valid=not was_corrected,
        was_corrected=was_corrected,
        original_value=original if was_corrected else None,
        warning=warning,
    )


def validate_rate(
    value: float | int | None,
    *,
    field_name: str = "rate",
    max_rate: float = 1_000_000.0,
) -> ValidationResult:
    """Validate a rate metric (requests/second, etc.).

    Rates must be non-negative and below a reasonable maximum.

    Args:
        value: The rate value.
        field_name: Name of the field for warnings.
        max_rate: Maximum allowed rate (default 1M/s).

    Returns:
        ValidationResult with sanitized rate.
    """
    return validate_numeric(
        value,
        default=0.0,
        min_value=0.0,
        max_value=max_rate,
        field_name=field_name,
    )


def validate_latency(
    value: float | int | None,
    *,
    field_name: str = "latency",
    max_latency_ms: float = 300_000.0,  # 5 minutes
) -> ValidationResult:
    """Validate a latency metric in milliseconds.

    Latencies must be non-negative and below a reasonable maximum.

    Args:
        value: The latency value in ms.
        field_name: Name of the field for warnings.
        max_latency_ms: Maximum allowed latency (default 5 minutes).

    Returns:
        ValidationResult with sanitized latency.
    """
    return validate_numeric(
        value,
        default=0.0,
        min_value=0.0,
        max_value=max_latency_ms,
        field_name=field_name,
    )


def validate_ratio(
    value: float | int | None,
    *,
    field_name: str = "ratio",
    min_ratio: float = 0.0,
    max_ratio: float = 1.0,
) -> ValidationResult:
    """Validate a ratio/percentage value (0.0 to 1.0).

    Args:
        value: The ratio value.
        field_name: Name of the field for warnings.
        min_ratio: Minimum ratio (default 0.0).
        max_ratio: Maximum ratio (default 1.0).

    Returns:
        ValidationResult with sanitized ratio.
    """
    return validate_numeric(
        value,
        default=0.0,
        min_value=min_ratio,
        max_value=max_ratio,
        field_name=field_name,
    )


def validate_error_rate(
    value: float | int | None,
    *,
    field_name: str = "error_rate",
) -> ValidationResult:
    """Validate an error rate metric (0.0 to 1.0).

    Args:
        value: The error rate value.
        field_name: Name of the field for warnings.

    Returns:
        ValidationResult with sanitized error rate.
    """
    return validate_ratio(value, field_name=field_name)


@dataclass
class MetricsValidationResult:
    """Result of validating a complete metrics set."""

    metrics: dict[str, float]
    warnings: list[str]
    has_warnings: bool


def validate_inference_metrics(
    metrics: dict[str, float | int | None],
) -> MetricsValidationResult:
    """Validate a complete set of inference metrics.

    Applies appropriate validation for each metric type:
    - request_rate: non-negative rate
    - *_latency: non-negative latency
    - error_rate: ratio 0.0-1.0

    Args:
        metrics: Dictionary of metric name to value.

    Returns:
        MetricsValidationResult with sanitized metrics and any warnings.
    """
    validated: dict[str, float] = {}
    warnings: list[str] = []

    for name, value in metrics.items():
        if "latency" in name.lower():
            result = validate_latency(value, field_name=name)
        elif "error" in name.lower() and "rate" in name.lower():
            result = validate_error_rate(value, field_name=name)
        elif "rate" in name.lower():
            result = validate_rate(value, field_name=name)
        else:
            # Generic numeric validation
            result = validate_numeric(value, field_name=name, min_value=0.0)

        validated[name] = result.value
        if result.warning:
            warnings.append(result.warning)

    return MetricsValidationResult(
        metrics=validated,
        warnings=warnings,
        has_warnings=len(warnings) > 0,
    )


def safe_divide(
    numerator: float,
    denominator: float,
    *,
    default: float = 0.0,
) -> float:
    """Safely divide two numbers, handling zero division.

    Args:
        numerator: The numerator.
        denominator: The denominator.
        default: Default value if denominator is zero.

    Returns:
        Division result or default.
    """
    if denominator == 0 or math.isnan(denominator) or math.isinf(denominator):
        return default
    result = numerator / denominator
    if math.isnan(result) or math.isinf(result):
        return default
    return result


def clamp(value: float, min_value: float, max_value: float) -> float:
    """Clamp a value between min and max bounds.

    Args:
        value: The value to clamp.
        min_value: Minimum bound.
        max_value: Maximum bound.

    Returns:
        Clamped value.
    """
    return max(min_value, min(max_value, value))
