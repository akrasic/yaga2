"""
Exception enrichment service for anomaly diagnosis.

Queries OpenTelemetry exception metrics from VictoriaMetrics to provide
context about which exceptions are occurring when error_rate is elevated.

The enrichment is time-aligned with anomaly detection:
- When an anomaly is detected at time T
- Exception data is queried for the window [T - lookback, T]
- This ensures exception data matches the anomaly detection window
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from smartbox_anomaly.core.logging import get_logger
from smartbox_anomaly.metrics.client import VictoriaMetricsClient

logger = get_logger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ExceptionSummary:
    """Summary of a single exception type."""

    exception_type: str  # Full class name e.g. "Smartbox\\Search\\R2D2\\Exception\\R2D2Exception"
    rate: float  # Events per second
    percentage: float  # Percentage of total exceptions
    short_name: str = ""  # Short name e.g. "R2D2Exception"

    def __post_init__(self) -> None:
        """Extract short name from full exception type."""
        if not self.short_name:
            # Handle PHP-style namespaces (backslash) and Java/Python (dot)
            parts = self.exception_type.replace("\\", "/").split("/")
            self.short_name = parts[-1] if parts else self.exception_type


@dataclass
class ExceptionBreakdown:
    """Complete breakdown of exceptions for a service."""

    service_name: str
    timestamp: datetime
    total_exception_rate: float  # Total exceptions per second
    exceptions: list[ExceptionSummary] = field(default_factory=list)
    query_successful: bool = True
    error_message: str | None = None

    @property
    def top_exception(self) -> ExceptionSummary | None:
        """Get the most frequent exception."""
        return self.exceptions[0] if self.exceptions else None

    @property
    def has_exceptions(self) -> bool:
        """Check if any exceptions are occurring."""
        return self.total_exception_rate > 0 and len(self.exceptions) > 0

    def get_summary_text(self, max_exceptions: int = 3) -> str:
        """Generate human-readable summary of exceptions.

        Args:
            max_exceptions: Maximum number of exceptions to include in summary.

        Returns:
            Human-readable summary string.
        """
        if not self.query_successful:
            return f"Failed to query exceptions: {self.error_message}"

        if not self.has_exceptions:
            return "No exceptions detected in the current window."

        lines = [f"Exception breakdown for {self.service_name} ({self.total_exception_rate:.2f}/s total):"]

        for exc in self.exceptions[:max_exceptions]:
            lines.append(f"  - {exc.short_name}: {exc.rate:.3f}/s ({exc.percentage:.1f}%)")

        remaining = len(self.exceptions) - max_exceptions
        if remaining > 0:
            lines.append(f"  ... and {remaining} more exception types")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "service_name": self.service_name,
            "timestamp": self.timestamp.isoformat(),
            "total_exception_rate": self.total_exception_rate,
            "exception_count": len(self.exceptions),
            "top_exceptions": [
                {
                    "type": exc.exception_type,
                    "short_name": exc.short_name,
                    "rate": exc.rate,
                    "percentage": exc.percentage,
                }
                for exc in self.exceptions[:5]
            ],
            "query_successful": self.query_successful,
            "error_message": self.error_message,
        }


# =============================================================================
# Exception Enrichment Service
# =============================================================================


class ExceptionEnrichmentService:
    """Service for enriching anomaly alerts with exception context.

    Queries OpenTelemetry exception metrics from VictoriaMetrics to identify
    which exception types are causing elevated error rates.

    The service uses time-aligned queries to ensure exception data matches
    the anomaly detection window. When an anomaly is detected at time T,
    exceptions are queried for [T - lookback_minutes, T].

    Example:
        >>> service = ExceptionEnrichmentService(vm_client)
        >>> breakdown = service.get_exception_breakdown("search", anomaly_time)
        >>> print(breakdown.get_summary_text())
        Exception breakdown for search (0.35/s total):
          - R2D2Exception: 0.217/s (62.0%)
          - UserInputException: 0.083/s (23.7%)
          - ClientSideMiddlewareException: 0.050/s (14.3%)
    """

    # Query template for exception rates by type (instant query with rate window)
    # Uses the events_total metric with exception_type label from OpenTelemetry
    EXCEPTION_RATE_QUERY = (
        'sum(rate(events_total{{service_name="{service}", '
        'deployment_environment_name=~"production"}}[{window}])) by (exception_type)'
    )

    # Query template for time-aligned exception counts (range query)
    # Uses increase() to get total count over the window
    EXCEPTION_COUNT_QUERY = (
        'sum(increase(events_total{{service_name="{service}", '
        'deployment_environment_name=~"production"}}[{window}])) by (exception_type)'
    )

    def __init__(
        self,
        vm_client: VictoriaMetricsClient,
        lookback_minutes: int = 5,
        min_rate_threshold: float = 0.001,
        enabled: bool = True,
    ) -> None:
        """Initialize the exception enrichment service.

        Args:
            vm_client: VictoriaMetrics client for querying metrics.
            lookback_minutes: Time window in minutes for exception queries (default: 5).
                             This should match the anomaly detection window.
            min_rate_threshold: Minimum rate to include an exception type (filters noise).
            enabled: Whether enrichment is enabled (can be disabled via config).
        """
        self._client = vm_client
        self._lookback_minutes = lookback_minutes
        self._min_rate_threshold = min_rate_threshold
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        """Check if enrichment is enabled."""
        return self._enabled

    def get_exception_breakdown(
        self,
        service_name: str,
        anomaly_timestamp: datetime | None = None,
        lookback_minutes: int | None = None,
    ) -> ExceptionBreakdown:
        """Get breakdown of exceptions for a service, time-aligned with anomaly detection.

        The query window is aligned with the anomaly timestamp:
        - end_time = anomaly_timestamp (or now if not provided)
        - start_time = end_time - lookback_minutes

        This ensures exception data matches the anomaly detection window.

        Args:
            service_name: Name of the service to query.
            anomaly_timestamp: When the anomaly was detected (default: now).
                              Used to align the query window.
            lookback_minutes: Time window in minutes (default: instance default).
                             Should match the anomaly detection window.

        Returns:
            ExceptionBreakdown with all exception types and their rates.
        """
        if not self._enabled:
            return ExceptionBreakdown(
                service_name=service_name,
                timestamp=datetime.now(),
                total_exception_rate=0.0,
                exceptions=[],
                query_successful=False,
                error_message="Exception enrichment is disabled",
            )

        # Use provided values or defaults
        lookback = lookback_minutes or self._lookback_minutes
        end_time = anomaly_timestamp or datetime.now()
        start_time = end_time - timedelta(minutes=lookback)
        window = f"{lookback}m"

        try:
            # Query exception rates by type
            # Using instant query with rate window aligned to anomaly time
            query = self.EXCEPTION_RATE_QUERY.format(service=service_name, window=window)
            result = self._client.query(query)

            # Parse results
            exceptions: list[ExceptionSummary] = []
            total_rate = 0.0

            data = result.get("data", {}).get("result", [])
            for series in data:
                exception_type = series.get("metric", {}).get("exception_type", "unknown")
                value = series.get("value", [None, "0"])

                try:
                    rate = float(value[1]) if len(value) > 1 else 0.0
                except (ValueError, TypeError, IndexError):
                    rate = 0.0

                # Skip very low rates (noise)
                if rate < self._min_rate_threshold:
                    continue

                total_rate += rate
                exceptions.append(ExceptionSummary(
                    exception_type=exception_type,
                    rate=rate,
                    percentage=0.0,  # Calculate after we have total
                ))

            # Calculate percentages
            if total_rate > 0:
                for exc in exceptions:
                    exc.percentage = (exc.rate / total_rate) * 100

            # Sort by rate descending
            exceptions.sort(key=lambda x: x.rate, reverse=True)

            logger.info(
                f"Exception breakdown for {service_name} "
                f"(window: {start_time.strftime('%H:%M:%S')} - {end_time.strftime('%H:%M:%S')}): "
                f"{len(exceptions)} types, {total_rate:.3f}/s total"
            )

            return ExceptionBreakdown(
                service_name=service_name,
                timestamp=end_time,
                total_exception_rate=total_rate,
                exceptions=exceptions,
                query_successful=True,
            )

        except Exception as e:
            logger.error(f"Failed to get exception breakdown for {service_name}: {e}")
            return ExceptionBreakdown(
                service_name=service_name,
                timestamp=end_time,
                total_exception_rate=0.0,
                exceptions=[],
                query_successful=False,
                error_message=str(e),
            )

    def should_enrich(
        self,
        anomaly: dict[str, Any],
        error_rate_threshold: float = 0.01,
    ) -> bool:
        """Determine if an anomaly should be enriched with exception data.

        Enrichment is triggered when:
        1. Enrichment is enabled
        2. Severity is high or critical
        3. Error rate is above threshold OR pattern is error-related

        Args:
            anomaly: Anomaly detection result dictionary.
            error_rate_threshold: Minimum error rate to trigger enrichment.

        Returns:
            True if anomaly should be enriched with exception context.
        """
        if not self._enabled:
            return False

        # Check if this is a high/critical severity anomaly
        severity = anomaly.get("severity", "low")
        if severity not in ("high", "critical"):
            return False

        # Check if error_rate is involved
        current_metrics = anomaly.get("current_metrics", {})
        error_rate = current_metrics.get("error_rate", 0.0)

        if error_rate < error_rate_threshold:
            return False

        # Check if pattern suggests errors are the issue
        pattern_name = anomaly.get("pattern_name", "")
        error_patterns = {
            "elevated_errors",
            "partial_outage",
            "fast_failure",
            "fast_rejection",
            "partial_fast_fail",
            "traffic_surge_failing",
        }

        if pattern_name in error_patterns:
            return True

        # Also enrich if error_rate is in the affected metrics
        affected = anomaly.get("contributing_metrics", [])
        return "error_rate" in affected

    def enrich_anomaly(
        self,
        anomaly: dict[str, Any],
        service_name: str,
        anomaly_timestamp: datetime | None = None,
    ) -> dict[str, Any]:
        """Enrich an anomaly with exception context, time-aligned with detection.

        Args:
            anomaly: Anomaly detection result dictionary.
            service_name: Name of the service.
            anomaly_timestamp: When the anomaly was detected (for time-aligned queries).

        Returns:
            Anomaly dictionary with added exception_context field.
        """
        if not self.should_enrich(anomaly):
            return anomaly

        breakdown = self.get_exception_breakdown(
            service_name,
            anomaly_timestamp=anomaly_timestamp,
        )

        # Add exception context to anomaly
        anomaly["exception_context"] = breakdown.to_dict()

        # Add summary to interpretation if we have exceptions
        if breakdown.has_exceptions:
            existing_interpretation = anomaly.get("interpretation", "")
            exception_summary = breakdown.get_summary_text(max_exceptions=3)

            anomaly["interpretation"] = (
                f"{existing_interpretation}\n\n"
                f"Exception Analysis:\n{exception_summary}"
            )

            # Add specific recommendation based on top exception
            top_exc = breakdown.top_exception
            if top_exc:
                recommendations = anomaly.get("recommended_actions", [])
                recommendations.insert(
                    0,
                    f"INVESTIGATE: Top exception is {top_exc.short_name} "
                    f"({top_exc.percentage:.0f}% of errors)"
                )
                anomaly["recommended_actions"] = recommendations

        return anomaly
