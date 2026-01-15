"""
VictoriaMetrics client for collecting service metrics.

This module provides a robust client for querying VictoriaMetrics,
with retry logic, circuit breaker pattern, and proper error handling.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from types import TracebackType
from typing import Any, Final

import requests
import urllib3

from smartbox_anomaly.core.config import VictoriaMetricsConfig, get_config
from smartbox_anomaly.core.constants import MetricName
from smartbox_anomaly.core.exceptions import (
    CircuitBreakerOpenError,
    MetricsCollectionError,
    MetricsValidationError,
)
from smartbox_anomaly.core.logging import get_logger
from smartbox_anomaly.core.protocols import MetricsCollector
from smartbox_anomaly.metrics.validation import ValidationResult, validate_metrics

# Suppress urllib3 warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = get_logger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class InferenceMetrics:
    """Input metrics for inference with validation and availability tracking."""

    service_name: str
    timestamp: datetime
    request_rate: float
    application_latency: float | None = None
    dependency_latency: float | None = None
    database_latency: float | None = None
    error_rate: float | None = None

    # Track which metrics were successfully collected vs failed
    failed_metrics: list[str] | None = None
    collection_errors: dict[str, str] | None = None

    def to_dict(self) -> dict[str, float]:
        """Convert to model input format."""
        return {
            MetricName.REQUEST_RATE: self.request_rate,
            MetricName.APPLICATION_LATENCY: self.application_latency or 0.0,
            MetricName.DEPENDENCY_LATENCY: self.dependency_latency or 0.0,
            MetricName.DATABASE_LATENCY: self.database_latency or 0.0,
            MetricName.ERROR_RATE: self.error_rate or 0.0,
        }

    def validate(self) -> ValidationResult:
        """Validate input data quality."""
        return validate_metrics(self.to_dict(), self.service_name)

    def is_valid(self) -> bool:
        """Check if metrics are valid."""
        return self.validate().is_valid

    def has_critical_failures(self) -> bool:
        """Check if critical metrics (request_rate) failed to collect.

        If request_rate failed, we cannot reliably detect anomalies
        because 0.0 looks like a traffic cliff.
        """
        if not self.failed_metrics:
            return False
        return MetricName.REQUEST_RATE in self.failed_metrics

    def has_any_failures(self) -> bool:
        """Check if any metrics failed to collect."""
        return bool(self.failed_metrics)

    def get_failure_summary(self) -> str | None:
        """Get a summary of collection failures."""
        if not self.failed_metrics:
            return None
        failed_count = len(self.failed_metrics)
        total_count = 5  # Total metrics we collect
        return f"{failed_count}/{total_count} metrics failed: {', '.join(self.failed_metrics)}"

    def is_reliable_for_detection(self) -> bool:
        """Check if metrics are reliable enough for anomaly detection.

        Returns False if:
        - request_rate failed (would cause false traffic cliff alerts)
        - More than 2 metrics failed (too much missing data)
        """
        if not self.failed_metrics:
            return True
        if self.has_critical_failures():
            return False
        # Allow detection if only 1-2 non-critical metrics failed
        return len(self.failed_metrics) <= 2


@dataclass
class QueryResult:
    """Structured result from VictoriaMetrics queries.

    Replaces silent failure pattern with explicit success/failure tracking.
    This allows callers to distinguish between "no data exists" and "query failed".
    """

    success: bool
    data: dict[str, Any]
    error_message: str | None = None
    warning_message: str | None = None
    query: str | None = None
    duration_ms: float | None = None

    @classmethod
    def ok(
        cls,
        data: dict[str, Any],
        query: str | None = None,
        duration_ms: float | None = None,
    ) -> QueryResult:
        """Create a successful result."""
        return cls(success=True, data=data, query=query, duration_ms=duration_ms)

    @classmethod
    def error(
        cls,
        message: str,
        query: str | None = None,
        duration_ms: float | None = None,
    ) -> QueryResult:
        """Create a failed result."""
        return cls(
            success=False,
            data={"data": {"result": []}},
            error_message=message,
            query=query,
            duration_ms=duration_ms,
        )

    @classmethod
    def empty_with_warning(
        cls,
        warning: str,
        query: str | None = None,
        duration_ms: float | None = None,
    ) -> QueryResult:
        """Create successful but empty result with warning."""
        return cls(
            success=True,
            data={"data": {"result": []}},
            warning_message=warning,
            query=query,
            duration_ms=duration_ms,
        )

    def get_result_data(self) -> list[Any]:
        """Extract the result array from the response."""
        return self.data.get("data", {}).get("result", [])

    def is_empty(self) -> bool:
        """Check if result contains no data."""
        return len(self.get_result_data()) == 0

    def get_values_count(self) -> int:
        """Count total data points across all series."""
        total = 0
        for series in self.get_result_data():
            values = series.get("values", [])
            total += len(values)
        return total


@dataclass
class CircuitBreakerState:
    """State tracking for circuit breaker pattern."""

    failure_count: int = 0
    last_failure_time: float | None = None
    threshold: int = 5
    timeout_seconds: int = 300

    def record_success(self) -> None:
        """Record a successful operation, resetting the circuit."""
        self.failure_count = 0
        self.last_failure_time = None

    def record_failure(self) -> None:
        """Record a failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()

    def is_open(self) -> bool:
        """Check if the circuit breaker is open."""
        if self.failure_count < self.threshold:
            return False

        if self.last_failure_time is None:
            return False

        time_since_failure = time.time() - self.last_failure_time
        return time_since_failure < self.timeout_seconds

    def time_until_reset(self) -> float | None:
        """Get time until circuit breaker resets, if open."""
        if not self.is_open():
            return None
        if self.last_failure_time is None:
            return None
        return self.timeout_seconds - (time.time() - self.last_failure_time)


# =============================================================================
# VictoriaMetrics Client
# =============================================================================


class VictoriaMetricsClient(MetricsCollector):
    """Robust VictoriaMetrics client with retry logic and circuit breaker.

    This client implements the MetricsCollector protocol and provides:
    - Connection pooling for efficient HTTP connections
    - Automatic retry with exponential backoff
    - Circuit breaker to prevent cascading failures
    - Comprehensive error handling

    Example:
        >>> client = VictoriaMetricsClient()
        >>> metrics = client.collect_service_metrics("booking")
        >>> print(metrics.request_rate)
    """

    # PromQL queries for each metric
    QUERIES: Final[dict[str, str]] = {
        MetricName.REQUEST_RATE: "http_requests:count:rate_5m",
        MetricName.APPLICATION_LATENCY: (
            'sum(rate(duration_milliseconds_sum{span_kind="SPAN_KIND_SERVER", '
            'deployment_environment_name=~"production"}[5m])) by (service_name) / '
            'sum(rate(duration_milliseconds_count{span_kind="SPAN_KIND_SERVER", '
            'deployment_environment_name=~"production"}[5m])) by (service_name)'
        ),
        MetricName.DEPENDENCY_LATENCY: (
            'sum(rate(duration_milliseconds_sum{span_kind="SPAN_KIND_CLIENT", '
            'deployment_environment_name=~"production", db_system="", db_system_name=""}[5m])) '
            'by (service_name) / sum(rate(duration_milliseconds_count{span_kind="SPAN_KIND_CLIENT", '
            'deployment_environment_name=~"production", db_system="", db_system_name=""}[5m])) '
            'by (service_name)'
        ),
        MetricName.DATABASE_LATENCY: (
            'sum(rate(duration_milliseconds_sum{span_kind="SPAN_KIND_CLIENT", '
            'deployment_environment_name=~"production", db_system_name!=""}[5m])) by (service_name) / '
            'sum(rate(duration_milliseconds_count{span_kind="SPAN_KIND_CLIENT", '
            'deployment_environment_name=~"production", db_system_name!=""}[5m])) by (service_name)'
        ),
        MetricName.ERROR_RATE: (
            'sum(rate(duration_milliseconds_count{span_kind="SPAN_KIND_SERVER", '
            'deployment_environment_name=~"production", http_response_status_code=~"5.*|"}[5m])) '
            'by (service_name) / sum(rate(duration_milliseconds_count{span_kind="SPAN_KIND_SERVER", '
            'deployment_environment_name=~"production"}[5m])) by (service_name)'
        ),
    }

    def __init__(
        self,
        endpoint: str | None = None,
        config: VictoriaMetricsConfig | None = None,
    ) -> None:
        """Initialize the VictoriaMetrics client."""
        self._config = config or get_config().victoria_metrics
        self._endpoint = (endpoint or self._config.endpoint).rstrip("/")
        self._session = self._create_session()
        self._circuit_breaker = CircuitBreakerState(
            threshold=self._config.circuit_breaker_threshold,
            timeout_seconds=self._config.circuit_breaker_timeout_seconds,
        )

    @property
    def endpoint(self) -> str:
        """Get the configured endpoint."""
        return self._endpoint

    def _create_session(self) -> requests.Session:
        """Create a configured requests session with connection pooling."""
        session = requests.Session()

        adapter = requests.adapters.HTTPAdapter(
            pool_connections=self._config.pool_connections,
            pool_maxsize=self._config.pool_maxsize,
            max_retries=urllib3.util.retry.Retry(
                total=2,
                backoff_factor=self._config.retry_backoff_factor,
                status_forcelist=list(self._config.retry_status_forcelist),
            ),
            pool_block=False,
        )

        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def is_circuit_open(self) -> bool:
        """Check if circuit breaker is open."""
        return self._circuit_breaker.is_open()

    def health_check(self, max_latency_ms: float = 5000) -> tuple[bool, dict[str, Any]]:
        """Check if VictoriaMetrics is accessible and responding within acceptable latency.

        Args:
            max_latency_ms: Maximum acceptable response time in milliseconds.

        Returns:
            Tuple of (is_healthy, details_dict) where details contains:
            - healthy: bool
            - latency_ms: response time in milliseconds
            - status_code: HTTP status code (if available)
            - error: error message (if any)
            - circuit_breaker_open: bool
        """
        details: dict[str, Any] = {
            "healthy": False,
            "latency_ms": None,
            "status_code": None,
            "error": None,
            "circuit_breaker_open": self.is_circuit_open(),
        }

        if self.is_circuit_open():
            details["error"] = "Circuit breaker is open"
            return False, details

        try:
            start_time = time.time()
            response = self._session.get(
                f"{self._endpoint}/health",
                timeout=self._config.timeout_seconds,
            )
            latency_ms = (time.time() - start_time) * 1000

            details["latency_ms"] = round(latency_ms, 2)
            details["status_code"] = response.status_code

            if response.status_code != 200:
                details["error"] = f"Unhealthy status code: {response.status_code}"
                return False, details

            if latency_ms > max_latency_ms:
                details["error"] = f"Response too slow: {latency_ms:.0f}ms > {max_latency_ms:.0f}ms threshold"
                return False, details

            details["healthy"] = True
            return True, details

        except requests.exceptions.Timeout:
            details["error"] = f"Request timed out after {self._config.timeout_seconds}s"
            return False, details
        except requests.exceptions.ConnectionError as e:
            details["error"] = f"Connection failed: {e}"
            return False, details
        except requests.RequestException as e:
            details["error"] = f"Request failed: {e}"
            return False, details

    def is_healthy(self, max_latency_ms: float = 5000) -> bool:
        """Simple health check returning only boolean.

        Args:
            max_latency_ms: Maximum acceptable response time in milliseconds.

        Returns:
            True if VictoriaMetrics is healthy and responding quickly.
        """
        healthy, _ = self.health_check(max_latency_ms)
        return healthy

    def collect_service_metrics(self, service_name: str) -> InferenceMetrics:  # type: ignore[override]
        """Collect current metrics for a service with robust error handling.

        Tracks which metrics failed to collect so detection can decide
        whether to proceed or skip (avoiding false traffic cliff alerts).
        """
        if self.is_circuit_open():
            raise CircuitBreakerOpenError(
                "VictoriaMetrics",
                failures=self._circuit_breaker.failure_count,
                timeout_remaining=self._circuit_breaker.time_until_reset(),
            )

        current_time = datetime.now()
        metrics_data: dict[str, Any] = {
            "service_name": service_name,
            "timestamp": current_time,
        }

        # Track failed metrics for reliability assessment
        failed_metrics: list[str] = []
        collection_errors: dict[str, str] = {}

        try:
            for metric_name, query in self.QUERIES.items():
                try:
                    value = self._query_metric_with_retry(query, service_name)
                    metrics_data[metric_name] = value
                except Exception as e:
                    error_msg = str(e)
                    # Check for connection errors specifically
                    is_connection_error = any(
                        err_type in error_msg.lower()
                        for err_type in ["connection refused", "connection error", "max retries", "timeout"]
                    )

                    if is_connection_error:
                        logger.warning(
                            f"Metrics server unreachable for {metric_name}/{service_name}: {error_msg[:100]}..."
                        )
                    else:
                        logger.warning(f"Failed to collect {metric_name} for {service_name}: {error_msg[:100]}...")

                    # Track the failure instead of silently defaulting to 0
                    failed_metrics.append(metric_name)
                    collection_errors[metric_name] = error_msg[:200]
                    metrics_data[metric_name] = 0.0  # Still need a value for the dataclass

                time.sleep(0.01)  # Reduced delay (was 0.1s)

            # Add failure tracking to the metrics object
            metrics_data["failed_metrics"] = failed_metrics if failed_metrics else None
            metrics_data["collection_errors"] = collection_errors if collection_errors else None

            metrics = InferenceMetrics(**metrics_data)

            # Log summary if there were failures
            if failed_metrics:
                failure_summary = metrics.get_failure_summary()
                if metrics.has_critical_failures():
                    logger.error(
                        f"Critical metrics collection failure for {service_name}: {failure_summary}. "
                        "Detection will be skipped to avoid false alerts."
                    )
                else:
                    logger.warning(
                        f"Partial metrics collection failure for {service_name}: {failure_summary}. "
                        "Detection will proceed with available metrics."
                    )

            validation_result = metrics.validate()
            if not validation_result.is_valid:
                raise MetricsValidationError(
                    service_name,
                    metric_name="multiple",
                    value=metrics_data,
                    reason="; ".join(validation_result.errors),
                )

            for warning in validation_result.warnings:
                logger.warning(f"Metrics warning for {service_name}: {warning}")

            # Only record full success if no metrics failed
            if not failed_metrics:
                self._circuit_breaker.record_success()
            elif len(failed_metrics) >= 3:
                # Multiple failures suggest systemic issue
                self._circuit_breaker.record_failure()

            return metrics

        except MetricsValidationError:
            self._circuit_breaker.record_failure()
            raise
        except Exception as e:
            self._circuit_breaker.record_failure()
            raise MetricsCollectionError(service_name, reason=str(e), cause=e) from e

    def query(self, query: str) -> dict[str, Any]:
        """Query VictoriaMetrics for current value.

        Note: This method returns raw dict for backward compatibility.
        For new code that needs error tracking, use query_instant() instead.
        """
        result = self.query_instant(query)
        return result.data

    def query_instant(self, query: str) -> QueryResult:
        """Query VictoriaMetrics for current value with structured result.

        Returns QueryResult with success status and error tracking.
        """
        params = {"query": query}
        start_ts = time.time()

        try:
            response = self._session.get(
                f"{self._endpoint}/api/v1/query",
                params=params,
                timeout=self._config.timeout_seconds,
            )
            duration_ms = (time.time() - start_ts) * 1000
            response.raise_for_status()
            result: dict[str, Any] = response.json()

            if result.get("status") == "error":
                error_msg = result.get("error", "Unknown VictoriaMetrics error")
                logger.error(
                    "VictoriaMetrics query error: %s (query: %s...)",
                    error_msg,
                    query[:100],
                )
                return QueryResult.error(
                    message=f"VictoriaMetrics error: {error_msg}",
                    query=query,
                    duration_ms=duration_ms,
                )

            return QueryResult.ok(data=result, query=query, duration_ms=duration_ms)

        except requests.exceptions.Timeout as e:
            duration_ms = (time.time() - start_ts) * 1000
            logger.error(
                "query timeout after %.1fs: %s...",
                duration_ms / 1000,
                query[:100],
            )
            return QueryResult.error(
                message=f"Request timed out after {self._config.timeout_seconds}s",
                query=query,
                duration_ms=duration_ms,
            )

        except requests.exceptions.ConnectionError as e:
            duration_ms = (time.time() - start_ts) * 1000
            logger.error(
                "query connection error: %s (query: %s...)",
                str(e)[:100],
                query[:100],
            )
            return QueryResult.error(
                message=f"Connection error: {str(e)[:100]}",
                query=query,
                duration_ms=duration_ms,
            )

        except requests.exceptions.RequestException as e:
            duration_ms = (time.time() - start_ts) * 1000
            logger.error(
                "query request failed: %s (query: %s...)",
                str(e)[:100],
                query[:100],
            )
            return QueryResult.error(
                message=f"Request failed: {str(e)[:100]}",
                query=query,
                duration_ms=duration_ms,
            )

    def query_range(
        self,
        query: str,
        start_time: datetime,
        end_time: datetime,
        step: str = "5m",
        timeout_seconds: int | None = None,
    ) -> QueryResult:
        """Query VictoriaMetrics for a time range.

        Args:
            query: PromQL query string.
            start_time: Start of the time range.
            end_time: End of the time range.
            step: Query resolution (default: 5m).
            timeout_seconds: Request timeout. Defaults to config value.
                Use longer timeouts for training queries (e.g., 120s for 30-day ranges).

        Returns:
            QueryResult with success status, data, and any error/warning messages.
        """
        effective_timeout = timeout_seconds or self._config.timeout_seconds
        params = {
            "query": query,
            "start": start_time.isoformat() + "Z",
            "end": end_time.isoformat() + "Z",
            "step": step,
        }

        start_ts = time.time()

        try:
            response = self._session.get(
                f"{self._endpoint}/api/v1/query_range",
                params=params,
                timeout=effective_timeout,
            )
            duration_ms = (time.time() - start_ts) * 1000
            response.raise_for_status()
            result: dict[str, Any] = response.json()

            if result.get("status") == "error":
                error_msg = result.get("error", "Unknown VictoriaMetrics error")
                logger.error(
                    "VictoriaMetrics query_range error: %s (query: %s...)",
                    error_msg,
                    query[:100],
                )
                return QueryResult.error(
                    message=f"VictoriaMetrics error: {error_msg}",
                    query=query,
                    duration_ms=duration_ms,
                )

            query_result = QueryResult.ok(data=result, query=query, duration_ms=duration_ms)

            # Log warning if result is unexpectedly empty
            if query_result.is_empty():
                logger.debug(
                    "query_range returned empty result for query: %s... (this may be expected)",
                    query[:100],
                )

            return query_result

        except requests.exceptions.Timeout as e:
            duration_ms = (time.time() - start_ts) * 1000
            logger.error(
                "query_range timeout after %.1fs (timeout=%ds): %s...",
                duration_ms / 1000,
                effective_timeout,
                query[:100],
            )
            return QueryResult.error(
                message=f"Request timed out after {effective_timeout}s",
                query=query,
                duration_ms=duration_ms,
            )

        except requests.exceptions.ConnectionError as e:
            duration_ms = (time.time() - start_ts) * 1000
            logger.error(
                "query_range connection error: %s (query: %s...)",
                str(e)[:100],
                query[:100],
            )
            return QueryResult.error(
                message=f"Connection error: {str(e)[:100]}",
                query=query,
                duration_ms=duration_ms,
            )

        except requests.exceptions.RequestException as e:
            duration_ms = (time.time() - start_ts) * 1000
            logger.error(
                "query_range request failed: %s (query: %s...)",
                str(e)[:100],
                query[:100],
            )
            return QueryResult.error(
                message=f"Request failed: {str(e)[:100]}",
                query=query,
                duration_ms=duration_ms,
            )

    def query_range_raw(
        self,
        query: str,
        start_time: datetime,
        end_time: datetime,
        step: str = "5m",
        timeout_seconds: int | None = None,
    ) -> dict[str, Any]:
        """Query VictoriaMetrics for a time range, returning raw dict.

        Backward-compatible method that returns raw dict format.
        New code should use query_range() which returns QueryResult.
        """
        result = self.query_range(query, start_time, end_time, step, timeout_seconds)
        if not result.success:
            logger.warning(
                "query_range_raw: query failed (%s), returning empty result",
                result.error_message,
            )
        return result.data

    def _query_metric_with_retry(self, query: str, service_name: str) -> float:
        """Query single metric with retry logic."""
        if "by (service_name)" in query:
            filtered_query = query.replace(
                'deployment_environment_name=~"production"',
                f'deployment_environment_name=~"production", service_name="{service_name}"',
            )
        else:
            filtered_query = f'{query}{{service_name="{service_name}"}}'

        last_exception: Exception | None = None

        for attempt in range(self._config.max_retries):
            try:
                response = self._session.get(
                    f"{self._endpoint}/api/v1/query",
                    params={"query": filtered_query},
                    timeout=self._config.timeout_seconds,
                )
                response.raise_for_status()

                result = response.json()
                if result.get("status") == "error":
                    raise MetricsCollectionError(
                        service_name,
                        reason=f"VictoriaMetrics error: {result.get('error')}",
                    )

                if result.get("data", {}).get("result"):
                    for series in result["data"]["result"]:
                        if series.get("value") and len(series["value"]) > 1:
                            try:
                                return float(series["value"][1])
                            except (ValueError, TypeError, IndexError):
                                pass

                return 0.0

            except Exception as e:
                last_exception = e
                if attempt < self._config.max_retries - 1:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                break

        raise last_exception or MetricsCollectionError(service_name, reason="Max retries exceeded")

    def close(self) -> None:
        """Close the session and clean up resources."""
        self._session.close()

    def __enter__(self) -> VictoriaMetricsClient:
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Context manager exit."""
        self.close()
