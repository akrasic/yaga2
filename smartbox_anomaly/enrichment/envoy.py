"""
Envoy edge metrics enrichment for ingress context.

Queries Envoy proxy metrics from Mimir to provide edge-level context about how
a service appears from the ingress perspective. This includes:
- Request rates by HTTP response class (2xx/3xx/4xx/5xx)
- Latency percentiles at the edge
- Active connection counts
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from smartbox_anomaly.core.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class EnvoyRequestRates:
    """Request rates by HTTP response class."""

    rate_2xx: float = 0.0
    rate_3xx: float = 0.0
    rate_4xx: float = 0.0
    rate_5xx: float = 0.0

    @property
    def total(self) -> float:
        """Total request rate across all classes."""
        return self.rate_2xx + self.rate_3xx + self.rate_4xx + self.rate_5xx

    @property
    def error_rate(self) -> float:
        """Combined 4xx + 5xx error rate as ratio (0-1)."""
        if self.total == 0:
            return 0.0
        return (self.rate_4xx + self.rate_5xx) / self.total

    @property
    def server_error_rate(self) -> float:
        """5xx server error rate as ratio (0-1)."""
        if self.total == 0:
            return 0.0
        return self.rate_5xx / self.total

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary for JSON serialization."""
        return {
            "2xx": round(self.rate_2xx, 3),
            "3xx": round(self.rate_3xx, 3),
            "4xx": round(self.rate_4xx, 3),
            "5xx": round(self.rate_5xx, 3),
            "total": round(self.total, 3),
        }


@dataclass
class EnvoyLatencyPercentiles:
    """Latency percentiles from Envoy edge."""

    p50_ms: float | None = None
    p90_ms: float | None = None
    p99_ms: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        """Convert to dictionary for JSON serialization."""
        return {
            "p50_ms": round(self.p50_ms, 2) if self.p50_ms else None,
            "p90_ms": round(self.p90_ms, 2) if self.p90_ms else None,
            "p99_ms": round(self.p99_ms, 2) if self.p99_ms else None,
        }


@dataclass
class EnvoyMetricsContext:
    """Complete Envoy edge metrics context for a service."""

    service_name: str
    envoy_cluster_name: str | None
    timestamp: datetime

    request_rates: EnvoyRequestRates = field(default_factory=EnvoyRequestRates)
    latency_percentiles: EnvoyLatencyPercentiles = field(
        default_factory=EnvoyLatencyPercentiles
    )
    active_connections: int = 0

    query_successful: bool = True
    error_message: str | None = None

    @property
    def has_data(self) -> bool:
        """Check if any Envoy data is available."""
        return self.request_rates.total > 0

    @property
    def edge_error_rate(self) -> float:
        """Combined 4xx + 5xx error rate."""
        return self.request_rates.error_rate

    @property
    def edge_server_error_rate(self) -> float:
        """5xx server error rate."""
        return self.request_rates.server_error_rate

    def get_summary_text(self) -> str:
        """Generate human-readable summary of Envoy metrics."""
        if not self.query_successful:
            return f"Failed to query Envoy metrics: {self.error_message}"

        if not self.has_data:
            return "No Envoy edge data available for this service."

        rates = self.request_rates
        lines = [
            f"Edge: {rates.total:.2f} req/s "
            f"({rates.rate_2xx:.2f} 2xx, {rates.rate_4xx:.2f} 4xx, {rates.rate_5xx:.3f} 5xx)"
        ]

        # Add latency if available
        if self.latency_percentiles.p99_ms:
            lines.append(f"P99: {self.latency_percentiles.p99_ms:.0f}ms")

        # Add connections
        if self.active_connections > 0:
            lines.append(f"{self.active_connections} active connections")

        return ". ".join(lines) + "."

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "service_name": self.service_name,
            "envoy_cluster_name": self.envoy_cluster_name,
            "timestamp": self.timestamp.isoformat(),
            "request_rates": self.request_rates.to_dict(),
            "edge_error_rate": round(self.edge_error_rate, 5),
            "edge_server_error_rate": round(self.edge_server_error_rate, 5),
            "latency_percentiles": self.latency_percentiles.to_dict(),
            "active_connections": self.active_connections,
            "summary": self.get_summary_text(),
            "query_successful": self.query_successful,
            "error_message": self.error_message,
        }


# =============================================================================
# Default Cluster Mapping
# =============================================================================

DEFAULT_CLUSTER_MAPPING: dict[str, str] = {
    # Exact matches
    "booking": "booking",
    "mobile-api": "mobile-api",
    "titan": "titan",
    "friday": "friday",
    # Mapped names
    "search": "search_k8s",
    "shire-api": "shireapi_cluster",
    "fa5": "fa5-public",
}


# =============================================================================
# Envoy Enrichment Service
# =============================================================================


class EnvoyEnrichmentService:
    """Service for enriching anomaly alerts with Envoy edge metrics.

    Queries Envoy proxy metrics from Mimir to provide edge-level context about
    how a service appears from the ingress perspective.

    Example:
        >>> service = EnvoyEnrichmentService()
        >>> context = service.get_envoy_context("booking", anomaly_time)
        >>> print(context.get_summary_text())
        Edge: 75.19 req/s (68.88 2xx, 0.51 4xx, 0.003 5xx). P99: 1920ms. 48 active connections.
    """

    # PromQL query templates for Mimir
    REQUEST_RATE_QUERY = (
        'sum(rate(envoy_cluster_upstream_rq_xx{{envoy_cluster_name="{cluster}"}}[{window}])) '
        "by (envoy_response_code_class)"
    )

    LATENCY_PERCENTILE_QUERY = (
        "histogram_quantile({percentile}, "
        'sum(rate(envoy_cluster_upstream_rq_time_bucket{{envoy_cluster_name="{cluster}"}}[{window}])) '
        "by (le))"
    )

    ACTIVE_CONNECTIONS_QUERY = (
        'sum(envoy_cluster_upstream_cx_active{{envoy_cluster_name="{cluster}"}})'
    )

    def __init__(
        self,
        mimir_endpoint: str = "https://mimir.sbxtest.net/prometheus",
        lookback_minutes: int = 5,
        timeout_seconds: int = 10,
        cluster_mapping: dict[str, str] | None = None,
        enabled: bool = True,
    ) -> None:
        """Initialize the Envoy enrichment service.

        Args:
            mimir_endpoint: Mimir Prometheus API endpoint.
            lookback_minutes: Time window in minutes for queries (default: 5).
            timeout_seconds: HTTP request timeout.
            cluster_mapping: OTel service name to Envoy cluster name mapping.
            enabled: Whether enrichment is enabled.
        """
        self._endpoint = mimir_endpoint.rstrip("/")
        self._lookback_minutes = lookback_minutes
        self._timeout = timeout_seconds
        self._cluster_mapping = cluster_mapping or DEFAULT_CLUSTER_MAPPING
        self._enabled = enabled

        # Set up HTTP session with retries
        self._session = requests.Session()
        retries = Retry(
            total=2,
            backoff_factor=0.3,
            status_forcelist=[500, 502, 503, 504],
        )
        self._session.mount("https://", HTTPAdapter(max_retries=retries))
        self._session.mount("http://", HTTPAdapter(max_retries=retries))

    @property
    def enabled(self) -> bool:
        """Check if enrichment is enabled."""
        return self._enabled

    def get_cluster_name(self, service_name: str) -> str | None:
        """Get Envoy cluster name for a service.

        Args:
            service_name: OTel service name.

        Returns:
            Envoy cluster name or None if no mapping exists.
        """
        return self._cluster_mapping.get(service_name)

    def _query_mimir(self, query: str) -> dict[str, Any] | None:
        """Execute a PromQL query against Mimir.

        Args:
            query: PromQL query string.

        Returns:
            Query result or None on failure.
        """
        try:
            url = f"{self._endpoint}/api/v1/query"
            response = self._session.get(
                url, params={"query": query}, timeout=self._timeout
            )
            response.raise_for_status()
            result = response.json()

            if result.get("status") != "success":
                logger.warning(f"Mimir query returned non-success: {result.get('error')}")
                return None

            return result

        except requests.exceptions.RequestException as e:
            logger.warning(f"Mimir query failed: {e}")
            return None

    def _parse_request_rates(
        self, result: dict[str, Any] | None
    ) -> EnvoyRequestRates:
        """Parse request rate query results."""
        rates = EnvoyRequestRates()

        if not result:
            return rates

        data = result.get("data", {}).get("result", [])
        for series in data:
            metric = series.get("metric", {})
            code_class = metric.get("envoy_response_code_class", "")
            value = series.get("value", [None, "0"])

            try:
                rate = float(value[1]) if len(value) > 1 else 0.0
            except (ValueError, TypeError, IndexError):
                rate = 0.0

            if code_class == "2":
                rates.rate_2xx = rate
            elif code_class == "3":
                rates.rate_3xx = rate
            elif code_class == "4":
                rates.rate_4xx = rate
            elif code_class == "5":
                rates.rate_5xx = rate

        return rates

    def _parse_latency_percentile(
        self, result: dict[str, Any] | None
    ) -> float | None:
        """Parse latency percentile query result."""
        if not result:
            return None

        data = result.get("data", {}).get("result", [])
        if not data:
            return None

        value = data[0].get("value", [None, "NaN"])
        try:
            latency = float(value[1]) if len(value) > 1 else None
            # Filter out NaN and unreasonable values
            if latency is None or latency != latency or latency < 0:  # noqa: PLR0124
                return None
            return latency
        except (ValueError, TypeError, IndexError):
            return None

    def _parse_connections(self, result: dict[str, Any] | None) -> int:
        """Parse active connections query result."""
        if not result:
            return 0

        data = result.get("data", {}).get("result", [])
        if not data:
            return 0

        value = data[0].get("value", [None, "0"])
        try:
            return int(float(value[1])) if len(value) > 1 else 0
        except (ValueError, TypeError, IndexError):
            return 0

    def get_envoy_context(
        self,
        service_name: str,
        anomaly_timestamp: datetime | None = None,
        lookback_minutes: int | None = None,
    ) -> EnvoyMetricsContext | None:
        """Get Envoy edge metrics context for a service.

        Args:
            service_name: OTel service name.
            anomaly_timestamp: When the anomaly was detected (for time alignment).
            lookback_minutes: Time window in minutes (default: instance default).

        Returns:
            EnvoyMetricsContext with edge metrics, or None if no mapping exists.
        """
        if not self._enabled:
            return None

        cluster_name = self.get_cluster_name(service_name)
        if not cluster_name:
            logger.debug(f"No Envoy cluster mapping for service: {service_name}")
            return None

        lookback = lookback_minutes or self._lookback_minutes
        timestamp = anomaly_timestamp or datetime.now()
        window = f"{lookback}m"

        try:
            # Query request rates by response class
            rate_query = self.REQUEST_RATE_QUERY.format(
                cluster=cluster_name, window=window
            )
            rate_result = self._query_mimir(rate_query)
            request_rates = self._parse_request_rates(rate_result)

            # Query latency percentiles
            latencies = EnvoyLatencyPercentiles()
            for percentile_value, attr_name in [
                (0.50, "p50_ms"),
                (0.90, "p90_ms"),
                (0.99, "p99_ms"),
            ]:
                latency_query = self.LATENCY_PERCENTILE_QUERY.format(
                    percentile=percentile_value, cluster=cluster_name, window=window
                )
                latency_result = self._query_mimir(latency_query)
                latency_ms = self._parse_latency_percentile(latency_result)
                setattr(latencies, attr_name, latency_ms)

            # Query active connections
            conn_query = self.ACTIVE_CONNECTIONS_QUERY.format(cluster=cluster_name)
            conn_result = self._query_mimir(conn_query)
            active_connections = self._parse_connections(conn_result)

            return EnvoyMetricsContext(
                service_name=service_name,
                envoy_cluster_name=cluster_name,
                timestamp=timestamp,
                request_rates=request_rates,
                latency_percentiles=latencies,
                active_connections=active_connections,
                query_successful=True,
            )

        except Exception as e:
            logger.warning(f"Envoy context query failed for {service_name}: {e}")
            return EnvoyMetricsContext(
                service_name=service_name,
                envoy_cluster_name=cluster_name,
                timestamp=timestamp,
                query_successful=False,
                error_message=str(e),
            )

    def is_service_supported(self, service_name: str) -> bool:
        """Check if a service has an Envoy cluster mapping.

        Args:
            service_name: OTel service name.

        Returns:
            True if the service has a cluster mapping.
        """
        return service_name in self._cluster_mapping
