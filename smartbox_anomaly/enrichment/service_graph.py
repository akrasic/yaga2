"""
Service graph enrichment for dependency latency diagnosis.

Queries OpenTelemetry service graph metrics from VictoriaMetrics to provide
context about which downstream services and routes are being called when
dependency_latency is elevated.

The enrichment shows:
- Which servers (downstream services) the client is calling
- Which HTTP routes are being hit
- Request rates and average latencies per route
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
class RouteSummary:
    """Summary of a single server/route combination."""

    server: str  # Downstream service name
    route: str | None  # HTTP route (may be None if not available)
    request_rate: float  # Requests per second
    avg_latency_ms: float | None  # Average latency in milliseconds (may be None)
    percentage: float = 0.0  # Percentage of total requests

    @property
    def display_name(self) -> str:
        """Get a display-friendly name for this route."""
        if self.route:
            # Shorten long route names
            route_short = self.route.split("_")[-1] if "_" in self.route else self.route
            return f"{self.server}/{route_short}"
        return self.server


@dataclass
class ServiceGraphBreakdown:
    """Complete breakdown of downstream service calls for a client service."""

    service_name: str
    timestamp: datetime
    total_request_rate: float  # Total requests per second to all downstream services
    routes: list[RouteSummary] = field(default_factory=list)
    query_successful: bool = True
    error_message: str | None = None

    @property
    def top_route(self) -> RouteSummary | None:
        """Get the route with highest request rate."""
        return self.routes[0] if self.routes else None

    @property
    def slowest_route(self) -> RouteSummary | None:
        """Get the route with highest latency."""
        routes_with_latency = [r for r in self.routes if r.avg_latency_ms is not None and r.avg_latency_ms > 0]
        if not routes_with_latency:
            return None
        return max(routes_with_latency, key=lambda r: r.avg_latency_ms or 0)

    @property
    def has_data(self) -> bool:
        """Check if any service graph data is available."""
        return self.total_request_rate > 0 and len(self.routes) > 0

    @property
    def unique_servers(self) -> list[str]:
        """Get list of unique downstream servers."""
        return list(set(r.server for r in self.routes))

    def get_summary_text(self, max_routes: int = 5) -> str:
        """Generate human-readable summary of service graph.

        Args:
            max_routes: Maximum number of routes to include in summary.

        Returns:
            Human-readable summary string.
        """
        if not self.query_successful:
            return f"Failed to query service graph: {self.error_message}"

        if not self.has_data:
            return "No downstream service calls detected in the current window."

        lines = [
            f"Service graph for {self.service_name} ({self.total_request_rate:.2f} req/s total):",
            f"  Downstream services: {', '.join(self.unique_servers[:5])}",
        ]

        # Show top routes by traffic
        lines.append("  Top routes by traffic:")
        for route in self.routes[:max_routes]:
            latency_str = f", {route.avg_latency_ms:.0f}ms" if route.avg_latency_ms else ""
            lines.append(f"    - {route.display_name}: {route.request_rate:.3f}/s ({route.percentage:.1f}%){latency_str}")

        # Show slowest route if different from top
        slowest = self.slowest_route
        if slowest and slowest not in self.routes[:max_routes]:
            lines.append(f"  Slowest route: {slowest.display_name} ({slowest.avg_latency_ms:.0f}ms)")

        remaining = len(self.routes) - max_routes
        if remaining > 0:
            lines.append(f"  ... and {remaining} more routes")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "service_name": self.service_name,
            "timestamp": self.timestamp.isoformat(),
            "total_request_rate": self.total_request_rate,
            "route_count": len(self.routes),
            "unique_servers": self.unique_servers,
            "routes": [
                {
                    "server": r.server,
                    "route": r.route,
                    "request_rate": r.request_rate,
                    "avg_latency_ms": r.avg_latency_ms,
                    "percentage": r.percentage,
                }
                for r in self.routes
            ],
            "top_route": {
                "server": self.top_route.server,
                "route": self.top_route.route,
                "request_rate": self.top_route.request_rate,
            } if self.top_route else None,
            "slowest_route": {
                "server": self.slowest_route.server,
                "route": self.slowest_route.route,
                "avg_latency_ms": self.slowest_route.avg_latency_ms,
            } if self.slowest_route else None,
            "summary": self.get_summary_text(),
            "query_successful": self.query_successful,
            "error_message": self.error_message,
        }


# =============================================================================
# Service Graph Enrichment Service
# =============================================================================


class ServiceGraphEnrichmentService:
    """Service for enriching anomaly alerts with downstream service context.

    Queries OpenTelemetry service graph metrics from VictoriaMetrics to identify
    which downstream services and routes are being called when dependency_latency
    is elevated.

    Example:
        >>> service = ServiceGraphEnrichmentService(vm_client)
        >>> breakdown = service.get_service_graph("cmhub", anomaly_time)
        >>> print(breakdown.get_summary_text())
        Service graph for cmhub (2.1 req/s total):
          Downstream services: r2d2, eai.production.smartbox.com
          Top routes by traffic:
            - r2d2/roomavailabilitylistener: 0.117/s (5.6%), 29ms
            - eai.production.smartbox.com: 0.087/s (4.1%)
    """

    # Query template for request rates by route
    REQUEST_RATE_QUERY = (
        'sum(rate(traces_service_graph_request_total{{client="{service}"}}[{window}])) '
        'by (client, server, server_http_route)'
    )

    # Query template for average latency by route (seconds)
    LATENCY_QUERY = (
        'sum(rate(traces_service_graph_request_server_seconds_sum{{client="{service}"}}[{window}])) '
        'by (client, server, server_http_route) / '
        'sum(rate(traces_service_graph_request_server_seconds_count{{client="{service}"}}[{window}])) '
        'by (client, server, server_http_route)'
    )

    def __init__(
        self,
        vm_client: VictoriaMetricsClient,
        lookback_minutes: int = 5,
        min_rate_threshold: float = 0.001,
        enabled: bool = True,
    ) -> None:
        """Initialize the service graph enrichment service.

        Args:
            vm_client: VictoriaMetrics client for querying metrics.
            lookback_minutes: Time window in minutes for queries (default: 5).
            min_rate_threshold: Minimum rate to include a route (filters noise).
            enabled: Whether enrichment is enabled.
        """
        self._client = vm_client
        self._lookback_minutes = lookback_minutes
        self._min_rate_threshold = min_rate_threshold
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        """Check if enrichment is enabled."""
        return self._enabled

    def get_service_graph(
        self,
        service_name: str,
        anomaly_timestamp: datetime | None = None,
        lookback_minutes: int | None = None,
    ) -> ServiceGraphBreakdown:
        """Get breakdown of downstream service calls for a client service.

        Args:
            service_name: Name of the client service to query.
            anomaly_timestamp: When the anomaly was detected (default: now).
            lookback_minutes: Time window in minutes (default: instance default).

        Returns:
            ServiceGraphBreakdown with all routes and their metrics.
        """
        if not self._enabled:
            return ServiceGraphBreakdown(
                service_name=service_name,
                timestamp=datetime.now(),
                total_request_rate=0.0,
                routes=[],
                query_successful=False,
                error_message="Service graph enrichment is disabled",
            )

        lookback = lookback_minutes or self._lookback_minutes
        end_time = anomaly_timestamp or datetime.now()
        window = f"{lookback}m"

        try:
            # Query request rates
            rate_query = self.REQUEST_RATE_QUERY.format(service=service_name, window=window)
            rate_result = self._client.query(rate_query)

            # Query latencies
            latency_query = self.LATENCY_QUERY.format(service=service_name, window=window)
            latency_result = self._client.query(latency_query)

            # Build latency lookup: (server, route) -> latency_seconds
            latency_map: dict[tuple[str, str | None], float] = {}
            latency_data = latency_result.get("data", {}).get("result", [])
            for series in latency_data:
                metric = series.get("metric", {})
                server = metric.get("server", "unknown")
                route = metric.get("server_http_route")
                value = series.get("value", [None, "0"])
                try:
                    latency_seconds = float(value[1]) if len(value) > 1 else 0.0
                    if latency_seconds > 0:
                        latency_map[(server, route)] = latency_seconds
                except (ValueError, TypeError, IndexError):
                    pass

            # Parse request rate results
            routes: list[RouteSummary] = []
            total_rate = 0.0

            rate_data = rate_result.get("data", {}).get("result", [])
            for series in rate_data:
                metric = series.get("metric", {})
                server = metric.get("server", "unknown")
                route = metric.get("server_http_route")
                value = series.get("value", [None, "0"])

                try:
                    rate = float(value[1]) if len(value) > 1 else 0.0
                except (ValueError, TypeError, IndexError):
                    rate = 0.0

                # Skip very low rates (noise)
                if rate < self._min_rate_threshold:
                    continue

                # Get latency for this route
                latency_seconds = latency_map.get((server, route))
                latency_ms = latency_seconds * 1000 if latency_seconds else None

                total_rate += rate
                routes.append(RouteSummary(
                    server=server,
                    route=route,
                    request_rate=rate,
                    avg_latency_ms=latency_ms,
                ))

            # Calculate percentages and sort by rate
            for route in routes:
                route.percentage = (route.request_rate / total_rate * 100) if total_rate > 0 else 0.0

            routes.sort(key=lambda r: r.request_rate, reverse=True)

            return ServiceGraphBreakdown(
                service_name=service_name,
                timestamp=end_time,
                total_request_rate=total_rate,
                routes=routes,
                query_successful=True,
            )

        except Exception as e:
            logger.warning(f"Service graph query failed for {service_name}: {e}")
            return ServiceGraphBreakdown(
                service_name=service_name,
                timestamp=end_time,
                total_request_rate=0.0,
                routes=[],
                query_successful=False,
                error_message=str(e),
            )
