"""
Tests for the enrichment modules (exceptions and service graph).
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from smartbox_anomaly.enrichment import (
    ExceptionBreakdown,
    ExceptionEnrichmentService,
    ExceptionSummary,
    RouteSummary,
    ServiceGraphBreakdown,
    ServiceGraphEnrichmentService,
)


class TestExceptionSummary:
    """Tests for ExceptionSummary dataclass."""

    def test_short_name_extraction_php_namespace(self):
        """Test short name extraction from PHP-style namespace."""
        exc = ExceptionSummary(
            exception_type="Smartbox\\Search\\R2D2\\Exception\\R2D2Exception",
            rate=0.5,
            percentage=50.0,
        )
        assert exc.short_name == "R2D2Exception"

    def test_short_name_extraction_java_namespace(self):
        """Test short name extraction from Java-style namespace."""
        exc = ExceptionSummary(
            exception_type="com.smartbox.search.R2D2Exception",
            rate=0.5,
            percentage=50.0,
        )
        # Java uses dots, which we don't currently split
        # This test documents current behavior
        assert exc.short_name == "com.smartbox.search.R2D2Exception"

    def test_short_name_simple(self):
        """Test short name for simple exception type."""
        exc = ExceptionSummary(
            exception_type="TypeError",
            rate=0.1,
            percentage=10.0,
        )
        assert exc.short_name == "TypeError"

    def test_explicit_short_name(self):
        """Test that explicit short_name is not overwritten."""
        exc = ExceptionSummary(
            exception_type="Smartbox\\Search\\Exception\\MyException",
            rate=0.5,
            percentage=50.0,
            short_name="CustomName",
        )
        assert exc.short_name == "CustomName"


class TestExceptionBreakdown:
    """Tests for ExceptionBreakdown dataclass."""

    @pytest.fixture
    def sample_breakdown(self) -> ExceptionBreakdown:
        """Create a sample exception breakdown."""
        return ExceptionBreakdown(
            service_name="search",
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
            total_exception_rate=0.35,
            exceptions=[
                ExceptionSummary(
                    exception_type="Smartbox\\Search\\R2D2\\Exception\\R2D2Exception",
                    rate=0.217,
                    percentage=62.0,
                ),
                ExceptionSummary(
                    exception_type="Smartbox\\Search\\Search\\Exception\\UserInputException",
                    rate=0.083,
                    percentage=23.7,
                ),
                ExceptionSummary(
                    exception_type="Smartbox\\Search\\SearchMiddleware\\Exception\\ClientSideMiddlewareException",
                    rate=0.050,
                    percentage=14.3,
                ),
            ],
        )

    def test_top_exception(self, sample_breakdown):
        """Test getting top exception."""
        top = sample_breakdown.top_exception
        assert top is not None
        assert top.short_name == "R2D2Exception"
        assert top.rate == 0.217

    def test_top_exception_empty(self):
        """Test top exception when no exceptions."""
        breakdown = ExceptionBreakdown(
            service_name="test",
            timestamp=datetime.now(),
            total_exception_rate=0.0,
            exceptions=[],
        )
        assert breakdown.top_exception is None

    def test_has_exceptions(self, sample_breakdown):
        """Test has_exceptions property."""
        assert sample_breakdown.has_exceptions is True

    def test_has_exceptions_false_when_empty(self):
        """Test has_exceptions when no exceptions."""
        breakdown = ExceptionBreakdown(
            service_name="test",
            timestamp=datetime.now(),
            total_exception_rate=0.0,
            exceptions=[],
        )
        assert breakdown.has_exceptions is False

    def test_get_summary_text(self, sample_breakdown):
        """Test summary text generation."""
        summary = sample_breakdown.get_summary_text(max_exceptions=2)

        assert "search" in summary
        assert "0.35/s" in summary
        assert "R2D2Exception" in summary
        assert "62.0%" in summary
        assert "UserInputException" in summary
        assert "1 more" in summary  # Third exception not shown

    def test_get_summary_text_no_exceptions(self):
        """Test summary text when no exceptions."""
        breakdown = ExceptionBreakdown(
            service_name="test",
            timestamp=datetime.now(),
            total_exception_rate=0.0,
            exceptions=[],
        )
        summary = breakdown.get_summary_text()
        assert "No exceptions detected" in summary

    def test_get_summary_text_query_failed(self):
        """Test summary text when query failed."""
        breakdown = ExceptionBreakdown(
            service_name="test",
            timestamp=datetime.now(),
            total_exception_rate=0.0,
            exceptions=[],
            query_successful=False,
            error_message="Connection refused",
        )
        summary = breakdown.get_summary_text()
        assert "Failed to query" in summary
        assert "Connection refused" in summary

    def test_to_dict(self, sample_breakdown):
        """Test dictionary conversion."""
        d = sample_breakdown.to_dict()

        assert d["service_name"] == "search"
        assert d["total_exception_rate"] == 0.35
        assert d["exception_count"] == 3
        assert len(d["top_exceptions"]) == 3
        assert d["top_exceptions"][0]["short_name"] == "R2D2Exception"
        assert d["query_successful"] is True


class TestExceptionEnrichmentService:
    """Tests for ExceptionEnrichmentService."""

    @pytest.fixture
    def mock_client(self) -> MagicMock:
        """Create a mock VictoriaMetrics client."""
        client = MagicMock()
        client.query.return_value = {
            "status": "success",
            "data": {
                "result": [
                    {
                        "metric": {"exception_type": "Smartbox\\Search\\R2D2Exception"},
                        "value": [1705312200, "0.217"],
                    },
                    {
                        "metric": {"exception_type": "Smartbox\\Search\\UserInputException"},
                        "value": [1705312200, "0.083"],
                    },
                    {
                        "metric": {"exception_type": "TypeError"},
                        "value": [1705312200, "0.0001"],  # Below threshold
                    },
                ]
            },
        }
        return client

    def test_get_exception_breakdown(self, mock_client):
        """Test getting exception breakdown."""
        service = ExceptionEnrichmentService(mock_client)
        breakdown = service.get_exception_breakdown("search")

        assert breakdown.query_successful is True
        assert breakdown.service_name == "search"
        assert len(breakdown.exceptions) == 2  # TypeError filtered out
        assert breakdown.exceptions[0].short_name == "R2D2Exception"

    def test_get_exception_breakdown_empty_result(self, mock_client):
        """Test handling empty results."""
        mock_client.query.return_value = {"data": {"result": []}}

        service = ExceptionEnrichmentService(mock_client)
        breakdown = service.get_exception_breakdown("search")

        assert breakdown.query_successful is True
        assert breakdown.has_exceptions is False
        assert len(breakdown.exceptions) == 0

    def test_get_exception_breakdown_query_error(self, mock_client):
        """Test handling query errors."""
        mock_client.query.side_effect = Exception("Connection refused")

        service = ExceptionEnrichmentService(mock_client)
        breakdown = service.get_exception_breakdown("search")

        assert breakdown.query_successful is False
        assert "Connection refused" in breakdown.error_message

    def test_should_enrich_high_severity_errors(self, mock_client):
        """Test should_enrich returns True for high severity error anomalies."""
        service = ExceptionEnrichmentService(mock_client)

        anomaly = {
            "severity": "high",
            "pattern_name": "elevated_errors",
            "current_metrics": {"error_rate": 0.05},
        }
        assert service.should_enrich(anomaly) is True

    def test_should_enrich_low_severity_false(self, mock_client):
        """Test should_enrich returns False for low severity."""
        service = ExceptionEnrichmentService(mock_client)

        anomaly = {
            "severity": "low",
            "pattern_name": "elevated_errors",
            "current_metrics": {"error_rate": 0.05},
        }
        assert service.should_enrich(anomaly) is False

    def test_should_enrich_low_error_rate_false(self, mock_client):
        """Test should_enrich returns False when error_rate is low."""
        service = ExceptionEnrichmentService(mock_client)

        anomaly = {
            "severity": "high",
            "pattern_name": "traffic_surge_healthy",
            "current_metrics": {"error_rate": 0.001},
        }
        assert service.should_enrich(anomaly) is False

    def test_should_enrich_error_in_contributing_metrics(self, mock_client):
        """Test should_enrich returns True when error_rate in contributing_metrics."""
        service = ExceptionEnrichmentService(mock_client)

        anomaly = {
            "severity": "critical",
            "pattern_name": "unknown",
            "current_metrics": {"error_rate": 0.1},
            "contributing_metrics": ["application_latency", "error_rate"],
        }
        assert service.should_enrich(anomaly) is True

    def test_enrich_anomaly(self, mock_client):
        """Test anomaly enrichment adds exception context."""
        service = ExceptionEnrichmentService(mock_client)

        anomaly = {
            "severity": "high",
            "pattern_name": "elevated_errors",
            "current_metrics": {"error_rate": 0.05},
            "interpretation": "Error rate is elevated.",
            "recommended_actions": ["CHECK: Logs"],
        }

        enriched = service.enrich_anomaly(anomaly, "search")

        # Check exception_context was added
        assert "exception_context" in enriched
        assert enriched["exception_context"]["service_name"] == "search"

        # Check interpretation was updated
        assert "Exception Analysis" in enriched["interpretation"]
        assert "R2D2Exception" in enriched["interpretation"]

        # Check recommendation was added
        assert "INVESTIGATE: Top exception" in enriched["recommended_actions"][0]

    def test_enrich_anomaly_skips_when_not_needed(self, mock_client):
        """Test enrichment is skipped when not needed."""
        service = ExceptionEnrichmentService(mock_client)

        anomaly = {
            "severity": "low",
            "pattern_name": "traffic_surge_healthy",
            "current_metrics": {"error_rate": 0.001},
        }

        enriched = service.enrich_anomaly(anomaly, "search")

        # Should not have exception_context
        assert "exception_context" not in enriched
        # Client should not have been called
        mock_client.query.assert_not_called()


class TestTimeAlignedQueries:
    """Tests for time-aligned exception queries."""

    @pytest.fixture
    def mock_client(self) -> MagicMock:
        """Create a mock VictoriaMetrics client."""
        client = MagicMock()
        client.query.return_value = {
            "status": "success",
            "data": {
                "result": [
                    {
                        "metric": {"exception_type": "TestException"},
                        "value": [1705312200, "0.5"],
                    },
                ]
            },
        }
        return client

    def test_get_exception_breakdown_with_timestamp(self, mock_client):
        """Test that anomaly_timestamp is used for time-aligned queries."""
        service = ExceptionEnrichmentService(mock_client, lookback_minutes=5)
        anomaly_time = datetime(2024, 1, 15, 10, 30, 0)

        breakdown = service.get_exception_breakdown("search", anomaly_timestamp=anomaly_time)

        assert breakdown.query_successful is True
        assert breakdown.timestamp == anomaly_time

    def test_get_exception_breakdown_custom_lookback(self, mock_client):
        """Test custom lookback_minutes override."""
        service = ExceptionEnrichmentService(mock_client, lookback_minutes=5)

        # Override with 10 minute window
        breakdown = service.get_exception_breakdown("search", lookback_minutes=10)

        assert breakdown.query_successful is True
        # Verify the query used 10m window
        call_args = mock_client.query.call_args[0][0]
        assert "[10m]" in call_args

    def test_enrich_anomaly_passes_timestamp(self, mock_client):
        """Test that enrich_anomaly passes timestamp to get_exception_breakdown."""
        service = ExceptionEnrichmentService(mock_client)
        anomaly_time = datetime(2024, 1, 15, 10, 30, 0)

        anomaly = {
            "severity": "high",
            "pattern_name": "elevated_errors",
            "current_metrics": {"error_rate": 0.05},
        }

        enriched = service.enrich_anomaly(anomaly, "search", anomaly_timestamp=anomaly_time)

        assert "exception_context" in enriched
        assert enriched["exception_context"]["timestamp"] == anomaly_time.isoformat()

    def test_disabled_service_returns_early(self, mock_client):
        """Test that disabled service returns without querying."""
        service = ExceptionEnrichmentService(mock_client, enabled=False)

        breakdown = service.get_exception_breakdown("search")

        assert breakdown.query_successful is False
        assert "disabled" in breakdown.error_message
        mock_client.query.assert_not_called()

    def test_should_enrich_returns_false_when_disabled(self, mock_client):
        """Test should_enrich returns False when service is disabled."""
        service = ExceptionEnrichmentService(mock_client, enabled=False)

        anomaly = {
            "severity": "critical",
            "pattern_name": "elevated_errors",
            "current_metrics": {"error_rate": 0.5},
        }

        assert service.should_enrich(anomaly) is False


# =============================================================================
# Service Graph Enrichment Tests
# =============================================================================


class TestRouteSummary:
    """Tests for RouteSummary dataclass."""

    def test_display_name_with_route(self):
        """Test display name with server and route."""
        route = RouteSummary(
            server="r2d2",
            route="app_broadcastlistener_roomavailabilitylistener",
            request_rate=0.117,
            avg_latency_ms=29.0,
            percentage=5.6,
        )
        assert route.display_name == "r2d2/roomavailabilitylistener"

    def test_display_name_simple_route(self):
        """Test display name with simple route without underscores."""
        route = RouteSummary(
            server="api-gateway",
            route="health",
            request_rate=0.5,
            avg_latency_ms=10.0,
            percentage=25.0,
        )
        assert route.display_name == "api-gateway/health"

    def test_display_name_no_route(self):
        """Test display name when route is None."""
        route = RouteSummary(
            server="external-api.com",
            route=None,
            request_rate=0.087,
            avg_latency_ms=None,
            percentage=4.1,
        )
        assert route.display_name == "external-api.com"


class TestServiceGraphBreakdown:
    """Tests for ServiceGraphBreakdown dataclass."""

    @pytest.fixture
    def sample_breakdown(self) -> ServiceGraphBreakdown:
        """Create a sample service graph breakdown."""
        return ServiceGraphBreakdown(
            service_name="cmhub",
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
            total_request_rate=2.1,
            routes=[
                RouteSummary(
                    server="r2d2",
                    route="app_broadcastlistener_roomavailabilitylistener",
                    request_rate=0.117,
                    avg_latency_ms=29.0,
                    percentage=5.6,
                ),
                RouteSummary(
                    server="eai.production.smartbox.com",
                    route=None,
                    request_rate=0.087,
                    avg_latency_ms=None,
                    percentage=4.1,
                ),
                RouteSummary(
                    server="database",
                    route="query",
                    request_rate=0.050,
                    avg_latency_ms=150.0,
                    percentage=2.4,
                ),
            ],
        )

    def test_top_route(self, sample_breakdown):
        """Test getting top route by request rate."""
        top = sample_breakdown.top_route
        assert top is not None
        assert top.server == "r2d2"
        assert top.request_rate == 0.117

    def test_top_route_empty(self):
        """Test top route when no routes."""
        breakdown = ServiceGraphBreakdown(
            service_name="test",
            timestamp=datetime.now(),
            total_request_rate=0.0,
            routes=[],
        )
        assert breakdown.top_route is None

    def test_slowest_route(self, sample_breakdown):
        """Test getting slowest route by latency."""
        slowest = sample_breakdown.slowest_route
        assert slowest is not None
        assert slowest.server == "database"
        assert slowest.avg_latency_ms == 150.0

    def test_slowest_route_no_latency_data(self):
        """Test slowest route when no latency data available."""
        breakdown = ServiceGraphBreakdown(
            service_name="test",
            timestamp=datetime.now(),
            total_request_rate=0.5,
            routes=[
                RouteSummary(
                    server="external",
                    route=None,
                    request_rate=0.5,
                    avg_latency_ms=None,
                    percentage=100.0,
                ),
            ],
        )
        assert breakdown.slowest_route is None

    def test_has_data(self, sample_breakdown):
        """Test has_data property."""
        assert sample_breakdown.has_data is True

    def test_has_data_false_when_empty(self):
        """Test has_data when no routes."""
        breakdown = ServiceGraphBreakdown(
            service_name="test",
            timestamp=datetime.now(),
            total_request_rate=0.0,
            routes=[],
        )
        assert breakdown.has_data is False

    def test_unique_servers(self, sample_breakdown):
        """Test getting unique downstream servers."""
        servers = sample_breakdown.unique_servers
        assert len(servers) == 3
        assert "r2d2" in servers
        assert "eai.production.smartbox.com" in servers
        assert "database" in servers

    def test_get_summary_text(self, sample_breakdown):
        """Test summary text generation."""
        summary = sample_breakdown.get_summary_text(max_routes=2)

        assert "cmhub" in summary
        assert "2.10 req/s" in summary
        assert "r2d2" in summary
        assert "roomavailabilitylistener" in summary
        assert "29ms" in summary
        assert "1 more" in summary

    def test_get_summary_text_no_data(self):
        """Test summary text when no downstream calls."""
        breakdown = ServiceGraphBreakdown(
            service_name="test",
            timestamp=datetime.now(),
            total_request_rate=0.0,
            routes=[],
        )
        summary = breakdown.get_summary_text()
        assert "No downstream service calls" in summary

    def test_get_summary_text_query_failed(self):
        """Test summary text when query failed."""
        breakdown = ServiceGraphBreakdown(
            service_name="test",
            timestamp=datetime.now(),
            total_request_rate=0.0,
            routes=[],
            query_successful=False,
            error_message="Connection timeout",
        )
        summary = breakdown.get_summary_text()
        assert "Failed to query" in summary
        assert "Connection timeout" in summary

    def test_to_dict(self, sample_breakdown):
        """Test dictionary conversion."""
        d = sample_breakdown.to_dict()

        assert d["service_name"] == "cmhub"
        assert d["total_request_rate"] == 2.1
        assert d["route_count"] == 3
        assert len(d["unique_servers"]) == 3
        assert len(d["routes"]) == 3
        assert d["top_route"]["server"] == "r2d2"
        assert d["slowest_route"]["server"] == "database"
        assert d["slowest_route"]["avg_latency_ms"] == 150.0
        assert d["query_successful"] is True


class TestServiceGraphEnrichmentService:
    """Tests for ServiceGraphEnrichmentService."""

    @pytest.fixture
    def mock_client(self) -> MagicMock:
        """Create a mock VictoriaMetrics client."""
        client = MagicMock()
        # Mock response for request rate query
        client.query.side_effect = [
            # First call: request rates
            {
                "status": "success",
                "data": {
                    "result": [
                        {
                            "metric": {
                                "client": "cmhub",
                                "server": "r2d2",
                                "server_http_route": "app_broadcastlistener_roomavailabilitylistener",
                            },
                            "value": [1705312200, "0.117"],
                        },
                        {
                            "metric": {
                                "client": "cmhub",
                                "server": "eai.production.smartbox.com",
                                "server_http_route": None,
                            },
                            "value": [1705312200, "0.087"],
                        },
                        {
                            "metric": {
                                "client": "cmhub",
                                "server": "noise",
                                "server_http_route": "low_traffic",
                            },
                            "value": [1705312200, "0.0001"],  # Below threshold
                        },
                    ]
                },
            },
            # Second call: latencies
            {
                "status": "success",
                "data": {
                    "result": [
                        {
                            "metric": {
                                "client": "cmhub",
                                "server": "r2d2",
                                "server_http_route": "app_broadcastlistener_roomavailabilitylistener",
                            },
                            "value": [1705312200, "0.029"],  # 29ms
                        },
                    ]
                },
            },
        ]
        return client

    def test_get_service_graph(self, mock_client):
        """Test getting service graph breakdown."""
        service = ServiceGraphEnrichmentService(mock_client)
        breakdown = service.get_service_graph("cmhub")

        assert breakdown.query_successful is True
        assert breakdown.service_name == "cmhub"
        assert len(breakdown.routes) == 2  # noise filtered out
        assert breakdown.routes[0].server == "r2d2"
        assert breakdown.routes[0].avg_latency_ms == 29.0

    def test_get_service_graph_empty_result(self, mock_client):
        """Test handling empty results."""
        mock_client.query.side_effect = [
            {"data": {"result": []}},
            {"data": {"result": []}},
        ]

        service = ServiceGraphEnrichmentService(mock_client)
        breakdown = service.get_service_graph("unknown-service")

        assert breakdown.query_successful is True
        assert breakdown.has_data is False
        assert len(breakdown.routes) == 0

    def test_get_service_graph_query_error(self, mock_client):
        """Test handling query errors."""
        mock_client.query.side_effect = Exception("Connection refused")

        service = ServiceGraphEnrichmentService(mock_client)
        breakdown = service.get_service_graph("cmhub")

        assert breakdown.query_successful is False
        assert "Connection refused" in breakdown.error_message

    def test_get_service_graph_with_timestamp(self, mock_client):
        """Test that anomaly_timestamp is used for time-aligned queries."""
        service = ServiceGraphEnrichmentService(mock_client, lookback_minutes=5)
        anomaly_time = datetime(2024, 1, 15, 10, 30, 0)

        breakdown = service.get_service_graph("cmhub", anomaly_timestamp=anomaly_time)

        assert breakdown.query_successful is True
        assert breakdown.timestamp == anomaly_time

    def test_get_service_graph_custom_lookback(self, mock_client):
        """Test custom lookback_minutes override."""
        service = ServiceGraphEnrichmentService(mock_client, lookback_minutes=5)

        # Override with 10 minute window
        breakdown = service.get_service_graph("cmhub", lookback_minutes=10)

        assert breakdown.query_successful is True
        # Verify the query used 10m window
        call_args = mock_client.query.call_args_list[0][0][0]
        assert "[10m]" in call_args

    def test_disabled_service_returns_early(self, mock_client):
        """Test that disabled service returns without querying."""
        service = ServiceGraphEnrichmentService(mock_client, enabled=False)

        breakdown = service.get_service_graph("cmhub")

        assert breakdown.query_successful is False
        assert "disabled" in breakdown.error_message
        mock_client.query.assert_not_called()

    def test_percentage_calculation(self, mock_client):
        """Test that percentages are calculated correctly."""
        service = ServiceGraphEnrichmentService(mock_client)
        breakdown = service.get_service_graph("cmhub")

        # Total rate is 0.117 + 0.087 = 0.204
        # First route: 0.117 / 0.204 * 100 = 57.35%
        # Second route: 0.087 / 0.204 * 100 = 42.65%
        assert breakdown.routes[0].percentage == pytest.approx(57.35, rel=0.01)
        assert breakdown.routes[1].percentage == pytest.approx(42.65, rel=0.01)

    def test_routes_sorted_by_rate(self, mock_client):
        """Test that routes are sorted by request rate descending."""
        service = ServiceGraphEnrichmentService(mock_client)
        breakdown = service.get_service_graph("cmhub")

        # Routes should be sorted by rate descending
        rates = [r.request_rate for r in breakdown.routes]
        assert rates == sorted(rates, reverse=True)
