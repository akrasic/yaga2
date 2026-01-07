#!/usr/bin/env python3
"""
Test script for exception enrichment prototype.

This script tests the ExceptionEnrichmentService by querying VictoriaMetrics
for exception data for a specified service.

Usage:
    # Test with search service (default)
    uv run python scripts/test_exception_enrichment.py

    # Test with a specific service
    uv run python scripts/test_exception_enrichment.py --service booking

    # Test with different time window
    uv run python scripts/test_exception_enrichment.py --service search --window 1m

    # Just test the raw query without the service
    uv run python scripts/test_exception_enrichment.py --raw-query
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime

from smartbox_anomaly.core.config import get_config
from smartbox_anomaly.enrichment import ExceptionBreakdown, ExceptionEnrichmentService
from smartbox_anomaly.metrics.client import VictoriaMetricsClient


def test_raw_query(client: VictoriaMetricsClient, service: str, window: str) -> None:
    """Test the raw PromQL query directly."""
    print(f"\n{'='*60}")
    print(f"Testing RAW QUERY for service: {service}")
    print(f"{'='*60}\n")

    query = (
        f'sum(rate(events_total{{service_name="{service}", '
        f'deployment_environment_name=~"production"}}[{window}])) by (exception_type)'
    )

    print(f"Query:\n{query}\n")
    print("Executing query...")

    result = client.query(query)

    print(f"\nRaw response status: {result.get('status', 'unknown')}")

    data = result.get("data", {}).get("result", [])
    print(f"Number of results: {len(data)}\n")

    if not data:
        print("⚠️  No data returned. Possible reasons:")
        print("   - Service name might be different")
        print("   - No exceptions in the time window")
        print("   - Metric name might be different")
        print("\nTry listing available services with:")
        print('   sum(rate(events_total{deployment_environment_name=~"production"}[5m])) by (service_name)')
        return

    print("Results:")
    print("-" * 60)

    for series in data:
        exception_type = series.get("metric", {}).get("exception_type", "unknown")
        value = series.get("value", [None, "0"])
        try:
            rate = float(value[1]) if len(value) > 1 else 0.0
        except (ValueError, TypeError):
            rate = 0.0

        # Extract short name
        parts = exception_type.replace("\\", "/").split("/")
        short_name = parts[-1] if parts else exception_type

        print(f"{short_name:50} {rate:.6f}/s")

    print("-" * 60)


def test_enrichment_service(
    client: VictoriaMetricsClient,
    service: str,
    window: str,
) -> ExceptionBreakdown:
    """Test the ExceptionEnrichmentService."""
    print(f"\n{'='*60}")
    print(f"Testing ENRICHMENT SERVICE for service: {service}")
    print(f"{'='*60}\n")

    # Parse window to minutes
    lookback_minutes = int(window.rstrip('m'))
    anomaly_time = datetime.now()

    enrichment = ExceptionEnrichmentService(client, lookback_minutes=lookback_minutes)
    breakdown = enrichment.get_exception_breakdown(service, anomaly_timestamp=anomaly_time)

    print(f"Query successful: {breakdown.query_successful}")
    print(f"Timestamp: {breakdown.timestamp}")
    print(f"Total exception rate: {breakdown.total_exception_rate:.4f}/s")
    print(f"Exception types found: {len(breakdown.exceptions)}")

    if breakdown.error_message:
        print(f"Error: {breakdown.error_message}")

    print(f"\n{breakdown.get_summary_text()}")

    return breakdown


def test_anomaly_enrichment(
    client: VictoriaMetricsClient,
    service: str,
    window: str,
) -> None:
    """Test enriching a mock anomaly."""
    print(f"\n{'='*60}")
    print("Testing ANOMALY ENRICHMENT")
    print(f"{'='*60}\n")

    # Create a mock anomaly that should trigger enrichment
    mock_anomaly = {
        "pattern_name": "elevated_errors",
        "severity": "high",
        "current_metrics": {
            "request_rate": 100.0,
            "application_latency": 50.0,
            "error_rate": 0.05,  # 5% errors
        },
        "interpretation": "Error rate elevated above normal threshold.",
        "recommended_actions": [
            "CHECK: Application logs for error details",
            "VERIFY: Recent deployments",
        ],
        "contributing_metrics": ["error_rate"],
    }

    print("Mock anomaly (before enrichment):")
    print(json.dumps(mock_anomaly, indent=2))

    # Parse window to minutes
    lookback_minutes = int(window.rstrip('m'))
    anomaly_time = datetime.now()

    enrichment = ExceptionEnrichmentService(client, lookback_minutes=lookback_minutes)
    enriched = enrichment.enrich_anomaly(mock_anomaly, service, anomaly_timestamp=anomaly_time)

    print("\n" + "-" * 60)
    print("\nEnriched anomaly:")
    print(json.dumps(enriched, indent=2, default=str))


def list_services(client: VictoriaMetricsClient) -> None:
    """List available services with exception metrics."""
    print(f"\n{'='*60}")
    print("Listing AVAILABLE SERVICES with exception metrics")
    print(f"{'='*60}\n")

    query = 'sum(rate(events_total{deployment_environment_name=~"production"}[5m])) by (service_name)'
    print(f"Query:\n{query}\n")

    result = client.query(query)
    data = result.get("data", {}).get("result", [])

    if not data:
        print("No services found with events_total metric.")
        return

    print(f"Found {len(data)} services:\n")
    print(f"{'Service Name':40} {'Exception Rate':>15}")
    print("-" * 60)

    services = []
    for series in data:
        service_name = series.get("metric", {}).get("service_name", "unknown")
        value = series.get("value", [None, "0"])
        try:
            rate = float(value[1]) if len(value) > 1 else 0.0
        except (ValueError, TypeError):
            rate = 0.0
        services.append((service_name, rate))

    # Sort by rate descending
    services.sort(key=lambda x: x[1], reverse=True)

    for service_name, rate in services:
        print(f"{service_name:40} {rate:>12.4f}/s")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test exception enrichment prototype",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--service",
        default="search",
        help="Service name to query (default: search)",
    )
    parser.add_argument(
        "--window",
        default="5m",
        help="Time window for rate calculation (default: 5m)",
    )
    parser.add_argument(
        "--raw-query",
        action="store_true",
        help="Only test the raw PromQL query",
    )
    parser.add_argument(
        "--list-services",
        action="store_true",
        help="List available services with exception metrics",
    )
    parser.add_argument(
        "--endpoint",
        help="VictoriaMetrics endpoint (overrides config)",
    )
    parser.add_argument(
        "--full-test",
        action="store_true",
        help="Run all tests including anomaly enrichment",
    )

    args = parser.parse_args()

    # Get config
    config = get_config()
    endpoint = args.endpoint or config.victoria_metrics.endpoint

    print(f"VictoriaMetrics endpoint: {endpoint}")
    print(f"Service: {args.service}")
    print(f"Window: {args.window}")

    # Create client
    client = VictoriaMetricsClient(endpoint=endpoint)

    # Check health first
    healthy, details = client.health_check()
    if not healthy:
        print(f"\n⚠️  VictoriaMetrics health check failed: {details.get('error')}")
        print("Continuing anyway...\n")

    try:
        if args.list_services:
            list_services(client)
            return 0

        if args.raw_query:
            test_raw_query(client, args.service, args.window)
            return 0

        # Run enrichment service test
        breakdown = test_enrichment_service(client, args.service, args.window)

        if args.full_test:
            test_anomaly_enrichment(client, args.service, args.window)

        # Print JSON output for programmatic use
        print(f"\n{'='*60}")
        print("JSON OUTPUT")
        print(f"{'='*60}\n")
        print(json.dumps(breakdown.to_dict(), indent=2))

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
