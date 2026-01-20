#!/usr/bin/env python3
"""
Prefetch metrics data from VictoriaMetrics into local parquet cache.

This script fetches metrics data slowly (in chunks) and caches it locally
for later use in training. This allows:
- Higher resolution data (1m step) without VM timeouts
- Faster training (no network queries)
- Incremental updates (only fetch new data)

Usage:
    # Prefetch all services for Aug-Oct 2025
    python prefetch_metrics.py --start-date 2025-08-01 --end-date 2025-10-31

    # Prefetch specific service
    python prefetch_metrics.py --service booking --start-date 2025-08-01 --end-date 2025-10-31

    # Check cache status
    python prefetch_metrics.py --status

    # Clear cache
    python prefetch_metrics.py --clear
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from smartbox_anomaly.core.config import get_config
from smartbox_anomaly.core.logging import configure_logging, get_logger
from smartbox_anomaly.metrics.client import VictoriaMetricsClient
from smartbox_anomaly.metrics.cache import MetricsCache

logger = get_logger(__name__)


def parse_date(date_str: str) -> datetime:
    """Parse date string in YYYY-MM-DD format."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: {date_str}. Use YYYY-MM-DD.")


def load_services(config_path: str = "./config.json") -> list[str]:
    """Load service list from config file."""
    with open(config_path) as f:
        config = json.load(f)

    services = []
    for category in ["critical", "standard", "micro", "admin", "core"]:
        services.extend(config.get("services", {}).get(category, []))

    return services


def show_status(cache: MetricsCache, services: list[str]) -> None:
    """Show cache status for all services."""
    print("\nCache Status")
    print("=" * 80)

    for service in services:
        status = cache.get_cache_status(service)
        cached_metrics = [m for m, info in status["metrics"].items() if info.get("cached")]

        if cached_metrics:
            # Get date range from first cached metric
            first_metric = cached_metrics[0]
            info = status["metrics"][first_metric]
            start = info.get("start", "")[:10]
            end = info.get("end", "")[:10]
            points = sum(
                status["metrics"][m].get("points", 0)
                for m in cached_metrics
            )
            print(f"  {service:20} {len(cached_metrics)}/5 metrics  {start} to {end}  ({points:,} points)")
        else:
            print(f"  {service:20} No cache")


def prefetch_service(
    cache: MetricsCache,
    service: str,
    start_date: datetime,
    end_date: datetime,
    force: bool = False,
) -> dict:
    """Prefetch a single service."""
    return cache.prefetch(
        service_name=service,
        start_date=start_date,
        end_date=end_date,
        force_refresh=force,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Prefetch metrics data from VictoriaMetrics into local cache",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prefetch all services for Aug-Oct 2025 (baseline without holiday traffic)
  python prefetch_metrics.py --start-date 2025-08-01 --end-date 2025-10-31

  # Prefetch specific service
  python prefetch_metrics.py --service booking --start-date 2025-08-01 --end-date 2025-10-31

  # Prefetch with force refresh (re-download even if cached)
  python prefetch_metrics.py --service booking --start-date 2025-08-01 --end-date 2025-10-31 --force

  # Check cache status
  python prefetch_metrics.py --status

  # Clear all cache
  python prefetch_metrics.py --clear

  # Clear cache for specific service
  python prefetch_metrics.py --clear --service booking
        """,
    )

    parser.add_argument("--service", "-s", type=str, help="Prefetch specific service (default: all)")
    parser.add_argument("--start-date", type=parse_date, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=parse_date, help="End date (YYYY-MM-DD)")
    parser.add_argument("--status", action="store_true", help="Show cache status")
    parser.add_argument("--clear", action="store_true", help="Clear cache")
    parser.add_argument("--force", "-f", action="store_true", help="Force re-fetch even if cached")
    parser.add_argument("--cache-dir", type=str, default="./metrics_cache", help="Cache directory")
    parser.add_argument("--step", type=str, default="5m", help="Query step (default: 5m)")
    parser.add_argument("--chunk-days", type=int, default=7, help="Days per fetch chunk (default: 7)")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between chunks in seconds")
    parser.add_argument("--delay-metrics", type=float, default=0.5, help="Delay between metrics in seconds (default: 0.5)")
    parser.add_argument("--delay-services", type=float, default=2.0, help="Delay between services in seconds (default: 2.0)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Configure logging
    log_level = "DEBUG" if args.verbose else "INFO"
    configure_logging(log_level)

    # Load config and services
    config = get_config()
    services = load_services()

    # Initialize VM client and cache
    vm_client = VictoriaMetricsClient(
        endpoint=config.victoria_metrics.endpoint,
        timeout=config.victoria_metrics.timeout_seconds,
    )

    cache = MetricsCache(
        vm_client=vm_client,
        cache_dir=args.cache_dir,
        step=args.step,
        chunk_days=args.chunk_days,
        delay_between_chunks=args.delay,
        delay_between_metrics=args.delay_metrics,
    )

    # Handle clear command
    if args.clear:
        if args.service:
            cache.clear_cache(args.service)
            print(f"Cleared cache for {args.service}")
        else:
            cache.clear_cache()
            print("Cleared all cache")
        return

    # Handle status command
    if args.status:
        show_status(cache, services)
        return

    # Prefetch requires date range
    if not args.start_date or not args.end_date:
        parser.error("--start-date and --end-date are required for prefetch")

    # Determine which services to prefetch
    if args.service:
        target_services = [args.service]
    else:
        target_services = services

    # Calculate total days
    total_days = (args.end_date - args.start_date).days
    print(f"\nPrefetching {len(target_services)} services")
    print(f"Date range: {args.start_date.date()} to {args.end_date.date()} ({total_days} days)")
    print(f"Step: {args.step}, Chunk: {args.chunk_days} days")
    print(f"Cache directory: {args.cache_dir}")
    print("=" * 80)

    # Prefetch each service
    results = {}
    for i, service in enumerate(target_services, 1):
        print(f"\n[{i}/{len(target_services)}] {service}")

        try:
            result = prefetch_service(
                cache=cache,
                service=service,
                start_date=args.start_date,
                end_date=args.end_date,
                force=args.force,
            )
            results[service] = result

            # Show result
            total_points = sum(result.values())
            cached_metrics = sum(1 for v in result.values() if v > 0)
            print(f"  → Cached {cached_metrics}/5 metrics ({total_points:,} points)")

        except Exception as e:
            logger.error(f"  Error prefetching {service}: {e}")
            results[service] = {"error": str(e)}

        # Delay between services to avoid hammering VictoriaMetrics
        if i < len(target_services) and args.delay_services > 0:
            time.sleep(args.delay_services)

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    successful = [s for s, r in results.items() if "error" not in r]
    failed = [s for s, r in results.items() if "error" in r]

    print(f"  Successful: {len(successful)}/{len(target_services)}")
    if failed:
        print(f"  Failed: {', '.join(failed)}")

    # Show cache status
    show_status(cache, target_services)


if __name__ == "__main__":
    main()
