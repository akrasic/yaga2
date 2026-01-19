"""
Metrics cache for storing VictoriaMetrics data locally in parquet files.

This module provides a caching layer that:
- Fetches data from VictoriaMetrics in chunks to avoid timeouts
- Stores data in parquet files for fast local access
- Supports incremental updates (only fetch missing data)
- Enables high-resolution (1m step) data collection

Uses Polars internally for fast data manipulation, returns pandas DataFrames
for compatibility with the rest of the codebase.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import polars as pl

from smartbox_anomaly.core.logging import get_logger

if TYPE_CHECKING:
    from smartbox_anomaly.metrics.client import VictoriaMetricsClient

logger = get_logger(__name__)

# Default queries for each metric
# Note: Uses 5m rate windows for stability (step should be >= rate window)
DEFAULT_QUERIES = {
    "request_rate": 'http_requests:count:rate_5m',
    "application_latency": """
        sum(rate(duration_milliseconds_sum{span_kind="SPAN_KIND_SERVER", deployment_environment_name=~"production"}[5m])) by (service_name)
        /
        sum(rate(duration_milliseconds_count{span_kind="SPAN_KIND_SERVER", deployment_environment_name=~"production"}[5m])) by (service_name)
    """,
    "dependency_latency": """
        sum(rate(duration_milliseconds_sum{span_kind="SPAN_KIND_CLIENT", deployment_environment_name=~"production", db_system="", db_system_name=""}[5m])) by (service_name)
        /
        sum(rate(duration_milliseconds_count{span_kind="SPAN_KIND_CLIENT", deployment_environment_name=~"production", db_system="", db_system_name=""}[5m])) by (service_name)
    """,
    "database_latency": """
        sum(rate(duration_milliseconds_sum{span_kind="SPAN_KIND_CLIENT", deployment_environment_name=~"production", db_system_name!=""}[5m])) by (service_name)
        /
        sum(rate(duration_milliseconds_count{span_kind="SPAN_KIND_CLIENT", deployment_environment_name=~"production", db_system_name!=""}[5m])) by (service_name)
    """,
    "error_rate": """
        sum(rate(duration_milliseconds_count{span_kind="SPAN_KIND_SERVER", deployment_environment_name=~"production", http_response_status_code=~"5.*|"}[5m])) by (service_name)
        /
        sum(rate(duration_milliseconds_count{span_kind="SPAN_KIND_SERVER", deployment_environment_name=~"production"}[5m])) by (service_name)
    """,
}


class MetricsCache:
    """Cache for storing VictoriaMetrics data in parquet files.

    Directory structure:
        cache_dir/
        ├── booking/
        │   ├── request_rate.parquet
        │   ├── application_latency.parquet
        │   ├── dependency_latency.parquet
        │   ├── database_latency.parquet
        │   ├── error_rate.parquet
        │   └── metadata.json
        ├── search/
        │   └── ...
    """

    def __init__(
        self,
        vm_client: VictoriaMetricsClient,
        cache_dir: str | Path = "./metrics_cache",
        step: str = "1m",
        chunk_days: int = 7,
        delay_between_chunks: float = 1.0,
        delay_between_metrics: float = 0.5,
    ):
        """Initialize the metrics cache.

        Args:
            vm_client: VictoriaMetrics client for fetching data.
            cache_dir: Directory to store cached parquet files.
            step: Query step size (default: 1m for high resolution).
            chunk_days: Days per chunk when fetching (default: 7).
            delay_between_chunks: Seconds to wait between chunk fetches.
            delay_between_metrics: Seconds to wait between different metrics.
        """
        self.vm_client = vm_client
        self.cache_dir = Path(cache_dir)
        self.step = step
        self.chunk_days = chunk_days
        self.delay_between_chunks = delay_between_chunks
        self.delay_between_metrics = delay_between_metrics
        self.queries = DEFAULT_QUERIES.copy()

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_service_dir(self, service_name: str) -> Path:
        """Get the cache directory for a service."""
        service_dir = self.cache_dir / service_name
        service_dir.mkdir(parents=True, exist_ok=True)
        return service_dir

    def _get_metadata_path(self, service_name: str) -> Path:
        """Get the metadata file path for a service."""
        return self._get_service_dir(service_name) / "metadata.json"

    def _get_parquet_path(self, service_name: str, metric_name: str) -> Path:
        """Get the parquet file path for a service metric."""
        return self._get_service_dir(service_name) / f"{metric_name}.parquet"

    def _load_metadata(self, service_name: str) -> dict:
        """Load metadata for a service."""
        metadata_path = self._get_metadata_path(service_name)
        if metadata_path.exists():
            with open(metadata_path) as f:
                return json.load(f)
        return {"metrics": {}}

    def _save_metadata(self, service_name: str, metadata: dict) -> None:
        """Save metadata for a service."""
        metadata_path = self._get_metadata_path(service_name)
        metadata["updated_at"] = datetime.now().isoformat()
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def _parse_metric_result(self, data: dict, metric_name: str) -> pl.DataFrame:
        """Parse VictoriaMetrics result into Polars DataFrame."""
        result = data.get("data", {}).get("result", [])
        if not result:
            return pl.DataFrame()

        # Combine all series (should be one per service after filtering)
        timestamps = []
        values = []
        for series in result:
            series_values = series.get("values", [])
            for ts, val in series_values:
                try:
                    timestamps.append(datetime.fromtimestamp(ts))
                    values.append(float(val) if val != "NaN" else None)
                except (ValueError, TypeError):
                    continue

        if not timestamps:
            return pl.DataFrame()

        df = pl.DataFrame({
            "timestamp": timestamps,
            metric_name: values,
        })
        # Sort by timestamp and deduplicate (keep first occurrence)
        df = df.sort("timestamp").unique(subset=["timestamp"], keep="first")
        return df

    def _fetch_metric_chunk(
        self,
        service_name: str,
        metric_name: str,
        start_time: datetime,
        end_time: datetime,
    ) -> pl.DataFrame:
        """Fetch a single chunk of metric data."""
        base_query = self.queries.get(metric_name)
        if not base_query:
            logger.warning(f"Unknown metric: {metric_name}")
            return pl.DataFrame()

        # Clean up query
        clean_query = " ".join(base_query.split())

        # Add service filter
        if "service_name" in clean_query and "by (service_name)" in clean_query:
            query = clean_query.replace(
                'deployment_environment_name=~"production"',
                f'deployment_environment_name=~"production", service_name="{service_name}"',
            )
        else:
            query = f'{clean_query}{{service_name="{service_name}"}}'

        # Execute query
        result = self.vm_client.query_range(
            query=query,
            start_time=start_time,
            end_time=end_time,
            step=self.step,
            timeout_seconds=120,
        )

        if not result.success:
            logger.error(f"  {metric_name}: Query failed - {result.error_message}")
            return pl.DataFrame()

        return self._parse_metric_result(result.data, metric_name)

    def prefetch(
        self,
        service_name: str,
        start_date: datetime,
        end_date: datetime,
        metrics: list[str] | None = None,
        force_refresh: bool = False,
    ) -> dict[str, int]:
        """Prefetch metrics data for a service and cache to parquet.

        Fetches data in chunks to avoid VM timeouts. Only fetches data
        that isn't already cached (unless force_refresh=True).

        Args:
            service_name: Service to fetch data for.
            start_date: Start of date range.
            end_date: End of date range.
            metrics: List of metrics to fetch (default: all).
            force_refresh: If True, re-fetch even if cached.

        Returns:
            Dict mapping metric names to number of data points fetched.
        """
        if metrics is None:
            metrics = list(self.queries.keys())

        metadata = self._load_metadata(service_name)
        results = {}

        logger.info(f"Prefetching {service_name} from {start_date.date()} to {end_date.date()}")

        for metric_idx, metric_name in enumerate(metrics):
            parquet_path = self._get_parquet_path(service_name, metric_name)

            # Determine what date range we need to fetch
            existing_df = None
            fetch_start = start_date
            fetch_end = end_date

            # Track what ranges need fetching
            fetch_ranges: list[tuple[datetime, datetime]] = []

            if not force_refresh and parquet_path.exists():
                existing_df = pl.read_parquet(parquet_path)
                metric_meta = metadata.get("metrics", {}).get(metric_name, {})

                if metric_meta:
                    cached_start = datetime.fromisoformat(metric_meta["start"])
                    cached_end = datetime.fromisoformat(metric_meta["end"])

                    # Check if fully cached
                    if start_date >= cached_start and end_date <= cached_end:
                        logger.info(f"  {metric_name}: Already cached ({len(existing_df)} points)")
                        results[metric_name] = len(existing_df)
                        continue

                    # Build list of ranges to fetch (can be both before and after)
                    if start_date < cached_start:
                        fetch_ranges.append((start_date, cached_start))
                    if end_date > cached_end:
                        fetch_ranges.append((cached_end, end_date))
                else:
                    # No metadata, fetch everything
                    fetch_ranges.append((start_date, end_date))
            else:
                # No cache or force refresh
                fetch_ranges.append((start_date, end_date))

            if not fetch_ranges:
                fetch_ranges.append((start_date, end_date))

            # Log what we're fetching
            range_strs = [f"{s.date()} to {e.date()}" for s, e in fetch_ranges]
            logger.info(f"  {metric_name}: Fetching {', '.join(range_strs)}...")

            chunks = []
            chunk_num = 0

            for range_start, range_end in fetch_ranges:
                current_start = range_start

                while current_start < range_end:
                    chunk_end = min(current_start + timedelta(days=self.chunk_days), range_end)
                    chunk_num += 1

                    logger.debug(f"    Chunk {chunk_num}: {current_start.date()} to {chunk_end.date()}")

                    chunk_df = self._fetch_metric_chunk(
                        service_name, metric_name, current_start, chunk_end
                    )

                    if not chunk_df.is_empty():
                        chunks.append(chunk_df)
                        logger.debug(f"    Chunk {chunk_num}: {len(chunk_df)} points")

                    current_start = chunk_end

                    # Delay between chunks to be nice to VM
                    if current_start < range_end:
                        time.sleep(self.delay_between_chunks)

            # Combine chunks with existing data
            if chunks:
                new_df = pl.concat(chunks)

                if existing_df is not None and not existing_df.is_empty():
                    # Ensure consistent column order before concat
                    cols = new_df.columns
                    existing_df = existing_df.select(cols)
                    combined_df = pl.concat([existing_df, new_df])
                    # Deduplicate by timestamp, keeping last value
                    combined_df = combined_df.unique(subset=["timestamp"], keep="last")
                    combined_df = combined_df.sort("timestamp")
                else:
                    combined_df = new_df.sort("timestamp")

                # Save to parquet
                combined_df.write_parquet(parquet_path)

                # Update metadata
                if "metrics" not in metadata:
                    metadata["metrics"] = {}
                ts_col = combined_df["timestamp"]
                metadata["metrics"][metric_name] = {
                    "start": ts_col.min().isoformat(),
                    "end": ts_col.max().isoformat(),
                    "points": len(combined_df),
                    "step": self.step,
                }

                logger.info(f"  {metric_name}: Cached {len(combined_df)} points")
                results[metric_name] = len(combined_df)
            else:
                logger.warning(f"  {metric_name}: No data fetched")
                results[metric_name] = len(existing_df) if existing_df is not None else 0

            # Delay between metrics to avoid hammering VictoriaMetrics
            if metric_idx < len(metrics) - 1 and self.delay_between_metrics > 0:
                time.sleep(self.delay_between_metrics)

        # Save metadata
        self._save_metadata(service_name, metadata)

        return results

    def get_data(
        self,
        service_name: str,
        start_date: datetime,
        end_date: datetime,
        metrics: list[str] | None = None,
        auto_fetch: bool = True,
    ) -> pd.DataFrame:
        """Get cached metrics data for a service.

        Args:
            service_name: Service to get data for.
            start_date: Start of date range.
            end_date: End of date range.
            metrics: List of metrics to include (default: all available).
            auto_fetch: If True, fetch missing data automatically.

        Returns:
            DataFrame with timestamp index and metric columns.
        """
        if metrics is None:
            metrics = list(self.queries.keys())

        # Check if we need to fetch
        if auto_fetch:
            metadata = self._load_metadata(service_name)
            needs_fetch = False

            for metric_name in metrics:
                metric_meta = metadata.get("metrics", {}).get(metric_name, {})
                if not metric_meta:
                    needs_fetch = True
                    break

                cached_start = datetime.fromisoformat(metric_meta["start"])
                cached_end = datetime.fromisoformat(metric_meta["end"])

                if start_date < cached_start or end_date > cached_end:
                    needs_fetch = True
                    break

            if needs_fetch:
                self.prefetch(service_name, start_date, end_date, metrics)

        # Load from parquet files using Polars for speed
        dfs: list[pl.DataFrame] = []
        for metric_name in metrics:
            parquet_path = self._get_parquet_path(service_name, metric_name)
            if parquet_path.exists():
                df = pl.read_parquet(parquet_path)
                # Filter to requested date range
                df = df.filter(
                    (pl.col("timestamp") >= start_date) &
                    (pl.col("timestamp") <= end_date)
                )
                # Remove duplicate timestamps (keep last value)
                df = df.unique(subset=["timestamp"], keep="last")
                if not df.is_empty():
                    dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        # Combine all metrics using outer join on timestamp
        combined = dfs[0]
        for df in dfs[1:]:
            combined = combined.join(df, on="timestamp", how="full", coalesce=True)
        combined = combined.sort("timestamp")

        # Convert to pandas with timestamp as index for API compatibility
        pandas_df = combined.to_pandas()
        pandas_df = pandas_df.set_index("timestamp")
        return pandas_df

    def get_cache_status(self, service_name: str) -> dict:
        """Get cache status for a service.

        Returns:
            Dict with cache information per metric.
        """
        metadata = self._load_metadata(service_name)
        status = {
            "service": service_name,
            "metrics": {},
        }

        for metric_name in self.queries:
            parquet_path = self._get_parquet_path(service_name, metric_name)
            metric_meta = metadata.get("metrics", {}).get(metric_name, {})

            if parquet_path.exists() and metric_meta:
                status["metrics"][metric_name] = {
                    "cached": True,
                    "start": metric_meta.get("start"),
                    "end": metric_meta.get("end"),
                    "points": metric_meta.get("points"),
                    "step": metric_meta.get("step"),
                }
            else:
                status["metrics"][metric_name] = {"cached": False}

        return status

    def clear_cache(self, service_name: str | None = None) -> None:
        """Clear cached data.

        Args:
            service_name: Service to clear (None = all services).
        """
        if service_name:
            service_dir = self._get_service_dir(service_name)
            if service_dir.exists():
                import shutil
                shutil.rmtree(service_dir)
                logger.info(f"Cleared cache for {service_name}")
        else:
            import shutil
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Cleared all cache")
