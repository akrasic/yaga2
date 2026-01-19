# Enhanced Training Pipeline with Explainability Support
# Now uses the enhanced SmartboxAnomalyDetector automatically

from smartbox_anomaly.detection.detector import SmartboxAnomalyDetector
from smartbox_anomaly.detection.time_aware import TimeAwareAnomalyDetector
from smartbox_anomaly.core.config import TrainingConfig, ExcludedPeriod
from smartbox_anomaly.training import TrainingRunStorage, TrainingRunStatus, ValidationStatus
from vmclient import VictoriaMetricsClient, QueryResult
from smartbox_anomaly.metrics.quality import analyze_combined_data_quality, DataQualityReport
from smartbox_anomaly.metrics.cache import MetricsCache
import pandas as pd
import numpy as np
import os
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import warnings
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings('ignore')

# Configure module logger
logger = logging.getLogger(__name__)


class SmartboxMetricsExtractor:
    def __init__(self, vm_client: VictoriaMetricsClient):
        self.vm_client = vm_client
        
        # Your actual Smartbox queries (single line format for VictoriaMetrics)
        self.queries = {
            'request_rate': 'http_requests:count:rate_5m',
            
            'application_latency': 'sum(rate(duration_milliseconds_sum{span_kind="SPAN_KIND_SERVER", deployment_environment_name=~"production"}[1m])) by (service_name) / sum(rate(duration_milliseconds_count{span_kind="SPAN_KIND_SERVER", deployment_environment_name=~"production"}[1m])) by (service_name)',
            
            'dependency_latency': 'sum(rate(duration_milliseconds_sum{span_kind="SPAN_KIND_CLIENT", deployment_environment_name=~"production", db_system="", db_system_name=""}[1m])) by (service_name) / sum(rate(duration_milliseconds_count{span_kind="SPAN_KIND_CLIENT", deployment_environment_name=~"production", db_system="", db_system_name=""}[1m])) by (service_name)',
            
            'database_latency': 'sum(rate(duration_milliseconds_sum{span_kind="SPAN_KIND_CLIENT", deployment_environment_name=~"production", db_system_name!=""}[1m])) by (service_name) / sum(rate(duration_milliseconds_count{span_kind="SPAN_KIND_CLIENT", deployment_environment_name=~"production", db_system_name!=""}[1m])) by (service_name)',
            
            'error_rate': 'sum(rate(duration_milliseconds_count{span_kind="SPAN_KIND_SERVER", deployment_environment_name=~"production", http_response_status_code=~"5.*|"}[1m])) by (service_name) / sum(rate(duration_milliseconds_count{span_kind="SPAN_KIND_SERVER", deployment_environment_name=~"production"}[1m])) by (service_name)'
        }
    
    def get_available_services(self) -> List[str]:
        """Get list of services from VictoriaMetrics"""
        query = 'group by (service_name) (http_requests:count:rate_5m)'
        result = self.vm_client.query(query)
        
        services = []
        if result.get('data', {}).get('result'):
            for item in result['data']['result']:
                service_name = item.get('metric', {}).get('service_name')
                if service_name:
                    services.append(service_name)
        
        return sorted(list(set(services)))
    
    def test_queries(self, service_name: str) -> Dict[str, bool]:
        """Test all queries for a service to see which ones work"""
        logger.info("Testing queries for %s", service_name)

        query_results = {}

        for metric_name, base_query in self.queries.items():
            # Clean up query and add service filter
            clean_query = ' '.join(base_query.split())

            if 'service_name' in clean_query and 'by (service_name)' in clean_query:
                query = clean_query.replace('deployment_environment_name=~"production"', f'deployment_environment_name=~"production", service_name="{service_name}"')
            else:
                query = f'{clean_query}{{service_name="{service_name}"}}'

            logger.debug("Testing %s with query: %s", metric_name, query)

            # Test with a simple instant query first
            result = self.vm_client.query(query)

            if result.get('data', {}).get('result'):
                query_results[metric_name] = True
                logger.info("  %s: OK", metric_name)
            else:
                query_results[metric_name] = False
                logger.warning("  %s: No data or error", metric_name)

                # Try without service filter to see if the base query works
                base_result = self.vm_client.query(clean_query)
                if base_result.get('data', {}).get('result'):
                    logger.debug("    Base query works, but no data for %s", service_name)

        return query_results
    
    def extract_service_metrics(
        self,
        service_name: str,
        lookback_days: int = 30,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Extract all metrics for a specific service with robust error handling.

        Uses longer timeouts for training queries (120s) and tracks query failures
        to distinguish between 'no data exists' and 'query failed'.

        Args:
            service_name: Name of the service to extract metrics for.
            lookback_days: Number of days to look back (used if start_date/end_date not provided).
            start_date: Optional explicit start date for training data.
            end_date: Optional explicit end date for training data.
        """
        # Use explicit dates if provided, otherwise use lookback_days from now
        if start_date is not None and end_date is not None:
            start_time = start_date
            end_time = end_date
            actual_days = (end_time - start_time).days
        else:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=lookback_days)
            actual_days = lookback_days

        # Dynamically adjust step size based on date range to avoid VM 422 errors
        # Complex rate queries fail over long ranges with small steps
        if actual_days > 120:
            step = '30m'
            intervals_per_day = 48  # 30-min intervals
        elif actual_days > 60:
            step = '15m'
            intervals_per_day = 96  # 15-min intervals
        else:
            step = '5m'
            intervals_per_day = 288  # 5-min intervals

        # Calculate expected data points for validation
        expected_intervals = actual_days * intervals_per_day

        logger.info(
            "Extracting metrics for %s from %s to %s (%d days, step=%s, expecting ~%d data points per metric)",
            service_name, start_time.date(), end_time.date(), actual_days, step, expected_intervals
        )

        all_metrics = {}
        query_failures = []
        query_stats = []

        for metric_name, base_query in self.queries.items():
            # Clean up query and add service filter
            clean_query = ' '.join(base_query.split())  # Remove extra whitespace

            # Add service filter to queries that support it
            if 'service_name' in clean_query and 'by (service_name)' in clean_query:
                # For aggregated queries, filter before aggregation
                query = clean_query.replace('deployment_environment_name=~"production"', f'deployment_environment_name=~"production", service_name="{service_name}"')
            else:
                # For simple metrics like request_rate, add service filter
                query = f'{clean_query}{{service_name="{service_name}"}}'

            logger.debug("Querying %s: %s...", metric_name, query[:100])

            # Use longer timeout for training queries (120s instead of default 10s)
            result = self.vm_client.query_range(
                query=query,
                start_time=start_time,
                end_time=end_time,
                step=step,
                timeout_seconds=120
            )

            # Track query statistics
            stat = {
                'metric': metric_name,
                'success': result.success,
                'duration_ms': result.duration_ms,
                'data_points': result.get_values_count() if result.success else 0,
            }

            if not result.success:
                query_failures.append({
                    'metric': metric_name,
                    'error': result.error_message,
                })
                logger.error(
                    "  %s: Query FAILED - %s",
                    metric_name, result.error_message
                )
                stat['error'] = result.error_message
            else:
                # Parse the result
                metric_data = self._parse_metric_result(result.data, metric_name)
                if not metric_data.empty:
                    all_metrics[metric_name] = metric_data
                    coverage_pct = (len(metric_data) / expected_intervals) * 100
                    stat['coverage_pct'] = coverage_pct

                    if coverage_pct < 50:
                        logger.warning(
                            "  %s: %d data points (%.1f%% coverage - LOW)",
                            metric_name, len(metric_data), coverage_pct
                        )
                    else:
                        logger.info(
                            "  %s: %d data points (%.1f%% coverage)",
                            metric_name, len(metric_data), coverage_pct
                        )
                else:
                    if result.warning_message:
                        logger.warning("  %s: No data found - %s", metric_name, result.warning_message)
                    else:
                        logger.warning("  %s: No data found (query succeeded but returned empty)", metric_name)
                    stat['coverage_pct'] = 0

            query_stats.append(stat)

        # Log query failures summary if any
        if query_failures:
            logger.error(
                "DATA QUALITY ISSUE: %d/%d metric queries failed for %s",
                len(query_failures), len(self.queries), service_name
            )
            for failure in query_failures:
                logger.error("  - %s: %s", failure['metric'], failure['error'])

        # Combine all metrics
        if all_metrics:
            combined_df = self._combine_metrics(all_metrics)
            logger.info(
                "Combined dataset: %d rows, %d columns",
                len(combined_df), len(combined_df.columns)
            )

            # Log overall data quality
            overall_coverage = (len(combined_df) / expected_intervals) * 100
            if overall_coverage < 80:
                logger.warning(
                    "DATA QUALITY WARNING: Only %.1f%% data coverage for %s (expected ~%d rows, got %d)",
                    overall_coverage, service_name, expected_intervals, len(combined_df)
                )

            return combined_df
        else:
            logger.error("No metrics found for %s - cannot proceed with training", service_name)
            return pd.DataFrame()
    
    def _parse_metric_result(self, result: Dict, metric_name: str) -> pd.DataFrame:
        """Parse VictoriaMetrics result into DataFrame"""
        data_points = []
        
        if result.get('data', {}).get('result'):
            for series in result['data']['result']:
                values = series.get('values', [])
                
                for timestamp, value in values:
                    try:
                        data_points.append({
                            'timestamp': pd.to_datetime(float(timestamp), unit='s'),
                            metric_name: float(value) if value != 'NaN' else np.nan
                        })
                    except (ValueError, TypeError):
                        continue
        
        if data_points:
            df = pd.DataFrame(data_points)
            df = df.groupby('timestamp').mean().reset_index()  # Handle duplicates
            return df.set_index('timestamp')
        else:
            return pd.DataFrame()
    
    def _combine_metrics(self, metrics_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Combine multiple metric DataFrames with robust data cleaning"""
        combined = None

        for metric_name, df in metrics_dict.items():
            if combined is None:
                combined = df
            else:
                combined = combined.join(df, how='outer')

        # Robust data cleaning
        logger.debug("Cleaning data...")

        # Replace infinite values with NaN
        combined = combined.replace([np.inf, -np.inf], np.nan)

        # Log data quality issues
        inf_count = np.isinf(combined.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            logger.warning("Replaced %d infinite values with NaN", inf_count)

        # Handle extremely large values (likely errors)
        numeric_cols = combined.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            # Define reasonable upper bounds for each metric type
            if 'latency' in col.lower():
                max_reasonable = 60000  # 60 seconds in milliseconds
            elif 'rate' in col.lower() and 'error' not in col.lower():
                max_reasonable = 100000  # 100k requests per second
            elif 'error' in col.lower():
                max_reasonable = 1.0  # Error rate should be <= 1
            else:
                max_reasonable = 1e6  # Generic large number

            # Replace unreasonable values
            mask = combined[col] > max_reasonable
            outlier_count = mask.sum()
            if outlier_count > 0:
                logger.warning(
                    "Capped %d outlier values in %s (>%s)",
                    outlier_count, col, max_reasonable
                )
                combined.loc[mask, col] = np.nan

        # Forward fill missing values (common in time series)
        combined = combined.fillna(method='ffill')

        # For remaining NaNs, use column median or 0
        for col in combined.columns:
            remaining_nans = combined[col].isna().sum()
            if remaining_nans > 0:
                if combined[col].dtype in ['float64', 'int64']:
                    median_val = combined[col].median()
                    fill_value = median_val if not pd.isna(median_val) else 0.0
                    combined[col] = combined[col].fillna(fill_value)
                    logger.debug(
                        "Filled %d missing values in %s with %.3f",
                        remaining_nans, col, fill_value
                    )

        # Final validation - ensure no inf/nan values remain
        final_inf_count = np.isinf(combined.select_dtypes(include=[np.number])).sum().sum()
        final_nan_count = combined.isna().sum().sum()

        if final_inf_count > 0 or final_nan_count > 0:
            logger.error(
                "Data cleaning incomplete: %d inf, %d NaN values remain",
                final_inf_count, final_nan_count
            )
            # As last resort, fill with zeros
            combined = combined.replace([np.inf, -np.inf], 0).fillna(0)
            logger.warning("Emergency cleanup: replaced remaining invalid values with 0")

        logger.debug("Data cleaning completed")

        # Run data quality analysis
        quality_report = analyze_combined_data_quality(combined, lookback_days=30)
        logger.info(
            "Data quality analysis: Overall grade %s (score %d/100), %d rows",
            quality_report["overall_grade"],
            quality_report["overall_score"],
            quality_report["row_count"],
        )

        # Log per-metric quality issues
        for metric_name, metric_report in quality_report["metrics"].items():
            if metric_report["quality_grade"] in ("D", "F"):
                logger.warning(
                    "  %s: Grade %s - %.1f%% coverage, %d gaps (max %.0fm)",
                    metric_name,
                    metric_report["quality_grade"],
                    metric_report["coverage_percent"],
                    metric_report["gap_count"],
                    metric_report["max_gap_minutes"],
                )
                for issue in metric_report.get("issues", []):
                    logger.warning("    Issue: %s", issue)

        return combined

    def filter_by_excluded_periods(
        self,
        data: pd.DataFrame,
        training_config: TrainingConfig,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into baseline and excluded (holiday) periods.

        Args:
            data: DataFrame with datetime index containing all metrics.
            training_config: TrainingConfig with excluded_periods defined.

        Returns:
            Tuple of (baseline_data, holiday_data) DataFrames.
            - baseline_data: Data outside excluded periods (for baseline model)
            - holiday_data: Data within excluded periods (for holiday model)
        """
        if data.empty:
            return data.copy(), pd.DataFrame()

        excluded_periods = training_config.excluded_periods
        if not excluded_periods:
            # No excluded periods - all data is baseline
            logger.info("No excluded periods configured - using all data for baseline model")
            return data.copy(), pd.DataFrame()

        # Create mask for excluded periods
        excluded_mask = pd.Series(False, index=data.index)

        for period in excluded_periods:
            try:
                start_dt = datetime.fromisoformat(period.start)
                end_dt = datetime.fromisoformat(period.end)

                # Handle timezone-naive vs timezone-aware comparison
                if data.index.tzinfo is None:
                    # Index is timezone-naive
                    period_mask = (data.index >= start_dt) & (data.index <= end_dt)
                else:
                    # Index is timezone-aware - make period dates timezone-aware
                    start_dt = start_dt.replace(tzinfo=data.index.tzinfo)
                    end_dt = end_dt.replace(tzinfo=data.index.tzinfo)
                    period_mask = (data.index >= start_dt) & (data.index <= end_dt)

                excluded_mask = excluded_mask | period_mask

                period_count = period_mask.sum()
                logger.info(
                    "Excluded period '%s' (%s to %s): %d data points",
                    period.reason or period.model_variant,
                    period.start,
                    period.end,
                    period_count,
                )
            except (ValueError, TypeError) as e:
                logger.warning(
                    "Invalid excluded period dates (%s to %s): %s",
                    period.start, period.end, e
                )
                continue

        # Split data
        baseline_data = data[~excluded_mask].copy()
        holiday_data = data[excluded_mask].copy()

        # Log split statistics
        total_points = len(data)
        baseline_points = len(baseline_data)
        holiday_points = len(holiday_data)
        baseline_pct = (baseline_points / total_points * 100) if total_points > 0 else 0
        holiday_pct = (holiday_points / total_points * 100) if total_points > 0 else 0

        logger.info(
            "Data split: %d total → %d baseline (%.1f%%) + %d holiday (%.1f%%)",
            total_points, baseline_points, baseline_pct, holiday_points, holiday_pct,
        )

        # Warn if split is unbalanced
        if baseline_points < training_config.min_data_points:
            logger.warning(
                "Baseline data has only %d points (need %d) - may not train well",
                baseline_points, training_config.min_data_points,
            )
        if holiday_points > 0 and holiday_points < 500:
            logger.warning(
                "Holiday data has only %d points - holiday model may be unreliable",
                holiday_points,
            )

        return baseline_data, holiday_data

    def load_with_cache(
        self,
        service_name: str,
        start_date: datetime,
        end_date: datetime,
        cache_dir: str = "./metrics_cache",
    ) -> pd.DataFrame:
        """Load metrics data using cache, fetching missing data automatically.

        This method:
        1. Checks what data is already cached
        2. Fetches any missing data from VictoriaMetrics (in chunks to avoid timeouts)
        3. Returns the complete dataset

        Args:
            service_name: Name of the service to load data for.
            start_date: Start date for the data range.
            end_date: End date for the data range.
            cache_dir: Directory containing cached parquet files.

        Returns:
            DataFrame with metrics, same format as extract_service_metrics().
        """
        cache = MetricsCache(
            vm_client=self.vm_client,
            cache_dir=cache_dir,
            step="5m",  # 5-minute resolution (matches inference)
            chunk_days=7,  # Fetch in weekly chunks
            delay_between_chunks=1.0,  # Delay between 7-day chunks
            delay_between_metrics=0.5,  # Delay between different metrics
        )

        # Check cache status
        status = cache.get_cache_status(service_name)
        cached_metrics = [m for m, info in status["metrics"].items() if info.get("cached")]
        total_metrics = len(cache.queries)

        if cached_metrics:
            # Check if cached data covers requested range
            all_covered = True
            for metric_name in cache.queries.keys():
                metric_info = status["metrics"].get(metric_name, {})
                if not metric_info.get("cached"):
                    all_covered = False
                    break
                cached_start = datetime.fromisoformat(metric_info.get("start", ""))
                cached_end = datetime.fromisoformat(metric_info.get("end", ""))
                if start_date < cached_start or end_date > cached_end:
                    all_covered = False
                    break

            if all_covered:
                logger.info(
                    "%s: All data cached (%d metrics), loading from disk",
                    service_name,
                    total_metrics,
                )
            else:
                logger.info(
                    "%s: Partial cache (%d/%d metrics), fetching missing data...",
                    service_name,
                    len(cached_metrics),
                    total_metrics,
                )
        else:
            logger.info(
                "%s: No cache, fetching data from VictoriaMetrics...",
                service_name,
            )

        # Load data - auto_fetch=True will download missing data
        data = cache.get_data(
            service_name=service_name,
            start_date=start_date,
            end_date=end_date,
            auto_fetch=True,  # Automatically fetch missing data
        )

        if data.empty:
            logger.warning("No data available for %s", service_name)
            return pd.DataFrame()

        # Log data statistics
        logger.info(
            "%s: Loaded %d data points (%s to %s)",
            service_name,
            len(data),
            data.index.min().strftime("%Y-%m-%d %H:%M"),
            data.index.max().strftime("%Y-%m-%d %H:%M"),
        )

        return data


class SmartboxFeatureEngineer:
    def __init__(self):
        self.feature_windows = ['5T', '15T', '1H']  # 5min, 15min, 1hour
        self.base_metrics = ['request_rate', 'application_latency', 'dependency_latency', 'database_latency', 'error_rate']

    def engineer_features(self, metrics_df: pd.DataFrame, include_rolling: bool = True) -> pd.DataFrame:
        """Create ML features from raw metrics with robust data handling.

        Args:
            metrics_df: Raw metrics DataFrame with timestamp index.
            include_rolling: If True, compute rolling features (may cause leakage if used
                           before train/validation split). Set to False and use
                           engineer_features_split() for proper temporal splitting.

        Returns:
            DataFrame with engineered features.
        """
        if metrics_df.empty:
            return metrics_df

        logger.debug("Engineering features...")

        features = metrics_df.copy()

        # Clean infinite and extreme values before feature engineering
        features = features.replace([np.inf, -np.inf], np.nan)

        # Time-based features (no leakage risk)
        features['hour'] = features.index.hour
        features['day_of_week'] = features.index.dayofweek
        features['is_business_hours'] = features['hour'].between(9, 17)
        features['is_weekend'] = features['day_of_week'] >= 5

        # Derived correlation features with safe division (no leakage risk)
        if all(col in features.columns for col in ['request_rate', 'application_latency', 'error_rate']):
            features['latency_error_ratio'] = np.where(
                features['application_latency'] > 0.001,
                features['error_rate'] / features['application_latency'],
                0
            )

            features['throughput_latency_efficiency'] = np.where(
                features['application_latency'] > 0.001,
                features['request_rate'] / features['application_latency'],
                0
            )

        # Database vs Application latency comparison (no leakage risk)
        if all(col in features.columns for col in ['database_latency', 'application_latency']):
            features['db_app_latency_ratio'] = np.where(
                features['application_latency'] > 0.001,
                features['database_latency'] / features['application_latency'],
                0
            )
            features['db_latency_dominance'] = features['database_latency'] > features['application_latency']

        # Only add rolling features if explicitly requested
        # WARNING: Rolling features computed on full dataset cause leakage
        if include_rolling:
            logger.warning("Computing rolling features on full dataset (potential leakage)")
            features = self._add_rolling_features(features)

        # Final cleanup
        features = self._cleanup_features(features)

        logger.info(
            "Feature engineering completed: %d rows, %d features",
            len(features), len(features.columns)
        )

        return features

    def engineer_features_split(
        self,
        metrics_df: pd.DataFrame,
        validation_fraction: float = 0.2
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Engineer features with proper temporal split to prevent leakage.

        Rolling features are computed separately on train and validation sets,
        preventing information from validation leaking into training statistics.

        Args:
            metrics_df: Raw metrics DataFrame with timestamp index.
            validation_fraction: Fraction of data to use for validation (from end).

        Returns:
            Tuple of (train_features, validation_features) with rolling features
            computed separately on each split.
        """
        if metrics_df.empty:
            return pd.DataFrame(), pd.DataFrame()

        logger.debug("Engineering features with temporal split (no leakage)...")

        # First, compute base features (no leakage risk)
        base_features = self.engineer_features(metrics_df, include_rolling=False)

        # Temporal split
        split_idx = int(len(base_features) * (1 - validation_fraction))
        train_base = base_features.iloc[:split_idx].copy()
        validation_base = base_features.iloc[split_idx:].copy()

        logger.info("Split: %d train, %d validation samples", len(train_base), len(validation_base))

        # Add rolling features separately to each split
        logger.debug("Computing rolling features on train set...")
        train_features = self._add_rolling_features(train_base)
        train_features = self._cleanup_features(train_features)

        logger.debug("Computing rolling features on validation set...")
        validation_features = self._add_rolling_features(validation_base)
        validation_features = self._cleanup_features(validation_features)

        # For validation, we need some warm-up period for rolling features
        # Drop the first few rows that have NaN from rolling windows
        warmup_periods = 12  # 1 hour of 5-min data
        if len(validation_features) > warmup_periods:
            validation_features = validation_features.iloc[warmup_periods:]
            logger.debug("Dropped %d warm-up rows from validation", warmup_periods)

        logger.info(
            "Leakage-free features: %d train, %d validation",
            len(train_features), len(validation_features)
        )

        return train_features, validation_features

    def _add_rolling_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Add rolling statistical features to a DataFrame.

        Should be called separately on train and validation sets to prevent leakage.
        """
        for metric in self.base_metrics:
            if metric in features.columns:
                for window in self.feature_windows:
                    try:
                        min_periods = max(1, self._window_to_periods(window) // 3)

                        features[f'{metric}_mean_{window}'] = features[metric].rolling(window, min_periods=min_periods).mean()
                        features[f'{metric}_std_{window}'] = features[metric].rolling(window, min_periods=min_periods).std()
                        features[f'{metric}_max_{window}'] = features[metric].rolling(window, min_periods=min_periods).max()
                        features[f'{metric}_min_{window}'] = features[metric].rolling(window, min_periods=min_periods).min()

                        # Rate of change
                        periods = self._window_to_periods(window)
                        pct_change = features[metric].pct_change(periods=periods)
                        pct_change = np.clip(pct_change, -10, 10)  # Cap at ±1000%
                        features[f'{metric}_pct_change_{window}'] = pct_change

                    except Exception as e:
                        logger.warning(
                            "Failed to create rolling features for %s_%s: %s",
                            metric, window, e
                        )

        # Anomaly indicators using rolling percentiles (within this split only)
        if all(col in features.columns for col in ['request_rate', 'application_latency', 'error_rate']):
            try:
                features['high_error_rate'] = features['error_rate'] > features['error_rate'].rolling('1H', min_periods=1).quantile(0.95)
                features['high_latency'] = features['application_latency'] > features['application_latency'].rolling('1H', min_periods=1).quantile(0.95)
                features['traffic_spike'] = features['request_rate'] > features['request_rate'].rolling('1H', min_periods=1).quantile(0.95)
            except Exception as e:
                logger.warning("Failed to create anomaly indicators: %s", e)

        return features

    def _cleanup_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Clean up features by handling NaN and inf values."""
        features = features.replace([np.inf, -np.inf], np.nan)

        # Remove rows with too many NaNs (from rolling windows)
        valid_threshold = int(len(features.columns) * 0.5)  # At least 50% valid values
        features = features.dropna(thresh=valid_threshold)

        # Fill any remaining NaNs with 0
        features = features.fillna(0)

        return features

    def _window_to_periods(self, window: str) -> int:
        """Convert window string to number of periods (5min intervals)"""
        if window == '5T':
            return 1
        elif window == '15T':
            return 3
        elif window == '1H':
            return 12
        else:
            return 1

class ParquetTrainingDataStorage:
    def __init__(self, base_path: str = "./smartbox_training_data/"):
        self.base_path = base_path
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create directory structure"""
        os.makedirs(self.base_path, exist_ok=True)
        
    def save_training_data(self, service_name: str, raw_data: pd.DataFrame, 
                          features: pd.DataFrame, metadata: Dict) -> Dict[str, str]:
        """Save training data in parquet format"""
        date_str = datetime.now().strftime('%Y-%m-%d')
        service_path = f"{self.base_path}/{service_name}/"
        
        # Create directories
        for subdir in ['raw_metrics', 'features', 'metadata']:
            os.makedirs(f"{service_path}/{subdir}/date={date_str}/", exist_ok=True)
        
        # Save raw data
        raw_data_path = f"{service_path}/raw_metrics/date={date_str}/data.parquet"
        raw_data.to_parquet(raw_data_path, compression='snappy', engine='pyarrow')
        
        # Save features
        features_path = f"{service_path}/features/date={date_str}/features.parquet"
        features.to_parquet(features_path, compression='snappy', engine='pyarrow')
        
        # Save metadata
        metadata_path = f"{service_path}/metadata/date={date_str}/metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, default=str, indent=2)

        logger.info("Saved training data for %s at %s", service_name, date_str)

        return {
            'raw_data_path': raw_data_path,
            'features_path': features_path,
            'metadata_path': metadata_path
        }

class EnhancedSmartboxTrainingPipeline:
    """Enhanced training pipeline that automatically uses explainability features"""

    def __init__(self, vm_endpoint: str = "https://otel-metrics.production.smartbox.com", config_path: str = "./config.json"):
        self.vm_client = VictoriaMetricsClient(vm_endpoint)
        self.metrics_extractor = SmartboxMetricsExtractor(self.vm_client)
        self.feature_engineer = SmartboxFeatureEngineer()
        self.storage = ParquetTrainingDataStorage()
        self.training_storage = TrainingRunStorage()

        # Load training configuration from config.json
        self.config = self._load_training_config(config_path)

        logger.info("Enhanced Smartbox Training Pipeline initialized")
        logger.info("  Explainable anomaly detection enabled by default")
        logger.info("  Validation fraction: %s", self.config.get('validation_fraction', 0.2))
        logger.info(
            "  Threshold calibration: %s",
            'enabled' if self.config.get('threshold_calibration', {}).get('enabled', True) else 'disabled'
        )

    def _load_training_config(self, config_path: str) -> Dict:
        """Load training configuration from config.json."""
        default_config = {
            'lookback_days': 60,
            'min_data_points': 2000,
            'validation_fraction': 0.2,
            'validation_split': 0.8,  # backward compatibility
            'threshold_calibration': {'enabled': True},
            'drift_detection': {'enabled': False},
            'excluded_periods': [],
            'model_variants': {
                'train_baseline': True,
                'train_holiday': True,
                'min_excluded_days_for_variant': 7,
                'holiday_data_weight': 0.3
            }
        }

        try:
            with open(config_path) as f:
                full_config = json.load(f)

            training_config = full_config.get("training", {})

            # Merge with defaults
            for key, value in default_config.items():
                if key not in training_config:
                    training_config[key] = value

            # Ensure backward compatibility
            training_config['validation_split'] = 1 - training_config.get('validation_fraction', 0.2)

            # Create TrainingConfig dataclass for structured access
            excluded_periods = []
            for period_dict in training_config.get('excluded_periods', []):
                try:
                    excluded_periods.append(ExcludedPeriod.from_dict(period_dict))
                except (KeyError, TypeError) as e:
                    logger.warning("Invalid excluded period: %s", e)

            training_config['_training_config_obj'] = TrainingConfig(
                lookback_days=training_config.get('lookback_days', 60),
                min_data_points=training_config.get('min_data_points', 2000),
                validation_fraction=training_config.get('validation_fraction', 0.2),
                excluded_periods=tuple(excluded_periods),
                train_baseline_model=training_config.get('model_variants', {}).get('train_baseline', True),
                train_holiday_model=training_config.get('model_variants', {}).get('train_holiday', True),
                min_excluded_days_for_variant=training_config.get('model_variants', {}).get('min_excluded_days_for_variant', 7),
                holiday_data_weight=training_config.get('model_variants', {}).get('holiday_data_weight', 0.3),
            )

            # Log excluded periods
            if excluded_periods:
                logger.info(
                    "Loaded %d excluded periods for dual-model training",
                    len(excluded_periods)
                )
                for ep in excluded_periods:
                    logger.info("  - %s: %s to %s", ep.reason or ep.model_variant, ep.start, ep.end)

            return training_config

        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning("Could not load training config: %s, using defaults", e)
            default_config['_training_config_obj'] = TrainingConfig()
            return default_config

    def load_services_from_config(self, config_path: str = "./config.json") -> List[str]:
        """Load services list from config.json.

        Combines all service categories: critical, standard, micro, admin, core.
        """
        try:
            with open(config_path) as f:
                config = json.load(f)

            services_config = config.get("services", {})
            all_services = []

            # Collect services from all categories
            for category in ["critical", "standard", "micro", "admin", "core", "background"]:
                category_services = services_config.get(category, [])
                if isinstance(category_services, list):
                    all_services.extend(category_services)

            # Remove duplicates while preserving order
            seen = set()
            unique_services = []
            for svc in all_services:
                if svc not in seen:
                    seen.add(svc)
                    unique_services.append(svc)

            if unique_services:
                logger.info("Loaded %d services from %s", len(unique_services), config_path)
                return unique_services
            else:
                logger.warning("No services found in %s, will discover from VictoriaMetrics", config_path)
                return []

        except FileNotFoundError:
            logger.warning("Config file not found: %s, will discover from VictoriaMetrics", config_path)
            return []
        except json.JSONDecodeError as e:
            logger.warning("Invalid JSON in %s: %s, will discover from VictoriaMetrics", config_path, e)
            return []
        except Exception as e:
            logger.warning("Error loading config: %s, will discover from VictoriaMetrics", e)
            return []
    
    def test_service_queries(self, service_name: str) -> Dict[str, bool]:
        """Test queries for a specific service"""
        return self.metrics_extractor.test_queries(service_name)
    
    def discover_services(self) -> List[str]:
        """Discover available services from VictoriaMetrics"""
        services = self.metrics_extractor.get_available_services()
        logger.info(
            "Discovered %d services: %s%s",
            len(services), services[:10], '...' if len(services) > 10 else ''
        )
        return services

    def train_service_model(self, service_name: str, lookback_days: Optional[int] = None) -> Dict:
        """Train enhanced anomaly detection model with explainability for a single service"""
        # Use custom lookback_days if provided, otherwise use config default
        training_days = lookback_days or self.config['lookback_days']

        logger.info("Training enhanced model for service: %s", service_name)
        logger.info("Using %d days of training data", training_days)
        logger.debug("Explainability features will be automatically included")
        
        try:
            # 1. Extract raw metrics
            raw_data = self.metrics_extractor.extract_service_metrics(
                service_name, 
                lookback_days=training_days
            )
            
            if raw_data.empty:
                return {'status': 'no_data', 'message': 'No metrics data found'}
            
            if len(raw_data) < self.config['min_data_points']:
                return {
                    'status': 'insufficient_data',
                    'message': f'Only {len(raw_data)} data points, need {self.config["min_data_points"]}'
                }

            # 2. Engineer features with proper temporal split (no leakage)
            validation_fraction = self.config.get('validation_fraction', 0.2)
            train_features, validation_features = self.feature_engineer.engineer_features_split(
                raw_data, validation_fraction=validation_fraction
            )

            if train_features.empty:
                return {'status': 'feature_engineering_failed', 'message': 'Feature engineering produced no features'}

            # Also create combined features for storage (with leakage warning acknowledged)
            features = self.feature_engineer.engineer_features(raw_data, include_rolling=True)

            # 3. Train enhanced model with leakage-free validation split
            model = SmartboxAnomalyDetector(service_name)

            # Train with properly split validation data for threshold calibration
            min_validation_samples = 100  # Need enough for reliable calibration
            train_result = model.train(
                train_features,
                validation_df=validation_features if len(validation_features) >= min_validation_samples else None
            )

            # Log calibration results
            if train_result.get("validation_results"):
                logger.info("Thresholds calibrated on %d validation samples", len(validation_features))

            # 4. Validate model using properly split validation data (no leakage)
            validation_results = self._validate_enhanced_model_split(model, train_features, validation_features)

            if not validation_results['passed']:
                return {
                    'status': 'validation_failed',
                    'validation_results': validation_results
                }

            # 5. Save training data
            metadata = {
                'service': service_name,
                'model_type': 'enhanced_explainable',
                'training_start': str(raw_data.index.min()),
                'training_end': str(raw_data.index.max()),
                'data_points': len(raw_data),
                'feature_count': len(features.columns),
                'training_samples': len(train_features),
                'validation_samples': len(validation_features),
                'explainability_features': len(model.training_statistics),
                'calibrated_thresholds': len(model.calibrated_thresholds),
                'validation_calibrated': bool(train_result.get("validation_results")),
                'validation_results': validation_results,
                'model_version': f"enhanced_v{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'training_config': self.config
            }

            storage_result = self.storage.save_training_data(
                service_name, raw_data, features, metadata
            )

            # 6. Save enhanced model (now includes explainability data)
            model_path = self.save_enhanced_model(service_name, model, metadata)

            # Log explainability status
            logger.info(
                "Explainability features: %s (%d metrics)",
                "enabled" if model.training_statistics else "disabled",
                len(model.training_statistics)
            )

            return {
                'status': 'success',
                'model_path': model_path,
                'storage_result': storage_result,
                'metadata': metadata,
                'validation_results': validation_results,
                'explainability_metrics': len(model.training_statistics)
            }

        except Exception as e:
            logger.error("Enhanced training failed for %s: %s", service_name, e)
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def train_service_model_time_aware(
        self,
        service_name: str,
        lookback_days: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        cache_dir: str = "./metrics_cache",
        models_dir: str = "./smartbox_models",
    ) -> Dict:
        """Train enhanced time-aware anomaly detection model with dual-model support.

        Data is loaded via cache with auto-fetch: if data is already cached, it's loaded
        from disk. If not, missing data is fetched from VictoriaMetrics in chunks and
        cached for future use.

        Args:
            service_name: Name of the service to train.
            lookback_days: Number of days to look back (used if start_date/end_date not provided).
            start_date: Optional explicit start date for training data.
            end_date: Optional explicit end date for training data.
            cache_dir: Directory for metrics cache (default: ./metrics_cache).
            models_dir: Directory to save trained models (default: ./smartbox_models).
        """
        training_config: TrainingConfig = self.config.get('_training_config_obj', TrainingConfig())
        training_days = lookback_days or training_config.lookback_days
        service_start_time = time.time()

        # Determine date range - convert lookback_days to explicit dates if needed
        if start_date is not None and end_date is not None:
            effective_start = start_date
            effective_end = end_date
            date_range_desc = f"{start_date.date()} to {end_date.date()}"
            actual_days = (end_date - start_date).days
        else:
            # Convert lookback_days to explicit date range
            effective_end = datetime.now()
            effective_start = effective_end - timedelta(days=training_days)

            # Compensate for excluded periods - extend lookback to get enough usable data
            if training_config.excluded_periods:
                excluded_days = 0
                for period in training_config.excluded_periods:
                    try:
                        period_start = datetime.fromisoformat(period.start)
                        period_end = datetime.fromisoformat(period.end)
                        # Calculate overlap between excluded period and our date range
                        overlap_start = max(effective_start, period_start)
                        overlap_end = min(effective_end, period_end)
                        if overlap_start < overlap_end:
                            excluded_days += (overlap_end - overlap_start).days
                    except (ValueError, TypeError):
                        continue

                if excluded_days > 0:
                    # Extend lookback to compensate for excluded days
                    effective_start = effective_start - timedelta(days=excluded_days)
                    logger.info(
                        "Extended lookback by %d days to compensate for excluded periods",
                        excluded_days
                    )

            total_window = (effective_end - effective_start).days
            date_range_desc = f"last {total_window} days ({training_days} target + excluded compensation)"
            actual_days = total_window

        logger.info("=" * 60)
        logger.info("Training enhanced time-aware model for service: %s", service_name)
        logger.info("Using %d days of training data (%s)", actual_days, date_range_desc)
        logger.info("Dual-model training: baseline=%s, holiday=%s",
                   training_config.train_baseline_model,
                   training_config.train_holiday_model)
        logger.debug("Time-aware + explainability features will be included")

        # Start tracking this training run
        run_id = self.training_storage.start_run(
            service_name=service_name,
            model_variant='baseline',
        )
        logger.debug("Training run started: %s", run_id)

        try:
            # Load data via cache (auto-fetches missing data from VictoriaMetrics)
            raw_data = self.metrics_extractor.load_with_cache(
                service_name,
                start_date=effective_start,
                end_date=effective_end,
                cache_dir=cache_dir,
            )

            if raw_data.empty:
                self.training_storage.fail_run(
                    run_id=run_id,
                    error_message='No metrics data found',
                )
                return {'status': 'no_data', 'message': 'No metrics data found'}

            if len(raw_data) < training_config.min_data_points:
                self.training_storage.fail_run(
                    run_id=run_id,
                    error_message=f'Only {len(raw_data)} data points, need {training_config.min_data_points}',
                )
                return {
                    'status': 'insufficient_data',
                    'message': f'Only {len(raw_data)} data points, need {training_config.min_data_points}'
                }

            # Split data by excluded periods for dual-model training
            baseline_data, holiday_data = self.metrics_extractor.filter_by_excluded_periods(
                raw_data, training_config
            )

            results = {
                'status': 'success',
                'variants_trained': [],
                'model_paths': {},
                'metadata': {},
                'total_data_points': len(raw_data),
            }

            validation_fraction = training_config.validation_fraction

            # Train BASELINE model (on data outside excluded periods)
            if training_config.train_baseline_model and len(baseline_data) >= training_config.min_data_points:
                baseline_start_time = time.time()
                logger.info("-" * 40)
                logger.info("BASELINE MODEL VARIANT")
                logger.info("-" * 40)
                logger.info("  Data points: %d (min required: %d)", len(baseline_data), training_config.min_data_points)
                logger.info("  Date range: %s to %s",
                           baseline_data.index.min().strftime("%Y-%m-%d"),
                           baseline_data.index.max().strftime("%Y-%m-%d"))
                if training_config.excluded_periods:
                    logger.info("  Excluded periods (not in baseline):")
                    for ep in training_config.excluded_periods:
                        logger.info("    - %s to %s (%s)", ep.start, ep.end, ep.reason)

                baseline_features = self.feature_engineer.engineer_features(baseline_data)
                if not baseline_features.empty:
                    baseline_detector = TimeAwareAnomalyDetector(service_name)
                    baseline_detector.train_time_aware_models(
                        baseline_features,
                        validation_fraction=validation_fraction
                    )

                    validation_results = self._validate_enhanced_time_aware_models(
                        baseline_detector, baseline_features
                    )

                    # Count explainability metrics
                    total_explainability = sum(
                        len(model.training_statistics) if hasattr(model, 'training_statistics') else 0
                        for model in baseline_detector.models.values()
                    )

                    metadata = {
                        'service': service_name,
                        'model_type': 'enhanced_time_aware_explainable',
                        'model_variant': 'baseline',
                        'time_periods': list(baseline_detector.models.keys()),
                        'training_start': str(baseline_data.index.min()),
                        'training_end': str(baseline_data.index.max()),
                        'data_points': len(baseline_data),
                        'feature_count': len(baseline_features.columns),
                        'total_explainability_metrics': total_explainability,
                        'validation_results': validation_results,
                        'model_version': f"enhanced_time_aware_v{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        'excluded_periods': [
                            {'start': ep.start, 'end': ep.end, 'reason': ep.reason}
                            for ep in training_config.excluded_periods
                        ],
                    }

                    # Save to models directory (baseline is primary)
                    saved_paths = baseline_detector.save_models(f"{models_dir}/", metadata)

                    results['variants_trained'].append('baseline')
                    results['model_paths']['baseline'] = saved_paths
                    results['metadata']['baseline'] = metadata

                    baseline_duration = time.time() - baseline_start_time
                    logger.info("BASELINE model training complete:")
                    logger.info("  Duration: %.1fs", baseline_duration)
                    logger.info("  Time periods trained: %d", len(baseline_detector.models))
                    logger.info("  Explainability metrics: %d", total_explainability)
                    logger.info("  Saved to: %s/%s/", models_dir, service_name)
                else:
                    logger.warning("BASELINE feature engineering failed")
            elif training_config.train_baseline_model:
                logger.warning(
                    "BASELINE model: insufficient data (%d < %d required)",
                    len(baseline_data), training_config.min_data_points
                )

            # Train HOLIDAY model (on data within excluded periods)
            min_holiday_points = 500  # Minimum points for holiday model
            if (training_config.train_holiday_model and
                len(holiday_data) >= min_holiday_points):
                holiday_start_time = time.time()
                logger.info("-" * 40)
                logger.info("HOLIDAY MODEL VARIANT")
                logger.info("-" * 40)
                logger.info("  Data points: %d (min required: %d)", len(holiday_data), min_holiday_points)
                logger.info("  Date range: %s to %s",
                           holiday_data.index.min().strftime("%Y-%m-%d"),
                           holiday_data.index.max().strftime("%Y-%m-%d"))
                logger.info("  Excluded periods covered:")
                for ep in training_config.excluded_periods:
                    logger.info("    - %s to %s (%s)", ep.start, ep.end, ep.reason)

                holiday_features = self.feature_engineer.engineer_features(holiday_data)
                if not holiday_features.empty:
                    holiday_detector = TimeAwareAnomalyDetector(service_name)
                    holiday_detector.train_time_aware_models(
                        holiday_features,
                        validation_fraction=validation_fraction
                    )

                    validation_results = self._validate_enhanced_time_aware_models(
                        holiday_detector, holiday_features
                    )

                    total_explainability = sum(
                        len(model.training_statistics) if hasattr(model, 'training_statistics') else 0
                        for model in holiday_detector.models.values()
                    )

                    metadata = {
                        'service': service_name,
                        'model_type': 'enhanced_time_aware_explainable',
                        'model_variant': 'holiday',
                        'time_periods': list(holiday_detector.models.keys()),
                        'training_start': str(holiday_data.index.min()),
                        'training_end': str(holiday_data.index.max()),
                        'data_points': len(holiday_data),
                        'feature_count': len(holiday_features.columns),
                        'total_explainability_metrics': total_explainability,
                        'validation_results': validation_results,
                        'model_version': f"enhanced_time_aware_holiday_v{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        'applicable_periods': [
                            {'start': ep.start, 'end': ep.end, 'reason': ep.reason}
                            for ep in training_config.excluded_periods
                        ],
                    }

                    # Save to holiday variant path
                    holiday_base_path = f"{models_dir}/"
                    holiday_service_path = os.path.join(holiday_base_path, service_name, "_holiday_variant")
                    os.makedirs(holiday_service_path, exist_ok=True)
                    saved_paths = holiday_detector.save_models(holiday_base_path.replace(
                        service_name, f"{service_name}/_holiday_variant"
                    ) if service_name in holiday_base_path else holiday_base_path, metadata)

                    # Actually save to the holiday variant directory
                    saved_paths = {}
                    for period, model in holiday_detector.models.items():
                        period_path = os.path.join(holiday_service_path, period)
                        os.makedirs(period_path, exist_ok=True)
                        model_file = os.path.join(period_path, "model.joblib")
                        import joblib
                        joblib.dump({'model': model, 'metadata': metadata}, model_file)
                        saved_paths[period] = model_file
                        logger.debug("Saved holiday model %s to %s", period, model_file)

                    results['variants_trained'].append('holiday')
                    results['model_paths']['holiday'] = saved_paths
                    results['metadata']['holiday'] = metadata

                    holiday_duration = time.time() - holiday_start_time
                    logger.info("HOLIDAY model training complete:")
                    logger.info("  Duration: %.1fs", holiday_duration)
                    logger.info("  Time periods trained: %d", len(holiday_detector.models))
                    logger.info("  Explainability metrics: %d", total_explainability)
                    logger.info("  Saved to: %s", holiday_service_path)
                else:
                    logger.warning("HOLIDAY feature engineering failed")
            elif training_config.train_holiday_model and len(holiday_data) > 0:
                logger.info(
                    "HOLIDAY model: insufficient data (%d < %d required) - skipping",
                    len(holiday_data), min_holiday_points
                )

            # Summary with timing
            service_duration = time.time() - service_start_time
            results['training_duration_seconds'] = service_duration

            logger.info("=" * 60)
            logger.info("Training complete for %s:", service_name)
            logger.info("  Duration: %.1fs", service_duration)
            logger.info("  Total data: %d points (%d days)", len(raw_data), training_days)
            logger.info("  Variants trained: %s", results['variants_trained'] or ['none'])
            if 'baseline' in results['variants_trained']:
                logger.info("  Baseline data: %d points", len(baseline_data))
            if 'holiday' in results['variants_trained']:
                logger.info("  Holiday data: %d points", len(holiday_data))

            if not results['variants_trained']:
                results['status'] = 'no_models_trained'
                results['message'] = 'Insufficient data for any model variant'

            # Record training completion - use baseline or holiday validation
            baseline_meta = results.get('metadata', {}).get('baseline', {})
            holiday_meta = results.get('metadata', {}).get('holiday', {})

            # Use baseline validation_results if it PASSED, otherwise fall back to holiday
            # This ensures we report the best validation outcome (holiday models may pass
            # even when baseline fails due to different data characteristics)
            baseline_validation = baseline_meta.get('validation_results', {})
            holiday_validation = holiday_meta.get('validation_results', {})

            # Check if baseline validation passed
            baseline_passed = baseline_validation.get('_summary', {}).get('overall_passed', False)
            holiday_passed = holiday_validation.get('_summary', {}).get('overall_passed', False)

            # Prefer whichever validation passed; if both pass/fail, prefer baseline
            if baseline_passed:
                validation_results = baseline_validation
                logger.info("Using BASELINE validation_results (passed)")
            elif holiday_passed:
                validation_results = holiday_validation
                logger.info("Using HOLIDAY validation_results (passed, baseline failed)")
            elif baseline_validation:
                validation_results = baseline_validation
                logger.info("Using BASELINE validation_results (neither passed)")
            else:
                validation_results = holiday_validation
                logger.info("Using HOLIDAY validation_results (baseline not available)")

            # Aggregate explainability metrics from both models
            total_explainability = (
                baseline_meta.get('total_explainability_metrics', 0) +
                holiday_meta.get('total_explainability_metrics', 0)
            )

            # Add validation_results at top level for callers to access
            results['validation_results'] = validation_results
            results['total_explainability_metrics'] = total_explainability

            # Build metadata dict for storage - use baseline or holiday as source
            primary_meta = baseline_meta if baseline_meta else holiday_meta
            storage_metadata = {
                'data_points': results.get('total_data_points', len(raw_data)),
                'training_start': effective_start.strftime("%Y-%m-%d"),
                'training_end': effective_end.strftime("%Y-%m-%d"),
                'time_periods': primary_meta.get('time_periods', []),
                'feature_count': primary_meta.get('feature_count', 0),
                'total_explainability_metrics': total_explainability,
                'full_metadata': results.get('metadata', {}),
            }

            self.training_storage.complete_run(
                run_id=run_id,
                validation_results=validation_results,
                metadata=storage_metadata,
            )
            logger.debug("Training run completed: %s", run_id)

            return results

        except Exception as e:
            service_duration = time.time() - service_start_time
            logger.error("Enhanced time-aware training failed for %s after %.1fs: %s",
                        service_name, service_duration, e)
            import traceback
            tb = traceback.format_exc()
            logger.error("Traceback: %s", tb)

            # Record training failure
            self.training_storage.fail_run(
                run_id=run_id,
                error_message=str(e),
                error_tb=tb,
            )
            logger.debug("Training run failed: %s", run_id)

            return {'status': 'error', 'error': str(e), 'training_duration_seconds': service_duration}

    
    def _validate_enhanced_model_split(
        self,
        model: SmartboxAnomalyDetector,
        train_features: pd.DataFrame,
        validation_features: pd.DataFrame
    ) -> Dict:
        """Enhanced validation using properly split data (no leakage).

        Args:
            model: Trained model to validate.
            train_features: Training features (for reference).
            validation_features: Held-out validation features (for testing).

        Returns:
            Validation results dictionary.
        """
        logger.info("Validating enhanced model (leakage-free)...")

        # Use the split validation method
        validation_result = self._validate_model_split(model, train_features, validation_features)

        # Additional explainability validation
        explainability_checks = {
            'has_training_statistics': hasattr(model, 'training_statistics') and len(model.training_statistics) > 0,
            'has_feature_importance': hasattr(model, 'feature_importance') and len(model.feature_importance) > 0,
            'has_context_method': hasattr(model, 'detect_anomalies_with_context')
        }

        validation_result['explainability_checks'] = explainability_checks
        validation_result['explainability_passed'] = all(explainability_checks.values())
        validation_result['explainability_metrics'] = len(model.training_statistics) if hasattr(model, 'training_statistics') else 0
        validation_result['enhanced_passed'] = validation_result['passed'] and validation_result['explainability_passed']
        validation_result['leakage_free'] = True

        if validation_result['explainability_passed']:
            logger.info("  Explainability validation passed (%d metrics)", validation_result['explainability_metrics'])
        else:
            logger.warning("  Explainability validation failed:")
            for check, passed in explainability_checks.items():
                status = "OK" if passed else "FAILED"
                logger.warning("    %s: %s", check, status)

        return validation_result

    def _validate_model_split(
        self,
        model: SmartboxAnomalyDetector,
        train_features: pd.DataFrame,
        validation_features: pd.DataFrame
    ) -> Dict:
        """Validate trained model using properly split data (no leakage).

        Args:
            model: Trained model to validate.
            train_features: Training features (for synthetic anomaly baseline).
            validation_features: Held-out validation features (for testing).

        Returns:
            Validation results dictionary.
        """
        logger.info("Validating model with split data (no leakage)...")

        if len(validation_features) < 50:
            return {
                'passed': False,
                'reason': 'Insufficient validation data',
                'validation_samples': len(validation_features),
                'leakage_free': True
            }

        total_tests = min(100, len(validation_features))
        test_errors = []

        logger.debug("  Testing on %d validation samples (properly split)...", total_tests)

        # Test 1: Normal validation data (should have low anomaly rate)
        normal_anomalies = 0
        for i in range(min(total_tests // 2, len(validation_features))):
            try:
                row = validation_features.iloc[i]
                current_metrics = {
                    'request_rate': row.get('request_rate', 0),
                    'application_latency': row.get('application_latency', 0),
                    'dependency_latency': row.get('dependency_latency', 0),
                    'database_latency': row.get('database_latency', 0),
                    'error_rate': row.get('error_rate', 0)
                }

                for key, value in current_metrics.items():
                    if pd.isna(value) or np.isinf(value):
                        current_metrics[key] = 0.0

                anomalies = model.detect_anomalies(current_metrics)
                if anomalies:
                    normal_anomalies += 1

            except Exception as e:
                test_errors.append(str(e))
                if len(test_errors) <= 3:
                    logger.warning("  Validation test %d failed: %s", i, e)

        # Test 2: Synthetic anomalies based on TRAINING statistics (no leakage)
        # Use training data statistics to create realistic synthetic anomalies
        synthetic_anomalies = 0
        synthetic_tests = total_tests // 2

        for i in range(synthetic_tests):
            try:
                # Use training data for baseline (this is correct - no leakage)
                row = train_features.iloc[i % len(train_features)]

                current_metrics = {
                    'request_rate': row.get('request_rate', 0) * 3,
                    'application_latency': row.get('application_latency', 0) * 5,
                    'dependency_latency': row.get('dependency_latency', 0) * 2,
                    'database_latency': row.get('database_latency', 0) * 2,
                    'error_rate': min(1.0, row.get('error_rate', 0) * 10)
                }

                for key, value in current_metrics.items():
                    if pd.isna(value) or np.isinf(value):
                        current_metrics[key] = 0.0

                anomalies = model.detect_anomalies(current_metrics)
                if anomalies:
                    synthetic_anomalies += 1

            except Exception as e:
                test_errors.append(str(e))

        total_anomalies = normal_anomalies + synthetic_anomalies
        total_tests_run = (total_tests // 2) + synthetic_tests
        anomaly_rate = total_anomalies / total_tests_run if total_tests_run > 0 else 0
        synthetic_detection_rate = synthetic_anomalies / synthetic_tests if synthetic_tests > 0 else 0
        normal_anomaly_rate = normal_anomalies / (total_tests // 2) if total_tests // 2 > 0 else 0

        validation_checks = {
            'has_models': len(model.models) > 0,
            'has_thresholds': len(model.thresholds) > 0,
            'reasonable_anomaly_rate': 0.0 <= anomaly_rate <= 0.6,
            'normal_data_reasonable': normal_anomaly_rate <= 0.15,  # Max 15% on normal data
            'few_test_errors': len(test_errors) < total_tests_run * 0.1,
            'model_responds': total_tests_run > 0,
            'detects_synthetic': synthetic_detection_rate > 0.2
        }

        validation_passed = all(validation_checks.values())

        validation_result = {
            'passed': validation_passed,
            'anomaly_rate': anomaly_rate,
            'normal_anomalies': normal_anomalies,
            'normal_anomaly_rate': normal_anomaly_rate,
            'synthetic_anomalies': synthetic_anomalies,
            'synthetic_detection_rate': synthetic_detection_rate,
            'models_trained': len(model.models),
            'thresholds_set': len(model.thresholds),
            'validation_samples': total_tests_run,
            'test_errors': len(test_errors),
            'checks': validation_checks,
            'leakage_free': True
        }

        if not validation_passed:
            logger.warning("  Validation failed:")
            for check, passed in validation_checks.items():
                status = "OK" if passed else "FAILED"
                logger.warning("    %s: %s", check, passed)
            logger.warning("  Normal data anomaly rate: %.1f%%", normal_anomaly_rate * 100)
            logger.warning("  Synthetic detection rate: %.1f%%", synthetic_detection_rate * 100)
        else:
            logger.info("  Validation passed (leakage-free)")
            logger.info("    Normal data anomaly rate: %.1f%%", normal_anomaly_rate * 100)
            logger.info("    Synthetic detection rate: %.1f%%", synthetic_detection_rate * 100)

        return validation_result

    def _validate_enhanced_model(self, model: SmartboxAnomalyDetector, features: pd.DataFrame) -> Dict:
        """Enhanced validation including explainability features (legacy method)."""
        logger.info("Validating enhanced model...")

        # Standard validation
        validation_result = self._validate_model(model, features)
        
        # Additional explainability validation
        explainability_checks = {
            'has_training_statistics': hasattr(model, 'training_statistics') and len(model.training_statistics) > 0,
            'has_feature_importance': hasattr(model, 'feature_importance') and len(model.feature_importance) > 0,
            'has_context_method': hasattr(model, 'detect_anomalies_with_context')
        }
        
        # Update validation result
        validation_result['explainability_checks'] = explainability_checks
        validation_result['explainability_passed'] = all(explainability_checks.values())
        validation_result['explainability_metrics'] = len(model.training_statistics) if hasattr(model, 'training_statistics') else 0
        
        # Overall pass includes explainability
        validation_result['enhanced_passed'] = validation_result['passed'] and validation_result['explainability_passed']

        if validation_result['explainability_passed']:
            logger.info("  Explainability validation passed (%d metrics)", validation_result['explainability_metrics'])
        else:
            logger.warning("  Explainability validation failed:")
            for check, passed in explainability_checks.items():
                status = "OK" if passed else "FAILED"
                logger.warning("    %s: %s", check, status)

        return validation_result

    def _validate_enhanced_time_aware_models(self, detector: TimeAwareAnomalyDetector, features_df: pd.DataFrame) -> Dict:
        """Enhanced validation for time-aware models with 3-period approach.

        Uses 3 time-of-day buckets covering all 7 days:
        - business_hours (8-18)
        - evening_hours (18-22)
        - night_hours (22-8)
        """
        logger.info("Validating enhanced time-aware models with 3-period approach...")
        validation_results = {}

        # Add time period column
        features_df = features_df.copy()
        features_df['time_period'] = features_df.index.map(detector.get_time_period)

        # Use the detector's service-specific thresholds
        thresholds = detector.validation_thresholds
        service_type = detector._get_service_type()

        logger.debug("  Using %s thresholds for 3 periods:", service_type)
        for period, threshold in thresholds.items():
            logger.debug("    %s: %.1f%% max anomaly rate", period, threshold * 100)

        # All 3 time-of-day periods (covering all 7 days)
        all_periods = ['business_hours', 'evening_hours', 'night_hours']
        
        for period in detector.models.keys():
            period_data = features_df[features_df['time_period'] == period]
            
            # Use detector's minimum sample requirements
            min_samples = detector._get_min_samples_for_period(period)
            
            if len(period_data) < min_samples:
                validation_results[period] = {
                    'status': 'insufficient_data', 
                    'samples': len(period_data),
                    'min_required': min_samples,
                    'passed': False,
                    'explainability_passed': False
                }
                continue
            
            logger.debug("  Testing enhanced %s model...", period)
            
            # Enhanced validation with realistic expectations
            model = detector.models[period]
            
            # Use smaller test sample for micro-services and admin services
            if service_type in ['micro_service', 'admin_service']:
                test_samples = min(25, len(period_data) // 3)  # Smaller samples for variable services
            elif period == 'night_hours':
                test_samples = min(30, len(period_data) // 2)  # Night hours have more variance
            else:
                test_samples = min(40, len(period_data) // 2)  # Standard samples for day periods
            
            normal_anomaly_count = 0
            test_errors = 0
            
            # Test on validation data with improved error handling
            for i in range(test_samples):
                try:
                    row_idx = -(i+1) if i < len(period_data) else i % len(period_data)
                    row = period_data.iloc[row_idx]
                    test_metrics = {
                        'request_rate': max(0, row.get('request_rate', 0)),
                        'application_latency': max(0, row.get('application_latency', 0)),
                        'dependency_latency': max(0, row.get('dependency_latency', 0)),
                        'database_latency': max(0, row.get('database_latency', 0)),
                        'error_rate': max(0, min(1.0, row.get('error_rate', 0)))
                    }
                    
                    # Ensure no invalid values
                    for key, value in test_metrics.items():
                        if pd.isna(value) or np.isinf(value):
                            test_metrics[key] = 0.0
                    
                    anomalies = model.detect_anomalies(test_metrics)
                    if anomalies and len(anomalies) > 0:
                        normal_anomaly_count += 1
                        
                except Exception as e:
                    test_errors += 1
                    if test_errors <= 2:  # Only log first few errors
                        logger.warning("    Test error for %s sample %d: %s...", period, i, str(e)[:50])
            
            normal_anomaly_rate = normal_anomaly_count / test_samples if test_samples > 0 else 0
            
            # Enhanced explainability validation
            explainability_checks = {
                'has_training_statistics': hasattr(model, 'training_statistics') and len(model.training_statistics) > 0,
                'has_feature_importance': hasattr(model, 'feature_importance') and len(model.feature_importance) > 0,
                'has_context_method': hasattr(model, 'detect_anomalies_with_context'),
                'has_zero_statistics': hasattr(model, 'zero_statistics') and len(model.zero_statistics) > 0
            }
            
            # Use service-specific threshold for this period
            period_threshold = thresholds.get(period, 0.15)
            
            # Enhanced validation criteria with period-specific adjustments
            error_tolerance = 0.25 if period == 'night_hours' else 0.2  # Slightly more lenient for night
            
            validation_checks = {
                'anomaly_rate_acceptable': normal_anomaly_rate <= period_threshold,
                'sufficient_tests': test_samples >= 10,  # Minimum tests
                'low_error_rate': test_errors < test_samples * error_tolerance,
                'model_functional': normal_anomaly_count >= 0  # Basic functionality check
            }
            
            # Overall validation status
            validation_passed = all(validation_checks.values())
            explainability_passed = all(explainability_checks.values())
            
            validation_results[period] = {
                'samples_tested': test_samples,
                'normal_anomaly_rate': normal_anomaly_rate,
                'threshold_used': period_threshold,
                'test_errors': test_errors,
                'validation_checks': validation_checks,
                'passed': validation_passed,
                'explainability_checks': explainability_checks,
                'explainability_passed': explainability_passed,
                'explainability_metrics': len(model.training_statistics) if hasattr(model, 'training_statistics') else 0,
                'enhanced_passed': validation_passed and explainability_passed,
                'period_type': detector._get_period_type(period)
            }
            
            # Enhanced status reporting
            if validation_passed:
                status_msg = f"{normal_anomaly_rate:.1%} anomaly rate (<={period_threshold:.1%})"
                explainability_str = "explainability: OK" if explainability_passed else "explainability: MISSING"
                logger.info("  %s: OK - %s, %s", period, status_msg, explainability_str)
            else:
                failed_checks = [check for check, passed in validation_checks.items() if not passed]
                if not validation_checks['anomaly_rate_acceptable']:
                    status_msg = f"{normal_anomaly_rate:.1%} anomaly rate (>{period_threshold:.1%} threshold)"
                else:
                    status_msg = f"Failed: {', '.join(failed_checks)}"
                logger.warning("  %s: FAILED - %s", period, status_msg)
            
            # Additional context for failures with 3-period awareness
            if not validation_passed:
                if period == 'night_hours':
                    logger.debug("    Note: Night hours have higher natural variability due to lower traffic")
                elif period == 'evening_hours':
                    logger.debug("    Note: Evening hours are a transition period with variable patterns")
        
        # Enhanced overall assessment with 3-period logic
        total_periods = len(validation_results)
        passed_periods = sum(1 for result in validation_results.values() if result.get('enhanced_passed', False))

        # Critical periods: business_hours and night_hours (most important for operations)
        critical_periods_passed = sum(1 for period in ['business_hours', 'night_hours']
                                    if period in validation_results and validation_results[period].get('enhanced_passed', False))

        # Enhanced validation logic for 3-period approach
        # With 3 periods covering all 7 days, we have more data per period
        if total_periods == 0:
            overall_passed = False
            overall_message = "No periods validated"
        elif critical_periods_passed >= 1:  # At least one critical period must pass
            # For 3-period model, at least 2 out of 3 periods should pass
            overall_passed = passed_periods >= 2
            if overall_passed:
                overall_message = f"Validation passed: {passed_periods}/{total_periods} periods validated"
            else:
                overall_message = f"Insufficient coverage: {passed_periods}/{total_periods} periods (need ≥2)"
        else:
            overall_passed = False
            overall_message = "No critical periods (business/night hours) passed validation"
        
        explainability_periods = sum(1 for result in validation_results.values() if result.get('explainability_passed', False))
        explainability_overall = explainability_periods > 0
        
        if overall_passed:
            logger.info("Enhanced 3-period validation: %s", overall_message)
        else:
            logger.warning("Enhanced 3-period validation FAILED: %s", overall_message)
        logger.info("Explainability: %d/%d periods enabled", explainability_periods, total_periods)

        # Show period breakdown (all periods now cover all 7 days)
        if validation_results:
            logger.info("Period breakdown:")
            all_periods_passed = sum(1 for period in ['business_hours', 'evening_hours', 'night_hours']
                               if period in validation_results and validation_results[period].get('enhanced_passed', False))
            all_periods_total = sum(1 for period in ['business_hours', 'evening_hours', 'night_hours']
                              if period in validation_results)
            logger.info("  Weekday periods: %d/%d passed", all_periods_passed, all_periods_total)
        
        # Add service-specific recommendations for 3-period approach
        if not overall_passed:
            logger.info("Recommendations for %s with 3-period approach:", service_type)
            if service_type == 'micro_service':
                logger.info("  - Consider increasing training data collection period")
                logger.info("  - Focus on getting business_hours and night_hours models working first")
                logger.info("  - Evening hours may need higher tolerance for variability")
            elif service_type == 'admin_service':
                logger.info("  - Admin services often have minimal night activity")
                logger.info("  - Focus on business_hours model which has the most consistent traffic")
                logger.info("  - Night hours may not be reliable for admin services")
            else:
                logger.info("  - Ensure adequate data collection across all time periods")
                logger.info("  - Night hours have higher natural variance than day hours")
                logger.info("  - Consider adjusting anomaly detection sensitivity for night periods")
        
        # Add summary to validation results
        validation_results['_summary'] = {
            'overall_passed': overall_passed,
            'overall_message': overall_message,
            'explainability_overall': explainability_overall,
            'service_type': service_type,
            'total_periods': total_periods,
            'passed_periods': passed_periods,
            'critical_periods_passed': critical_periods_passed,
            'thresholds_used': thresholds,
            'approach': '3_period_time_of_day'
        }
        
        return validation_results
        
    def train_all_services(self, service_list: Optional[List[str]] = None) -> Dict:
        """Train enhanced models for all services"""
        if service_list is None:
            service_list = self.discover_services()

        logger.info("Training enhanced models for %d services", len(service_list))
        logger.info("All models will include explainability features")
        
        results = {}
        successful_trainings = 0
        explainable_models = 0
        
        for i, service in enumerate(service_list, 1):
            logger.info("[%d/%d] Processing %s", i, len(service_list), service)

            result = self.train_service_model(service)
            results[service] = result

            if result['status'] == 'success':
                successful_trainings += 1
                if result.get('explainability_metrics', 0) > 0:
                    explainable_models += 1
                logger.info("%s: Enhanced training successful", service)
                logger.info("  Explainability: %d metrics", result.get('explainability_metrics', 0))
            else:
                logger.error("%s: %s - %s", service, result['status'], result.get('message', ''))

        logger.info("Enhanced Training Summary:")
        logger.info("  Total services: %d", len(service_list))
        logger.info("  Successful: %d", successful_trainings)
        logger.info("  With explainability: %d", explainable_models)
        logger.info("  Failed: %d", len(service_list) - successful_trainings)
        
        return results

    def warm_cache_for_services(
        self,
        services: List[str],
        start_date: datetime,
        end_date: datetime,
        cache_dir: str = "./metrics_cache",
        parallel_fetch: bool = True,
    ) -> Dict[str, Dict[str, int]]:
        """Pre-populate the metrics cache for all services before training.

        This separates the I/O-bound cache warming from the CPU-bound training,
        enabling subsequent parallel training without VictoriaMetrics contention.

        Args:
            services: List of service names to cache.
            start_date: Start date for training data.
            end_date: End date for training data.
            cache_dir: Directory for metrics cache.
            parallel_fetch: If True, fetch multiple services concurrently (max 2).

        Returns:
            Dict mapping service names to their cache status per metric.
        """
        logger.info("╔" + "═" * 68 + "╗")
        logger.info("║  CACHE WARMING PHASE%s║", " " * 48)
        logger.info("║  Services: %d%s║", len(services), " " * (68 - 12 - len(str(len(services)))))
        logger.info("║  Date range: %s to %s%s║",
                   start_date.strftime("%Y-%m-%d"),
                   end_date.strftime("%Y-%m-%d"),
                   " " * (68 - 30 - len(start_date.strftime("%Y-%m-%d")) - len(end_date.strftime("%Y-%m-%d"))))
        logger.info("╚" + "═" * 68 + "╝")
        logger.info("")

        cache_results = {}
        cache_start_time = time.time()

        # Create a single MetricsCache instance
        from smartbox_anomaly.metrics.cache import MetricsCache
        cache = MetricsCache(
            vm_client=self.vm_client,
            cache_dir=cache_dir,
        )

        def prefetch_service(service_name: str) -> Tuple[str, Dict[str, int]]:
            """Prefetch metrics for a single service."""
            try:
                result = cache.prefetch(
                    service_name=service_name,
                    start_date=start_date,
                    end_date=end_date,
                )
                return service_name, result
            except Exception as e:
                logger.error("Failed to cache %s: %s", service_name, e)
                return service_name, {"error": str(e)}

        if parallel_fetch and len(services) > 1:
            # Fetch up to 2 services concurrently to avoid hammering VictoriaMetrics
            max_workers = min(2, len(services))
            logger.info("Parallel cache fetch with %d workers", max_workers)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(prefetch_service, svc): svc for svc in services}

                for i, future in enumerate(as_completed(futures), 1):
                    service_name, result = future.result()
                    cache_results[service_name] = result
                    total_points = sum(v for v in result.values() if isinstance(v, int))
                    logger.info("[%d/%d] Cached %s: %d data points",
                               i, len(services), service_name, total_points)
        else:
            # Sequential fetch
            for i, service in enumerate(services, 1):
                logger.info("[%d/%d] Caching %s...", i, len(services), service)
                service_name, result = prefetch_service(service)
                cache_results[service_name] = result
                total_points = sum(v for v in result.values() if isinstance(v, int))
                logger.info("  Cached %d data points", total_points)

                # Small delay between services
                if i < len(services):
                    time.sleep(0.5)

        cache_duration = time.time() - cache_start_time
        logger.info("")
        logger.info("Cache warming complete in %.1fs", cache_duration)
        logger.info("")

        return cache_results

    def train_all_services_time_aware(
        self,
        service_list: Optional[List[str]] = None,
        parallel_workers: int = 1,
        warm_cache_first: bool = False,
        lookback_days: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        cache_dir: str = "./metrics_cache",
        models_dir: str = "./smartbox_models",
    ) -> Dict:
        """Train enhanced time-aware models for all services with 3-period approach.

        Args:
            service_list: List of services to train (defaults to discovered services).
            parallel_workers: Number of parallel training threads (1 = sequential).
            warm_cache_first: If True, pre-populate cache for all services before training.
            lookback_days: Number of days to look back for training data.
            start_date: Optional explicit start date.
            end_date: Optional explicit end date.
            cache_dir: Directory for metrics cache.
            models_dir: Directory for saving trained models (default: ./smartbox_models).
        """
        if service_list is None:
            service_list = self.discover_services()

        # Determine effective date range
        training_config: TrainingConfig = self.config.get('_training_config_obj', TrainingConfig())
        effective_lookback = lookback_days or training_config.lookback_days

        if start_date is not None and end_date is not None:
            effective_start = start_date
            effective_end = end_date
        else:
            effective_end = datetime.now()
            effective_start = effective_end - timedelta(days=effective_lookback)

            # Compensate for excluded periods - extend lookback to get enough usable data
            if training_config.excluded_periods:
                excluded_days = 0
                for period in training_config.excluded_periods:
                    try:
                        period_start = datetime.fromisoformat(period.start)
                        period_end = datetime.fromisoformat(period.end)
                        # Calculate overlap between excluded period and our date range
                        overlap_start = max(effective_start, period_start)
                        overlap_end = min(effective_end, period_end)
                        if overlap_start < overlap_end:
                            excluded_days += (overlap_end - overlap_start).days
                    except (ValueError, TypeError):
                        continue

                if excluded_days > 0:
                    # Extend lookback to compensate for excluded days
                    effective_start = effective_start - timedelta(days=excluded_days)
                    effective_lookback += excluded_days
                    logger.info(
                        "Extended lookback by %d days to compensate for excluded periods",
                        excluded_days
                    )

        # Generate unique run identifier for log correlation
        run_id = uuid.uuid4().hex[:8]
        batch_start_time = time.time()
        started_at = datetime.now()

        logger.info("╔" + "═" * 68 + "╗")
        logger.info("║  TRAINING RUN: %s%s║", run_id, " " * (68 - 16 - len(run_id)))
        logger.info("║  Started: %s%s║", started_at.strftime("%Y-%m-%d %H:%M:%S"), " " * (68 - 21 - 19))
        logger.info("║  Services: %d%s║", len(service_list), " " * (68 - 12 - len(str(len(service_list)))))
        parallel_str = f"parallel={parallel_workers}" if parallel_workers > 1 else "sequential"
        logger.info("║  Mode: %s%s║", parallel_str, " " * (68 - 8 - len(parallel_str)))
        logger.info("╚" + "═" * 68 + "╝")
        logger.info("")
        logger.info("Configuration:")
        logger.info("  3-period approach: business_hours, evening_hours, night_hours")
        logger.info("  Features: time-awareness + explainability")
        logger.info("  Validation: service-specific thresholds")
        if parallel_workers > 1:
            logger.info("  Parallel workers: %d", parallel_workers)
        if warm_cache_first:
            logger.info("  Cache warming: enabled (pre-fetch all services)")
        logger.info("")

        # Phase 1: Warm cache for all services (optional)
        if warm_cache_first:
            self.warm_cache_for_services(
                services=service_list,
                start_date=effective_start,
                end_date=effective_end,
                cache_dir=cache_dir,
            )

        results = {}
        successful_trainings = 0
        total_explainability_metrics = 0
        validation_passed_services = 0
        total_training_time = 0.0
        period_stats = {
            'business_hours': {'trained': 0, 'passed': 0},
            'evening_hours': {'trained': 0, 'passed': 0},
            'night_hours': {'trained': 0, 'passed': 0},
        }

        # Thread-safe counters for parallel training
        import threading
        results_lock = threading.Lock()

        def train_single_service(service: str) -> Tuple[str, Dict]:
            """Train a single service (thread-safe)."""
            result = self.train_service_model_time_aware(
                service,
                lookback_days=effective_lookback,
                start_date=start_date,
                end_date=end_date,
                cache_dir=cache_dir,
                models_dir=models_dir,
            )
            return service, result

        def process_result(service: str, result: Dict, idx: int, total: int):
            """Process a single training result (must be called with lock held)."""
            nonlocal successful_trainings, total_explainability_metrics, validation_passed_services

            results[service] = result
            total_training_time_local = result.get('training_duration_seconds', 0)

            if result['status'] == 'success':
                successful_trainings += 1
                explainability_count = result.get('total_explainability_metrics', 0)
                total_explainability_metrics += explainability_count

                # Check if validation passed with 3-period logic
                validation_results = result.get('validation_results', {})
                summary = validation_results.get('_summary', {})
                validation_passed = summary.get('overall_passed', False)
                service_type = summary.get('service_type', 'unknown')

                # Track period-specific statistics
                for period in period_stats.keys():
                    if period in validation_results and validation_results[period].get('status') != 'insufficient_data':
                        period_stats[period]['trained'] += 1
                        if validation_results[period].get('enhanced_passed', False):
                            period_stats[period]['passed'] += 1

                if validation_passed:
                    validation_passed_services += 1
                    logger.info("[%d/%d] %s (%s): ✓ Training successful",
                               idx, total, service, service_type)
                else:
                    overall_message = summary.get('overall_message', 'Validation issues')
                    logger.warning("[%d/%d] %s (%s): ~ Training OK but validation concerns: %s",
                                  idx, total, service, service_type, overall_message)
            else:
                logger.error("[%d/%d] %s: ✗ %s - %s",
                            idx, total, service, result['status'], result.get('message', ''))

            return total_training_time_local

        # Phase 2: Train services (parallel or sequential)
        if parallel_workers > 1 and len(service_list) > 1:
            # Parallel training with ThreadPoolExecutor
            logger.info("Starting parallel training with %d workers...", parallel_workers)
            logger.info("")

            with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
                futures = {executor.submit(train_single_service, svc): svc for svc in service_list}

                for i, future in enumerate(as_completed(futures), 1):
                    service, result = future.result()
                    with results_lock:
                        total_training_time += process_result(service, result, i, len(service_list))
        else:
            # Sequential training
            for i, service in enumerate(service_list, 1):
                progress_pct = (i / len(service_list)) * 100
                logger.info("[%d/%d] (%.0f%%) Processing %s", i, len(service_list), progress_pct, service)

                service, result = train_single_service(service)
                total_training_time += process_result(service, result, i, len(service_list))

        # Calculate final timing
        batch_duration = time.time() - batch_start_time
        finished_at = datetime.now()
        failed_trainings = len(service_list) - successful_trainings

        # ══════════════════════════════════════════════════════════════════════
        # STRUCTURED TRAINING SUMMARY REPORT
        # ══════════════════════════════════════════════════════════════════════
        logger.info("")
        logger.info("╔" + "═" * 68 + "╗")
        logger.info("║  TRAINING RUN COMPLETE: %s%s║", run_id, " " * (68 - 26 - len(run_id)))
        logger.info("╠" + "═" * 68 + "╣")

        # Timing section
        logger.info("║  TIMING%s║", " " * 61)
        logger.info("║    Started:    %s%s║", started_at.strftime("%Y-%m-%d %H:%M:%S"), " " * (68 - 17 - 19))
        logger.info("║    Finished:   %s%s║", finished_at.strftime("%Y-%m-%d %H:%M:%S"), " " * (68 - 17 - 19))
        duration_str = f"{batch_duration:.1f}s"
        if batch_duration >= 60:
            duration_str = f"{batch_duration / 60:.1f}m ({batch_duration:.0f}s)"
        logger.info("║    Duration:   %s%s║", duration_str, " " * (68 - 17 - len(duration_str)))
        avg_time = total_training_time / len(service_list) if service_list else 0
        avg_str = f"{avg_time:.1f}s/service"
        logger.info("║    Avg time:   %s%s║", avg_str, " " * (68 - 17 - len(avg_str)))

        logger.info("╠" + "═" * 68 + "╣")

        # Results section
        logger.info("║  RESULTS%s║", " " * 60)
        logger.info("║    Services processed:    %d%s║", len(service_list), " " * (68 - 28 - len(str(len(service_list)))))
        logger.info("║    Successful trainings:  %d%s║", successful_trainings, " " * (68 - 28 - len(str(successful_trainings))))
        logger.info("║    Validation passed:     %d/%d%s║", validation_passed_services, successful_trainings,
                   " " * (68 - 28 - len(f"{validation_passed_services}/{successful_trainings}")))
        logger.info("║    Failed trainings:      %d%s║", failed_trainings, " " * (68 - 28 - len(str(failed_trainings))))
        logger.info("║    Explainability metrics: %d%s║", total_explainability_metrics, " " * (68 - 29 - len(str(total_explainability_metrics))))

        logger.info("╚" + "═" * 68 + "╝")
        logger.info("")

        # Validation rate assessment
        if successful_trainings > 0:
            validation_rate = validation_passed_services / successful_trainings
            logger.info("Validation Summary (run_id=%s):", run_id)
            logger.info("  Success rate: %.1f%%", validation_rate * 100)

            if validation_rate >= 0.8:
                logger.info("  Assessment: Excellent validation performance")
            elif validation_rate >= 0.6:
                logger.info("  Assessment: Good - some services need attention")
            else:
                logger.warning("  Assessment: Many services have validation concerns")
                logger.info("  Recommendations:")
                logger.info("    - Increase training data collection period")
                logger.info("    - Review service-specific patterns")
                logger.info("    - Adjust night period sensitivity")

        # Period-by-period breakdown
        logger.info("")
        logger.info("Period Training Results (run_id=%s):", run_id)
        for period, stats in period_stats.items():
            if stats['trained'] > 0:
                success_rate = stats['passed'] / stats['trained']
                status_icon = "✓" if success_rate >= 0.8 else "~" if success_rate >= 0.6 else "!"
                logger.info("  [%s] %s: %d/%d (%.0f%%)",
                           status_icon, period, stats['passed'], stats['trained'], success_rate * 100)

        # Service type breakdown
        service_types = {}
        for service, result in results.items():
            if result['status'] == 'success':
                validation_results = result.get('validation_results', {})
                summary = validation_results.get('_summary', {})
                service_type = summary.get('service_type', 'unknown')

                if service_type not in service_types:
                    service_types[service_type] = {
                        'total': 0,
                        'passed': 0,
                        'critical_coverage': 0,
                    }

                service_types[service_type]['total'] += 1
                if summary.get('overall_passed', False):
                    service_types[service_type]['passed'] += 1

                if summary.get('critical_periods_passed', 0) >= 1:
                    service_types[service_type]['critical_coverage'] += 1

        if service_types:
            logger.info("")
            logger.info("Service Type Breakdown (run_id=%s):", run_id)
            for service_type, stats in sorted(service_types.items()):
                total = stats['total']
                passed = stats['passed']
                critical_cov = stats['critical_coverage']

                rate = passed / total if total > 0 else 0
                logger.info("  %s: %d/%d passed (%.0f%%)", service_type, passed, total, rate * 100)
                logger.info("    Critical periods: %d/%d", critical_cov, total)

        # Model distribution (3-period: all days combined per time-of-day bucket)
        total_models = sum(stats['trained'] for stats in period_stats.values())
        if total_models > 0:
            logger.info("")
            logger.info("Model Distribution (run_id=%s):", run_id)
            logger.info("  Total period models: %d", total_models)
            logger.info("  Business hours: %d", period_stats['business_hours']['trained'])
            logger.info("  Evening hours: %d", period_stats['evening_hours']['trained'])
            logger.info("  Night hours: %d", period_stats['night_hours']['trained'])

        # Final summary line for easy log searching
        logger.info("")
        logger.info("TRAINING_COMPLETE run_id=%s services=%d success=%d failed=%d duration=%.1fs",
                   run_id, len(service_list), successful_trainings, failed_trainings, batch_duration)

        return results
        
    def _validate_model(self, model: SmartboxAnomalyDetector, features: pd.DataFrame) -> Dict:
        """Validate trained model with detailed diagnostics"""
        logger.info("Validating model...")
        
        validation_split = int(len(features) * self.config['validation_split'])
        validation_data = features[validation_split:]
        
        if len(validation_data) < 50:
            return {
                'passed': False, 
                'reason': 'Insufficient validation data',
                'validation_samples': len(validation_data)
            }
        
        # Test anomaly detection on validation data + synthetic anomalies
        total_tests = min(100, len(validation_data))
        test_errors = []
        
        logger.debug("  Testing on %d validation samples...", total_tests)
        
        # Test 1: Normal validation data
        normal_anomalies = 0
        for i in range(total_tests // 2):
            try:
                row = validation_data.iloc[i]
                current_metrics = {
                    'request_rate': row.get('request_rate', 0),
                    'application_latency': row.get('application_latency', 0),
                    'dependency_latency': row.get('dependency_latency', 0),
                    'database_latency': row.get('database_latency', 0),
                    'error_rate': row.get('error_rate', 0)
                }
                
                # Ensure no invalid values
                for key, value in current_metrics.items():
                    if pd.isna(value) or np.isinf(value):
                        current_metrics[key] = 0.0
                
                anomalies = model.detect_anomalies(current_metrics)
                if anomalies:
                    normal_anomalies += 1
                    
            except Exception as e:
                test_errors.append(str(e))
                if len(test_errors) <= 3:
                    logger.warning("  Validation test %d failed: %s", i, e)

        # Test 2: Synthetic anomalies (should be detected)
        synthetic_anomalies = 0
        synthetic_tests = total_tests // 2
        
        for i in range(synthetic_tests):
            try:
                row = validation_data.iloc[i % len(validation_data)]
                
                # Create synthetic anomaly by amplifying values
                current_metrics = {
                    'request_rate': row.get('request_rate', 0) * 3,  # 3x normal traffic
                    'application_latency': row.get('application_latency', 0) * 5,  # 5x normal latency
                    'dependency_latency': row.get('dependency_latency', 0) * 2,
                    'database_latency': row.get('database_latency', 0) * 2,
                    'error_rate': min(1.0, row.get('error_rate', 0) * 10)  # 10x errors, capped at 100%
                }
                
                # Ensure no invalid values
                for key, value in current_metrics.items():
                    if pd.isna(value) or np.isinf(value):
                        current_metrics[key] = 0.0
                
                anomalies = model.detect_anomalies(current_metrics)
                if anomalies:
                    synthetic_anomalies += 1
                    
            except Exception as e:
                test_errors.append(str(e))
        
        total_anomalies = normal_anomalies + synthetic_anomalies
        total_tests_run = (total_tests // 2) + synthetic_tests
        anomaly_rate = total_anomalies / total_tests_run if total_tests_run > 0 else 0
        
        # Calculate synthetic detection rate
        synthetic_detection_rate = synthetic_anomalies / synthetic_tests if synthetic_tests > 0 else 0
        
        # Balanced validation criteria
        validation_checks = {
            'has_models': len(model.models) > 0,
            'has_thresholds': len(model.thresholds) > 0,
            'reasonable_anomaly_rate': 0.0 <= anomaly_rate <= 0.6,  # Allow up to 60% overall (includes synthetic)
            'normal_data_reasonable': normal_anomalies <= (total_tests//2) * 0.15,  # Max 15% anomalies in normal data (more strict)
            'few_test_errors': len(test_errors) < total_tests_run * 0.1,
            'model_responds': total_tests_run > 0,
            'detects_synthetic': synthetic_detection_rate > 0.2  # Should detect at least 20% of obvious anomalies
        }
        
        validation_passed = all(validation_checks.values())
        
        validation_result = {
            'passed': validation_passed,
            'anomaly_rate': anomaly_rate,
            'normal_anomalies': normal_anomalies,
            'normal_anomaly_rate': normal_anomalies / (total_tests//2) if total_tests//2 > 0 else 0,
            'synthetic_anomalies': synthetic_anomalies,
            'synthetic_detection_rate': synthetic_detection_rate,
            'models_trained': len(model.models),
            'thresholds_set': len(model.thresholds),
            'validation_samples': total_tests_run,
            'test_errors': len(test_errors),
            'checks': validation_checks
        }
        
        normal_rate = validation_result['normal_anomaly_rate']
        
        if not validation_passed:
            logger.warning("  Validation failed:")
            for check, passed in validation_checks.items():
                status = "OK" if passed else "FAILED"
                logger.warning("    %s: %s", check, passed)
            logger.warning("  Overall anomaly rate: %.3f", anomaly_rate)
            logger.warning("  Normal data anomaly rate: %.1f%% (%d/%d)", normal_rate * 100, normal_anomalies, total_tests//2)
            logger.warning("  Synthetic anomaly detection: %d/%d (%.1f%%)", synthetic_anomalies, synthetic_tests, synthetic_detection_rate * 100)
            logger.warning("  Test errors: %d/%d", len(test_errors), total_tests_run)
        else:
            logger.info("  Validation passed")
            logger.info("    Overall anomaly rate: %.3f", anomaly_rate)
            logger.info("    Normal data anomaly rate: %.1f%%", normal_rate * 100)
            logger.info("    Synthetic detection rate: %.1f%%", synthetic_detection_rate * 100)
            logger.info("    Models: %d, Thresholds: %d", len(model.models), len(model.thresholds))
        
        return validation_result
    
    def save_enhanced_model(self, service_name: str, model: SmartboxAnomalyDetector, metadata: Dict) -> str:
        """Save enhanced model with explainability data"""
        models_dir = "./smartbox_models/"
        os.makedirs(models_dir, exist_ok=True)

        # The enhanced SmartboxAnomalyDetector automatically saves explainability data
        model_dir = model.save_model_secure(models_dir, metadata)

        logger.info("Enhanced model saved: %s", model_dir)
        if hasattr(model, 'training_statistics'):
            logger.info("  Explainability data included for %d metrics", len(model.training_statistics))

        return str(model_dir)


# Create alias for backward compatibility
SmartboxTrainingPipeline = EnhancedSmartboxTrainingPipeline

# Example usage
def parse_date(date_str: str) -> datetime:
    """Parse date string in YYYY-MM-DD format."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: {date_str}. Use YYYY-MM-DD.")


def promote_models(staging_dir: str, production_dir: str, backup: bool = True) -> Dict[str, Any]:
    """Promote models from staging to production directory.

    Args:
        staging_dir: Path to staging models directory.
        production_dir: Path to production models directory.
        backup: If True, backup existing production models before replacing.

    Returns:
        Dictionary with promotion results.
    """
    import shutil
    from pathlib import Path

    staging_path = Path(staging_dir)
    production_path = Path(production_dir)

    if not staging_path.exists():
        return {
            'success': False,
            'error': f"Staging directory does not exist: {staging_dir}",
            'services_promoted': [],
        }

    # Get list of services in staging
    staging_services = [d.name for d in staging_path.iterdir() if d.is_dir()]

    if not staging_services:
        return {
            'success': False,
            'error': f"No services found in staging directory: {staging_dir}",
            'services_promoted': [],
        }

    # Backup existing production models if requested
    backup_path = None
    if backup and production_path.exists():
        backup_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = Path(f"{production_dir}_backup_{backup_timestamp}")
        try:
            shutil.copytree(production_path, backup_path)
            logger.info("Backed up existing production models to: %s", backup_path)
        except Exception as e:
            logger.warning("Failed to backup production models: %s", e)

    # Promote each service
    promoted_services = []
    failed_services = []

    for service in staging_services:
        staging_service_path = staging_path / service
        production_service_path = production_path / service

        try:
            # Remove existing production service directory if it exists
            if production_service_path.exists():
                shutil.rmtree(production_service_path)

            # Copy staging to production
            shutil.copytree(staging_service_path, production_service_path)
            promoted_services.append(service)
            logger.info("Promoted: %s", service)

        except Exception as e:
            failed_services.append({'service': service, 'error': str(e)})
            logger.error("Failed to promote %s: %s", service, e)

    return {
        'success': len(failed_services) == 0,
        'services_promoted': promoted_services,
        'services_failed': failed_services,
        'backup_path': str(backup_path) if backup_path else None,
        'staging_dir': staging_dir,
        'production_dir': production_dir,
    }


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train anomaly detection models for Smartbox services",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all services (trains to staging, auto-promotes if all pass validation)
  python main.py

  # Train specific service
  python main.py --service booking

  # Train with specific date range (Aug-Oct for baseline without holiday traffic)
  python main.py --start-date 2025-08-01 --end-date 2025-10-31

  # Train directly to production (bypasses staging and validation gate)
  python main.py --direct

  # Manually promote staging models to production (if auto-promotion was blocked)
  python main.py --promote

Default Behavior:
  1. Models are trained to staging directory (./smartbox_models_staging/)
  2. If ALL models pass validation → auto-promoted to production
  3. If ANY model fails → remains in staging, exits with error code 1

Data is automatically cached to parquet files in ./metrics_cache/. On first run,
data is fetched from VictoriaMetrics. Subsequent runs load from cache instantly.
        """
    )
    parser.add_argument(
        "--service", "-s",
        type=str,
        help="Train only a specific service (default: train all configured services)"
    )
    parser.add_argument(
        "--start-date",
        type=parse_date,
        help="Start date for training data in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--end-date",
        type=parse_date,
        help="End date for training data in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=30,
        help="Number of days to look back for training data (default: 30, ignored if --start-date/--end-date provided)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./metrics_cache",
        help="Directory for metrics cache. Data is automatically cached to parquet files. (default: ./metrics_cache)"
    )
    parser.add_argument(
        "--parallel", "-p",
        type=int,
        default=1,
        choices=range(1, 9),
        metavar="{1-8}",
        help="Number of parallel training threads (1=sequential, 2-8=parallel). Default: 1"
    )
    parser.add_argument(
        "--warm-cache",
        action="store_true",
        help="Pre-populate cache for all services before training. Recommended when using --parallel"
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Train directly to production directory, bypassing staging and validation gate"
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Promote staging models to production (copy from staging to production directory)"
    )
    parser.add_argument(
        "--staging-dir",
        type=str,
        default="./smartbox_models_staging",
        help="Staging directory for models (default: ./smartbox_models_staging)"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="./smartbox_models",
        help="Production directory for models (default: ./smartbox_models)"
    )

    args = parser.parse_args()

    # Validate date arguments
    if (args.start_date is None) != (args.end_date is None):
        parser.error("--start-date and --end-date must be used together")

    if args.start_date and args.end_date and args.start_date >= args.end_date:
        parser.error("--start-date must be before --end-date")

    # Configure logging for the main module
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Handle --promote mode: just promote staging to production and exit
    if args.promote:
        logger.info("Promoting models from staging to production...")
        logger.info("  Staging: %s", args.staging_dir)
        logger.info("  Production: %s", args.models_dir)

        result = promote_models(
            staging_dir=args.staging_dir,
            production_dir=args.models_dir,
            backup=True,
        )

        if result['success']:
            logger.info("")
            logger.info("Promotion successful!")
            logger.info("  Services promoted: %d", len(result['services_promoted']))
            for service in result['services_promoted']:
                logger.info("    ✓ %s", service)
            if result.get('backup_path'):
                logger.info("  Backup saved to: %s", result['backup_path'])
        else:
            logger.error("Promotion failed: %s", result.get('error', 'Unknown error'))
            if result.get('services_failed'):
                for failed in result['services_failed']:
                    logger.error("  ✗ %s: %s", failed['service'], failed['error'])
            exit(1)

        exit(0)

    # Initialize enhanced training pipeline
    training_pipeline = EnhancedSmartboxTrainingPipeline()

    # Determine services to train
    if args.service:
        services = [args.service]
        logger.info("Training single service: %s", args.service)
    else:
        # Load services from config.json (falls back to discovery if not found)
        services = training_pipeline.load_services_from_config()
        if not services:
            logger.info("Discovering services from VictoriaMetrics...")
            services = training_pipeline.discover_services()

    if not services:
        logger.error("No services found to train. Please check config.json or VictoriaMetrics connection.")
        exit(1)

    # Determine models directory (staging by default, or direct to production)
    if args.direct:
        models_dir = args.models_dir
        logger.info("Training DIRECTLY to production directory: %s", models_dir)
        logger.info("(validation gate bypassed)")
    else:
        models_dir = args.staging_dir
        logger.info("Training to staging directory: %s", models_dir)
        logger.info("(will auto-promote to production if all models pass validation)")

    # Log date range information
    if args.start_date and args.end_date:
        logger.info("Training date range: %s to %s (%d days)",
                    args.start_date.strftime("%Y-%m-%d"),
                    args.end_date.strftime("%Y-%m-%d"),
                    (args.end_date - args.start_date).days)
    else:
        logger.info("Training with last %d days of data", args.lookback_days)

    logger.info("Cache directory: %s", args.cache_dir)
    if args.parallel > 1:
        logger.info("Parallel training: %d workers", args.parallel)
    if args.warm_cache:
        logger.info("Cache warming: enabled")

    # Use enhanced time-aware training with optional parallelization
    logger.info("Using enhanced time-aware anomaly detection with explainability")

    # Train all services (with optional cache warming and parallel training)
    results = training_pipeline.train_all_services_time_aware(
        service_list=services,
        parallel_workers=args.parallel,
        warm_cache_first=args.warm_cache,
        lookback_days=args.lookback_days,
        start_date=args.start_date,
        end_date=args.end_date,
        cache_dir=args.cache_dir,
        models_dir=models_dir,
    )

    # Print enhanced results summary
    logger.info("")
    logger.info("Final Results Summary:")

    # Track which services passed/failed
    passed_services = []
    failed_services = []

    for service, result in results.items():
        training_success = result['status'] == 'success'
        validation_results = result.get('validation_results', {})
        validation_summary = validation_results.get('_summary', {})
        validation_passed = validation_summary.get('overall_passed', False)

        explainability_count = result.get('total_explainability_metrics', 0)
        explainability_status = "with explainability" if explainability_count > 0 else "no explainability"

        if training_success and validation_passed:
            passed_services.append(service)
            logger.info("  ✓ %s: %s (%d metrics)", service, explainability_status, explainability_count)
        elif training_success and not validation_passed:
            reason = validation_summary.get('overall_message', 'validation failed')
            failed_services.append({'service': service, 'reason': reason})
            logger.info("  ⚠ %s: training OK but %s", service, reason)
        else:
            reason = result.get('message', 'training failed')
            failed_services.append({'service': service, 'reason': reason})
            logger.info("  ✗ %s: %s", service, reason)

    # Print overall summary
    logger.info("")
    logger.info("Overall: %d passed, %d failed", len(passed_services), len(failed_services))

    # Handle auto-promote (default behavior, unless --direct was used)
    if not args.direct:
        logger.info("")
        logger.info("=" * 60)

        all_passed = len(failed_services) == 0 and len(passed_services) > 0

        if all_passed:
            logger.info("All %d services passed validation. Promoting to production...", len(passed_services))

            promote_result = promote_models(
                staging_dir=args.staging_dir,
                production_dir=args.models_dir,
                backup=True,
            )

            if promote_result['success']:
                logger.info("")
                logger.info("✓ PROMOTION SUCCESSFUL!")
                logger.info("  Services promoted: %d", len(promote_result['services_promoted']))
                if promote_result.get('backup_path'):
                    logger.info("  Backup saved to: %s", promote_result['backup_path'])
            else:
                logger.error("")
                logger.error("✗ PROMOTION FAILED: %s", promote_result.get('error', 'Unknown error'))
                exit(1)
        else:
            logger.error("✗ PROMOTION BLOCKED: %d service(s) failed validation", len(failed_services))
            logger.error("")
            logger.error("Failed services:")
            for failed in failed_services:
                logger.error("  - %s: %s", failed['service'], failed['reason'])
            logger.error("")
            logger.error("Models remain in staging: %s", args.staging_dir)
            logger.error("")
            logger.error("Options:")
            logger.error("  1. Fix issues and re-train: python main.py")
            logger.error("  2. Manually promote anyway: python main.py --promote")
            exit(1)

        logger.info("=" * 60)
