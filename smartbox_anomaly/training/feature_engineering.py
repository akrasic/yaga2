"""Feature engineering for ML anomaly detection training.

This module provides feature engineering utilities for transforming
raw metrics into features suitable for Isolation Forest training.
Features include:
- Time-based features (hour, day of week, business hours)
- Derived correlation features (latency ratios, efficiency)
- Rolling statistical features (mean, std, max, min over windows)
- Anomaly indicator features (percentile-based flags)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from smartbox_anomaly.core.constants import TrainingValidation
from smartbox_anomaly.core.logging import get_logger

logger = get_logger(__name__)


class SmartboxFeatureEngineer:
    """Feature engineering for Smartbox anomaly detection.

    Creates ML features from raw metrics with proper handling of
    temporal splits to prevent data leakage in rolling features.
    """

    def __init__(self):
        """Initialize feature engineer with window configurations."""
        self.feature_windows = ['5T', '15T', '1H']  # 5min, 15min, 1hour
        self.base_metrics = [
            'request_rate',
            'application_latency',
            'dependency_latency',
            'database_latency',
            'error_rate',
        ]

    def engineer_features(
        self,
        metrics_df: pd.DataFrame,
        include_rolling: bool = True,
    ) -> pd.DataFrame:
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
        validation_fraction: float | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Engineer features with proper temporal split to prevent leakage.

        Rolling features are computed separately on train and validation sets,
        preventing information from validation leaking into training statistics.

        Args:
            metrics_df: Raw metrics DataFrame with timestamp index.
            validation_fraction: Fraction of data to use for validation (from end).
                               Defaults to TrainingValidation.DEFAULT_VALIDATION_FRACTION.

        Returns:
            Tuple of (train_features, validation_features) with rolling features
            computed separately on each split.
        """
        if validation_fraction is None:
            validation_fraction = TrainingValidation.DEFAULT_VALIDATION_FRACTION

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
        warmup_periods = TrainingValidation.VALIDATION_WARMUP_ROWS
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

        Args:
            features: DataFrame with base features.

        Returns:
            DataFrame with added rolling features.
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
        """Clean up features by handling NaN and inf values.

        Args:
            features: DataFrame with potentially NaN/inf values.

        Returns:
            Cleaned DataFrame with NaN/inf handled.
        """
        features = features.replace([np.inf, -np.inf], np.nan)

        # Remove rows with too many NaNs (from rolling windows)
        valid_threshold = int(len(features.columns) * 0.5)  # At least 50% valid values
        features = features.dropna(thresh=valid_threshold)

        # Fill any remaining NaNs with 0
        features = features.fillna(0)

        return features

    def _window_to_periods(self, window: str) -> int:
        """Convert window string to number of periods (5min intervals).

        Args:
            window: Window string (e.g., '5T', '15T', '1H').

        Returns:
            Number of 5-minute periods in the window.
        """
        if window == '5T':
            return 1
        elif window == '15T':
            return 3
        elif window == '1H':
            return 12
        else:
            return 1
