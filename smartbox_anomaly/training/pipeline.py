"""Training pipeline for Smartbox ML anomaly detection.

This module provides the main training pipeline for time-aware
anomaly detection models with dual-model architecture support.

Uses:
- MetricsCache for data loading (replaces SmartboxMetricsExtractor)
- SmartboxFeatureEngineer for feature engineering
- TrainingRunStorage for training history persistence
- TimeAwareAnomalyDetector for time-aware models
"""

from __future__ import annotations

import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from smartbox_anomaly.core.config import (
    ExcludedPeriod,
    TrainingConfig,
    get_config,
)
from smartbox_anomaly.core.constants import (
    TimePeriod,
    TrainingValidation,
)
from smartbox_anomaly.core.logging import get_logger
from smartbox_anomaly.detection.detector import SmartboxAnomalyDetector
from smartbox_anomaly.detection.time_aware import TimeAwareAnomalyDetector
from smartbox_anomaly.metrics.cache import MetricsCache
from smartbox_anomaly.metrics.client import VictoriaMetricsClient
from smartbox_anomaly.training.feature_engineering import SmartboxFeatureEngineer
from smartbox_anomaly.training.storage import TrainingRunStorage
from smartbox_anomaly.training.validators import (
    validate_model_split,
    validate_enhanced_model_split,
    validate_time_aware_models,
)

logger = get_logger(__name__)


class SmartboxTrainingPipeline:
    """Training pipeline for Smartbox anomaly detection models.

    Supports:
    - Time-aware 3-period models (business_hours, evening_hours, night_hours)
    - Dual-model architecture (baseline + holiday variants)
    - Metrics caching via MetricsCache
    - Training history tracking via TrainingRunStorage
    """

    def __init__(
        self,
        vm_endpoint: str | None = None,
        config: Dict[str, Any] | None = None,
        cache_dir: str = "./metrics_cache",
    ):
        """Initialize the training pipeline.

        Args:
            vm_endpoint: VictoriaMetrics endpoint URL. If None, loaded from config.
            config: Optional configuration dict. If None, loaded from get_config().
            cache_dir: Directory for metrics cache.
        """
        # Load config
        self.config = config or {}
        if not self.config:
            try:
                full_config = get_config()
                self.config = {
                    'lookback_days': TrainingValidation.DEFAULT_LOOKBACK_DAYS,
                    'min_data_points': 2000,
                    'validation_fraction': TrainingValidation.DEFAULT_VALIDATION_FRACTION,
                }
                # Merge training config from loaded config
                if hasattr(full_config, 'training'):
                    training_cfg = full_config.training
                    self.config['lookback_days'] = getattr(training_cfg, 'lookback_days', 60)
                    self.config['min_data_points'] = getattr(training_cfg, 'min_data_points', 2000)
                    self.config['validation_fraction'] = getattr(
                        training_cfg, 'validation_fraction',
                        TrainingValidation.DEFAULT_VALIDATION_FRACTION
                    )
            except Exception:
                self.config = {
                    'lookback_days': TrainingValidation.DEFAULT_LOOKBACK_DAYS,
                    'min_data_points': 2000,
                    'validation_fraction': TrainingValidation.DEFAULT_VALIDATION_FRACTION,
                }

        # Initialize VictoriaMetrics client
        if vm_endpoint is None:
            try:
                full_config = get_config()
                vm_endpoint = full_config.victoria_metrics.endpoint
            except Exception:
                vm_endpoint = "https://otel-metrics.production.smartbox.com"

        self.vm_client = VictoriaMetricsClient(endpoint=vm_endpoint)

        # Initialize metrics cache (replaces SmartboxMetricsExtractor)
        self.metrics_cache = MetricsCache(
            vm_client=self.vm_client,
            cache_dir=cache_dir,
        )

        # Initialize feature engineer
        self.feature_engineer = SmartboxFeatureEngineer()

        # Load training configuration
        self.training_config = self._load_training_config()

        # Store config object for validation access
        self.config['_training_config_obj'] = self.training_config

        logger.info("SmartboxTrainingPipeline initialized")
        logger.info("  VM endpoint: %s", vm_endpoint)
        logger.info("  Cache dir: %s", cache_dir)
        logger.info("  Lookback days: %d", self.config['lookback_days'])

    def _load_training_config(self) -> TrainingConfig:
        """Load training configuration from config.json."""
        try:
            config_path = Path("config.json")
            if config_path.exists():
                with open(config_path) as f:
                    config_data = json.load(f)

                training_data = config_data.get("training", {})

                # Parse excluded periods
                excluded_periods = []
                for period_data in training_data.get("excluded_periods", []):
                    excluded_periods.append(ExcludedPeriod.from_dict(period_data))

                return TrainingConfig(
                    lookback_days=training_data.get("lookback_days", 60),
                    min_data_points=training_data.get("min_data_points", 2000),
                    validation_fraction=training_data.get(
                        "validation_fraction",
                        TrainingValidation.DEFAULT_VALIDATION_FRACTION
                    ),
                    contamination_method=training_data.get(
                        "contamination_estimation", {}
                    ).get("method", "knee"),
                    contamination_fallback=training_data.get(
                        "contamination_estimation", {}
                    ).get("fallback", 0.05),
                    threshold_calibration_enabled=training_data.get(
                        "threshold_calibration", {}
                    ).get("enabled", True),
                    excluded_periods=excluded_periods,
                )
        except Exception as e:
            logger.warning("Failed to load training config: %s, using defaults", e)

        return TrainingConfig()

    def load_services_from_config(self) -> List[str]:
        """Load service list from config.json."""
        try:
            config_path = Path("config.json")
            if config_path.exists():
                with open(config_path) as f:
                    config_data = json.load(f)

                services_config = config_data.get("services", {})
                all_services = []
                for category in ["critical", "standard", "micro", "admin", "core", "background"]:
                    all_services.extend(services_config.get(category, []))
                return sorted(list(set(all_services)))
        except Exception as e:
            logger.warning("Failed to load services from config: %s", e)

        return []

    def discover_services(self) -> List[str]:
        """Discover available services from VictoriaMetrics."""
        query = 'group by (service_name) (http_requests:count:rate_5m)'
        result = self.vm_client.query(query)

        services = []
        if result.get('data', {}).get('result'):
            for item in result['data']['result']:
                service_name = item.get('metric', {}).get('service_name')
                if service_name:
                    services.append(service_name)

        services = sorted(list(set(services)))
        logger.info(
            "Discovered %d services: %s%s",
            len(services), services[:10], '...' if len(services) > 10 else ''
        )
        return services

    def train_service_model_time_aware(
        self,
        service_name: str,
        lookback_days: int | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        cache_dir: str = "./metrics_cache",
        models_dir: str = "./smartbox_models",
    ) -> Dict[str, Any]:
        """Train time-aware anomaly detection model with 3-period approach.

        Supports dual-model architecture:
        - Baseline model: Trained on normal operating data (excludes excluded_periods)
        - Holiday model: Trained on excluded period data (if sufficient)

        Args:
            service_name: Name of the service to train.
            lookback_days: Number of days of data to use.
            start_date: Optional explicit start date.
            end_date: Optional explicit end date.
            cache_dir: Directory for metrics cache.
            models_dir: Directory for saving models.

        Returns:
            Dict with training status and results.
        """
        training_start = time.time()
        effective_lookback = lookback_days or self.training_config.lookback_days

        logger.info("Training time-aware model for %s", service_name)
        logger.info("  Lookback: %d days", effective_lookback)

        # Determine date range
        if start_date is not None and end_date is not None:
            effective_start = start_date
            effective_end = end_date
        else:
            effective_end = datetime.now()
            effective_start = effective_end - timedelta(days=effective_lookback)

        logger.info("  Date range: %s to %s", effective_start.date(), effective_end.date())

        try:
            # Load metrics using cache
            raw_data = self.metrics_cache.get_data(
                service_name=service_name,
                start_date=effective_start,
                end_date=effective_end,
                auto_fetch=True,
            )

            if raw_data.empty:
                return {'status': 'no_data', 'message': 'No metrics data found'}

            if len(raw_data) < self.config['min_data_points']:
                return {
                    'status': 'insufficient_data',
                    'message': f'Only {len(raw_data)} data points, need {self.config["min_data_points"]}',
                    'data_points': len(raw_data),
                }

            logger.info("  Loaded %d data points", len(raw_data))

            # Separate baseline and excluded period data
            baseline_data, excluded_data = self._split_by_excluded_periods(
                raw_data, self.training_config.excluded_periods
            )

            logger.info("  Baseline data: %d points", len(baseline_data))
            logger.info("  Excluded period data: %d points", len(excluded_data))

            # Train baseline model
            baseline_result = self._train_time_aware_model(
                service_name=service_name,
                raw_data=baseline_data,
                models_dir=models_dir,
                model_variant=None,  # Standard baseline
            )

            # Optionally train holiday variant
            holiday_result = None
            min_holiday_points = TrainingValidation.MIN_PERIOD_SAMPLES * 3
            if len(excluded_data) >= min_holiday_points:
                logger.info("  Training holiday variant (%d points)", len(excluded_data))
                holiday_result = self._train_time_aware_model(
                    service_name=service_name,
                    raw_data=excluded_data,
                    models_dir=models_dir,
                    model_variant="holiday",
                )

            training_duration = time.time() - training_start

            return {
                'status': 'success',
                'service_name': service_name,
                'baseline_result': baseline_result,
                'holiday_result': holiday_result,
                'data_points': len(raw_data),
                'baseline_points': len(baseline_data),
                'excluded_points': len(excluded_data),
                'training_duration_seconds': training_duration,
                'validation_results': baseline_result.get('validation_results', {}),
                'total_explainability_metrics': baseline_result.get('explainability_metrics', 0),
            }

        except Exception as e:
            logger.exception("Training failed for %s: %s", service_name, e)
            return {
                'status': 'error',
                'message': str(e),
                'service_name': service_name,
            }

    def _split_by_excluded_periods(
        self,
        data: pd.DataFrame,
        excluded_periods: List[ExcludedPeriod],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into baseline and excluded period data.

        Args:
            data: DataFrame with datetime index.
            excluded_periods: List of ExcludedPeriod objects.

        Returns:
            Tuple of (baseline_data, excluded_data).
        """
        if not excluded_periods:
            return data, pd.DataFrame()

        # Create mask for excluded periods
        is_excluded = pd.Series(False, index=data.index)
        for period in excluded_periods:
            is_excluded = is_excluded | data.index.to_series().apply(period.contains)

        baseline_data = data[~is_excluded]
        excluded_data = data[is_excluded]

        return baseline_data, excluded_data

    def _train_time_aware_model(
        self,
        service_name: str,
        raw_data: pd.DataFrame,
        models_dir: str,
        model_variant: str | None = None,
    ) -> Dict[str, Any]:
        """Train a time-aware model on the provided data.

        Args:
            service_name: Service name.
            raw_data: Raw metrics DataFrame.
            models_dir: Directory for saving models.
            model_variant: Optional variant name (e.g., "holiday").

        Returns:
            Training result dict.
        """
        # Engineer features using temporal split to prevent leakage
        train_features, validation_features = self.feature_engineer.engineer_features_split(
            raw_data,
            validation_fraction=self.config.get(
                'validation_fraction',
                TrainingValidation.DEFAULT_VALIDATION_FRACTION
            ),
        )

        if train_features.empty:
            return {
                'status': 'insufficient_features',
                'message': 'Feature engineering produced no data',
            }

        # Create time-aware detector
        detector = TimeAwareAnomalyDetector(
            service_name=service_name,
        )

        # Train on the features
        periods_trained = detector.train_time_aware_models(train_features)

        if not periods_trained:
            return {
                'status': 'no_periods_trained',
                'message': 'No time periods had sufficient data',
            }

        # Validate the model
        validation_results = validate_time_aware_models(
            detector=detector,
            features_df=validation_features,
        )

        # Save the model
        if model_variant:
            save_dir = Path(models_dir) / f"_{model_variant}_variant" / service_name
        else:
            save_dir = Path(models_dir) / service_name

        save_dir.mkdir(parents=True, exist_ok=True)
        detector.save_models(str(save_dir))

        logger.info("  Saved model to %s", save_dir)

        return {
            'status': 'success',
            'periods_trained': periods_trained,
            'validation_results': validation_results,
            'explainability_metrics': len(detector.training_statistics) if hasattr(detector, 'training_statistics') else 0,
            'model_path': str(save_dir),
        }

    def train_all_services_time_aware(
        self,
        service_list: List[str] | None = None,
        parallel_workers: int = 1,
        warm_cache_first: bool = False,
        lookback_days: int | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        cache_dir: str = "./metrics_cache",
        models_dir: str = "./smartbox_models",
    ) -> Dict[str, Any]:
        """Train time-aware models for all services.

        Args:
            service_list: List of services to train (defaults to discovered services).
            parallel_workers: Number of parallel training threads.
            warm_cache_first: If True, pre-populate cache before training.
            lookback_days: Number of days to look back.
            start_date: Optional explicit start date.
            end_date: Optional explicit end date.
            cache_dir: Directory for metrics cache.
            models_dir: Directory for saving models.

        Returns:
            Dict with training results for all services.
        """
        if service_list is None:
            service_list = self.load_services_from_config() or self.discover_services()

        # Determine effective date range
        effective_lookback = lookback_days or self.training_config.lookback_days

        if start_date is not None and end_date is not None:
            effective_start = start_date
            effective_end = end_date
        else:
            effective_end = datetime.now()
            effective_start = effective_end - timedelta(days=effective_lookback)

            # Compensate for excluded periods
            if self.training_config.excluded_periods:
                excluded_days = self._calculate_excluded_days(
                    effective_start, effective_end, self.training_config.excluded_periods
                )
                if excluded_days > 0:
                    effective_start = effective_start - timedelta(days=excluded_days)
                    logger.info("Extended lookback by %d days for excluded periods", excluded_days)

        # Generate run ID
        run_id = uuid.uuid4().hex[:8]
        batch_start_time = time.time()
        started_at = datetime.now()

        logger.info("=" * 70)
        logger.info("TRAINING RUN: %s", run_id)
        logger.info("  Started: %s", started_at.strftime("%Y-%m-%d %H:%M:%S"))
        logger.info("  Services: %d", len(service_list))
        logger.info("  Mode: %s", f"parallel={parallel_workers}" if parallel_workers > 1 else "sequential")
        logger.info("=" * 70)

        # Optional cache warming
        if warm_cache_first:
            self._warm_cache_for_services(
                services=service_list,
                start_date=effective_start,
                end_date=effective_end,
                cache_dir=cache_dir,
            )

        results = {}
        successful_trainings = 0
        validation_passed_services = 0
        total_explainability_metrics = 0

        def train_single_service(service: str) -> Tuple[str, Dict]:
            """Train a single service."""
            result = self.train_service_model_time_aware(
                service,
                lookback_days=effective_lookback,
                start_date=start_date,
                end_date=end_date,
                cache_dir=cache_dir,
                models_dir=models_dir,
            )
            return service, result

        # Train services
        if parallel_workers > 1 and len(service_list) > 1:
            import threading
            results_lock = threading.Lock()

            with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
                futures = {executor.submit(train_single_service, svc): svc for svc in service_list}

                for i, future in enumerate(as_completed(futures), 1):
                    service, result = future.result()
                    with results_lock:
                        results[service] = result
                        if result['status'] == 'success':
                            successful_trainings += 1
                            total_explainability_metrics += result.get('total_explainability_metrics', 0)
                            validation = result.get('validation_results', {})
                            if validation.get('_summary', {}).get('overall_passed', False):
                                validation_passed_services += 1
                            logger.info("[%d/%d] %s: Success", i, len(service_list), service)
                        else:
                            logger.error("[%d/%d] %s: %s", i, len(service_list), service, result['status'])
        else:
            for i, service in enumerate(service_list, 1):
                logger.info("[%d/%d] Processing %s", i, len(service_list), service)
                service, result = train_single_service(service)
                results[service] = result

                if result['status'] == 'success':
                    successful_trainings += 1
                    total_explainability_metrics += result.get('total_explainability_metrics', 0)
                    validation = result.get('validation_results', {})
                    if validation.get('_summary', {}).get('overall_passed', False):
                        validation_passed_services += 1
                    logger.info("  Success")
                else:
                    logger.error("  %s: %s", result['status'], result.get('message', ''))

        batch_duration = time.time() - batch_start_time
        failed_trainings = len(service_list) - successful_trainings

        # Log summary
        logger.info("")
        logger.info("=" * 70)
        logger.info("TRAINING COMPLETE: %s", run_id)
        logger.info("  Duration: %.1fs", batch_duration)
        logger.info("  Successful: %d/%d", successful_trainings, len(service_list))
        logger.info("  Validation passed: %d/%d", validation_passed_services, successful_trainings)
        logger.info("  Explainability metrics: %d", total_explainability_metrics)
        logger.info("=" * 70)

        return results

    def _calculate_excluded_days(
        self,
        start_date: datetime,
        end_date: datetime,
        excluded_periods: List[ExcludedPeriod],
    ) -> int:
        """Calculate total excluded days that overlap with the date range."""
        excluded_days = 0
        for period in excluded_periods:
            try:
                period_start = datetime.fromisoformat(period.start)
                period_end = datetime.fromisoformat(period.end)
                overlap_start = max(start_date, period_start)
                overlap_end = min(end_date, period_end)
                if overlap_start < overlap_end:
                    excluded_days += (overlap_end - overlap_start).days
            except (ValueError, TypeError):
                continue
        return excluded_days

    def _warm_cache_for_services(
        self,
        services: List[str],
        start_date: datetime,
        end_date: datetime,
        cache_dir: str,
        parallel_fetch: bool = True,
    ) -> Dict[str, Dict[str, int]]:
        """Pre-populate metrics cache for all services.

        Args:
            services: List of service names.
            start_date: Start date for data.
            end_date: End date for data.
            cache_dir: Cache directory.
            parallel_fetch: If True, fetch multiple services concurrently.

        Returns:
            Dict mapping service names to cache results.
        """
        logger.info("CACHE WARMING PHASE")
        logger.info("  Services: %d", len(services))
        logger.info("  Date range: %s to %s", start_date.date(), end_date.date())

        cache_results = {}
        cache_start_time = time.time()

        # Use existing metrics cache
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
            max_workers = min(2, len(services))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(prefetch_service, svc): svc for svc in services}
                for i, future in enumerate(as_completed(futures), 1):
                    service_name, result = future.result()
                    cache_results[service_name] = result
                    total_points = sum(v for v in result.values() if isinstance(v, int))
                    logger.info("[%d/%d] Cached %s: %d points", i, len(services), service_name, total_points)
        else:
            for i, service in enumerate(services, 1):
                service_name, result = prefetch_service(service)
                cache_results[service_name] = result
                if i < len(services):
                    time.sleep(0.5)

        cache_duration = time.time() - cache_start_time
        logger.info("Cache warming complete in %.1fs", cache_duration)

        return cache_results


# Alias for backward compatibility
EnhancedSmartboxTrainingPipeline = SmartboxTrainingPipeline
