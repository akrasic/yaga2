"""
TimeAwareAnomalyDetector - Time-period-specific anomaly detection.

This module provides time-aware detection with:
- Separate models for different time periods (business, evening, night, weekend)
- Lazy model loading for efficiency
- Automatic fallback between time periods
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from smartbox_anomaly.core.constants import (
    PERIOD_CONFIDENCE_SCORES,
    PERIOD_FALLBACK_MAP,
    TimePeriod,
)
from smartbox_anomaly.core.logging import get_logger
from smartbox_anomaly.core.types import DependencyContext
from smartbox_anomaly.core.utils import get_time_period, parse_service_model
from smartbox_anomaly.detection.detector import SmartboxAnomalyDetector
from smartbox_anomaly.detection.service_config import (
    get_min_samples_for_period,
    get_validation_thresholds,
)

if TYPE_CHECKING:
    import pandas as pd

logger = get_logger(__name__)


class TimeAwareAnomalyDetector:
    """Time-aware anomaly detector with lazy loading and fallback support.

    This detector maintains separate models for different time periods
    (business hours, evening, night, weekend day, weekend night) and
    automatically selects the appropriate model based on the current time.

    Features:
    - Lazy loading: Models are loaded only when needed
    - Fallback support: Uses fallback periods if primary model unavailable
    - Confidence scoring: Reports confidence based on model match

    Example:
        >>> detector = TimeAwareAnomalyDetector("booking")
        >>> detector.load_from_directory("./smartbox_models/")
        >>> result = detector.detect(current_metrics)
    """

    TIME_PERIODS: dict[str, dict[str, Any]] = {
        "business_hours": {"start": 8, "end": 18, "weekdays_only": True},
        "evening_hours": {"start": 18, "end": 22, "weekdays_only": True},
        "night_hours": {"start": 22, "end": 6, "weekdays_only": True},
        "weekend_day": {"start": 8, "end": 22, "weekends_only": True},
        "weekend_night": {"start": 22, "end": 8, "weekends_only": True},
    }

    def __init__(self, service_name: str) -> None:
        """Initialize the time-aware detector.

        Args:
            service_name: Base service name (without time period suffix).
        """
        self.service_name = service_name
        self.models: dict[str, SmartboxAnomalyDetector] = {}
        self._available_periods: set[str] = set()
        self._models_directory: Path | None = None
        self._loaded_periods: set[str] = set()
        self.validation_thresholds = get_validation_thresholds(service_name)

    @property
    def available_periods(self) -> list[str]:
        """Get list of available time periods."""
        return sorted(self._available_periods)

    def train_time_aware_models(
        self,
        features_df: pd.DataFrame,
        validation_fraction: float = 0.2,
    ) -> dict[str, SmartboxAnomalyDetector]:
        """Train separate models for each time period with temporal validation split.

        Args:
            features_df: Training data with datetime index.
            validation_fraction: Fraction of data to hold out for validation (0-0.5).
                               Uses temporal split: first (1-fraction) for training,
                               last fraction for validation.

        Returns:
            Dictionary mapping period names to trained detectors.

        Raises:
            ValueError: If no models could be trained.
        """
        if features_df.empty:
            raise ValueError("No training data provided")

        # Clamp validation fraction to reasonable range
        validation_fraction = max(0.0, min(0.5, validation_fraction))

        features_df = features_df.copy()
        features_df["time_period"] = features_df.index.map(
            lambda ts: get_time_period(ts).value if isinstance(ts, datetime) else "business_hours"
        )

        trained_models = {}

        for period in self.TIME_PERIODS:
            period_data = features_df[features_df["time_period"] == period]
            min_samples = get_min_samples_for_period(self.service_name, period)

            if len(period_data) < min_samples:
                logger.warning(
                    f"Insufficient data for {period}: {len(period_data)} samples "
                    f"(need {min_samples})"
                )
                continue

            # Temporal split: first (1-fraction) for training, last fraction for validation
            # Sort by index (datetime) to ensure temporal ordering
            period_data_sorted = period_data.sort_index()
            split_idx = int(len(period_data_sorted) * (1 - validation_fraction))

            train_data = period_data_sorted.iloc[:split_idx]
            validation_data = period_data_sorted.iloc[split_idx:] if validation_fraction > 0 else None

            # Check if we have enough training data after split
            if len(train_data) < min_samples * 0.5:  # Allow slightly less for training
                logger.warning(
                    f"Insufficient training data for {period} after split: "
                    f"{len(train_data)} samples"
                )
                continue

            logger.info(
                f"Training model for {period} ({len(train_data)} train, "
                f"{len(validation_data) if validation_data is not None else 0} validation samples)"
            )

            model = SmartboxAnomalyDetector(f"{self.service_name}_{period}")

            # Train with validation data for threshold calibration
            train_result = model.train(
                train_data.drop(columns=["time_period"]),
                validation_df=validation_data.drop(columns=["time_period"]) if validation_data is not None and len(validation_data) > 20 else None,
            )

            trained_models[period] = model
            self.models[period] = model
            self._available_periods.add(period)
            self._loaded_periods.add(period)  # Mark as loaded since it's in memory

            # Log validation results if available
            val_results = train_result.get("validation_results", {})
            if val_results:
                logger.info(f"Trained model for {period} (validation calibrated)")
            else:
                logger.info(f"Trained model for {period}")

        if not trained_models:
            raise ValueError("No models trained. Insufficient data for all time periods.")

        logger.info(f"Trained {len(trained_models)} time-aware models for {self.service_name}")
        return trained_models

    def save_models(
        self,
        directory: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        """Save all trained models to disk.

        Args:
            directory: Base directory for saving models.
            metadata: Additional metadata to include.

        Returns:
            Dictionary mapping periods to saved paths.
        """
        saved_paths = {}

        for period, model in self.models.items():
            model_path = model.save_model(directory, metadata)
            saved_paths[period] = str(model_path)

        return saved_paths

    def load_from_directory(self, directory: str) -> None:
        """Discover available models in a directory.

        This method scans the directory for available period models
        but does not load them until needed (lazy loading).

        Args:
            directory: Directory containing model subdirectories.
        """
        self._models_directory = Path(directory)
        self._available_periods.clear()
        self._loaded_periods.clear()
        self.models.clear()

        if not self._models_directory.exists():
            logger.warning(f"Models directory does not exist: {directory}")
            return

        for item in self._models_directory.iterdir():
            if not item.is_dir():
                continue

            # Check if this is a model for our service
            model_data_file = item / "model_data.json"
            if not model_data_file.exists():
                continue

            dir_name = item.name
            if not dir_name.startswith(self.service_name):
                continue

            # Extract period from directory name
            base_name, period = parse_service_model(dir_name)
            if base_name == self.service_name and period in self.TIME_PERIODS:
                self._available_periods.add(period)

        logger.info(
            f"Discovered {len(self._available_periods)} models for {self.service_name}: "
            f"{sorted(self._available_periods)}"
        )

    def _lazy_load_model(self, period: str) -> SmartboxAnomalyDetector | None:
        """Load a model for a specific period (lazy loading).

        Args:
            period: Time period to load.

        Returns:
            Loaded detector or None if not available.
        """
        if period in self._loaded_periods:
            return self.models.get(period)

        if period not in self._available_periods:
            return None

        if self._models_directory is None:
            return None

        try:
            model_name = f"{self.service_name}_{period}"
            model = SmartboxAnomalyDetector.load_model(
                str(self._models_directory),
                model_name,
            )
            self.models[period] = model
            self._loaded_periods.add(period)
            logger.debug(f"Lazy-loaded model for {period}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model for {period}: {e}")
            return None

    def get_current_period(self, timestamp: datetime | None = None) -> TimePeriod:
        """Get the time period for a timestamp.

        Args:
            timestamp: Timestamp to classify (defaults to now).

        Returns:
            TimePeriod enum value.
        """
        if timestamp is None:
            timestamp = datetime.now()
        return get_time_period(timestamp)

    def detect(
        self,
        metrics: dict[str, float],
        timestamp: datetime | None = None,
        dependency_context: DependencyContext | None = None,
        check_drift: bool = False,
    ) -> dict[str, Any]:
        """Detect anomalies using the appropriate time-period model.

        Args:
            metrics: Current metric values.
            timestamp: Timestamp for period selection (defaults to now).
            dependency_context: Optional dependency status for cascade detection.
            check_drift: Whether to check for distribution drift from training data.

        Returns:
            Detection result with anomalies and metadata.
        """
        if timestamp is None:
            timestamp = datetime.now()

        current_period = self.get_current_period(timestamp)
        model, used_period, confidence = self._get_best_model(current_period.value)

        if model is None:
            return {
                "anomalies": {},
                "metadata": {
                    "service_name": self.service_name,
                    "timestamp": timestamp.isoformat(),
                    "status": "no_model",
                    "error": f"No model available for {self.service_name}",
                },
            }

        # Run detection with optional dependency context and drift checking
        result = model.detect(metrics, timestamp, dependency_context, check_drift=check_drift)

        # Enhance metadata
        result["metadata"].update({
            "time_period": current_period.value,
            "model_period": used_period,
            "is_fallback": used_period != current_period.value,
            "confidence": confidence,
            "lazy_loaded": used_period in self._loaded_periods,
            "available_periods": list(self._available_periods),
        })

        # Adjust severity based on time period characteristics
        self._adjust_for_time_period(result, current_period, confidence)

        # Add training baselines for SLO evaluation
        if model is not None and hasattr(model, "training_statistics") and model.training_statistics:
            training_baselines = {}
            for metric_name, stats in model.training_statistics.items():
                if hasattr(stats, "mean"):
                    training_baselines[f"{metric_name}_mean"] = stats.mean
            if training_baselines:
                result["training_baselines"] = training_baselines
                # Also add to metrics dict for SLO evaluator compatibility
                result["metrics"] = {**metrics, **training_baselines}

        return result

    def _get_best_model(
        self,
        target_period: str,
    ) -> tuple[SmartboxAnomalyDetector | None, str, float]:
        """Get the best available model for a target period.

        Uses fallback chain if target period model is not available.

        Args:
            target_period: Target time period.

        Returns:
            Tuple of (model, used_period, confidence).
        """
        # Try exact match first
        if target_period in self._available_periods:
            model = self._lazy_load_model(target_period)
            if model is not None:
                confidence = PERIOD_CONFIDENCE_SCORES.get(target_period, 0.85)
                return model, target_period, confidence

        # Try fallbacks
        fallback_chain: tuple[str, ...] = PERIOD_FALLBACK_MAP.get(target_period, ())
        for fallback_period in fallback_chain:
            if fallback_period in self._available_periods:
                model = self._lazy_load_model(fallback_period)
                if model is not None:
                    # Reduce confidence for fallback
                    base_confidence = PERIOD_CONFIDENCE_SCORES.get(fallback_period, 0.75)
                    confidence = base_confidence * 0.85
                    logger.debug(
                        f"Using fallback model {fallback_period} for {target_period} "
                        f"(confidence: {confidence:.2f})"
                    )
                    return model, fallback_period, confidence

        # No model available
        return None, "", 0.0

    def _adjust_for_time_period(
        self,
        result: dict[str, Any],
        period: TimePeriod,
        confidence: float,
    ) -> None:
        """Adjust detection results based on time period characteristics.

        Args:
            result: Detection result to modify.
            period: Current time period.
            confidence: Model confidence score.
        """
        anomalies = result.get("anomalies", {})

        for _name, anomaly in anomalies.items():
            # Add time context
            anomaly["time_period"] = period.value
            anomaly["time_confidence"] = confidence

            # Adjust severity for low-confidence detections
            if confidence < 0.7:
                current_severity = anomaly.get("severity", "medium")
                severity_order = ["low", "medium", "high", "critical"]

                if current_severity in severity_order:
                    idx = severity_order.index(current_severity)
                    if idx > 0:
                        anomaly["severity"] = severity_order[idx - 1]
                        anomaly["severity_adjusted"] = True
                        anomaly["original_severity"] = current_severity

            # Add period-specific context
            if period.is_weekend:
                anomaly["period_context"] = "Weekend activity typically differs from weekdays"
            elif period.is_night:
                anomaly["period_context"] = "Night hours may have different traffic patterns"

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about the time-aware detector.

        Returns:
            Dictionary with detector statistics.
        """
        return {
            "service_name": self.service_name,
            "available_periods": list(self._available_periods),
            "loaded_periods": list(self._loaded_periods),
            "total_available": len(self._available_periods),
            "total_loaded": len(self._loaded_periods),
            "validation_thresholds": self.validation_thresholds,
        }

    # Backward compatibility methods
    def get_time_period(self, timestamp: datetime | None = None) -> str:
        """Get time period string for a timestamp (backward compatibility)."""
        return self.get_current_period(timestamp).value

    def _discover_available_periods(self, directory: str) -> set[str]:
        """Discover available periods (backward compatibility)."""
        self.load_from_directory(directory)
        return self._available_periods

    def _find_fallback_period(self, target_period: str) -> str | None:
        """Find fallback period (backward compatibility)."""
        fallback_chain = PERIOD_FALLBACK_MAP.get(target_period, ())
        for fallback in fallback_chain:
            if fallback in self._available_periods:
                return fallback
        return None

    def _get_service_type(self) -> str:
        """Get service type for validation thresholds."""
        name = self.service_name.lower()
        if "admin" in name or "adm" in name:
            return "admin_service"
        elif name in ("mobile-api", "shire-api"):
            return "high_volume"
        elif len(name) <= 5 or "-" in name:
            return "micro_service"
        return "standard"

    def _get_period_type(self, period: str) -> str:
        """Get period type (weekday/weekend)."""
        if period.startswith("weekend"):
            return "weekend"
        return "weekday"

    def _get_min_samples_for_period(self, period: str) -> int:
        """Get minimum samples for period (backward compatibility)."""
        return get_min_samples_for_period(self.service_name, period)

    def detect_anomalies(
        self,
        metrics: dict[str, float],
        timestamp: datetime | None = None,
        models_directory: str | None = None,
        verbose: bool = False,
        check_drift: bool = False,
    ) -> dict[str, Any]:
        """Detect anomalies (backward compatibility method).

        Args:
            metrics: Current metric values.
            timestamp: Timestamp for period selection.
            models_directory: Optional models directory to load from.
            verbose: Enable verbose logging.
            check_drift: Whether to check for distribution drift.

        Returns:
            Dictionary of detected anomalies.
        """
        if models_directory and self._models_directory is None:
            self.load_from_directory(models_directory)

        result = self.detect(metrics, timestamp, check_drift=check_drift)
        return result.get("anomalies", {})

    def detect_anomalies_with_context(
        self,
        metrics: dict[str, float],
        timestamp: datetime | None = None,
        models_directory: str | None = None,
        verbose: bool = False,
        dependency_context: DependencyContext | None = None,
        check_drift: bool = False,
    ) -> dict[str, Any]:
        """Detect anomalies with full context (backward compatibility).

        Args:
            metrics: Current metric values.
            timestamp: Timestamp for period selection.
            models_directory: Optional models directory to load from.
            verbose: Enable verbose logging.
            dependency_context: Optional dependency status for cascade detection.
            check_drift: Whether to check for distribution drift.

        Returns:
            Full detection result with context.
        """
        if timestamp is None:
            timestamp = datetime.now()

        if models_directory and self._models_directory is None:
            self.load_from_directory(models_directory)

        result = self.detect(metrics, timestamp, dependency_context, check_drift=check_drift)
        anomalies = result.get("anomalies", {})
        metadata = result.get("metadata", {})

        # Determine overall severity
        severity_order = ["low", "medium", "high", "critical"]
        max_severity = "low"
        for anomaly in anomalies.values():
            sev = anomaly.get("severity", "low")
            if sev in severity_order and severity_order.index(sev) > severity_order.index(max_severity):
                max_severity = sev

        # Build response with drift analysis if available
        response = {
            "service": self.service_name,
            "timestamp": timestamp.isoformat(),
            "time_period": metadata.get("time_period", "unknown"),
            "model_type": "time_aware_5period",
            "anomaly_count": len(anomalies),
            "anomalies": anomalies,
            "overall_severity": max_severity,
            "current_metrics": metrics,
            "metrics": metrics,
            "explainable": True,
            "performance_info": {
                "lazy_loaded": True,
                "models_loaded": list(self._loaded_periods),
                "period_used": metadata.get("model_period"),
                "total_available": len(self._available_periods),
                "drift_check_enabled": check_drift,
            },
        }

        # Include drift analysis if present
        if "drift_analysis" in result:
            response["drift_analysis"] = result["drift_analysis"]

        # Add training baselines for SLO evaluation
        # Extract from the model's training_statistics (not validation_metrics!)
        model_period = metadata.get("model_period")
        if model_period and model_period in self.models:
            model = self.models[model_period]
            if hasattr(model, "training_statistics") and model.training_statistics:
                training_baselines = {}
                for metric_name, stats in model.training_statistics.items():
                    if hasattr(stats, "mean"):
                        training_baselines[f"{metric_name}_mean"] = stats.mean
                if training_baselines:
                    response["training_baselines"] = training_baselines
                    # Also add to metrics dict for SLO evaluator compatibility
                    response["metrics"] = {**metrics, **training_baselines}

        return response

    @classmethod
    def load_models(
        cls,
        directory: str,
        service_name: str,
        verbose: bool = False,
    ) -> TimeAwareAnomalyDetector:
        """Load time-aware models from directory (backward compatibility).

        Args:
            directory: Models directory.
            service_name: Base service name.
            verbose: Enable verbose logging.

        Returns:
            Configured TimeAwareAnomalyDetector instance.
        """
        detector = cls(service_name)
        detector.load_from_directory(directory)
        return detector


def create_time_aware_detector(
    service_name: str,
    models_directory: str | None = None,
) -> TimeAwareAnomalyDetector:
    """Factory function to create a time-aware detector.

    Args:
        service_name: Base service name.
        models_directory: Optional directory to load models from.

    Returns:
        Configured TimeAwareAnomalyDetector instance.
    """
    detector = TimeAwareAnomalyDetector(service_name)

    if models_directory:
        detector.load_from_directory(models_directory)

    return detector
