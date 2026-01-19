"""
Enhanced time-aware detector with efficient lazy loading.

Supports dual-model architecture (baseline + holiday variants) when excluded
periods are configured. Holiday variant models are used when the detection
timestamp falls within an excluded period.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from smartbox_anomaly.core import DependencyContext, ModelLoadError
from smartbox_anomaly.core.config import ExcludedPeriod

logger = logging.getLogger(__name__)


class EnhancedTimeAwareDetector:
    """Enhanced time-aware detector with efficient lazy loading.

    Supports dual-model architecture for handling seasonal/holiday traffic patterns:
    - Baseline models: Trained on normal operating periods
    - Holiday variant models: Trained on excluded periods (e.g., Christmas)

    The appropriate model variant is selected based on the detection timestamp
    and the configured excluded periods.
    """

    def __init__(
        self,
        models_directory: str,
        excluded_periods: Sequence[ExcludedPeriod] | None = None,
    ):
        """Initialize the enhanced time-aware detector.

        Args:
            models_directory: Directory containing trained models.
            excluded_periods: Optional sequence of excluded periods for holiday
                variant selection. When detection timestamp falls within an
                excluded period, holiday variant models will be used if available.
        """
        self.models_directory = models_directory
        self._excluded_periods: tuple[ExcludedPeriod, ...] = (
            tuple(excluded_periods) if excluded_periods else ()
        )
        self._detector_cache: Dict[str, Any] = {}
        self._load_times: Dict[str, float] = {}

    def set_excluded_periods(
        self,
        excluded_periods: Sequence[ExcludedPeriod],
    ) -> None:
        """Set excluded periods for holiday variant selection.

        This also updates any cached detectors with the new periods.

        Args:
            excluded_periods: Sequence of excluded periods.
        """
        self._excluded_periods = tuple(excluded_periods)
        # Update cached detectors with new excluded periods
        for detector in self._detector_cache.values():
            if hasattr(detector, "set_excluded_periods"):
                detector.set_excluded_periods(self._excluded_periods)

    def _calculate_drift_penalty(self, drift_score: float) -> float:
        """Calculate confidence penalty based on drift severity.

        Args:
            drift_score: The overall drift score (z-score based)

        Returns:
            Confidence penalty factor (0.0 to 0.3)
        """
        if drift_score > 5:
            return 0.3  # 30% confidence reduction for severe drift
        elif drift_score > 3:
            return 0.15  # 15% reduction for moderate drift
        return 0.0

    def _apply_drift_adjustments(
        self,
        result: Dict[str, Any],
        drift_analysis: Dict[str, Any],
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Apply drift-based adjustments to detection results.

        When drift is detected, this method:
        1. Adds a warning to the result
        2. Adjusts confidence scores downward
        3. Marks affected anomalies with drift_warning flag

        Args:
            result: The detection result dict
            drift_analysis: Drift analysis from the detector
            verbose: Whether to log adjustments

        Returns:
            Modified result dict with drift adjustments
        """
        if not drift_analysis.get("has_drift", False):
            return result

        drift_score = drift_analysis.get("overall_drift_score", 0.0)
        penalty = self._calculate_drift_penalty(drift_score)

        if penalty == 0:
            return result

        # Add drift warning to result
        result["drift_warning"] = {
            "type": "model_drift",
            "overall_drift_score": drift_score,
            "recommendation": drift_analysis.get("recommendation", "Monitor closely"),
            "affected_metrics": list(drift_analysis.get("drift_metrics", {}).keys()),
            "confidence_penalty_applied": penalty,
            "multivariate_drift": drift_analysis.get("multivariate_drift", False),
        }

        if verbose:
            logger.warning(
                f"Drift detected (score={drift_score:.2f}), "
                f"applying {penalty*100:.0f}% confidence penalty"
            )

        # Adjust confidence scores in anomalies
        anomalies = result.get("anomalies", {})
        for anomaly_name, anomaly_data in anomalies.items():
            if isinstance(anomaly_data, dict) and "confidence" in anomaly_data:
                original_confidence = anomaly_data["confidence"]
                anomaly_data["original_confidence"] = original_confidence
                anomaly_data["confidence"] = original_confidence * (1 - penalty)
                anomaly_data["drift_warning"] = True

                if verbose:
                    logger.info(
                        f"  {anomaly_name}: confidence {original_confidence:.2f} "
                        f"→ {anomaly_data['confidence']:.2f}"
                    )

        return result

    def load_time_aware_detector(
        self, service_name: str, verbose: bool = False
    ) -> Optional[Any]:
        """Load time-aware detector with lazy loading discovery - no models loaded yet.

        This method creates or retrieves a cached TimeAwareAnomalyDetector for the
        given service. The detector is initialized with:
        - Lazy loading (models loaded on-demand)
        - Holiday variant discovery (if _holiday_variant subdirectory exists)
        - Excluded periods for holiday model selection

        Args:
            service_name: Name of the service to load detector for.
            verbose: Whether to log detailed loading information.

        Returns:
            TimeAwareAnomalyDetector instance or None if no models found.
        """
        cache_key = service_name

        # Check if we need to reload the detector (not the models)
        service_model_dirs = []
        models_path = Path(self.models_directory)

        # Look for any of the 5 period models for this service
        period_suffixes = [
            "_business_hours",
            "_night_hours",
            "_evening_hours",
            "_weekend_day",
            "_weekend_night",
        ]

        for suffix in period_suffixes:
            period_service_dir = models_path / f"{service_name}{suffix}"
            if period_service_dir.exists():
                service_model_dirs.append(period_service_dir)

        # Also check holiday variant directory
        holiday_variant_dir = models_path / "_holiday_variant"
        if holiday_variant_dir.exists():
            for suffix in period_suffixes:
                holiday_period_dir = holiday_variant_dir / f"{service_name}{suffix}"
                if holiday_period_dir.exists():
                    service_model_dirs.append(holiday_period_dir)

        if not service_model_dirs:
            if verbose:
                logger.info(f"No time-aware models found for {service_name}")
            return None

        # Get the most recent modification time across all period models
        most_recent_mod_time = max(
            dir_path.stat().st_mtime for dir_path in service_model_dirs
        )
        cached_mod_time = self._load_times.get(cache_key, 0)

        if cache_key not in self._detector_cache or most_recent_mod_time > cached_mod_time:
            if verbose:
                logger.info(f"Initializing lazy-loading detector for {service_name}")
                logger.info(f"Found {len(service_model_dirs)} period models available")

            try:
                # Import the efficient TimeAwareAnomalyDetector class
                from smartbox_anomaly.detection.time_aware import TimeAwareAnomalyDetector

                # Create detector with lazy loading and excluded periods
                detector = TimeAwareAnomalyDetector(
                    service_name,
                    excluded_periods=self._excluded_periods,
                )

                # Use load_from_directory for proper discovery of baseline and holiday variants
                detector.load_from_directory(self.models_directory)

                if detector._available_periods or detector._available_holiday_periods:
                    self._detector_cache[cache_key] = detector
                    self._load_times[cache_key] = most_recent_mod_time

                    if verbose:
                        logger.info(f"Detector ready for {service_name}")
                        logger.info(
                            f"Available baseline periods: {sorted(list(detector._available_periods))}"
                        )
                        if detector.has_holiday_variants:
                            logger.info(
                                f"Available holiday periods: {sorted(list(detector._available_holiday_periods))}"
                            )
                        if self._excluded_periods:
                            logger.info(
                                f"Excluded periods configured: {len(self._excluded_periods)}"
                            )
                        logger.info("Models will load on-demand")
                else:
                    if verbose:
                        logger.warning(f"No valid periods found for {service_name}")
                    return None

            except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
                # Expected file/model loading errors
                if verbose:
                    logger.warning(f"Model not available for {service_name}: {e}")
                return None
            except (KeyError, ValueError, TypeError) as e:
                # Data structure errors in model metadata
                logger.warning(
                    f"Invalid model data for {service_name}: {e}", exc_info=verbose
                )
                return None
            except Exception as e:
                # Unexpected error - always log with trace
                logger.error(
                    f"Unexpected error initializing detector for {service_name}: {e}",
                    exc_info=True,
                )
                return None

        return self._detector_cache.get(cache_key)

    def detect_with_explainability(
        self,
        service_name: str,
        metrics: Dict[str, Any],
        timestamp: datetime,
        verbose: bool = False,
        dependency_context: Optional[DependencyContext] = None,
        check_drift: bool = False,
    ) -> Dict[str, Any]:
        """Enhanced time-aware detection with explainability - EFFICIENT with lazy loading"""

        try:
            detector = self.load_time_aware_detector(service_name, verbose)

            if not detector:
                raise ModelLoadError(
                    f"Could not initialize time-aware detector for {service_name}"
                )

            # Determine current period first
            current_period = detector.get_time_period(timestamp)

            if verbose:
                logger.info(f"Current period for {service_name}: {current_period}")

            # Check if this period is available
            if current_period not in detector._available_periods:
                available_periods = list(detector._available_periods)
                if verbose:
                    logger.warning(
                        f"No model for {current_period}, available: {available_periods}"
                    )

                # Try fallback to similar period
                fallback_period = detector._find_fallback_period(current_period)
                if fallback_period:
                    if verbose:
                        logger.info(f"Will use fallback period: {fallback_period}")
                    current_period = fallback_period
                else:
                    raise ModelLoadError(
                        f"No model available for time period: {current_period} "
                        f"(available: {available_periods})"
                    )

            # Use lazy loading detection - only loads the specific period model needed
            if hasattr(detector, "detect_anomalies_with_context"):
                if verbose:
                    logger.info(
                        f"Using explainable detection with lazy loading for {current_period}"
                    )

                try:
                    # This will lazy load only the current period model
                    enhanced_result = detector.detect_anomalies_with_context(
                        metrics,
                        timestamp,
                        self.models_directory,
                        verbose,
                        dependency_context=dependency_context,
                        check_drift=check_drift,
                    )

                    # Apply drift adjustments if drift was detected
                    if check_drift and "drift_analysis" in enhanced_result:
                        enhanced_result = self._apply_drift_adjustments(
                            enhanced_result, enhanced_result["drift_analysis"], verbose
                        )

                    # Add efficiency info
                    enhanced_result["performance_info"] = {
                        "lazy_loaded": True,
                        "models_loaded": list(detector.models.keys()),
                        "period_used": current_period,
                        "total_available": len(detector._available_periods),
                        "drift_check_enabled": check_drift,
                        "drift_penalty_applied": enhanced_result.get(
                            "drift_warning", {}
                        ).get("confidence_penalty_applied", 0.0),
                    }

                    return enhanced_result

                except Exception as e:
                    if verbose:
                        logger.warning(
                            f"Explainable detection failed: {e}, falling back to standard"
                        )
                    # Fall through to standard detection

            # Fallback to standard detection with lazy loading
            if verbose:
                logger.info(
                    f"Using standard detection with lazy loading for {current_period}"
                )

            # This will also lazy load only the needed model
            anomalies = detector.detect_anomalies(
                metrics, timestamp, self.models_directory, verbose, check_drift=check_drift
            )

            # Handle both dict and other formats properly
            if isinstance(anomalies, dict):
                anomaly_count = len(anomalies)
            else:
                anomaly_count = len(anomalies) if hasattr(anomalies, "__len__") else 0
                anomalies = anomalies if isinstance(anomalies, dict) else {}

            return {
                "service": service_name,
                "time_period": current_period,
                "model_type": "time_aware_5period_standard_lazy",
                "anomaly_count": anomaly_count,
                "anomalies": anomalies,
                "metrics": metrics,
                "timestamp": timestamp.isoformat(),
                "explainable": False,
                "performance_info": {
                    "lazy_loaded": True,
                    "models_loaded": list(detector.models.keys()),
                    "period_used": current_period,
                    "total_available": len(detector._available_periods),
                    "drift_check_enabled": check_drift,
                },
            }

        except Exception as e:
            if verbose:
                logger.error(f"All detection methods failed for {service_name}: {e}")

            return {
                "service": service_name,
                "error": str(e),
                "timestamp": timestamp.isoformat(),
                "model_type": "time_aware_failed",
                "explainable": False,
            }

    def get_efficiency_stats(self) -> Dict[str, Any]:
        """Get statistics about lazy loading efficiency"""
        total_detectors = len(self._detector_cache)
        total_models_loaded = 0
        total_models_available = 0

        for detector in self._detector_cache.values():
            if hasattr(detector, "models") and hasattr(detector, "_available_periods"):
                total_models_loaded += len(detector.models)
                total_models_available += len(detector._available_periods)

        efficiency = (1 - (total_models_loaded / max(1, total_models_available))) * 100

        return {
            "total_services": total_detectors,
            "total_models_available": total_models_available,
            "total_models_loaded": total_models_loaded,
            "memory_efficiency": f"{efficiency:.1f}%",
            "models_saved_from_loading": total_models_available - total_models_loaded,
            "lazy_loading_enabled": True,
        }
