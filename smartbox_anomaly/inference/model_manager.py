"""
Enhanced model management for inference pipeline.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from smartbox_anomaly.core import ModelLoadError
from smartbox_anomaly.detection import SmartboxAnomalyDetector

logger = logging.getLogger(__name__)


class EnhancedModelManager:
    """Enhanced model management - UPDATED to not interfere with lazy loading"""

    def __init__(self, models_directory: str = "./smartbox_models/"):
        self.models_directory = Path(models_directory)
        self._model_cache: Dict[str, SmartboxAnomalyDetector] = {}
        self._model_metadata: Dict[str, Dict[str, Any]] = {}
        self._load_times: Dict[str, float] = {}
        self._model_validators: Dict[str, Any] = {}

    def load_services_from_config(self, config_path: str = "./config.json") -> List[str]:
        """Load services list from config.json.

        Combines all service categories: critical, standard, micro, admin, core.
        Returns empty list if config not found or invalid.
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
            seen: set[str] = set()
            unique_services = []
            for svc in all_services:
                if svc not in seen:
                    seen.add(svc)
                    unique_services.append(svc)

            return unique_services

        except (FileNotFoundError, json.JSONDecodeError, Exception) as e:
            logger.debug(f"Could not load services from config: {e}")
            return []

    def get_services_with_models(self, config_services: List[str]) -> Tuple[List[str], List[str]]:
        """Check which config services have trained models.

        Args:
            config_services: List of services from config.json

        Returns:
            Tuple of (services_with_models, services_missing_models)
        """
        available = set(self.get_base_services())
        with_models = [s for s in config_services if s in available]
        missing = [s for s in config_services if s not in available]
        return with_models, missing

    def get_base_services(self) -> List[str]:
        """Get list of base service names (without time period suffixes) - FIXED for 5-period"""
        all_services = self.get_available_services()
        base_services: set[str] = set()

        # Extract base service names by removing time period suffixes
        # Updated for 5-period approach
        time_suffixes = [
            "_business_hours",
            "_evening_hours",
            "_night_hours",
            "_weekend_day",  # New 5-period
            "_weekend_night",  # New 5-period
            "_weekend",  # Legacy 4-period (for backward compatibility)
        ]

        for service in all_services:
            base_name = service
            for suffix in time_suffixes:
                if service.endswith(suffix):
                    base_name = service[: -len(suffix)]
                    break
            base_services.add(base_name)

        return sorted(list(base_services))

    def get_available_services(self) -> List[str]:
        """Get list of services with valid models - includes period-specific models"""
        if not self.models_directory.exists():
            logger.warning(f"Models directory does not exist: {self.models_directory}")
            return []

        services = []
        for item in self.models_directory.iterdir():
            if item.is_dir():
                model_data_file = item / "model_data.json"
                if model_data_file.exists():
                    services.append(item.name)

        return sorted(services)

    def load_model(self, service_name: str) -> SmartboxAnomalyDetector:
        """Load individual period model - ONLY used for non-time-aware detection"""
        # This should NOT be called for time-aware inference
        # Only used for individual period model loading in specific cases

        cache_key = service_name

        try:
            service_dir = self.models_directory / service_name
            if not service_dir.exists():
                raise ModelLoadError(f"Model directory not found: {service_dir}")

            # Check if reload is needed
            current_mod_time = service_dir.stat().st_mtime
            cached_mod_time = self._load_times.get(cache_key, 0)

            if cache_key not in self._model_cache or current_mod_time > cached_mod_time:
                logger.info(f"Loading individual model for {service_name}")

                # Load using secure method
                model = SmartboxAnomalyDetector.load_model_secure(
                    str(self.models_directory), service_name
                )

                # Cache model and metadata
                self._model_cache[cache_key] = model
                self._model_metadata[cache_key] = model.model_metadata
                self._load_times[cache_key] = current_mod_time

                model_version = model.model_metadata.get("model_version", "unknown")
                explainability_status = (
                    "enabled"
                    if hasattr(model, "training_statistics") and model.training_statistics
                    else "disabled"
                )

                logger.info(
                    f"Successfully loaded individual model for {service_name}: {model_version}"
                )
                logger.info(f"Explainability features: {explainability_status}")

            return self._model_cache[cache_key]

        except Exception as e:
            logger.error(f"Failed to load individual model for {service_name}: {e}")
            raise ModelLoadError(f"Model loading failed for {service_name}: {e}")

    def get_model_metadata(self, service_name: str) -> Dict[str, Any]:
        """Get model metadata for a service"""
        return self._model_metadata.get(service_name, {})
