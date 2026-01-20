"""Training module for Smartbox ML anomaly detection.

This module provides:
- Training pipeline for time-aware anomaly detection models
- Feature engineering with proper temporal splits
- Model validation with leakage prevention
- Training run storage and reporting
- Disk/backup utilities for model management
"""

from smartbox_anomaly.training.storage import (
    TrainingRunStorage,
    TrainingRunStatus,
    ValidationStatus,
)
from smartbox_anomaly.training.feature_engineering import (
    SmartboxFeatureEngineer,
)
from smartbox_anomaly.training.validators import (
    validate_model_split,
    validate_enhanced_model_split,
    validate_time_aware_models,
)
from smartbox_anomaly.training.pipeline import (
    SmartboxTrainingPipeline,
    EnhancedSmartboxTrainingPipeline,
)
from smartbox_anomaly.training.utils import (
    parse_date,
    check_disk_space,
    create_backup,
    rollback_from_backup,
    cleanup_directory,
    promote_models,
    get_directory_size,
    format_bytes,
)

__all__ = [
    # Storage
    "TrainingRunStorage",
    "TrainingRunStatus",
    "ValidationStatus",
    # Feature engineering
    "SmartboxFeatureEngineer",
    # Validation
    "validate_model_split",
    "validate_enhanced_model_split",
    "validate_time_aware_models",
    # Pipeline
    "SmartboxTrainingPipeline",
    "EnhancedSmartboxTrainingPipeline",
    # Utilities
    "parse_date",
    "check_disk_space",
    "create_backup",
    "rollback_from_backup",
    "cleanup_directory",
    "promote_models",
    "get_directory_size",
    "format_bytes",
]
