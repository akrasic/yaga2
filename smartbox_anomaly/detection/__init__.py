"""
Detection module - ML-based anomaly detection.

This module contains:
    - detector: SmartboxAnomalyDetector for single-model detection
    - time_aware: TimeAwareAnomalyDetector for period-specific detection
    - service_config: Service-specific configuration and tuning
    - explainability: Feature importance and explanation utilities
"""

from smartbox_anomaly.detection.detector import (
    SmartboxAnomalyDetector,
    create_detector,
)
from smartbox_anomaly.detection.service_config import (
    ServiceParameters,
    detect_service_category,
    get_service_parameters,
)
from smartbox_anomaly.detection.time_aware import (
    TimeAwareAnomalyDetector,
    create_time_aware_detector,
)

__all__ = [
    # Main detectors
    "SmartboxAnomalyDetector",
    "create_detector",
    "TimeAwareAnomalyDetector",
    "create_time_aware_detector",
    # Service configuration
    "ServiceParameters",
    "get_service_parameters",
    "detect_service_category",
]
