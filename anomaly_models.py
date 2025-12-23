"""
Backward compatibility stub for anomaly_models.

This module redirects imports to the new package structure.
New code should import from smartbox_anomaly.detection instead.
"""

from smartbox_anomaly.detection import (
    ServiceParameters,
    SmartboxAnomalyDetector,
    TimeAwareAnomalyDetector,
    create_detector,
    create_time_aware_detector,
    detect_service_category,
    get_service_parameters,
)

__all__ = [
    "SmartboxAnomalyDetector",
    "TimeAwareAnomalyDetector",
    "create_detector",
    "create_time_aware_detector",
    "ServiceParameters",
    "get_service_parameters",
    "detect_service_category",
]
