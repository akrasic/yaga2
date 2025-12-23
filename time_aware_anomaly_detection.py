"""
Backward compatibility stub for time_aware_anomaly_detection.

This module redirects imports to the new package structure.
New code should import from smartbox_anomaly.detection instead.
"""

from smartbox_anomaly.detection import (
    TimeAwareAnomalyDetector,
    create_time_aware_detector,
)

__all__ = [
    "TimeAwareAnomalyDetector",
    "create_time_aware_detector",
]
