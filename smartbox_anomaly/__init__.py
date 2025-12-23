"""
Smartbox Anomaly Detection Pipeline.

A production-grade ML pipeline for real-time anomaly detection in microservices,
using Isolation Forest models with time-aware period separation and enhanced
explainability features.

Package Structure:
    - core: Configuration, constants, exceptions, logging, protocols, utils
    - detection: ML models and anomaly detection algorithms
    - metrics: VictoriaMetrics client and metric validation
    - fingerprinting: Incident lifecycle tracking
    - api: Data models and API client

Example usage:
    from smartbox_anomaly import get_config, PipelineConfig
    from smartbox_anomaly.detection import SmartboxAnomalyDetector, TimeAwareAnomalyDetector
    from smartbox_anomaly.metrics import VictoriaMetricsClient
    from smartbox_anomaly.fingerprinting import AnomalyFingerprinter
    from smartbox_anomaly.api import AnomalyDetectedPayload
"""

__version__ = "2.1.0"

# Core exports - most commonly used items
from smartbox_anomaly.core.config import PipelineConfig, get_config
from smartbox_anomaly.core.constants import (
    AnomalySeverity,
    IncidentAction,
    IncidentStatus,
    MetricName,
    TimePeriod,
)
from smartbox_anomaly.core.exceptions import PipelineError
from smartbox_anomaly.core.logging import configure_logging, get_logger

__all__ = [
    # Version
    "__version__",
    # Config
    "get_config",
    "PipelineConfig",
    # Constants
    "AnomalySeverity",
    "IncidentAction",
    "IncidentStatus",
    "TimePeriod",
    "MetricName",
    # Exceptions
    "PipelineError",
    # Logging
    "get_logger",
    "configure_logging",
]
