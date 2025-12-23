"""
API module - data models and client for observability service.

This module contains:
    - models: Pydantic data models for API payloads
"""

from smartbox_anomaly.api.models import (
    # Schema version
    API_SCHEMA_VERSION,
    # Fingerprinting models
    ActionSummary,
    # Enums
    AlertType,
    Anomaly,
    # Batch payloads
    AnomalyBatchRequest,
    # Alert payloads
    AnomalyDetectedPayload,
    AnomalyMetadata,
    # Cascade analysis
    CascadeInfo,
    ComparisonData,
    # Metrics models
    CurrentMetrics,
    DetectionContext,
    DetectionMethod,
    DetectionSignal,
    ErrorPayload,
    # Anomaly models
    FeatureContribution,
    FingerprintingMetadata,
    HeartbeatPayload,
    IncidentAction,
    IncidentListResponse,
    IncidentResolvedPayload,
    IncidentSummary,
    # Response models
    IngestionResponse,
    ModelType,
    PayloadMetadata,
    ResolutionBatchRequest,
    ResolutionDetails,
    Severity,
    # Factory functions
    create_anomaly_payload,
    create_resolution_payload,
)

__all__ = [
    # Schema version
    "API_SCHEMA_VERSION",
    # Enums
    "AlertType",
    "Severity",
    "IncidentAction",
    "ModelType",
    "DetectionMethod",
    # Metrics models
    "CurrentMetrics",
    # Anomaly models
    "DetectionSignal",
    "FeatureContribution",
    "ComparisonData",
    "AnomalyMetadata",
    "Anomaly",
    # Cascade analysis
    "CascadeInfo",
    # Fingerprinting models
    "ActionSummary",
    "DetectionContext",
    "FingerprintingMetadata",
    # Alert payloads
    "PayloadMetadata",
    "AnomalyDetectedPayload",
    "ResolutionDetails",
    "IncidentResolvedPayload",
    "ErrorPayload",
    "HeartbeatPayload",
    # Batch payloads
    "AnomalyBatchRequest",
    "ResolutionBatchRequest",
    # Response models
    "IngestionResponse",
    "IncidentSummary",
    "IncidentListResponse",
    # Factory functions
    "create_anomaly_payload",
    "create_resolution_payload",
]
