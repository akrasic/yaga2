"""
Pydantic data models for the ML Anomaly Detection API.

This module defines stable, versioned data contracts for:
- API request/response payloads
- Internal data structures
- Serialization/deserialization

These models serve as the single source of truth for API contracts
and enable automatic validation, documentation, and schema generation.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

# =============================================================================
# Schema Version - Increment when breaking changes are made
# =============================================================================

API_SCHEMA_VERSION = "1.2.0"  # Added service_graph_context, database_latency_evaluation, request_rate_evaluation


# =============================================================================
# Enumerations
# =============================================================================


class AlertType(str, Enum):
    """Types of alerts sent to the observability service."""

    ANOMALY_DETECTED = "anomaly_detected"
    NO_ANOMALY = "no_anomaly"
    INCIDENT_RESOLVED = "incident_resolved"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


class Severity(str, Enum):
    """Anomaly severity levels."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @classmethod
    def from_score(cls, score: float) -> Severity:
        """Map isolation forest score to severity."""
        if score < -0.6:
            return cls.CRITICAL
        if score < -0.3:
            return cls.HIGH
        if score < -0.1:
            return cls.MEDIUM
        return cls.LOW

    @classmethod
    def max_severity(cls, severities: list[Severity]) -> Severity:
        """Return the highest severity from a list."""
        order = [cls.NONE, cls.LOW, cls.MEDIUM, cls.HIGH, cls.CRITICAL]
        if not severities:
            return cls.NONE
        return max(severities, key=lambda s: order.index(s))


class IncidentAction(str, Enum):
    """Actions taken on incidents during processing."""

    CREATE = "CREATE"
    CONTINUE = "CONTINUE"
    UPDATE = "UPDATE"
    RESOLVE = "RESOLVE"
    CLOSE = "CLOSE"
    NO_CHANGE = "NO_CHANGE"
    MIXED = "MIXED"


class ModelType(str, Enum):
    """Types of ML models used for detection."""

    TIME_AWARE = "time_aware"
    TIME_AWARE_EXPLAINABLE = "time_aware_explainable"
    TIME_AWARE_FALLBACK = "time_aware_fallback"
    STANDARD_ML = "standard_ml"
    EXPLAINABLE_ML = "explainable_ml"
    INCIDENT_RESOLUTION = "incident_resolution"


class DetectionMethod(str, Enum):
    """Methods used for anomaly detection."""

    ISOLATION_FOREST = "isolation_forest"
    ENHANCED_ISOLATION_FOREST = "enhanced_isolation_forest"
    MULTIVARIATE = "multivariate"
    THRESHOLD = "threshold"
    PATTERN = "pattern"
    CORRELATION = "correlation"
    NAMED_PATTERN_MATCHING = "named_pattern_matching"
    ZERO_NORMAL_THRESHOLD = "zero_normal_threshold"
    FAST_FAIL = "fast_fail"


# =============================================================================
# Metrics Models
# =============================================================================


class CurrentMetrics(BaseModel):
    """Current metric values at time of detection.

    All metrics are optional to support partial data.
    """

    request_rate: float | None = Field(None, ge=0, description="Requests per second")
    application_latency: float | None = Field(None, ge=0, description="App latency in ms")
    client_latency: float | None = Field(None, ge=0, description="Client call latency in ms")
    database_latency: float | None = Field(None, ge=0, description="Database latency in ms")
    error_rate: float | None = Field(None, ge=0, le=1, description="Error rate 0.0-1.0")

    class Config:
        extra = "allow"  # Allow additional metrics


# =============================================================================
# Anomaly Models
# =============================================================================


class FeatureContribution(BaseModel):
    """Feature contribution to anomaly detection."""

    feature: str = Field(..., description="Feature/metric name")
    contribution: float = Field(..., description="Contribution score")
    direction: str = Field("elevated", description="Direction: elevated, reduced, unusual")
    percentile: float | None = Field(None, ge=0, le=100, description="Percentile position")


class ComparisonData(BaseModel):
    """Historical comparison data for explainability."""

    training_mean: float | None = Field(None, description="Mean from training data")
    training_std: float | None = Field(None, description="Std dev from training data")
    training_median: float | None = Field(None, description="Median from training data")
    training_q25: float | None = Field(None, description="25th percentile")
    training_q75: float | None = Field(None, description="75th percentile")
    current_value: float | None = Field(None, description="Current observed value")
    deviation_sigma: float | None = Field(None, description="Standard deviations from mean")


class AnomalyMetadata(BaseModel):
    """Metadata associated with a detected anomaly.

    Contains fingerprinting, time-awareness, and explainability data.
    """

    # Fingerprinting fields
    fingerprint_id: str | None = Field(None, description="Deterministic pattern identifier")
    incident_id: str | None = Field(None, description="Unique incident occurrence ID")
    anomaly_name: str | None = Field(None, description="Generated anomaly name")
    fingerprint_action: IncidentAction | None = Field(None, description="Pattern-level action")
    incident_action: IncidentAction | None = Field(None, description="Incident-level action")
    occurrence_count: int | None = Field(None, ge=1, description="Times this incident detected")
    first_seen: str | None = Field(None, description="ISO timestamp of first detection")
    last_updated: str | None = Field(None, description="ISO timestamp of last update")
    incident_duration_minutes: int | None = Field(None, ge=0, description="Duration in minutes")
    detected_by_model: str | None = Field(None, description="Model that detected this")

    # Time-awareness fields
    time_period: str | None = Field(None, description="Time period: business_hours, etc.")
    time_confidence: float | None = Field(None, ge=0, le=1, description="Period confidence")
    lazy_loaded: bool | None = Field(None, description="Whether model was lazy-loaded")

    # Explainability fields
    feature_contributions: list[FeatureContribution] | None = Field(
        None, description="Feature importance breakdown"
    )
    comparison_data: dict[str, ComparisonData] | None = Field(
        None, description="Historical comparison per metric"
    )
    business_impact: str | None = Field(None, description="Business impact assessment")
    percentile_position: float | None = Field(None, description="Overall percentile position")

    class Config:
        extra = "allow"  # Allow additional metadata fields


class CascadeInfo(BaseModel):
    """Information about dependency cascade for an anomaly.

    When an anomaly is caused by a downstream service failure, this provides
    the cascade analysis results including root cause service identification.
    """

    is_cascade: bool = Field(..., description="Whether this anomaly is part of a cascade")
    root_cause_service: str | None = Field(
        None, description="Service identified as the root cause"
    )
    affected_chain: list[str] = Field(
        default_factory=list, description="Chain of affected services"
    )
    cascade_type: str = Field(
        "none",
        description="Type: upstream_cascade, chain_degraded, dependencies_healthy, none"
    )
    confidence: float = Field(0.0, ge=0, le=1, description="Confidence in cascade analysis")
    propagation_path: list[dict[str, Any]] | None = Field(
        None, description="Detailed propagation path with service status"
    )

    class Config:
        extra = "allow"


class DetectionSignal(BaseModel):
    """Individual detection signal from a detection method."""

    method: str = Field(..., description="Detection method: isolation_forest, named_pattern_matching, etc.")
    type: str = Field(..., description="Detection type: ml_isolation, multivariate_pattern, etc.")
    severity: Severity = Field(..., description="Severity from this method")
    score: float = Field(..., description="Anomaly score from this method")
    direction: str | None = Field(None, description="Direction: high, low, activated")
    percentile: float | None = Field(None, ge=0, le=100, description="Percentile position")
    pattern: str | None = Field(None, description="Pattern name for pattern-based methods")


class Anomaly(BaseModel):
    """Single detected anomaly - supports both single and consolidated anomalies.

    For consolidated anomalies (multiple detection methods), type="consolidated"
    and detection_signals contains all contributing signals.
    """

    type: str = Field(..., description="'consolidated' or detection type (ml_isolation, etc.)")
    severity: Severity = Field(..., description="Severity level (max of all signals if consolidated)")
    confidence: float = Field(..., ge=0, le=1, description="Detection confidence 0-1")
    score: float = Field(..., description="Best anomaly score (most negative)")
    description: str = Field(..., description="Human-readable description")

    # Root cause identification
    root_metric: str | None = Field(None, description="Primary metric: application_latency, error_rate, etc.")
    signal_count: int | None = Field(None, ge=1, description="Number of detection methods (consolidated)")
    pattern_name: str | None = Field(None, description="Named pattern if pattern-based")
    interpretation: str | None = Field(None, description="Semantic interpretation of the anomaly")
    value: float | None = Field(None, description="Current value of root metric")

    # Detection transparency
    detection_signals: list[DetectionSignal] = Field(
        default_factory=list, description="Individual signals from each detection method"
    )

    # Actionable information
    possible_causes: list[str] | None = Field(None, description="Possible root causes")
    recommended_actions: list[str] | None = Field(None, description="Prioritized actions to take")
    checks: list[str] | None = Field(None, description="Diagnostic checks to perform")

    # Statistical context
    comparison_data: dict[str, Any] | None = Field(None, description="Historical comparison per metric")
    business_impact: str | None = Field(None, description="Business impact assessment")

    # Cascade/dependency analysis
    cascade_analysis: CascadeInfo | None = Field(
        None, description="Cascade analysis when anomaly is caused by upstream service"
    )

    # Fingerprinting fields (at anomaly level)
    fingerprint_id: str | None = Field(None, description="Deterministic pattern identifier")
    fingerprint_action: str | None = Field(None, description="Pattern-level action: CREATE, UPDATE, RESOLVE")
    incident_id: str | None = Field(None, description="Unique incident occurrence ID")
    incident_action: str | None = Field(None, description="Incident action: CREATE, CONTINUE, CLOSE")
    incident_duration_minutes: int | None = Field(None, ge=0, description="Duration in minutes")
    first_seen: str | None = Field(None, description="ISO timestamp of first detection")
    last_updated: str | None = Field(None, description="ISO timestamp of last update")
    occurrence_count: int | None = Field(None, ge=1, description="Times this incident detected")
    time_confidence: float | None = Field(None, ge=0, le=1, description="Time period confidence")
    detected_by_model: str | None = Field(None, description="Model that detected this")

    @field_validator("confidence", mode="before")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Ensure confidence is in valid range by clamping."""
        if v is None:
            return 0.5
        return max(0.0, min(1.0, float(v)))

    class Config:
        extra = "allow"  # Allow additional fields for extensibility


# =============================================================================
# Fingerprinting Models
# =============================================================================


class ActionSummary(BaseModel):
    """Summary of fingerprinting actions taken."""

    incident_creates: int = Field(0, ge=0, description="New incidents created")
    incident_continues: int = Field(0, ge=0, description="Existing incidents continued")
    incident_closes: int = Field(0, ge=0, description="Incidents resolved/closed")


class DetectionContext(BaseModel):
    """Context about the detection process."""

    model_used: str = Field(..., description="Model identifier used")
    inference_timestamp: str = Field(..., description="ISO timestamp of inference")


class FingerprintingMetadata(BaseModel):
    """Fingerprinting metadata attached to alerts.

    Provides incident lifecycle tracking information.
    """

    service_name: str = Field(..., description="Base service name")
    model_name: str = Field(..., description="Time period model name")
    timestamp: str = Field(..., description="ISO timestamp")
    action_summary: ActionSummary = Field(..., description="Actions taken this cycle")
    overall_action: IncidentAction = Field(..., description="Primary action for this payload")
    resolved_incidents: list[dict[str, Any]] = Field(
        default_factory=list, description="Incidents resolved this cycle"
    )
    total_open_incidents: int = Field(0, ge=0, description="Currently open incidents")
    detection_context: DetectionContext = Field(..., description="Detection context")


# =============================================================================
# Alert Payloads - Primary API Contracts
# =============================================================================


class PayloadMetadata(BaseModel):
    """Metadata about the detection process and features."""

    service_name: str = Field(..., description="Service name")
    detection_timestamp: str = Field(..., description="ISO timestamp of detection")
    models_used: list[str] = Field(default_factory=list, description="Detection models used")
    enhanced_messaging: bool = Field(True, description="Enhanced messaging enabled")
    features: dict[str, bool] = Field(
        default_factory=lambda: {
            "contextual_severity": True,
            "named_patterns": True,
            "recommendations": True,
            "interpretations": True,
            "anomaly_correlation": True,
        },
        description="Feature flags"
    )


class AnomalyDetectedPayload(BaseModel):
    """Payload for anomaly_detected and no_anomaly alerts.

    This is the primary payload sent to POST /api/anomalies/batch.
    Anomalies are stored as a dict keyed by anomaly name for easy lookup.

    Example:
        {
            "alert_type": "anomaly_detected",
            "service_name": "booking",
            "timestamp": "2024-01-15T10:30:00",
            "time_period": "business_hours",
            "model_name": "business_hours",
            "model_type": "time_aware_5period",
            "anomalies": {"recent_degradation": {...}},
            "anomaly_count": 1,
            "overall_severity": "high",
            "current_metrics": {...},
            "fingerprinting": {...},
            "metadata": {...}
        }
    """

    # Required fields
    alert_type: AlertType = Field(..., description="'anomaly_detected' or 'no_anomaly'")
    service_name: str = Field(..., min_length=1, description="Service name")
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    time_period: str = Field(..., description="Time period: business_hours, evening_hours, etc.")
    model_name: str = Field(..., description="Model name (typically matches time_period)")
    model_type: str = Field("time_aware_5period", description="Model type")

    # Anomaly data - dict keyed by anomaly name
    anomalies: dict[str, Anomaly] = Field(
        default_factory=dict, description="Anomalies keyed by name"
    )
    anomaly_count: int = Field(0, ge=0, description="Number of anomalies")
    overall_severity: Severity = Field(Severity.NONE, description="Highest severity")

    # Metrics at detection time
    current_metrics: CurrentMetrics = Field(
        default_factory=CurrentMetrics, description="Metrics at detection time"
    )

    # Fingerprinting data
    fingerprinting: FingerprintingMetadata | None = Field(
        None, description="Fingerprinting/incident tracking data"
    )

    # Performance info
    performance_info: dict[str, Any] | None = Field(
        None, description="Model loading and performance metadata"
    )

    # Exception context for error-related anomalies
    exception_context: dict[str, Any] | None = Field(
        None, description="Exception breakdown when error-related anomalies detected"
    )

    # Service graph context for latency-related anomalies
    service_graph_context: dict[str, Any] | None = Field(
        None, description="Downstream service call breakdown when latency anomalies detected"
    )

    # Metadata about detection
    metadata: PayloadMetadata | None = Field(None, description="Detection metadata")

    @model_validator(mode="after")
    def validate_anomaly_count(self) -> AnomalyDetectedPayload:
        """Ensure anomaly_count matches actual anomalies."""
        if len(self.anomalies) != self.anomaly_count:
            self.anomaly_count = len(self.anomalies)
        return self

    @model_validator(mode="after")
    def validate_alert_type(self) -> AnomalyDetectedPayload:
        """Set alert_type based on anomaly presence."""
        if not self.anomalies:
            self.alert_type = AlertType.NO_ANOMALY
            self.overall_severity = Severity.NONE
        return self


class ResolutionDetails(BaseModel):
    """Details about an incident resolution."""

    final_severity: Severity = Field(..., description="Severity at resolution")
    total_occurrences: int = Field(..., ge=1, description="Total detection count")
    incident_duration_minutes: int = Field(..., ge=0, description="Total duration")
    first_seen: str = Field(..., description="ISO timestamp of first detection")
    last_detected_by_model: str | None = Field(None, description="Last model to detect")


class IncidentResolvedPayload(BaseModel):
    """Payload for incident_resolved alerts.

    Sent to POST /api/incidents/resolve when an anomaly clears.

    Example:
        {
            "alert_type": "incident_resolved",
            "service_name": "booking",
            "timestamp": "2024-01-15T11:00:00",
            "incident_id": "incident_abc123",
            "fingerprint_id": "anomaly_xyz789",
            "anomaly_name": "multivariate_isolation_forest",
            "resolution_details": {...}
        }
    """

    alert_type: AlertType = Field(
        default=AlertType.INCIDENT_RESOLVED,
        description="Must be 'incident_resolved'"
    )
    service_name: str = Field(..., min_length=1, description="Service name")
    timestamp: str = Field(..., description="ISO timestamp of resolution")
    incident_id: str = Field(..., description="Unique incident identifier")
    fingerprint_id: str = Field(..., description="Pattern fingerprint identifier")
    anomaly_name: str = Field(..., description="Name of the resolved anomaly")
    resolution_details: ResolutionDetails = Field(..., description="Resolution details")
    model_type: str = Field(
        default="incident_resolution",
        description="Always 'incident_resolution'"
    )


class ErrorPayload(BaseModel):
    """Payload for error alerts.

    Sent when inference fails for a service.
    """

    alert_type: AlertType = Field(AlertType.ERROR, description="Must be 'error'")
    service: str = Field(..., description="Service that failed")
    timestamp: str = Field(..., description="ISO timestamp of error")
    error_message: str = Field(..., description="Error description")
    error_code: str | None = Field(None, description="Error code if applicable")


class HeartbeatPayload(BaseModel):
    """Payload for heartbeat/status messages.

    Can be sent periodically to indicate pipeline health.
    """

    alert_type: AlertType = Field(AlertType.HEARTBEAT, description="Must be 'heartbeat'")
    timestamp: str = Field(..., description="ISO timestamp")
    status: str = Field("healthy", description="Pipeline status")
    services_evaluated: int = Field(0, ge=0, description="Services checked this cycle")
    anomalies_detected: int = Field(0, ge=0, description="Anomalies found")
    incidents_resolved: int = Field(0, ge=0, description="Incidents resolved")
    inference_duration_ms: float | None = Field(None, description="Total inference time")


# =============================================================================
# Batch Payloads - For API Endpoints
# =============================================================================


class AnomalyBatchRequest(BaseModel):
    """Request body for POST /api/anomalies/batch.

    Contains one or more anomaly alerts.
    """

    alerts: list[AnomalyDetectedPayload] = Field(..., description="Anomaly alerts to ingest")
    schema_version: str = Field(API_SCHEMA_VERSION, description="API schema version")


class ResolutionBatchRequest(BaseModel):
    """Request body for POST /api/incidents/resolve.

    Contains one or more resolution notifications.
    """

    resolutions: list[IncidentResolvedPayload] = Field(..., description="Resolutions to process")
    schema_version: str = Field(API_SCHEMA_VERSION, description="API schema version")


# =============================================================================
# Response Models
# =============================================================================


class IngestionResponse(BaseModel):
    """Response from ingestion endpoints."""

    success: bool = Field(..., description="Whether ingestion succeeded")
    processed_count: int = Field(..., ge=0, description="Items successfully processed")
    failed_count: int = Field(0, ge=0, description="Items that failed")
    errors: list[str] = Field(default_factory=list, description="Error messages if any")
    timestamp: str = Field(..., description="Server timestamp")


class IncidentSummary(BaseModel):
    """Summary of a single incident for API responses."""

    incident_id: str
    fingerprint_id: str
    service_name: str
    anomaly_name: str
    severity: Severity
    status: str = Field(..., description="OPEN or CLOSED")
    first_seen: str
    last_updated: str
    occurrence_count: int
    duration_minutes: int | None = None


class IncidentListResponse(BaseModel):
    """Response for incident listing endpoints."""

    incidents: list[IncidentSummary]
    total_count: int
    open_count: int
    closed_count: int


# =============================================================================
# Utility Functions
# =============================================================================


def create_anomaly_payload(
    service_name: str,
    anomalies: dict[str, dict[str, Any]],
    metrics: dict[str, float],
    time_period: str,
    model_name: str | None = None,
    timestamp: datetime | None = None,
    **kwargs: Any,
) -> AnomalyDetectedPayload:
    """Factory function to create a validated anomaly payload.

    Args:
        service_name: Service name.
        anomalies: Dict of anomalies keyed by name.
        metrics: Current metrics dictionary.
        time_period: Time period for detection.
        model_name: Model name (defaults to time_period).
        timestamp: Optional timestamp (defaults to now).
        **kwargs: Additional payload fields.

    Returns:
        Validated AnomalyDetectedPayload instance.
    """
    if timestamp is None:
        timestamp = datetime.now()
    if model_name is None:
        model_name = time_period

    # Convert anomaly dicts to Anomaly objects
    anomaly_objects: dict[str, Anomaly] = {}
    severities: list[Severity] = []

    for name, a in anomalies.items():
        # Build detection signals if present
        signals = []
        for sig in a.get("detection_signals", []):
            signals.append(DetectionSignal(
                method=sig.get("method", "isolation_forest"),
                type=sig.get("type", "ml_isolation"),
                severity=Severity(sig.get("severity", "medium")),
                score=float(sig.get("score", 0.0)),
                direction=sig.get("direction"),
                percentile=sig.get("percentile"),
                pattern=sig.get("pattern"),
            ))

        # Build cascade info if present
        cascade_info = None
        if a.get("cascade_analysis"):
            ca = a["cascade_analysis"]
            cascade_info = CascadeInfo(
                is_cascade=ca.get("is_cascade", False),
                root_cause_service=ca.get("root_cause_service"),
                affected_chain=ca.get("affected_chain", []),
                cascade_type=ca.get("cascade_type", "none"),
                confidence=ca.get("confidence", 0.0),
                propagation_path=ca.get("propagation_path"),
            )

        anomaly = Anomaly(
            type=a.get("type", "ml_isolation"),
            severity=Severity(a.get("severity", "medium")),
            confidence=float(a.get("confidence", a.get("score", 0.5))),
            score=float(a.get("score", a.get("confidence", 0.0))),
            description=a.get("description", name.replace("_", " ").title()),
            root_metric=a.get("root_metric"),
            signal_count=a.get("signal_count"),
            pattern_name=a.get("pattern_name"),
            interpretation=a.get("interpretation"),
            value=a.get("value"),
            detection_signals=signals,
            possible_causes=a.get("possible_causes"),
            recommended_actions=a.get("recommended_actions"),
            checks=a.get("checks"),
            comparison_data=a.get("comparison_data"),
            business_impact=a.get("business_impact"),
            cascade_analysis=cascade_info,
            fingerprint_id=a.get("fingerprint_id"),
            fingerprint_action=a.get("fingerprint_action"),
            incident_id=a.get("incident_id"),
            incident_action=a.get("incident_action"),
            incident_duration_minutes=a.get("incident_duration_minutes"),
            first_seen=a.get("first_seen"),
            last_updated=a.get("last_updated"),
            occurrence_count=a.get("occurrence_count"),
            time_confidence=a.get("time_confidence"),
            detected_by_model=a.get("detected_by_model"),
        )
        anomaly_objects[name] = anomaly
        severities.append(anomaly.severity)

    # Determine overall severity
    overall_severity = Severity.max_severity(severities) if severities else Severity.NONE
    alert_type = AlertType.ANOMALY_DETECTED if anomaly_objects else AlertType.NO_ANOMALY

    return AnomalyDetectedPayload(
        alert_type=alert_type,
        service_name=service_name,
        timestamp=timestamp.isoformat(),
        time_period=time_period,
        model_name=model_name,
        overall_severity=overall_severity,
        anomaly_count=len(anomaly_objects),
        current_metrics=CurrentMetrics(**metrics),
        anomalies=anomaly_objects,
        **kwargs,
    )


def create_resolution_payload(
    service_name: str,
    incident_id: str,
    fingerprint_id: str,
    anomaly_name: str,
    final_severity: str,
    total_occurrences: int,
    duration_minutes: int,
    first_seen: str,
    timestamp: datetime | None = None,
    **kwargs: Any,
) -> IncidentResolvedPayload:
    """Factory function to create a validated resolution payload.

    Args:
        service_name: Service name.
        incident_id: Unique incident ID.
        fingerprint_id: Pattern fingerprint ID.
        anomaly_name: Name of the anomaly.
        final_severity: Severity at resolution.
        total_occurrences: Total times detected.
        duration_minutes: Incident duration.
        first_seen: ISO timestamp of first detection.
        timestamp: Optional timestamp (defaults to now).
        **kwargs: Additional fields.

    Returns:
        Validated IncidentResolvedPayload instance.
    """
    if timestamp is None:
        timestamp = datetime.now()

    return IncidentResolvedPayload(
        service_name=service_name,
        timestamp=timestamp.isoformat(),
        incident_id=incident_id,
        fingerprint_id=fingerprint_id,
        anomaly_name=anomaly_name,
        resolution_details=ResolutionDetails(
            final_severity=Severity(final_severity),
            total_occurrences=total_occurrences,
            incident_duration_minutes=duration_minutes,
            first_seen=first_seen,
            last_detected_by_model=kwargs.get("last_detected_by_model"),
        ),
    )
