"""
Dependency-aware types for cascade detection.

These types enable cross-service anomaly correlation by tracking
dependency relationships and their status at inference time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict, NotRequired


# =============================================================================
# TypedDict definitions for inference pipeline
# =============================================================================


class MetricsDict(TypedDict):
    """Current metrics at detection time."""

    request_rate: float
    application_latency: float
    dependency_latency: float
    database_latency: float
    error_rate: float


class ComparisonMetric(TypedDict):
    """Statistical comparison for a single metric."""

    current: float
    training_mean: float
    training_std: NotRequired[float]
    training_p95: NotRequired[float]
    deviation_sigma: float
    percentile_estimate: float
    status: NotRequired[str]  # "normal", "elevated", "high", "low", "very_low"


class DetectionSignalDict(TypedDict):
    """Single detection signal from a detection method."""

    method: str  # "isolation_forest", "named_pattern_matching", etc.
    type: str  # "ml_isolation", "multivariate_pattern", etc.
    severity: str
    score: float
    direction: NotRequired[str]  # "high", "low", "activated"
    percentile: NotRequired[float]
    pattern: NotRequired[str]


class AnomalyData(TypedDict):
    """Single anomaly with all detection and fingerprinting data."""

    type: str
    root_metric: NotRequired[str]
    severity: str
    confidence: NotRequired[float]
    score: float
    signal_count: NotRequired[int]
    description: str
    interpretation: NotRequired[str]
    pattern_name: NotRequired[str]
    value: NotRequired[float]
    detection_signals: NotRequired[list[DetectionSignalDict]]
    possible_causes: NotRequired[list[str]]
    recommended_actions: NotRequired[list[str]]
    checks: NotRequired[list[str]]
    comparison_data: NotRequired[dict[str, ComparisonMetric]]
    business_impact: NotRequired[str]
    # Fingerprinting fields
    fingerprint_id: NotRequired[str]
    fingerprint_action: NotRequired[str]  # "CREATE", "UPDATE", "RESOLVE"
    incident_id: NotRequired[str]
    incident_action: NotRequired[str]  # "CREATE", "CONTINUE", "CLOSE"
    status: NotRequired[str]  # "SUSPECTED", "OPEN", "RECOVERING", "CLOSED"
    previous_status: NotRequired[str]
    incident_duration_minutes: NotRequired[int]
    first_seen: NotRequired[str]
    last_updated: NotRequired[str]
    occurrence_count: NotRequired[int]
    consecutive_detections: NotRequired[int]
    confirmation_pending: NotRequired[bool]
    cycles_to_confirm: NotRequired[int]
    is_confirmed: NotRequired[bool]
    newly_confirmed: NotRequired[bool]


class FingerprintingActionSummary(TypedDict):
    """Summary of fingerprinting actions in a detection cycle."""

    incident_creates: int
    incident_continues: int
    incident_closes: int
    newly_confirmed: NotRequired[int]


class FingerprintingStatusSummary(TypedDict):
    """Count of incidents by status."""

    suspected: int
    confirmed: int
    recovering: int


class ResolvedIncidentData(TypedDict):
    """Data for a resolved incident."""

    fingerprint_id: str
    incident_id: str
    anomaly_name: str
    fingerprint_action: str
    incident_action: str
    final_severity: str
    resolved_at: str
    total_occurrences: int
    incident_duration_minutes: int
    first_seen: str
    service_name: str
    last_detected_by_model: NotRequired[str]
    resolution_reason: str  # "resolved", "auto_stale", "suspected_expired"
    resolution_context: NotRequired[dict[str, Any]]


class FingerprintingData(TypedDict):
    """Fingerprinting metadata in detection result."""

    service_name: str
    model_name: str
    timestamp: str
    overall_action: str  # "CREATE", "CONFIRMED", "UPDATE", "MIXED", "RESOLVE", "NO_CHANGE"
    total_active_incidents: int
    total_alerting_incidents: int
    action_summary: FingerprintingActionSummary
    status_summary: NotRequired[FingerprintingStatusSummary]
    detection_context: NotRequired[dict[str, Any]]
    resolved_incidents: list[ResolvedIncidentData]
    newly_confirmed_incidents: NotRequired[list[dict[str, Any]]]


class SLOLatencyEvaluation(TypedDict):
    """SLO evaluation for latency metrics."""

    status: str  # "ok", "warning", "breached"
    proximity: float
    value: float
    threshold_acceptable: float
    threshold_warning: float
    threshold_critical: float


class SLOErrorRateEvaluation(TypedDict):
    """SLO evaluation for error rate."""

    status: str
    proximity: float
    value: float
    value_percent: str
    threshold_acceptable: float
    threshold_warning: float
    within_acceptable: bool


class SLOEvaluationData(TypedDict):
    """Full SLO evaluation result."""

    original_severity: NotRequired[str]
    adjusted_severity: NotRequired[str]
    severity_changed: bool
    slo_status: str  # "ok", "elevated", "warning", "breached"
    slo_proximity: float
    operational_impact: str  # "none", "informational", "actionable", "critical"
    is_busy_period: bool
    latency_evaluation: NotRequired[SLOLatencyEvaluation]
    error_rate_evaluation: NotRequired[SLOErrorRateEvaluation]
    database_latency_evaluation: NotRequired[dict[str, Any]]
    request_rate_evaluation: NotRequired[dict[str, Any]]
    explanation: NotRequired[str]


class InferenceResultDict(TypedDict):
    """Complete inference result for a service."""

    alert_type: str  # "anomaly_detected", "no_anomaly", "metrics_unavailable"
    service_name: str
    timestamp: str
    time_period: str
    model_name: str
    model_type: str
    anomalies: dict[str, AnomalyData]
    anomaly_count: int
    overall_severity: str
    original_severity: NotRequired[str]
    current_metrics: MetricsDict
    exception_context: NotRequired[dict[str, Any]]
    service_graph_context: NotRequired[dict[str, Any]]
    fingerprinting: NotRequired[FingerprintingData]
    performance_info: NotRequired[dict[str, Any]]
    metadata: NotRequired[dict[str, Any]]
    drift_warning: NotRequired[dict[str, Any]]
    validation_warnings: NotRequired[list[str]]
    drift_analysis: NotRequired[dict[str, Any]]
    slo_evaluation: NotRequired[SLOEvaluationData]
    # Error cases
    error: NotRequired[str]
    skipped_reason: NotRequired[str]
    failed_metrics: NotRequired[list[str]]
    collection_errors: NotRequired[dict[str, str]]


class FingerprintingStats(TypedDict):
    """Statistics from fingerprinting pass."""

    enhanced_services: int
    creates: int
    updates: int
    resolves: int
    fingerprinting_errors: int
    resolved_incidents: list[ResolvedIncidentData]
    resolution_summary: NotRequired[dict[str, int]]


# =============================================================================
# Dataclass definitions for dependency tracking
# =============================================================================


@dataclass
class DependencyStatus:
    """Status of a single dependency service at inference time."""

    service_name: str
    has_anomaly: bool
    anomaly_type: str | None = None
    severity: str | None = None
    confidence: float | None = None
    latency_percentile: float | None = None
    error_rate: float | None = None
    timestamp: str | None = None


@dataclass
class DependencyContext:
    """Full dependency context passed to detector for cascade analysis.

    Contains the dependency graph and current status of all dependencies.
    """

    dependencies: dict[str, DependencyStatus] = field(default_factory=dict)
    graph: dict[str, list[str]] = field(default_factory=dict)
    detection_timestamp: str | None = None

    def get_status(self, service_name: str) -> DependencyStatus | None:
        """Get the status of a specific dependency."""
        return self.dependencies.get(service_name)

    def get_upstream_chain(
        self, service_name: str, max_depth: int = 5
    ) -> list[str]:
        """Get full upstream dependency chain for a service.

        Args:
            service_name: The service to get dependencies for.
            max_depth: Maximum depth to traverse (prevents infinite loops).

        Returns:
            List of service names in dependency order (immediate deps first).
        """
        chain: list[str] = []
        visited: set[str] = set()
        self._traverse_upstream(service_name, chain, visited, 0, max_depth)
        return chain

    def _traverse_upstream(
        self,
        service: str,
        chain: list[str],
        visited: set[str],
        depth: int,
        max_depth: int,
    ) -> None:
        """Recursively traverse upstream dependencies."""
        if depth >= max_depth or service in visited:
            return
        visited.add(service)
        for dep in self.graph.get(service, []):
            chain.append(dep)
            self._traverse_upstream(dep, chain, visited, depth + 1, max_depth)

    def find_root_cause_service(
        self, service_name: str
    ) -> tuple[str | None, list[str]]:
        """Find the root cause service in the dependency chain.

        Traverses the dependency graph to find the deepest service
        that has an anomaly or elevated latency.

        Args:
            service_name: The service experiencing issues.

        Returns:
            Tuple of (root_cause_service, affected_chain).
            root_cause_service is None if no upstream anomaly found.
        """
        chain = self.get_upstream_chain(service_name)
        root_cause: str | None = None

        # Start from deepest dependency and work back
        for dep in reversed(chain):
            status = self.get_status(dep)
            if status and status.has_anomaly:
                root_cause = dep
                break
            if status and status.latency_percentile and status.latency_percentile > 90:
                root_cause = dep
                break

        affected_chain: list[str] = []
        if root_cause and root_cause in chain:
            # Build affected chain from root cause to current service
            idx = chain.index(root_cause)
            affected_chain = chain[: idx + 1]

        return root_cause, affected_chain

    def get_affected_dependencies(self) -> list[str]:
        """Get list of dependencies that have anomalies."""
        return [
            name
            for name, status in self.dependencies.items()
            if status.has_anomaly
        ]

    def all_dependencies_healthy(self) -> bool:
        """Check if all known dependencies are healthy."""
        if not self.dependencies:
            return True
        return not any(status.has_anomaly for status in self.dependencies.values())


@dataclass
class CascadeAnalysis:
    """Result of cascade analysis for an anomaly.

    Provides detailed information about whether an anomaly is caused
    by an upstream dependency failure and the propagation path.
    """

    is_cascade: bool
    root_cause_service: str | None
    affected_chain: list[str]
    cascade_type: str  # "upstream_cascade", "chain_degraded", "shared_dependency", "none"
    confidence: float
    root_cause_anomaly_type: str | None = None
    propagation_path: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "is_cascade": self.is_cascade,
            "root_cause_service": self.root_cause_service,
            "affected_chain": self.affected_chain,
            "cascade_type": self.cascade_type,
            "confidence": self.confidence,
            "root_cause_anomaly_type": self.root_cause_anomaly_type,
            "propagation_path": self.propagation_path,
        }


@dataclass
class CascadeDetectionConfig:
    """Configuration for cascade detection behavior."""

    enabled: bool = True
    max_depth: int = 5
    latency_propagation_threshold: float = 0.6
    require_temporal_correlation: bool = True
    correlation_window_minutes: int = 5


@dataclass
class DependencyGraphConfig:
    """Configuration for the service dependency graph."""

    graph: dict[str, list[str]] = field(default_factory=dict)
    cascade_detection: CascadeDetectionConfig = field(
        default_factory=CascadeDetectionConfig
    )

    def get_dependencies(self, service_name: str) -> list[str]:
        """Get direct dependencies for a service."""
        return self.graph.get(service_name, [])

    def has_dependencies(self, service_name: str) -> bool:
        """Check if a service has any dependencies defined."""
        return service_name in self.graph and len(self.graph[service_name]) > 0
