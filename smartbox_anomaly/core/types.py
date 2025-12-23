"""
Dependency-aware types for cascade detection.

These types enable cross-service anomaly correlation by tracking
dependency relationships and their status at inference time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
