"""
Protocol definitions and abstract base classes for the ML pipeline.

This module defines interfaces that enable:
- Loose coupling between components
- Easy mocking for testing
- Clear contracts for implementations
- Type-safe dependency injection
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from smartbox_anomaly.core.constants import AnomalySeverity, TimePeriod

# =============================================================================
# Data Transfer Objects (DTOs)
# =============================================================================


@runtime_checkable
class MetricsData(Protocol):
    """Protocol for metrics data objects."""

    service_name: str
    timestamp: datetime
    request_rate: float
    application_latency: float | None
    dependency_latency: float | None
    database_latency: float | None
    error_rate: float | None

    def to_dict(self) -> dict[str, float]:
        """Convert metrics to dictionary format."""
        ...

    def validate(self) -> bool:
        """Validate the metrics data."""
        ...


@runtime_checkable
class AnomalyData(Protocol):
    """Protocol for anomaly detection results."""

    anomaly_type: str
    severity: AnomalySeverity
    confidence_score: float
    description: str
    threshold_value: float | None
    actual_value: float | None


# =============================================================================
# Metrics Collection Interface
# =============================================================================


class MetricsCollector(ABC):
    """Abstract base class for metrics collection.

    Implementations should handle:
    - Connection management
    - Retry logic
    - Circuit breaker patterns
    """

    @abstractmethod
    def collect_service_metrics(self, service_name: str) -> MetricsData:
        """Collect current metrics for a service.

        Args:
            service_name: Name of the service to collect metrics for.

        Returns:
            MetricsData containing collected metrics.

        Raises:
            MetricsCollectionError: If collection fails.
            CircuitBreakerOpenError: If circuit breaker is open.
        """
        ...

    @abstractmethod
    def is_circuit_open(self) -> bool:
        """Check if the circuit breaker is open.

        Returns:
            True if circuit breaker is open, False otherwise.
        """
        ...

    @abstractmethod
    def health_check(self) -> bool:
        """Check if the metrics source is healthy.

        Returns:
            True if healthy, False otherwise.
        """
        ...


# =============================================================================
# Model Management Interface
# =============================================================================


class ModelManager(ABC):
    """Abstract base class for model management.

    Handles model loading, caching, and lifecycle.
    """

    @abstractmethod
    def get_available_services(self) -> list[str]:
        """Get list of services with available models.

        Returns:
            List of service names with trained models.
        """
        ...

    @abstractmethod
    def get_base_services(self) -> list[str]:
        """Get list of base service names (without time period suffixes).

        Returns:
            List of base service names.
        """
        ...

    @abstractmethod
    def load_model(self, service_name: str) -> Any:
        """Load a model for a specific service.

        Args:
            service_name: Name of the service (may include time period suffix).

        Returns:
            Loaded model instance.

        Raises:
            ModelLoadError: If model cannot be loaded.
            ModelNotFoundError: If model doesn't exist.
        """
        ...

    @abstractmethod
    def get_model_metadata(self, service_name: str) -> dict[str, Any]:
        """Get metadata for a loaded model.

        Args:
            service_name: Name of the service.

        Returns:
            Dictionary containing model metadata.
        """
        ...


# =============================================================================
# Anomaly Detection Interface
# =============================================================================


class AnomalyDetector(ABC):
    """Abstract base class for anomaly detection.

    Implementations should provide methods for both
    simple detection and context-aware detection.
    """

    @abstractmethod
    def detect_anomalies(
        self,
        metrics: dict[str, Any],
        timestamp: datetime | None = None,
    ) -> dict[str, Any]:
        """Detect anomalies in the provided metrics.

        Args:
            metrics: Dictionary of metric name to value.
            timestamp: Optional timestamp for the metrics.

        Returns:
            Dictionary mapping anomaly names to anomaly data.
        """
        ...

    @abstractmethod
    def get_time_period(self, timestamp: datetime) -> TimePeriod:
        """Determine the time period for a timestamp.

        Args:
            timestamp: The timestamp to classify.

        Returns:
            The TimePeriod enum value.
        """
        ...


class TimeAwareDetector(AnomalyDetector):
    """Extended interface for time-aware anomaly detection."""

    @abstractmethod
    def detect_anomalies_with_context(
        self,
        metrics: dict[str, Any],
        timestamp: datetime,
        models_directory: str,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Detect anomalies with full context and explainability.

        Args:
            metrics: Dictionary of metric name to value.
            timestamp: Timestamp for the metrics.
            models_directory: Path to models directory.
            verbose: Enable verbose logging.

        Returns:
            Dictionary with anomalies and context information.
        """
        ...

    @abstractmethod
    def get_available_periods(self) -> list[TimePeriod]:
        """Get list of available time periods with trained models.

        Returns:
            List of TimePeriod values with available models.
        """
        ...


# =============================================================================
# Fingerprinting Interface
# =============================================================================


class IncidentTracker(ABC):
    """Abstract base class for incident tracking and fingerprinting.

    Handles incident lifecycle: creation, continuation, resolution.
    """

    @abstractmethod
    def process_anomalies(
        self,
        full_service_name: str,
        anomaly_result: dict[str, Any],
        current_metrics: dict[str, Any] | None = None,
        timestamp: datetime | None = None,
    ) -> dict[str, Any]:
        """Process anomaly results and track incidents.

        Args:
            full_service_name: Service name with model suffix.
            anomaly_result: Detection results to process.
            current_metrics: Optional current metric values.
            timestamp: Optional timestamp for this detection.

        Returns:
            Enhanced result with fingerprinting information.
        """
        ...

    @abstractmethod
    def get_open_incidents(self, service_name: str | None = None) -> dict[str, list[dict]]:
        """Get all open incidents, optionally filtered by service.

        Args:
            service_name: Optional service name filter.

        Returns:
            Dictionary mapping service names to incident lists.
        """
        ...

    @abstractmethod
    def get_incident_by_id(self, incident_id: str) -> dict[str, Any] | None:
        """Get a specific incident by ID.

        Args:
            incident_id: The incident ID to look up.

        Returns:
            Incident data or None if not found.
        """
        ...

    @abstractmethod
    def get_statistics(self) -> dict[str, Any]:
        """Get fingerprinting statistics.

        Returns:
            Dictionary with statistical information.
        """
        ...


# =============================================================================
# Results Processing Interface
# =============================================================================


class ResultsProcessor(ABC):
    """Abstract base class for processing and outputting results."""

    @abstractmethod
    def process_result(self, result: dict[str, Any]) -> None:
        """Process a single detection result.

        Args:
            result: The detection result to process.
        """
        ...

    @abstractmethod
    def output_anomalies(self) -> None:
        """Output all processed anomalies."""
        ...

    @abstractmethod
    def get_detected_anomalies(self) -> list[dict[str, Any]]:
        """Get list of detected anomalies.

        Returns:
            List of anomaly dictionaries.
        """
        ...

    @abstractmethod
    def clear(self) -> None:
        """Clear stored anomalies."""
        ...


# =============================================================================
# Observability Integration Interface
# =============================================================================


class ObservabilityClient(ABC):
    """Abstract base class for observability service integration."""

    @abstractmethod
    def send_anomalies(self, anomalies: list[dict[str, Any]]) -> bool:
        """Send anomalies to the observability service.

        Args:
            anomalies: List of anomaly dictionaries to send.

        Returns:
            True if successful, False otherwise.
        """
        ...

    @abstractmethod
    def send_resolutions(self, resolutions: list[dict[str, Any]]) -> bool:
        """Send incident resolutions to the observability service.

        Args:
            resolutions: List of resolution dictionaries to send.

        Returns:
            True if successful, False otherwise.
        """
        ...


# =============================================================================
# Pipeline Interface
# =============================================================================


class InferencePipeline(ABC):
    """Abstract base class for the inference pipeline."""

    @abstractmethod
    def run_inference(
        self,
        service_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run inference for specified services.

        Args:
            service_names: Optional list of services. If None, uses all available.

        Returns:
            Dictionary mapping service names to results.
        """
        ...

    @abstractmethod
    def get_system_status(self) -> dict[str, Any]:
        """Get comprehensive system status.

        Returns:
            Dictionary with system status information.
        """
        ...
