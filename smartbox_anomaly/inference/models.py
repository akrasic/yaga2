"""
Data models for inference results.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from smartbox_anomaly.core import AnomalySeverity
from vmclient import InferenceMetrics


@dataclass
class AnomalyResult:
    """Single anomaly detection result with explainability"""
    anomaly_type: str
    severity: AnomalySeverity
    confidence_score: float
    description: str
    threshold_value: Optional[float] = None
    actual_value: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    # Explainability fields
    comparison_data: Optional[Dict[str, Any]] = None
    business_impact: Optional[str] = None
    percentile_position: Optional[float] = None


@dataclass
class ServiceInferenceResult:
    """Complete inference result for a service with explainability"""
    service_name: str
    timestamp: datetime
    input_metrics: InferenceMetrics
    anomalies: List[AnomalyResult]
    model_version: str
    inference_time_ms: float
    status: str  # 'success', 'error', 'no_model'
    error_message: Optional[str] = None
    # Explainability fields
    historical_context: Optional[Dict[str, Any]] = None
    metric_analysis: Optional[Dict[str, Any]] = None
    explanation: Optional[Dict[str, Any]] = None
    recommended_actions: Optional[List[str]] = None
    # Exception enrichment context (populated when error_rate anomalies detected)
    exception_context: Optional[Dict[str, Any]] = None

    @property
    def has_anomalies(self) -> bool:
        return len(self.anomalies) > 0

    @property
    def max_severity(self) -> Optional[AnomalySeverity]:
        if not self.anomalies:
            return None
        severity_order = [
            AnomalySeverity.LOW,
            AnomalySeverity.MEDIUM,
            AnomalySeverity.HIGH,
            AnomalySeverity.CRITICAL,
        ]
        return max(
            (a.severity for a in self.anomalies),
            key=lambda x: severity_order.index(x)
        )
