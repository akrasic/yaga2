"""
Enhanced anomaly detection engine with explainability support.
"""

import logging
from typing import Any, Dict, List

from smartbox_anomaly.core import AnomalySeverity, ModelLoadError
from vmclient import InferenceMetrics

from .model_manager import EnhancedModelManager
from .models import AnomalyResult

logger = logging.getLogger(__name__)


class EnhancedAnomalyDetectionEngine:
    """Enhanced anomaly detection engine with explainability support"""

    def __init__(self, model_manager: EnhancedModelManager):
        self.model_manager = model_manager

    def detect_anomalies_with_context(
        self, metrics: InferenceMetrics, use_explainable: bool = True
    ) -> Dict[str, Any]:
        """Enhanced anomaly detection with full context and explainability"""
        try:
            # Load enhanced model
            model = self.model_manager.load_model(metrics.service_name)

            # Convert metrics to model format
            metrics_dict = metrics.to_dict()

            # Use explainable detection if available and requested
            if use_explainable and hasattr(model, "detect_anomalies_with_context"):
                enhanced_result = model.detect_anomalies_with_context(
                    metrics_dict, metrics.timestamp
                )
                return enhanced_result
            else:
                # Fallback to standard detection
                raw_anomalies = model.detect_anomalies(metrics_dict)
                return {
                    "service": metrics.service_name,
                    "timestamp": metrics.timestamp.isoformat(),
                    "anomalies": raw_anomalies,
                    "metrics": metrics_dict,
                    "explainable": False,
                }

        except ModelLoadError:
            raise
        except Exception as e:
            logger.error(
                f"Enhanced anomaly detection failed for {metrics.service_name}: {e}"
            )
            raise RuntimeError(f"Detection failed: {e}")

    def detect_anomalies(self, metrics: InferenceMetrics) -> List[AnomalyResult]:
        """Legacy method for backward compatibility"""
        try:
            model = self.model_manager.load_model(metrics.service_name)
            metrics_dict = metrics.to_dict()
            raw_anomalies = model.detect_anomalies(metrics_dict)
            anomalies = self._process_anomalies(raw_anomalies)
            return anomalies

        except ModelLoadError:
            raise
        except Exception as e:
            logger.error(f"Anomaly detection failed for {metrics.service_name}: {e}")
            raise RuntimeError(f"Detection failed: {e}")

    def _process_anomalies(self, raw_anomalies: Dict) -> List[AnomalyResult]:
        """Process raw anomalies into structured format with enhanced fields"""
        anomalies = []

        for anomaly_name, anomaly_data in raw_anomalies.items():
            try:
                if isinstance(anomaly_data, dict):
                    anomaly = AnomalyResult(
                        anomaly_type=anomaly_name,
                        severity=self._map_severity(
                            anomaly_data.get("severity", "medium")
                        ),
                        confidence_score=anomaly_data.get("score", 0.5),
                        description=anomaly_data.get(
                            "description", anomaly_name.replace("_", " ").title()
                        ),
                        threshold_value=anomaly_data.get("threshold"),
                        actual_value=anomaly_data.get("value"),
                        metadata=anomaly_data,
                        # Extract explainability data if available
                        comparison_data=anomaly_data.get("comparison_data"),
                        business_impact=anomaly_data.get("business_impact"),
                        percentile_position=anomaly_data.get("percentile_position"),
                    )
                else:
                    # Handle legacy format
                    anomaly = AnomalyResult(
                        anomaly_type=anomaly_name,
                        severity=AnomalySeverity.MEDIUM,
                        confidence_score=0.5,
                        description=anomaly_name.replace("_", " ").title(),
                    )

                anomalies.append(anomaly)

            except Exception as e:
                logger.warning(f"Failed to process anomaly {anomaly_name}: {e}")

        return anomalies

    def _map_severity(self, severity_str: str) -> AnomalySeverity:
        """Map string severity to enum"""
        mapping = {
            "low": AnomalySeverity.LOW,
            "medium": AnomalySeverity.MEDIUM,
            "high": AnomalySeverity.HIGH,
            "critical": AnomalySeverity.CRITICAL,
        }
        return mapping.get(severity_str.lower(), AnomalySeverity.MEDIUM)
