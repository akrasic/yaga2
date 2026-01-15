"""
Enhanced results processor with explainability support.

This module orchestrates result processing by delegating to specialized components:
- AlertFormatter: Formats detection results into API payloads
- File I/O: Persists alerts to disk
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from .alert_formatter import AlertFormatter
from .models import ServiceInferenceResult


class EnhancedResultsProcessor:
    """Enhanced results processor with explainability support.

    This class orchestrates result processing by:
    - Delegating formatting to AlertFormatter
    - Managing file I/O for alert persistence
    - Tracking detected anomalies for output
    """

    def __init__(self, alerts_directory: str = "./alerts/", verbose: bool = False):
        """Initialize the results processor.

        Args:
            alerts_directory: Directory for storing alert files.
            verbose: Enable verbose output mode.
        """
        self.alerts_directory = Path(alerts_directory)
        self.alerts_directory.mkdir(exist_ok=True)
        self.detected_anomalies: List[Dict[str, Any]] = []
        self.verbose = verbose
        self._formatter = AlertFormatter()

    def process_explainable_result(self, result: Dict[str, Any]) -> None:
        """Process explainable anomaly detection result.

        Args:
            result: Explainable detection result dictionary.
        """
        if result.get("anomaly_count", 0) > 0:
            alert_json = self._formatter.format_explainable_alert(result)
            self.detected_anomalies.append(alert_json)
            self._save_alert_to_file(alert_json, result.get("service", "unknown"))
        elif result.get("error"):
            self._handle_error_result(result)

    def process_result(self, result: Any) -> None:
        """Process different types of results.

        Routes results to appropriate handlers based on type.

        Args:
            result: Detection result (dict or ServiceInferenceResult).
        """
        # Handle explainable results (new format)
        if isinstance(result, dict) and "explainable" in result:
            if result.get("explainable", False):
                self.process_explainable_result(result)
                return

        # Handle time-aware results (existing format)
        if isinstance(result, dict):
            self._process_dict_result(result)
            return

        # Handle ServiceInferenceResult objects (legacy)
        self._process_service_inference_result(result)

    def _process_dict_result(self, result: dict) -> None:
        """Process dictionary-format detection result.

        Args:
            result: Detection result dictionary.
        """
        if result.get("anomalies") and len(result["anomalies"]) > 0:
            alert_json = self._formatter.format_time_aware_alert(result)
            self.detected_anomalies.append(alert_json)
            self._save_alert_to_file(
                alert_json, result.get("service", result.get("service_name", "unknown"))
            )
        elif result.get("error"):
            self._handle_error_result(result)

    def _process_service_inference_result(self, result: ServiceInferenceResult) -> None:
        """Process ServiceInferenceResult object.

        Args:
            result: ServiceInferenceResult object.
        """
        if result.status == "success" and result.has_anomalies:
            alert_json = self._formatter.format_standard_alert(result)
            self.detected_anomalies.append(alert_json)
            self._save_alert_to_file(alert_json, result.service_name)
        elif result.status == "error":
            error_json = {
                "alert_type": "error",
                "service": result.service_name,
                "error_message": result.error_message,
                "timestamp": result.timestamp.isoformat(),
            }
            self.detected_anomalies.append(error_json)

    def _handle_error_result(self, result: dict) -> None:
        """Handle error result by adding to detected anomalies.

        Args:
            result: Result dictionary containing error.
        """
        error_json = {
            "alert_type": "error",
            "service": result.get("service", "unknown"),
            "error_message": result["error"],
            "timestamp": result.get("timestamp", datetime.now().isoformat()),
        }
        self.detected_anomalies.append(error_json)

    def _save_alert_to_file(self, alert_data: dict, service_name: str) -> None:
        """Save alert to daily JSONL file.

        Args:
            alert_data: Formatted alert data.
            service_name: Name of the service.
        """
        alert_data = dict(alert_data)  # Copy to avoid modifying original
        alert_data["alert_id"] = f"{service_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        alert_file = (
            self.alerts_directory / f"alerts_{datetime.now().strftime('%Y-%m-%d')}.jsonl"
        )

        with open(alert_file, "a") as f:
            f.write(json.dumps(alert_data, default=str) + "\n")

    def output_anomalies(self) -> None:
        """Output all detected anomalies based on verbose mode."""
        if self.verbose:
            if self.detected_anomalies:
                print("\n" + "=" * 50)
                print("DETECTED ANOMALIES:")
                print("=" * 50)
                print(json.dumps(self.detected_anomalies, indent=2))
            else:
                print("\n" + "=" * 50)
                print("No anomalies detected across all services.")
                print("=" * 50)
        else:
            print(json.dumps(self.detected_anomalies, indent=2))

    def clear_anomalies(self) -> None:
        """Clear stored anomalies."""
        self.detected_anomalies = []

    # Legacy method aliases for backward compatibility
    def _format_time_aware_alert_json(self, result: dict) -> dict:
        """Format time-aware anomaly as structured JSON.

        Delegates to AlertFormatter for backward compatibility.

        Args:
            result: Raw detection result.

        Returns:
            Formatted alert payload.
        """
        return self._formatter.format_time_aware_alert(result)

    def _format_explainable_alert_json(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Format explainable anomaly result as enhanced JSON.

        Delegates to AlertFormatter for backward compatibility.

        Args:
            result: Raw explainable detection result.

        Returns:
            Formatted alert payload.
        """
        return self._formatter.format_explainable_alert(result)

    def _format_alert_json(self, result: ServiceInferenceResult) -> dict:
        """Format regular ServiceInferenceResult as structured JSON.

        Delegates to AlertFormatter for backward compatibility.

        Args:
            result: ServiceInferenceResult object.

        Returns:
            Formatted alert payload.
        """
        return self._formatter.format_standard_alert(result)

    # Legacy save methods for backward compatibility
    def _save_explainable_alert(self, result: Dict[str, Any]) -> None:
        """Save explainable alert to file.

        Args:
            result: Explainable detection result.
        """
        alert_data = self._formatter.format_explainable_alert(result)
        self._save_alert_to_file(alert_data, result.get("service", "unknown"))

    def _save_time_aware_alert(self, result: dict) -> None:
        """Save time-aware alert to file.

        Args:
            result: Time-aware detection result.
        """
        alert_data = self._formatter.format_time_aware_alert(result)
        self._save_alert_to_file(
            alert_data, result.get("service", result.get("service_name", "unknown"))
        )

    def _save_alert(self, result: ServiceInferenceResult) -> None:
        """Save regular alert to file.

        Args:
            result: ServiceInferenceResult object.
        """
        alert_data = self._formatter.format_standard_alert(result)
        self._save_alert_to_file(alert_data, result.service_name)
