"""
Enhanced Production-Grade ML Inference Pipeline for Smartbox Anomaly Detection
Now supports explainable anomaly detection with rich context and recommendations
"""

import json
import logging
import argparse
import math
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import time
import urllib3

# Import from smartbox_anomaly package (primary source)
from smartbox_anomaly.core import (
    AnomalySeverity,
    ModelLoadError,
    MetricsCollectionError,
    get_config,
    ObservabilityConfig,
    DependencyContext,
    DependencyStatus,
    SLOConfig,
)
from smartbox_anomaly.slo import SLOEvaluator
from smartbox_anomaly.detection import (
    SmartboxAnomalyDetector,
    TimeAwareAnomalyDetector,
)
from smartbox_anomaly.fingerprinting import AnomalyFingerprinter
from smartbox_anomaly.enrichment import ExceptionEnrichmentService, ServiceGraphEnrichmentService
from smartbox_anomaly.api import (
    AlertType,
    AnomalyDetectedPayload,
    Anomaly,
    CascadeInfo,
    CurrentMetrics,
    DetectionSignal,
    FingerprintingMetadata,
    PayloadMetadata,
    Severity,
)

# Import VictoriaMetrics client (still uses root-level stub for backward compatibility)
from vmclient import VictoriaMetricsClient, InferenceMetrics

# Disable urllib3 warnings for cleaner output
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Reduce urllib3 logging level to avoid connection pool warnings
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)


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
    # NEW: Explainability fields
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
    # NEW: Explainability fields
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
        severity_order = [AnomalySeverity.LOW, AnomalySeverity.MEDIUM, 
                         AnomalySeverity.HIGH, AnomalySeverity.CRITICAL]
        return max((a.severity for a in self.anomalies), key=lambda x: severity_order.index(x))


class EnhancedModelManager:
    """Enhanced model management - UPDATED to not interfere with lazy loading"""

    def __init__(self, models_directory: str = "./smartbox_models/"):
        self.models_directory = Path(models_directory)
        self._model_cache = {}
        self._model_metadata = {}
        self._load_times = {}
        self._model_validators = {}

    def load_services_from_config(self, config_path: str = "./config.json") -> List[str]:
        """Load services list from config.json.

        Combines all service categories: critical, standard, micro, admin, core.
        Returns empty list if config not found or invalid.
        """
        try:
            with open(config_path) as f:
                config = json.load(f)

            services_config = config.get("services", {})
            all_services = []

            # Collect services from all categories
            for category in ["critical", "standard", "micro", "admin", "core", "background"]:
                category_services = services_config.get(category, [])
                if isinstance(category_services, list):
                    all_services.extend(category_services)

            # Remove duplicates while preserving order
            seen = set()
            unique_services = []
            for svc in all_services:
                if svc not in seen:
                    seen.add(svc)
                    unique_services.append(svc)

            return unique_services

        except (FileNotFoundError, json.JSONDecodeError, Exception) as e:
            logger.debug(f"Could not load services from config: {e}")
            return []

    def get_services_with_models(self, config_services: List[str]) -> Tuple[List[str], List[str]]:
        """Check which config services have trained models.

        Args:
            config_services: List of services from config.json

        Returns:
            Tuple of (services_with_models, services_missing_models)
        """
        available = set(self.get_base_services())
        with_models = [s for s in config_services if s in available]
        missing = [s for s in config_services if s not in available]
        return with_models, missing

    def get_base_services(self) -> List[str]:
        """Get list of base service names (without time period suffixes) - FIXED for 5-period"""
        all_services = self.get_available_services()
        base_services = set()
        
        # Extract base service names by removing time period suffixes
        # Updated for 5-period approach
        time_suffixes = [
            '_business_hours', 
            '_evening_hours', 
            '_night_hours', 
            '_weekend_day',     # New 5-period
            '_weekend_night',   # New 5-period
            '_weekend'          # Legacy 4-period (for backward compatibility)
        ]
        
        for service in all_services:
            base_name = service
            for suffix in time_suffixes:
                if service.endswith(suffix):
                    base_name = service[:-len(suffix)]
                    break
            base_services.add(base_name)
        
        return sorted(list(base_services))
    
    def get_available_services(self) -> List[str]:
        """Get list of services with valid models - includes period-specific models"""
        if not self.models_directory.exists():
            logger.warning(f"Models directory does not exist: {self.models_directory}")
            return []
        
        services = []
        for item in self.models_directory.iterdir():
            if item.is_dir():
                model_data_file = item / "model_data.json"
                if model_data_file.exists():
                    services.append(item.name)
        
        return sorted(services)
    
    def load_model(self, service_name: str) -> SmartboxAnomalyDetector:
        """Load individual period model - ONLY used for non-time-aware detection"""
        # This should NOT be called for time-aware inference
        # Only used for individual period model loading in specific cases
        
        cache_key = service_name
        
        try:
            service_dir = self.models_directory / service_name
            if not service_dir.exists():
                raise ModelLoadError(f"Model directory not found: {service_dir}")
            
            # Check if reload is needed
            current_mod_time = service_dir.stat().st_mtime
            cached_mod_time = self._load_times.get(cache_key, 0)
            
            if cache_key not in self._model_cache or current_mod_time > cached_mod_time:
                logger.info(f"Loading individual model for {service_name}")
                
                # Load using secure method
                model = SmartboxAnomalyDetector.load_model_secure(
                    str(self.models_directory), service_name
                )
                
                # Cache model and metadata
                self._model_cache[cache_key] = model
                self._model_metadata[cache_key] = model.model_metadata
                self._load_times[cache_key] = current_mod_time
                
                model_version = model.model_metadata.get('model_version', 'unknown')
                explainability_status = "enabled" if hasattr(model, 'training_statistics') and model.training_statistics else "disabled"
                
                logger.info(f"Successfully loaded individual model for {service_name}: {model_version}")
                logger.info(f"Explainability features: {explainability_status}")
            
            return self._model_cache[cache_key]
            
        except Exception as e:
            logger.error(f"Failed to load individual model for {service_name}: {e}")
            raise ModelLoadError(f"Model loading failed for {service_name}: {e}")
    
    def get_model_metadata(self, service_name: str) -> Dict[str, Any]:
        """Get model metadata for a service"""
        return self._model_metadata.get(service_name, {})


class EnhancedAnomalyDetectionEngine:
    """Enhanced anomaly detection engine with explainability support"""
    
    def __init__(self, model_manager: EnhancedModelManager):
        self.model_manager = model_manager
    
    def detect_anomalies_with_context(self, metrics: InferenceMetrics, 
                                    use_explainable: bool = True) -> Dict[str, Any]:
        """Enhanced anomaly detection with full context and explainability"""
        try:
            # Load enhanced model
            model = self.model_manager.load_model(metrics.service_name)
            
            # Convert metrics to model format
            metrics_dict = metrics.to_dict()
            
            # Use explainable detection if available and requested
            if use_explainable and hasattr(model, 'detect_anomalies_with_context'):
                enhanced_result = model.detect_anomalies_with_context(metrics_dict, metrics.timestamp)
                return enhanced_result
            else:
                # Fallback to standard detection
                raw_anomalies = model.detect_anomalies(metrics_dict)
                return {
                    'service': metrics.service_name,
                    'timestamp': metrics.timestamp.isoformat(),
                    'anomalies': raw_anomalies,
                    'metrics': metrics_dict,
                    'explainable': False
                }
                
        except ModelLoadError:
            raise
        except Exception as e:
            logger.error(f"Enhanced anomaly detection failed for {metrics.service_name}: {e}")
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
                        severity=self._map_severity(anomaly_data.get('severity', 'medium')),
                        confidence_score=anomaly_data.get('score', 0.5),
                        description=anomaly_data.get('description', anomaly_name.replace('_', ' ').title()),
                        threshold_value=anomaly_data.get('threshold'),
                        actual_value=anomaly_data.get('value'),
                        metadata=anomaly_data,
                        # NEW: Extract explainability data if available
                        comparison_data=anomaly_data.get('comparison_data'),
                        business_impact=anomaly_data.get('business_impact'),
                        percentile_position=anomaly_data.get('percentile_position')
                    )
                else:
                    # Handle legacy format
                    anomaly = AnomalyResult(
                        anomaly_type=anomaly_name,
                        severity=AnomalySeverity.MEDIUM,
                        confidence_score=0.5,
                        description=anomaly_name.replace('_', ' ').title()
                    )
                
                anomalies.append(anomaly)
                
            except Exception as e:
                logger.warning(f"Failed to process anomaly {anomaly_name}: {e}")
        
        return anomalies
    
    def _map_severity(self, severity_str: str) -> AnomalySeverity:
        """Map string severity to enum"""
        mapping = {
            'low': AnomalySeverity.LOW,
            'medium': AnomalySeverity.MEDIUM,
            'high': AnomalySeverity.HIGH,
            'critical': AnomalySeverity.CRITICAL
        }
        return mapping.get(severity_str.lower(), AnomalySeverity.MEDIUM)


class EnhancedResultsProcessor:
    """Enhanced results processor with explainability support"""
    
    def __init__(self, alerts_directory: str = "./alerts/", verbose: bool = False):
        self.alerts_directory = Path(alerts_directory)
        self.alerts_directory.mkdir(exist_ok=True)
        self.detected_anomalies = []
        self.verbose = verbose
    
    def process_explainable_result(self, result: Dict[str, Any]):
        """Process explainable anomaly detection result"""
        if result.get('anomaly_count', 0) > 0:
            # Enhanced alert with explainability
            alert_json = self._format_explainable_alert_json(result)
            self.detected_anomalies.append(alert_json)
            self._save_explainable_alert(result)
        elif result.get('error'):
            error_json = {
                "alert_type": "error",
                "service": result.get('service', 'unknown'),
                "error_message": result['error'],
                "timestamp": result.get('timestamp', datetime.now().isoformat())
            }
            self.detected_anomalies.append(error_json)
    
    def process_result(self, result):
        """Process different types of results"""
        # Handle explainable results (new format)
        if isinstance(result, dict) and 'explainable' in result:
            if result.get('explainable', False):
                self.process_explainable_result(result)
                return
        
        # Handle time-aware results (existing format)
        if isinstance(result, dict):
            if result.get('anomalies') and len(result['anomalies']) > 0:
                alert_json = self._format_time_aware_alert_json(result)
                self.detected_anomalies.append(alert_json)
                self._save_time_aware_alert(result)
            elif result.get('error'):
                error_json = {
                    "alert_type": "error",
                    "service": result.get('service', 'unknown'),
                    "error_message": result['error'],
                    "timestamp": result.get('timestamp', datetime.now().isoformat())
                }
                self.detected_anomalies.append(error_json)
            return
        
        # Handle ServiceInferenceResult objects (legacy)
        if result.status == 'success' and result.has_anomalies:
            alert_json = self._format_alert_json(result)
            self.detected_anomalies.append(alert_json)
            self._save_alert(result)
        elif result.status == 'error':
            error_json = {
                "alert_type": "error", 
                "service": result.service_name,
                "error_message": result.error_message,
                "timestamp": result.timestamp.isoformat()
            }
            self.detected_anomalies.append(error_json)
    
    def _format_explainable_alert_json(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Format explainable anomaly result as enhanced JSON"""
        enhanced_anomalies = []
        
        # Process anomalies with explainability data
        for anomaly_data in result.get('anomalies', []):
            if isinstance(anomaly_data, dict):
                enhanced_anomaly = {
                    "type": anomaly_data.get('type', 'unknown'),
                    "severity": anomaly_data.get('severity', 'medium'),
                    "confidence_score": anomaly_data.get('score', 0.0),
                    "description": anomaly_data.get('description', 'Anomaly detected'),
                    "detection_method": anomaly_data.get('detection_method', anomaly_data.get('type', 'unknown')),
                    "threshold_value": anomaly_data.get('threshold_value'),
                    "actual_value": anomaly_data.get('actual_value'),
                    # NEW: Explainability fields
                    "comparison_data": anomaly_data.get('comparison_data'),
                    "feature_contributions": anomaly_data.get('feature_contributions'),
                    "business_impact": anomaly_data.get('business_impact'),
                    "metadata": anomaly_data.get('metadata', {})
                }
                enhanced_anomalies.append(enhanced_anomaly)
        
        base_alert = {
            "alert_type": "anomaly_detected",
            "service": result['service'],
            "timestamp": result['timestamp'],
            "overall_severity": result['overall_severity'],
            "anomaly_count": result['anomaly_count'],
            "current_metrics": result['current_metrics'],
            "anomalies": enhanced_anomalies,
            "model_type": "explainable_ml"
        }
        
        # Add explainability context if available
        if 'historical_context' in result:
            base_alert['historical_context'] = result['historical_context']
        
        if 'metric_analysis' in result:
            base_alert['metric_analysis'] = result['metric_analysis']
        
        if 'explanation' in result:
            base_alert['explanation'] = result['explanation']
        
        if 'recommended_actions' in result:
            base_alert['recommended_actions'] = result['recommended_actions']
        
        if 'model_metadata' in result:
            base_alert['model_metadata'] = result['model_metadata']
        
        return base_alert
    
    def _format_time_aware_alert_json(self, result: dict) -> dict:
        """Format time-aware anomaly as structured JSON matching API specification.

        Produces payload format documented in docs/INFERENCE_API_PAYLOAD.md
        Uses Pydantic models for validation and consistency.
        """
        # Handle both dict and list formats for anomalies
        anomalies_raw = result.get('anomalies', {})
        if isinstance(anomalies_raw, list):
            # Convert list to dict format keyed by anomaly name
            anomalies_dict = {}
            for i, anomaly in enumerate(anomalies_raw):
                if isinstance(anomaly, dict):
                    anomaly_name = (
                        anomaly.get('anomaly_name') or
                        anomaly.get('pattern_name') or
                        anomaly.get('type') or
                        f'anomaly_{i}'
                    )
                    anomalies_dict[anomaly_name] = anomaly
            anomalies_raw = anomalies_dict

        # Format anomalies using Pydantic models
        formatted_anomalies: Dict[str, Anomaly] = {}
        severities: List[Severity] = []

        for anomaly_name, anomaly_data in anomalies_raw.items():
            if isinstance(anomaly_data, dict):
                anomaly_model = self._build_anomaly_model(anomaly_name, anomaly_data)
                formatted_anomalies[anomaly_name] = anomaly_model
                severities.append(anomaly_model.severity)

        # Extract time period and model info
        time_period = result.get('time_period', 'unknown')
        model_name = result.get('model_name', time_period)
        service_name = result.get('service', result.get('service_name', 'unknown'))
        timestamp = result.get('timestamp', datetime.now().isoformat())

        # Determine alert type and overall severity
        alert_type = AlertType.ANOMALY_DETECTED if formatted_anomalies else AlertType.NO_ANOMALY

        # Use SLO-adjusted severity if available, otherwise calculate from anomalies
        slo_eval = result.get('slo_evaluation', {})
        if slo_eval.get('severity_changed') and result.get('overall_severity'):
            # SLO evaluation already adjusted the severity - use it
            severity_str = result['overall_severity']
            overall_severity = Severity(severity_str) if severity_str in [s.value for s in Severity] else Severity.NONE
        else:
            # No SLO adjustment - calculate from individual anomaly severities
            overall_severity = Severity.max_severity(severities) if severities else Severity.NONE

        # Build current metrics model
        metrics_data = result.get('metrics', result.get('current_metrics', {}))
        current_metrics = CurrentMetrics(**metrics_data) if metrics_data else CurrentMetrics()

        # Build fingerprinting metadata if present
        fingerprinting = None
        if 'fingerprinting' in result and result['fingerprinting']:
            try:
                fingerprinting = FingerprintingMetadata(**result['fingerprinting'])
            except Exception:
                # Fall back to dict if validation fails
                fingerprinting = result['fingerprinting']

        # Build payload metadata
        models_used = self._extract_models_used({k: v.model_dump() for k, v in formatted_anomalies.items()})
        metadata = PayloadMetadata(
            service_name=service_name,
            detection_timestamp=timestamp,
            models_used=models_used,
            enhanced_messaging=True,
            features={
                "contextual_severity": True,
                "named_patterns": True,
                "recommendations": True,
                "interpretations": True,
                "anomaly_correlation": True
            }
        )

        # Build the AnomalyDetectedPayload using Pydantic model
        payload = AnomalyDetectedPayload(
            alert_type=alert_type,
            service_name=service_name,
            timestamp=timestamp,
            time_period=time_period,
            model_name=model_name,
            model_type=result.get('model_type', 'time_aware_5period'),
            anomalies=formatted_anomalies,
            anomaly_count=len(formatted_anomalies),
            overall_severity=overall_severity,
            current_metrics=current_metrics,
            fingerprinting=fingerprinting if isinstance(fingerprinting, FingerprintingMetadata) else None,
            performance_info=result.get('performance_info'),
            exception_context=result.get('exception_context'),
            service_graph_context=result.get('service_graph_context'),
            metadata=metadata,
        )

        # Convert to dict and add fingerprinting as dict if it wasn't a valid model
        response = payload.model_dump(mode='json', exclude_none=True)
        if fingerprinting and not isinstance(fingerprinting, FingerprintingMetadata):
            response['fingerprinting'] = fingerprinting

        # Add SLO evaluation if present (not part of Pydantic model yet)
        if result.get('slo_evaluation'):
            response['slo_evaluation'] = result['slo_evaluation']

        # Add drift analysis if present
        if result.get('drift_analysis'):
            response['drift_analysis'] = result['drift_analysis']
        if result.get('drift_warning'):
            response['drift_warning'] = result['drift_warning']

        # Add validation warnings if present
        if result.get('validation_warnings'):
            response['validation_warnings'] = result['validation_warnings']

        return response

    def _format_single_anomaly(self, anomaly_name: str, anomaly_data: dict) -> dict:
        """Format a single anomaly to match API specification."""
        # Determine if this is a consolidated anomaly (multiple detection signals)
        detection_signals = anomaly_data.get('detection_signals', [])
        is_consolidated = len(detection_signals) > 1 or anomaly_data.get('type') == 'consolidated'

        # Build base anomaly structure
        formatted = {
            "type": "consolidated" if is_consolidated else anomaly_data.get('type', 'ml_isolation'),
            "severity": anomaly_data.get('severity', 'medium'),
            "confidence": float(anomaly_data.get('confidence', anomaly_data.get('score', 0.5))),
            "score": float(anomaly_data.get('score', anomaly_data.get('confidence', 0.0))),
            "description": anomaly_data.get('description', anomaly_name.replace('_', ' ').title()),
        }

        # Add root metric if available
        if anomaly_data.get('root_metric'):
            formatted['root_metric'] = anomaly_data['root_metric']
        elif 'latency' in anomaly_name.lower():
            formatted['root_metric'] = 'application_latency'
        elif 'error' in anomaly_name.lower():
            formatted['root_metric'] = 'error_rate'
        elif 'traffic' in anomaly_name.lower() or 'request' in anomaly_name.lower():
            formatted['root_metric'] = 'request_rate'

        # Add signal count for consolidated anomalies
        if is_consolidated:
            formatted['signal_count'] = len(detection_signals) if detection_signals else anomaly_data.get('signal_count', 1)

        # Add pattern name if present
        if anomaly_data.get('pattern_name'):
            formatted['pattern_name'] = anomaly_data['pattern_name']

        # Add interpretation if present
        if anomaly_data.get('interpretation'):
            formatted['interpretation'] = anomaly_data['interpretation']

        # Add current value
        if anomaly_data.get('value') is not None:
            formatted['value'] = anomaly_data['value']
        elif anomaly_data.get('actual_value') is not None:
            formatted['value'] = anomaly_data['actual_value']

        # Add detection signals array
        if detection_signals:
            formatted['detection_signals'] = detection_signals
        else:
            # Create a single detection signal from the anomaly data
            signal = {
                "method": anomaly_data.get('detection_method', 'isolation_forest'),
                "type": anomaly_data.get('type', 'ml_isolation'),
                "severity": anomaly_data.get('severity', 'medium'),
                "score": float(anomaly_data.get('score', 0.0)),
            }
            if anomaly_data.get('direction'):
                signal['direction'] = anomaly_data['direction']
            if anomaly_data.get('percentile_position') is not None:
                signal['percentile'] = anomaly_data['percentile_position']
            if anomaly_data.get('pattern_name'):
                signal['pattern'] = anomaly_data['pattern_name']
            formatted['detection_signals'] = [signal]

        # Add actionable information
        if anomaly_data.get('possible_causes'):
            formatted['possible_causes'] = anomaly_data['possible_causes']
        if anomaly_data.get('recommended_actions'):
            formatted['recommended_actions'] = anomaly_data['recommended_actions']
        if anomaly_data.get('checks'):
            formatted['checks'] = anomaly_data['checks']

        # Add comparison data
        if anomaly_data.get('comparison_data'):
            formatted['comparison_data'] = anomaly_data['comparison_data']

        # Add business impact
        if anomaly_data.get('business_impact'):
            formatted['business_impact'] = anomaly_data['business_impact']

        # Add fingerprinting fields at anomaly level
        fingerprint_fields = [
            'fingerprint_id', 'fingerprint_action', 'incident_id', 'incident_action',
            'incident_duration_minutes', 'first_seen', 'last_updated', 'occurrence_count',
            'time_confidence', 'detected_by_model'
        ]
        for field in fingerprint_fields:
            if anomaly_data.get(field) is not None:
                formatted[field] = anomaly_data[field]

        return formatted

    def _extract_models_used(self, anomalies: dict) -> list:
        """Extract list of detection models used from anomalies."""
        models = set()
        for anomaly in anomalies.values():
            if isinstance(anomaly, dict):
                signals = anomaly.get('detection_signals', [])
                for signal in signals:
                    if isinstance(signal, dict) and signal.get('method'):
                        models.add(signal['method'])
        return list(models) if models else []

    def _build_anomaly_model(self, anomaly_name: str, anomaly_data: dict) -> Anomaly:
        """Build an Anomaly Pydantic model from raw anomaly data.

        Handles detection_signals array format from the sequential IF → Pattern pipeline.
        """
        # Get detection signals
        detection_signals_raw = anomaly_data.get('detection_signals', [])
        anomaly_type = anomaly_data.get('type', 'unknown')
        is_consolidated = anomaly_type == 'consolidated' or len(detection_signals_raw) > 1

        # Build detection signals from array format
        detection_signals = []
        if detection_signals_raw:
            for signal in detection_signals_raw:
                if isinstance(signal, dict):
                    try:
                        detection_signals.append(DetectionSignal(
                            method=signal.get('method', 'unknown'),
                            type=signal.get('type', 'ml_isolation'),
                            severity=Severity(signal.get('severity', 'medium')),
                            score=float(signal.get('score', 0.0)),
                            direction=signal.get('direction'),
                            percentile=signal.get('percentile'),
                            pattern=signal.get('pattern'),
                        ))
                    except Exception:
                        pass

        # Fallback: create a single detection signal from the anomaly data
        if not detection_signals:
            detection_signals.append(DetectionSignal(
                method=anomaly_data.get('detection_method', 'isolation_forest'),
                type=anomaly_data.get('type', 'ml_isolation'),
                severity=Severity(anomaly_data.get('severity', 'medium')),
                score=float(anomaly_data.get('score', 0.0)),
                direction=anomaly_data.get('direction'),
                percentile=anomaly_data.get('percentile_position') or anomaly_data.get('percentile'),
                pattern=anomaly_data.get('pattern_name'),
            ))

        # Determine root metric from various sources
        root_metric = anomaly_data.get('root_metric')
        if not root_metric:
            # Try to get from contributing_metrics
            contributing = anomaly_data.get('contributing_metrics', [])
            if contributing:
                # Prioritize latency > error_rate > request_rate
                priority = ['application_latency', 'error_rate', 'request_rate',
                           'database_latency', 'client_latency']
                for metric in priority:
                    if metric in contributing:
                        root_metric = metric
                        break
                if not root_metric:
                    root_metric = contributing[0]

            # Fallback: infer from anomaly name
            elif 'latency' in anomaly_name.lower():
                root_metric = 'application_latency'
            elif 'error' in anomaly_name.lower():
                root_metric = 'error_rate'
            elif 'traffic' in anomaly_name.lower() or 'request' in anomaly_name.lower():
                root_metric = 'request_rate'

        # Use type directly (already API-compatible from detector)
        output_type = anomaly_data.get('type', 'ml_isolation')

        # Build cascade info if present
        cascade_info = None
        if anomaly_data.get('cascade_analysis'):
            ca = anomaly_data['cascade_analysis']
            cascade_info = CascadeInfo(
                is_cascade=ca.get('is_cascade', False),
                root_cause_service=ca.get('root_cause_service'),
                affected_chain=ca.get('affected_chain', []),
                cascade_type=ca.get('cascade_type', 'none'),
                confidence=ca.get('confidence', 0.0),
                propagation_path=ca.get('propagation_path'),
            )

        # Build the Anomaly model
        return Anomaly(
            type=output_type,
            severity=Severity(anomaly_data.get('severity', 'medium')),
            confidence=float(anomaly_data.get('confidence', anomaly_data.get('score', 0.5))),
            score=float(anomaly_data.get('score', anomaly_data.get('confidence', 0.0))),
            description=anomaly_data.get('description', anomaly_name.replace('_', ' ').title()),
            root_metric=root_metric,
            signal_count=anomaly_data.get('signal_count'),
            pattern_name=anomaly_data.get('pattern_name'),
            interpretation=anomaly_data.get('interpretation'),
            value=anomaly_data.get('value') or anomaly_data.get('actual_value'),
            detection_signals=detection_signals,
            possible_causes=anomaly_data.get('possible_causes'),
            recommended_actions=anomaly_data.get('recommended_actions'),
            checks=anomaly_data.get('checks'),
            comparison_data=anomaly_data.get('comparison_data'),
            business_impact=anomaly_data.get('business_impact'),
            cascade_analysis=cascade_info,
            fingerprint_id=anomaly_data.get('fingerprint_id'),
            fingerprint_action=anomaly_data.get('fingerprint_action'),
            incident_id=anomaly_data.get('incident_id'),
            incident_action=anomaly_data.get('incident_action'),
            incident_duration_minutes=anomaly_data.get('incident_duration_minutes'),
            first_seen=anomaly_data.get('first_seen'),
            last_updated=anomaly_data.get('last_updated'),
            occurrence_count=anomaly_data.get('occurrence_count'),
            time_confidence=anomaly_data.get('time_confidence'),
            detected_by_model=anomaly_data.get('detected_by_model'),
        )

    def _format_alert_json(self, result: ServiceInferenceResult) -> dict:
        """Format regular ServiceInferenceResult as structured JSON"""
        formatted_anomalies = []
        for anomaly in result.anomalies:
            anomaly_dict = {
                "type": anomaly.anomaly_type,
                "severity": anomaly.severity.value,
                "confidence_score": anomaly.confidence_score,
                "description": anomaly.description,
                "threshold_value": anomaly.threshold_value,
                "actual_value": anomaly.actual_value,
                "metadata": anomaly.metadata or {}
            }
            
            # Add explainability fields if available
            if anomaly.comparison_data:
                anomaly_dict["comparison_data"] = anomaly.comparison_data
            if anomaly.business_impact:
                anomaly_dict["business_impact"] = anomaly.business_impact
            if anomaly.percentile_position:
                anomaly_dict["percentile_position"] = anomaly.percentile_position
            
            formatted_anomalies.append(anomaly_dict)
        
        alert_data = {
            "alert_type": "anomaly_detected",
            "service": result.service_name,
            "timestamp": result.timestamp.isoformat(),
            "overall_severity": result.max_severity.value,
            "anomaly_count": len(result.anomalies),
            "model_version": result.model_version,
            "inference_time_ms": result.inference_time_ms,
            "current_metrics": result.input_metrics.to_dict(),
            "anomalies": formatted_anomalies,
            "model_type": "standard_ml"
        }
        
        # Add explainability context if available
        if result.historical_context:
            alert_data["historical_context"] = result.historical_context
        if result.metric_analysis:
            alert_data["metric_analysis"] = result.metric_analysis
        if result.explanation:
            alert_data["explanation"] = result.explanation
        if result.recommended_actions:
            alert_data["recommended_actions"] = result.recommended_actions
        
        return alert_data
    
    def output_anomalies(self):
        """Output all detected anomalies based on verbose mode"""
        if self.verbose:
            # Verbose mode: Show header and formatted output
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
            # Non-verbose mode: Just output JSON array
            print(json.dumps(self.detected_anomalies, indent=2))
    
    def clear_anomalies(self):
        """Clear stored anomalies"""
        self.detected_anomalies = []
    
    def _save_explainable_alert(self, result: Dict[str, Any]):
        """Save explainable alert to file"""
        alert_data = self._format_explainable_alert_json(result)
        alert_data['alert_id'] = f"{result['service']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save to daily file
        alert_file = self.alerts_directory / f"alerts_{datetime.now().strftime('%Y-%m-%d')}.jsonl"
        
        with open(alert_file, 'a') as f:
            f.write(json.dumps(alert_data, default=str) + '\n')
    
    def _save_time_aware_alert(self, result: dict):
        """Save time-aware alert to file"""
        alert_data = self._format_time_aware_alert_json(result)
        alert_data['alert_id'] = f"{result['service']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save to daily file
        alert_file = self.alerts_directory / f"alerts_{datetime.now().strftime('%Y-%m-%d')}.jsonl"
        
        with open(alert_file, 'a') as f:
            f.write(json.dumps(alert_data, default=str) + '\n')
    
    def _save_alert(self, result: ServiceInferenceResult):
        """Save regular alert to file"""
        alert_data = self._format_alert_json(result)
        alert_data['alert_id'] = f"{result.service_name}_{result.timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        # Save to daily file
        alert_file = self.alerts_directory / f"alerts_{datetime.now().strftime('%Y-%m-%d')}.jsonl"
        
        with open(alert_file, 'a') as f:
            f.write(json.dumps(alert_data, default=str) + '\n')


class EnhancedTimeAwareDetector:
    """Enhanced time-aware detector with efficient lazy loading - FIXED for performance"""

    def __init__(self, models_directory: str):
        self.models_directory = models_directory
        self._detector_cache = {}
        self._load_times = {}

    def _calculate_drift_penalty(self, drift_score: float) -> float:
        """Calculate confidence penalty based on drift severity.

        Args:
            drift_score: The overall drift score (z-score based)

        Returns:
            Confidence penalty factor (0.0 to 0.3)
        """
        if drift_score > 5:
            return 0.3  # 30% confidence reduction for severe drift
        elif drift_score > 3:
            return 0.15  # 15% reduction for moderate drift
        return 0.0

    def _apply_drift_adjustments(
        self,
        result: Dict[str, Any],
        drift_analysis: Dict[str, Any],
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Apply drift-based adjustments to detection results.

        When drift is detected, this method:
        1. Adds a warning to the result
        2. Adjusts confidence scores downward
        3. Marks affected anomalies with drift_warning flag

        Args:
            result: The detection result dict
            drift_analysis: Drift analysis from the detector
            verbose: Whether to log adjustments

        Returns:
            Modified result dict with drift adjustments
        """
        if not drift_analysis.get('has_drift', False):
            return result

        drift_score = drift_analysis.get('overall_drift_score', 0.0)
        penalty = self._calculate_drift_penalty(drift_score)

        if penalty == 0:
            return result

        # Add drift warning to result
        result['drift_warning'] = {
            'type': 'model_drift',
            'overall_drift_score': drift_score,
            'recommendation': drift_analysis.get('recommendation', 'Monitor closely'),
            'affected_metrics': list(drift_analysis.get('drift_metrics', {}).keys()),
            'confidence_penalty_applied': penalty,
            'multivariate_drift': drift_analysis.get('multivariate_drift', False),
        }

        if verbose:
            logger.warning(
                f"Drift detected (score={drift_score:.2f}), applying {penalty*100:.0f}% confidence penalty"
            )

        # Adjust confidence scores in anomalies
        anomalies = result.get('anomalies', {})
        for anomaly_name, anomaly_data in anomalies.items():
            if isinstance(anomaly_data, dict) and 'confidence' in anomaly_data:
                original_confidence = anomaly_data['confidence']
                anomaly_data['original_confidence'] = original_confidence
                anomaly_data['confidence'] = original_confidence * (1 - penalty)
                anomaly_data['drift_warning'] = True

                if verbose:
                    logger.info(
                        f"  {anomaly_name}: confidence {original_confidence:.2f} → {anomaly_data['confidence']:.2f}"
                    )

        return result
    
    def load_time_aware_detector(self, service_name: str, verbose: bool = False) -> 'TimeAwareAnomalyDetector':
        """Load time-aware detector with lazy loading discovery - no models loaded yet"""
        cache_key = service_name
        
        # Check if we need to reload the detector (not the models)
        service_model_dirs = []
        models_path = Path(self.models_directory)
        
        # Look for any of the 5 period models for this service
        period_suffixes = ['_business_hours', '_night_hours', '_evening_hours', '_weekend_day', '_weekend_night']
        
        for suffix in period_suffixes:
            period_service_dir = models_path / f"{service_name}{suffix}"
            if period_service_dir.exists():
                service_model_dirs.append(period_service_dir)
        
        if not service_model_dirs:
            if verbose:
                logger.info(f"No time-aware models found for {service_name}")
            return None
        
        # Get the most recent modification time across all period models
        most_recent_mod_time = max(dir_path.stat().st_mtime for dir_path in service_model_dirs)
        cached_mod_time = self._load_times.get(cache_key, 0)
        
        if cache_key not in self._detector_cache or most_recent_mod_time > cached_mod_time:
            if verbose:
                logger.info(f"Initializing lazy-loading detector for {service_name}")
                logger.info(f"Found {len(service_model_dirs)} period models available")
            
            try:
                # Import the efficient TimeAwareAnomalyDetector class
                from smartbox_anomaly.detection.time_aware import TimeAwareAnomalyDetector

                # Create detector with lazy loading (NO models loaded yet)
                detector = TimeAwareAnomalyDetector(service_name)
                detector._models_directory = self.models_directory
                detector._available_periods = detector._discover_available_periods(self.models_directory)
                
                if detector._available_periods:
                    self._detector_cache[cache_key] = detector
                    self._load_times[cache_key] = most_recent_mod_time
                    
                    if verbose:
                        logger.info(f"Detector ready for {service_name}")
                        logger.info(f"Available periods: {sorted(list(detector._available_periods))}")
                        logger.info("Models will load on-demand")
                else:
                    if verbose:
                        logger.warning(f"No valid periods found for {service_name}")
                    return None
                    
            except Exception as e:
                if verbose:
                    logger.error(f"Error initializing detector for {service_name}: {e}")
                return None
        
        return self._detector_cache.get(cache_key)
    
    def detect_with_explainability(
        self,
        service_name: str,
        metrics: Dict[str, Any],
        timestamp: datetime,
        verbose: bool = False,
        dependency_context: Optional[DependencyContext] = None,
        check_drift: bool = False,
    ) -> Dict[str, Any]:
        """Enhanced time-aware detection with explainability - EFFICIENT with lazy loading"""

        try:
            detector = self.load_time_aware_detector(service_name, verbose)

            if not detector:
                raise ModelLoadError(f"Could not initialize time-aware detector for {service_name}")
            
            # Determine current period first
            current_period = detector.get_time_period(timestamp)
            
            if verbose:
                logger.info(f"Current period for {service_name}: {current_period}")
            
            # Check if this period is available
            if current_period not in detector._available_periods:
                available_periods = list(detector._available_periods)
                if verbose:
                    logger.warning(f"No model for {current_period}, available: {available_periods}")
                
                # Try fallback to similar period
                fallback_period = detector._find_fallback_period(current_period)
                if fallback_period:
                    if verbose:
                        logger.info(f"Will use fallback period: {fallback_period}")
                    current_period = fallback_period
                else:
                    raise ModelLoadError(f"No model available for time period: {current_period} (available: {available_periods})")
            
            # Use lazy loading detection - only loads the specific period model needed
            if hasattr(detector, 'detect_anomalies_with_context'):
                if verbose:
                    logger.info(f"Using explainable detection with lazy loading for {current_period}")

                try:
                    # This will lazy load only the current period model
                    enhanced_result = detector.detect_anomalies_with_context(
                        metrics, timestamp, self.models_directory, verbose,
                        dependency_context=dependency_context,
                        check_drift=check_drift
                    )

                    # Apply drift adjustments if drift was detected
                    if check_drift and 'drift_analysis' in enhanced_result:
                        enhanced_result = self._apply_drift_adjustments(
                            enhanced_result,
                            enhanced_result['drift_analysis'],
                            verbose
                        )

                    # Add efficiency info
                    enhanced_result['performance_info'] = {
                        'lazy_loaded': True,
                        'models_loaded': list(detector.models.keys()),
                        'period_used': current_period,
                        'total_available': len(detector._available_periods),
                        'drift_check_enabled': check_drift,
                        'drift_penalty_applied': enhanced_result.get('drift_warning', {}).get('confidence_penalty_applied', 0.0)
                    }

                    return enhanced_result
                    
                except Exception as e:
                    if verbose:
                        logger.warning(f"Explainable detection failed: {e}, falling back to standard")
                    # Fall through to standard detection
            
            # Fallback to standard detection with lazy loading
            if verbose:
                logger.info(f"Using standard detection with lazy loading for {current_period}")

            # This will also lazy load only the needed model
            anomalies = detector.detect_anomalies(
                metrics, timestamp, self.models_directory, verbose,
                check_drift=check_drift
            )

            # Handle both dict and other formats properly
            if isinstance(anomalies, dict):
                anomaly_count = len(anomalies)
            else:
                anomaly_count = len(anomalies) if hasattr(anomalies, '__len__') else 0
                anomalies = anomalies if isinstance(anomalies, dict) else {}

            return {
                'service': service_name,
                'time_period': current_period,
                'model_type': 'time_aware_5period_standard_lazy',
                'anomaly_count': anomaly_count,
                'anomalies': anomalies,
                'metrics': metrics,
                'timestamp': timestamp.isoformat(),
                'explainable': False,
                'performance_info': {
                    'lazy_loaded': True,
                    'models_loaded': list(detector.models.keys()),
                    'period_used': current_period,
                    'total_available': len(detector._available_periods),
                    'drift_check_enabled': check_drift
                }
            }
            
        except Exception as e:
            if verbose:
                logger.error(f"All detection methods failed for {service_name}: {e}")
            
            return {
                'service': service_name,
                'error': str(e),
                'timestamp': timestamp.isoformat(),
                'model_type': 'time_aware_failed',
                'explainable': False
            }
    
    def get_efficiency_stats(self) -> Dict:
        """Get statistics about lazy loading efficiency"""
        total_detectors = len(self._detector_cache)
        total_models_loaded = 0
        total_models_available = 0
        
        for detector in self._detector_cache.values():
            if hasattr(detector, 'models') and hasattr(detector, '_available_periods'):
                total_models_loaded += len(detector.models)
                total_models_available += len(detector._available_periods)
        
        efficiency = (1 - (total_models_loaded / max(1, total_models_available))) * 100
        
        return {
            'total_services': total_detectors,
            'total_models_available': total_models_available,
            'total_models_loaded': total_models_loaded,
            'memory_efficiency': f"{efficiency:.1f}%",
            'models_saved_from_loading': total_models_available - total_models_loaded,
            'lazy_loading_enabled': True
        }


class SmartboxMLInferencePipeline:
    """Enhanced production-grade ML inference pipeline with explainability"""

    def __init__(self,
                 vm_endpoint: str | None = None,
                 models_directory: str | None = None,
                 alerts_directory: str | None = None,
                 max_workers: int | None = None,
                 verbose: bool = False,
                 check_drift: bool | None = None):

        # Load configuration
        config = get_config()

        # Use config values as defaults, allow overrides
        vm_endpoint = vm_endpoint or config.victoria_metrics.endpoint
        models_directory = models_directory or config.model.models_directory
        alerts_directory = alerts_directory or config.inference.alerts_directory
        max_workers = max_workers if max_workers is not None else config.inference.max_workers
        self.check_drift = check_drift if check_drift is not None else config.inference.check_drift

        self.vm_client = VictoriaMetricsClient(vm_endpoint)
        self.model_manager = EnhancedModelManager(models_directory)
        self.detection_engine = EnhancedAnomalyDetectionEngine(self.model_manager)
        self.results_processor = EnhancedResultsProcessor(alerts_directory, verbose)
        self.time_aware_detector = EnhancedTimeAwareDetector(models_directory)
        self.max_workers = max_workers
        self.verbose = verbose

        # Load dependency graph from config
        self.dependency_graph = self._load_dependency_graph()

        # Initialize SLO evaluator if configured
        self.slo_evaluator: SLOEvaluator | None = None
        if config.slo.enabled:
            self.slo_evaluator = SLOEvaluator(config.slo)
            logger.info("SLO-aware severity evaluation enabled")

        # Initialize exception enrichment service
        # Uses same VM client to query exception metrics from OpenTelemetry
        enrichment_enabled = config.inference.exception_enrichment_enabled if hasattr(config.inference, 'exception_enrichment_enabled') else True
        self.exception_enrichment = ExceptionEnrichmentService(
            vm_client=self.vm_client,
            lookback_minutes=5,  # Match detection window
            enabled=enrichment_enabled,
        )
        if enrichment_enabled:
            logger.info("Exception enrichment enabled for error-related anomalies")

        # Initialize service graph enrichment service
        # Queries downstream service calls when client_latency is elevated
        self.service_graph_enrichment = ServiceGraphEnrichmentService(
            vm_client=self.vm_client,
            lookback_minutes=5,  # Match detection window
            enabled=enrichment_enabled,
        )
        if enrichment_enabled:
            logger.info("Service graph enrichment enabled for client latency anomalies")

        logger.info("Enhanced Yaga2 ML Inference Pipeline initialized with explainability")
        if verbose:
            logger.info(f"  VM Endpoint: {vm_endpoint}")
            logger.info(f"  Models Directory: {models_directory}")
            logger.info(f"  Observability API: {config.observability.base_url}")
            if self.dependency_graph:
                logger.info(f"  Dependency graph loaded: {len(self.dependency_graph)} services")
            if self.slo_evaluator:
                logger.info(f"  SLO evaluation: enabled")

    def _load_dependency_graph(self) -> Dict[str, List[str]]:
        """Load dependency graph from config.json."""
        try:
            config_path = Path("config.json")
            if config_path.exists():
                with open(config_path) as f:
                    config_data = json.load(f)
                deps = config_data.get("dependencies", {})
                return deps.get("graph", {})
        except Exception as e:
            logger.warning(f"Failed to load dependency graph: {e}")
        return {}

    def _validate_metrics(
        self,
        metrics_dict: Dict[str, float],
        service_name: str,
    ) -> Tuple[Dict[str, float], List[str]]:
        """Validate and sanitize metrics before inference.

        Checks for:
        - NaN, inf, None values
        - Negative rates and latencies
        - Extreme outliers beyond reasonable bounds

        Args:
            metrics_dict: Raw metrics from VictoriaMetrics
            service_name: Name of the service (for logging)

        Returns:
            Tuple of (cleaned_metrics, validation_warnings)
        """
        warnings = []
        cleaned = {}

        # Validation bounds from config
        max_request_rate = 1_000_000.0  # 1M req/s
        max_latency_ms = 300_000.0  # 5 minutes
        max_error_rate = 1.0  # 100%

        for metric_name, value in metrics_dict.items():
            # Skip non-numeric values
            if not isinstance(value, (int, float)):
                warnings.append(f"{metric_name}: non-numeric value {type(value).__name__}, skipping")
                continue

            # Check for invalid values (NaN, inf, None)
            if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
                warnings.append(f"{metric_name}: invalid value {value}, using 0.0")
                cleaned[metric_name] = 0.0
                continue

            # Semantic validation based on metric type
            metric_lower = metric_name.lower()

            # Rate metrics should not be negative
            if 'rate' in metric_lower and value < 0:
                warnings.append(f"{metric_name}: negative rate {value}, using 0.0")
                cleaned[metric_name] = 0.0
                continue

            # Latency metrics should not be negative
            if 'latency' in metric_lower and value < 0:
                warnings.append(f"{metric_name}: negative latency {value}, using 0.0")
                cleaned[metric_name] = 0.0
                continue

            # Cap extreme request rates
            if 'request' in metric_lower and 'rate' in metric_lower and value > max_request_rate:
                warnings.append(f"{metric_name}: extreme rate {value}, capping at {max_request_rate}")
                cleaned[metric_name] = max_request_rate
                continue

            # Cap extreme latencies
            if 'latency' in metric_lower and value > max_latency_ms:
                warnings.append(f"{metric_name}: extreme latency {value}ms, capping at {max_latency_ms}ms")
                cleaned[metric_name] = max_latency_ms
                continue

            # Error rate should be between 0 and 1
            if 'error' in metric_lower and 'rate' in metric_lower:
                if value < 0:
                    warnings.append(f"{metric_name}: negative error rate {value}, using 0.0")
                    cleaned[metric_name] = 0.0
                    continue
                elif value > max_error_rate:
                    warnings.append(f"{metric_name}: error rate {value} > 1.0, capping at 1.0")
                    cleaned[metric_name] = max_error_rate
                    continue

            # Value is valid
            cleaned[metric_name] = value

        # Log warnings if any
        if warnings and self.verbose:
            logger.warning(f"Metrics validation for {service_name}: {len(warnings)} issues")
            for warning in warnings[:5]:  # Show first 5 warnings
                logger.warning(f"  {warning}")
            if len(warnings) > 5:
                logger.warning(f"  ... and {len(warnings) - 5} more")

        return cleaned, warnings

    def _has_latency_anomaly(self, result: Dict) -> bool:
        """Check if a result contains latency-related anomalies."""
        if "error" in result:
            return False
        anomalies = result.get("anomalies", {})
        for anomaly_name, anomaly_data in anomalies.items():
            # Check if it's a latency-related anomaly
            root_metric = anomaly_data.get("root_metric", "")
            if "latency" in root_metric.lower() or "latency" in anomaly_name.lower():
                return True
            # Check contributing metrics
            for metric in anomaly_data.get("contributing_metrics", []):
                if "latency" in metric.lower():
                    return True
        return False

    def _build_dependency_context(
        self,
        service_name: str,
        all_results: Dict[str, Dict],
    ) -> Optional[DependencyContext]:
        """Build dependency context from Pass 1 results.

        Args:
            service_name: The service to build context for
            all_results: Results from Pass 1 for all services

        Returns:
            DependencyContext if dependencies exist, None otherwise
        """
        if not self.dependency_graph:
            return None

        service_deps = self.dependency_graph.get(service_name, [])
        if not service_deps:
            return None

        dependencies = {}
        for dep_service in service_deps:
            if dep_service in all_results:
                result = all_results[dep_service]
                if "error" not in result:
                    anomalies = result.get("anomalies", {})
                    has_anomaly = len(anomalies) > 0

                    # Get primary anomaly info
                    primary_anomaly_type = None
                    severity = None
                    latency_percentile = None

                    if anomalies:
                        first_anomaly = list(anomalies.values())[0]
                        primary_anomaly_type = first_anomaly.get("pattern_name", list(anomalies.keys())[0])
                        severity = first_anomaly.get("severity")

                        # Try to extract latency percentile
                        for signal in first_anomaly.get("detection_signals", []):
                            if "latency" in signal.get("method", "").lower():
                                latency_percentile = signal.get("percentile")
                                break

                        # Fallback: check comparison_data
                        if latency_percentile is None:
                            comp_data = first_anomaly.get("comparison_data", {})
                            lat_data = comp_data.get("application_latency", {})
                            latency_percentile = lat_data.get("percentile_estimate")

                    dependencies[dep_service] = DependencyStatus(
                        service_name=dep_service,
                        has_anomaly=has_anomaly,
                        anomaly_type=primary_anomaly_type,
                        severity=severity,
                        latency_percentile=latency_percentile,
                        timestamp=result.get("timestamp"),
                    )

        if not dependencies:
            return None

        return DependencyContext(
            dependencies=dependencies,
            graph=self.dependency_graph,
            detection_timestamp=datetime.now().isoformat(),
        )

    def run_enhanced_time_aware_inference(self, service_names: Optional[List[str]] = None) -> Dict[str, Dict]:
        """Run inference with enhanced time-aware anomaly detection and explainability.

        Uses two-pass detection for dependency-aware cascade analysis:
        - Pass 1: Detect anomalies for all services (no dependency context)
        - Pass 2: Re-run detection for services with latency anomalies, using
                  dependency context built from Pass 1 results

        If service_names is None, loads services from config.json and filters
        to those with available models.
        """
        start_time = datetime.now()

        # Get services to run inference on
        if service_names is None:
            # Try to load from config.json first
            config_services = self.model_manager.load_services_from_config()

            if config_services:
                # Check which config services have models
                service_names, missing_services = self.model_manager.get_services_with_models(config_services)

                if missing_services and self.verbose:
                    logger.warning(
                        f"Services in config.json missing trained models: {missing_services}"
                    )
                    logger.info("Run main.py to train models for these services")

                if service_names:
                    logger.info(f"Loaded {len(service_names)} services from config.json (with models)")
            else:
                # Fall back to discovering from models directory
                service_names = self.model_manager.get_base_services()
                if service_names:
                    logger.info(f"Discovered {len(service_names)} services from models directory")

        if not service_names:
            logger.error("No services available for inference")
            return {}

        logger.info(f"Running enhanced time-aware inference for {len(service_names)} services: {service_names}")

        # Collect metrics for all services first
        metrics_cache: Dict[str, Any] = {}
        for service_name in service_names:
            try:
                if self.verbose:
                    logger.info(f"Collecting metrics from VictoriaMetrics for {service_name}")
                metrics_cache[service_name] = self.vm_client.collect_service_metrics(service_name)
            except Exception as e:
                logger.error(f"Failed to collect metrics for {service_name}: {e}")

        # ===== PASS 1: Detect anomalies without dependency context =====
        if self.verbose:
            logger.info("Pass 1: Running initial detection for all services")

        pass1_results: Dict[str, Dict] = {}
        validation_warnings_by_service: Dict[str, List[str]] = {}

        for service_name in service_names:
            if service_name not in metrics_cache:
                pass1_results[service_name] = {'service': service_name, 'error': 'No metrics collected'}
                continue

            # Use fresh timestamp per service to avoid period mismatch
            service_timestamp = datetime.now()

            if self.verbose:
                logger.info(f"Pass 1: Analyzing {service_name}")

            try:
                metrics = metrics_cache[service_name]

                # Check if metrics are reliable enough for detection
                # Skip detection if critical metrics (request_rate) failed to avoid false alerts
                if not metrics.is_reliable_for_detection():
                    failure_summary = metrics.get_failure_summary()
                    logger.warning(
                        f"Skipping detection for {service_name}: metrics unreliable - {failure_summary}"
                    )
                    pass1_results[service_name] = {
                        'service': service_name,
                        'alert_type': 'metrics_unavailable',
                        'error': f'Metrics collection failed: {failure_summary}',
                        'failed_metrics': metrics.failed_metrics,
                        'collection_errors': metrics.collection_errors,
                        'timestamp': service_timestamp.isoformat(),
                        'anomalies': {},
                        'anomaly_count': 0,
                        'overall_severity': 'none',
                        'skipped_reason': 'critical_metrics_unavailable',
                    }
                    continue

                metrics_dict = metrics.to_dict()

                # Validate metrics at inference boundary
                validated_metrics, validation_warnings = self._validate_metrics(metrics_dict, service_name)
                if validation_warnings:
                    validation_warnings_by_service[service_name] = validation_warnings

                enhanced_result = self.time_aware_detector.detect_with_explainability(
                    service_name, validated_metrics, service_timestamp, self.verbose,
                    check_drift=self.check_drift
                )

                # Add validation info to result
                if validation_warnings:
                    enhanced_result['validation_warnings'] = validation_warnings

                # Add partial failure info if some non-critical metrics failed
                if metrics.has_any_failures():
                    enhanced_result['partial_metrics_failure'] = {
                        'failed_metrics': metrics.failed_metrics,
                        'failure_summary': metrics.get_failure_summary(),
                    }

                pass1_results[service_name] = enhanced_result

            except Exception as model_error:
                # Fallback to standard time-aware detection
                if self.verbose:
                    logger.warning(f"Enhanced detection failed for {service_name}: {str(model_error)[:50]}...")

                try:
                    metrics = metrics_cache[service_name]
                    metrics_dict = metrics.to_dict()

                    # Validate metrics for fallback path too
                    validated_metrics, validation_warnings = self._validate_metrics(metrics_dict, service_name)

                    detector = TimeAwareAnomalyDetector.load_models(
                        str(self.model_manager.models_directory), service_name, False
                    )
                    anomalies = detector.detect_anomalies(validated_metrics, service_timestamp)
                    period = detector.get_time_period(service_timestamp)

                    if isinstance(anomalies, dict):
                        anomaly_count = len(anomalies)
                    else:
                        anomaly_count = len(anomalies) if hasattr(anomalies, '__len__') else 0
                        anomalies = {}

                    fallback_result = {
                        'service': service_name,
                        'time_period': period,
                        'model_type': 'time_aware_fallback',
                        'anomaly_count': anomaly_count,
                        'anomalies': anomalies,
                        'metrics': validated_metrics,
                        'timestamp': service_timestamp.isoformat(),
                        'explainable': False
                    }

                    # Add validation warnings to fallback result
                    if validation_warnings:
                        fallback_result['validation_warnings'] = validation_warnings

                    pass1_results[service_name] = fallback_result
                except Exception as fallback_error:
                    logger.error(f"Both enhanced and standard models failed for {service_name}: {fallback_error}")
                    pass1_results[service_name] = {'service': service_name, 'error': str(fallback_error)}

            time.sleep(0.1)  # Small delay between services

        # ===== PASS 2: Re-run detection with dependency context for latency anomalies =====
        results = dict(pass1_results)  # Start with Pass 1 results

        if self.dependency_graph:
            # Find services that have latency anomalies and have dependencies
            services_for_pass2 = [
                svc for svc in service_names
                if svc in pass1_results
                and 'error' not in pass1_results[svc]
                and self._has_latency_anomaly(pass1_results[svc])
                and svc in self.dependency_graph
            ]

            if services_for_pass2:
                if self.verbose:
                    logger.info(f"Pass 2: Re-analyzing {len(services_for_pass2)} services with dependency context")

                for service_name in services_for_pass2:
                    # Use fresh timestamp for Pass 2 as well
                    pass2_timestamp = datetime.now()

                    if self.verbose:
                        logger.info(f"Pass 2: Re-analyzing {service_name} with dependency context")

                    try:
                        # Build dependency context from Pass 1 results
                        dep_context = self._build_dependency_context(service_name, pass1_results)

                        if dep_context and any(
                            status.has_anomaly for status in dep_context.dependencies.values()
                        ):
                            # Re-run detection with dependency context
                            metrics = metrics_cache[service_name]
                            metrics_dict = metrics.to_dict()

                            # Re-validate metrics (use cached warnings if available)
                            validated_metrics, _ = self._validate_metrics(metrics_dict, service_name)

                            enhanced_result = self.time_aware_detector.detect_with_explainability(
                                service_name, validated_metrics, pass2_timestamp, self.verbose,
                                dependency_context=dep_context,
                                check_drift=self.check_drift
                            )

                            # Check if cascade pattern was detected
                            anomalies = enhanced_result.get('anomalies', {})
                            cascade_detected = any(
                                'cascade' in name.lower() or
                                anomaly.get('cascade_analysis', {}).get('is_cascade', False)
                                for name, anomaly in anomalies.items()
                            )

                            if cascade_detected:
                                if self.verbose:
                                    logger.info(f"Cascade detected for {service_name}")
                                # Preserve validation warnings from Pass 1
                                if service_name in validation_warnings_by_service:
                                    enhanced_result['validation_warnings'] = validation_warnings_by_service[service_name]
                                results[service_name] = enhanced_result

                    except Exception as e:
                        if self.verbose:
                            logger.warning(f"Pass 2 failed for {service_name}: {e}")
                        # Keep Pass 1 result on failure

        # ===== SLO Evaluation Layer =====
        # Apply SLO-aware severity adjustments if configured
        if self.slo_evaluator:
            if self.verbose:
                logger.info("Applying SLO-aware severity evaluation")

            for service_name, result in results.items():
                if not result.get('error') and result.get('alert_type') != 'metrics_unavailable':
                    try:
                        # Get timestamp from result for busy period check
                        result_timestamp = None
                        if 'timestamp' in result:
                            try:
                                result_timestamp = datetime.fromisoformat(result['timestamp'].replace('Z', '+00:00'))
                            except (ValueError, AttributeError):
                                pass

                        # Apply SLO evaluation
                        results[service_name] = self.slo_evaluator.evaluate_result(
                            result, timestamp=result_timestamp
                        )

                        # Log if severity was adjusted
                        slo_eval = results[service_name].get('slo_evaluation', {})
                        if slo_eval.get('severity_changed') and self.verbose:
                            logger.info(
                                f"SLO adjustment: {service_name} "
                                f"{slo_eval.get('original_severity')} -> {slo_eval.get('adjusted_severity')} "
                                f"(impact: {slo_eval.get('operational_impact')})"
                            )
                    except Exception as e:
                        logger.warning(f"SLO evaluation failed for {service_name}: {e}")
                        # Keep original result on failure

            # Log SLO evaluation stats
            slo_stats = self.slo_evaluator.get_stats()
            if self.verbose and slo_stats['evaluations_performed'] > 0:
                logger.info(
                    f"SLO evaluation complete: {slo_stats['severity_adjustments']} adjustments "
                    f"out of {slo_stats['evaluations_performed']} evaluations"
                )

        # ===== Exception Enrichment Layer =====
        # Enrich error-related anomalies with exception breakdown from OpenTelemetry
        # Only enrich when SLO confirms errors are actually above threshold
        if self.exception_enrichment.enabled:
            for service_name, result in results.items():
                if result.get('error') or result.get('alert_type') == 'metrics_unavailable':
                    continue

                anomalies = result.get('anomalies', {})
                if not anomalies:
                    continue

                # Check SLO evaluation - only enrich if errors are confirmed above threshold
                slo_eval = result.get('slo_evaluation', {})
                error_rate_eval = slo_eval.get('error_rate_evaluation', {})
                error_status = error_rate_eval.get('status', 'unknown')

                # Only enrich if SLO confirms error rate is NOT ok (warning, high, critical, breach)
                # Skip enrichment if errors are within acceptable SLO thresholds
                if error_status == 'ok':
                    continue

                # Check if any anomaly is error-related with high/critical severity
                should_enrich = False
                for anomaly_name, anomaly_data in anomalies.items():
                    severity = anomaly_data.get('severity', 'low')
                    if severity not in ('high', 'critical'):
                        continue

                    # Check if error-related by pattern name or root metric
                    pattern = anomaly_name.lower()
                    root_metric = anomaly_data.get('root_metric', '').lower()
                    description = anomaly_data.get('description', '').lower()

                    if any(term in pattern or term in root_metric or term in description
                           for term in ['error', 'failure', 'outage', 'fail']):
                        should_enrich = True
                        break

                if should_enrich:
                    try:
                        # Get timestamp from result
                        anomaly_timestamp = None
                        if 'timestamp' in result:
                            try:
                                ts_str = result['timestamp']
                                if isinstance(ts_str, str):
                                    anomaly_timestamp = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                            except (ValueError, AttributeError):
                                pass

                        breakdown = self.exception_enrichment.get_exception_breakdown(
                            service_name=service_name,
                            anomaly_timestamp=anomaly_timestamp,
                        )

                        if breakdown.query_successful and breakdown.has_exceptions:
                            result['exception_context'] = breakdown.to_dict()
                            if self.verbose:
                                logger.info(
                                    f"EXCEPTION_ENRICHMENT - {service_name}: "
                                    f"{len(breakdown.exceptions)} exception types, "
                                    f"top: {breakdown.top_exception.short_name if breakdown.top_exception else 'none'}"
                                )
                        elif not breakdown.query_successful:
                            if self.verbose:
                                logger.warning(f"Exception enrichment query failed for {service_name}: {breakdown.error_message}")
                    except Exception as e:
                        if self.verbose:
                            logger.warning(f"Exception enrichment failed for {service_name}: {e}")

        # ===== Service Graph Enrichment Layer =====
        # Enrich client_latency anomalies with downstream service call data
        if self.service_graph_enrichment.enabled:
            for service_name, result in results.items():
                if result.get('error') or result.get('alert_type') == 'metrics_unavailable':
                    continue

                anomalies = result.get('anomalies', {})
                if not anomalies:
                    continue

                # Check SLO evaluation - only enrich if latency is elevated
                slo_eval = result.get('slo_evaluation', {})
                latency_eval = slo_eval.get('latency_evaluation', {})
                latency_status = latency_eval.get('status', 'unknown')

                # Only enrich if SLO confirms latency is NOT ok
                if latency_status == 'ok':
                    continue

                # Check if any anomaly is latency-related with high/critical severity
                should_enrich = False
                for anomaly_name, anomaly_data in anomalies.items():
                    severity = anomaly_data.get('severity', 'low')
                    if severity not in ('high', 'critical'):
                        continue

                    # Check if latency-related by pattern name or root metric
                    pattern = anomaly_name.lower()
                    root_metric = anomaly_data.get('root_metric', '').lower()
                    description = anomaly_data.get('description', '').lower()

                    if any(term in pattern or term in root_metric or term in description
                           for term in ['latency', 'slow', 'degradation', 'client_latency']):
                        should_enrich = True
                        break

                if should_enrich:
                    try:
                        # Get timestamp from result
                        anomaly_timestamp = None
                        if 'timestamp' in result:
                            try:
                                ts_str = result['timestamp']
                                if isinstance(ts_str, str):
                                    anomaly_timestamp = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                            except (ValueError, AttributeError):
                                pass

                        breakdown = self.service_graph_enrichment.get_service_graph(
                            service_name=service_name,
                            anomaly_timestamp=anomaly_timestamp,
                        )

                        if breakdown.query_successful and breakdown.has_data:
                            result['service_graph_context'] = breakdown.to_dict()
                            if self.verbose:
                                logger.info(
                                    f"SERVICE_GRAPH_ENRICHMENT - {service_name}: "
                                    f"{len(breakdown.routes)} routes to {len(breakdown.unique_servers)} servers, "
                                    f"slowest: {breakdown.slowest_route.display_name if breakdown.slowest_route else 'none'}"
                                )
                        elif not breakdown.query_successful:
                            if self.verbose:
                                logger.warning(f"Service graph enrichment query failed for {service_name}: {breakdown.error_message}")
                    except Exception as e:
                        if self.verbose:
                            logger.warning(f"Service graph enrichment failed for {service_name}: {e}")

        # Process final results and log
        for service_name, result in results.items():
            if not result.get('error'):
                self.results_processor.process_result(result)

                anomaly_count = result.get('anomaly_count', len(result.get('anomalies', {})))
                period = result.get('time_period', 'unknown')
                model_type = result.get('model_type', 'unknown')
                is_explainable = result.get('explainable', model_type.endswith('explainable'))

                if anomaly_count > 0:
                    explainable_indicator = "explainable" if is_explainable else "standard"
                    if self.verbose:
                        logger.info(f"ANOMALIES DETECTED for {service_name}: {anomaly_count} anomalies found ({explainable_indicator})")

                        if 'explanation' in result:
                            explanation = result['explanation']
                            if 'detailed_explanations' in explanation:
                                logger.info(f"Explanation: {explanation['detailed_explanations'][0]}")

                        if 'recommended_actions' in result and result['recommended_actions']:
                            logger.info(f"Recommendation: {result['recommended_actions'][0]}")

                    logger.info(f"ANOMALY_DETECTED - {service_name} [{period}] {explainable_indicator}: {anomaly_count} anomalies detected")
                else:
                    if self.verbose:
                        logger.info(f"No anomalies detected for {service_name}")
                    logger.info(f"NO_ANOMALIES - {service_name} [{period}]: No anomalies detected")

        execution_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Enhanced time-aware inference completed in {execution_time:.2f}s")

        return results
    
    def run_time_aware_inference(self, service_names: Optional[List[str]] = None) -> Dict[str, Dict]:
        """Legacy method - now uses enhanced detection"""
        return self.run_enhanced_time_aware_inference(service_names)
    
    def run_inference(self, service_names: Optional[List[str]] = None) -> Dict[str, ServiceInferenceResult]:
        """Standard inference with enhanced detection engine"""
        start_time = datetime.now()
        
        # Get services to evaluate
        if service_names is None:
            service_names = self.model_manager.get_available_services()
        
        if not service_names:
            logger.error("No services available for inference")
            return {}
        
        logger.info(f"Running enhanced inference for {len(service_names)} services: {service_names}")
        
        results = {}
        
        # Process services sequentially
        for service_name in service_names:
            try:
                result = self._run_enhanced_service_inference(service_name)
                results[service_name] = result
                self.results_processor.process_result(result)
                
                time.sleep(0.2)
                
            except Exception as e:
                logger.error(f"Inference failed for {service_name}: {e}")
                results[service_name] = ServiceInferenceResult(
                    service_name=service_name,
                    timestamp=datetime.now(),
                    input_metrics=InferenceMetrics(service_name, datetime.now(), 0.0),
                    anomalies=[],
                    model_version="unknown",
                    inference_time_ms=0.0,
                    status="error",
                    error_message=str(e)
                )
        
        # Generate summary
        execution_time = (datetime.now() - start_time).total_seconds()
        total_anomalies = sum(len(r.anomalies) for r in results.values() if r.status == 'success')
        successful_services = sum(1 for r in results.values() if r.status == 'success')
        
        logger.info(f"Enhanced inference completed: {total_anomalies} anomalies across {successful_services}/{len(service_names)} services in {execution_time:.2f}s")
        
        return results
    
    def _run_enhanced_service_inference(self, service_name: str) -> ServiceInferenceResult:
        """Run enhanced inference for a single service"""
        start_time = datetime.now()
        
        try:
            # Collect metrics
            metrics = self.vm_client.collect_service_metrics(service_name)
            
            # Try enhanced detection
            try:
                enhanced_result = self.detection_engine.detect_anomalies_with_context(
                    metrics, use_explainable=True
                )
                
                # Convert to ServiceInferenceResult format
                if enhanced_result.get('explainable', False):
                    # Process explainable result
                    anomalies = self._convert_explainable_anomalies(enhanced_result.get('anomalies', []))

                    # Calculate inference time
                    inference_time_ms = (datetime.now() - start_time).total_seconds() * 1000

                    # Get model metadata
                    model_metadata = self.model_manager.get_model_metadata(service_name)
                    model_version = model_metadata.get('model_version', 'enhanced_unknown')

                    # Enrich with exception context if error-related anomalies detected
                    exception_context = None
                    if anomalies and self.exception_enrichment.enabled:
                        exception_context = self._enrich_with_exceptions(
                            service_name=service_name,
                            anomalies=anomalies,
                            metrics=metrics,
                            anomaly_timestamp=start_time,
                        )

                    result = ServiceInferenceResult(
                        service_name=service_name,
                        timestamp=start_time,
                        input_metrics=metrics,
                        anomalies=anomalies,
                        model_version=model_version,
                        inference_time_ms=inference_time_ms,
                        status='success',
                        # Enhanced fields
                        historical_context=enhanced_result.get('historical_context'),
                        metric_analysis=enhanced_result.get('metric_analysis'),
                        explanation=enhanced_result.get('explanation'),
                        recommended_actions=enhanced_result.get('recommended_actions'),
                        exception_context=exception_context,
                    )

                    if anomalies:
                        logger.info(f"ENHANCED_ANOMALY_DETECTED - {service_name}: {len(anomalies)} anomalies with context")
                    else:
                        logger.info(f"NO_ANOMALIES - {service_name}: No anomalies detected (enhanced)")

                    return result
            
            except Exception as enhanced_error:
                logger.warning(f"Enhanced detection failed for {service_name}, falling back: {enhanced_error}")
            
            # Fallback to standard detection
            anomalies = self.detection_engine.detect_anomalies(metrics)
            
            # Get model metadata
            model_metadata = self.model_manager.get_model_metadata(service_name)
            model_version = model_metadata.get('model_version', 'unknown')
            
            # Calculate inference time
            inference_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            result = ServiceInferenceResult(
                service_name=service_name,
                timestamp=start_time,
                input_metrics=metrics,
                anomalies=anomalies,
                model_version=model_version,
                inference_time_ms=inference_time_ms,
                status='success'
            )
            
            if anomalies:
                logger.info(f"ANOMALY_DETECTED - {service_name}: {len(anomalies)} anomalies detected")
            else:
                logger.info(f"NO_ANOMALIES - {service_name}: No anomalies detected")
            
            return result
            
        except ModelLoadError as e:
            logger.warning(f"MODEL_UNAVAILABLE - {service_name}: {e}")
            return ServiceInferenceResult(
                service_name=service_name,
                timestamp=start_time,
                input_metrics=InferenceMetrics(service_name, datetime.now(), 0.0),
                anomalies=[],
                model_version="unknown",
                inference_time_ms=0.0,
                status='no_model',
                error_message=str(e)
            )
        except MetricsCollectionError as e:
            logger.error(f"METRICS_FAILED - {service_name}: {e}")
            return ServiceInferenceResult(
                service_name=service_name,
                timestamp=start_time,
                input_metrics=InferenceMetrics(service_name, datetime.now(), 0.0),
                anomalies=[],
                model_version="unknown",
                inference_time_ms=0.0,
                status='error',
                error_message=str(e)
            )
        except Exception as e:
            logger.error(f"INFERENCE_ERROR - {service_name}: {e}")
            return ServiceInferenceResult(
                service_name=service_name,
                timestamp=start_time,
                input_metrics=InferenceMetrics(service_name, datetime.now(), 0.0),
                anomalies=[],
                model_version="unknown",
                inference_time_ms=0.0,
                status='error',
                error_message=str(e)
            )
    
    def _convert_explainable_anomalies(self, explainable_anomalies: List[Dict]) -> List[AnomalyResult]:
        """Convert explainable anomalies to AnomalyResult objects"""
        results = []
        
        for anomaly_data in explainable_anomalies:
            if isinstance(anomaly_data, dict):
                result = AnomalyResult(
                    anomaly_type=anomaly_data.get('type', 'unknown'),
                    severity=self.detection_engine._map_severity(anomaly_data.get('severity', 'medium')),
                    confidence_score=anomaly_data.get('score', 0.5),
                    description=anomaly_data.get('description', 'Anomaly detected'),
                    threshold_value=anomaly_data.get('threshold_value'),
                    actual_value=anomaly_data.get('actual_value'),
                    metadata=anomaly_data.get('metadata', {}),
                    # Explainability fields
                    comparison_data=anomaly_data.get('comparison_data'),
                    business_impact=anomaly_data.get('business_impact'),
                    percentile_position=anomaly_data.get('percentile_position')
                )
                results.append(result)

        return results

    def _enrich_with_exceptions(
        self,
        service_name: str,
        anomalies: List[AnomalyResult],
        metrics: InferenceMetrics,
        anomaly_timestamp: datetime,
    ) -> Optional[Dict[str, Any]]:
        """Enrich anomaly result with exception context if applicable.

        Queries OpenTelemetry exception metrics when error-related anomalies
        are detected. The query window is aligned with the anomaly timestamp.

        Args:
            service_name: Name of the service.
            anomalies: List of detected anomalies.
            metrics: Current metrics for the service.
            anomaly_timestamp: When the anomaly was detected.

        Returns:
            Exception context dictionary if enrichment was performed, None otherwise.
        """
        # Check if any anomaly is error-related and high/critical severity
        error_related = False
        high_severity = False

        for anomaly in anomalies:
            # Check severity
            if anomaly.severity in (AnomalySeverity.HIGH, AnomalySeverity.CRITICAL):
                high_severity = True

            # Check if error-related (by type or description)
            anomaly_type = anomaly.anomaly_type.lower()
            description = (anomaly.description or "").lower()

            if any(term in anomaly_type or term in description for term in
                   ["error", "failure", "outage", "fail"]):
                error_related = True

        # Also check error_rate from metrics
        error_rate = metrics.error_rate if metrics.error_rate else 0.0
        if error_rate > 0.01:  # More than 1% errors
            error_related = True

        # Only enrich if high/critical severity AND error-related
        if not (high_severity and error_related):
            return None

        try:
            breakdown = self.exception_enrichment.get_exception_breakdown(
                service_name=service_name,
                anomaly_timestamp=anomaly_timestamp,
            )

            if breakdown.query_successful and breakdown.has_exceptions:
                logger.info(
                    f"EXCEPTION_ENRICHMENT - {service_name}: "
                    f"{len(breakdown.exceptions)} exception types, "
                    f"top: {breakdown.top_exception.short_name if breakdown.top_exception else 'none'}"
                )
                return breakdown.to_dict()
            elif not breakdown.query_successful:
                logger.warning(f"Exception enrichment query failed for {service_name}: {breakdown.error_message}")
            else:
                logger.debug(f"No exceptions found for {service_name} in enrichment window")

            return None

        except Exception as e:
            logger.warning(f"Exception enrichment failed for {service_name}: {e}")
            return None

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status with explainability info"""
        available_services = self.model_manager.get_available_services()
        explainable_services = self._count_explainable_services(available_services)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'status': 'ready' if available_services else 'no_models',
            'available_services': len(available_services),
            'explainable_services': explainable_services,
            'explainability_coverage': f"{explainable_services}/{len(available_services)}" if available_services else "0/0",
            'services': available_services,
            'vm_endpoint': self.vm_client.endpoint,
            'vm_status': self._check_vm_connectivity(),
            'models_directory': str(self.model_manager.models_directory),
            'circuit_breaker_status': 'open' if self.vm_client.is_circuit_open() else 'closed',
            'features': self._get_feature_status()
        }

    def _check_vm_connectivity(self) -> str:
        """Test VictoriaMetrics connectivity with proper error handling"""
        try:
            base_services = self.model_manager.get_base_services()
            test_service = base_services[0] if base_services else "test"
            self.vm_client.collect_service_metrics(test_service)
            return "connected"
        except (ConnectionError, TimeoutError, requests.RequestException) as e:
            logger.debug(f"VictoriaMetrics connectivity test failed: {e}")
            return "disconnected"
        except Exception as e:
            logger.warning(f"Unexpected error testing VictoriaMetrics connectivity: {e}")
            return "error"

    def _count_explainable_services(self, available_services: List[str]) -> int:
        """Count services with explainability support"""
        explainable_count = 0
        for service in available_services:
            try:
                model = self.model_manager.load_model(service)
                if hasattr(model, 'detect_anomalies_with_context') and model.training_statistics:
                    explainable_count += 1
            except (ModelLoadError, FileNotFoundError, json.JSONDecodeError) as e:
                logger.debug(f"Could not check explainability for {service}: {e}")
            except Exception as e:
                logger.debug(f"Unexpected error checking explainability for {service}: {e}")
        return explainable_count

    def _get_feature_status(self) -> Dict[str, bool]:
        """Get current feature availability status"""
        return {
            'explainable_anomaly_detection': True,
            'time_aware_detection': True,
            'historical_context': True,
            'business_impact_analysis': True,
            'actionable_recommendations': True
        }


def main():
    """Main execution function with integrated anomaly fingerprinting"""
    parser = argparse.ArgumentParser(
        description="Smartbox ML Anomaly Detection - Enhanced Production Inference Pipeline"
    )
    parser.add_argument(
        '-v', '--verbose', 
        action='store_true', 
        help='Enable verbose output with progress information and explanations'
    )
    parser.add_argument(
        '--fingerprint-db',
        type=str,
        default="/app/data/anomaly_state.db",
        help='Path to fingerprinting database (default: /app/data/anomaly_state.db)'
    )
    
    args = parser.parse_args()
    
    # Set logging level based on verbose flag
    if not args.verbose:
        logging.getLogger().setLevel(logging.WARNING)
        logging.getLogger(__name__).setLevel(logging.WARNING)
    
    if args.verbose:
        logger.info("Smartbox Enhanced ML Anomaly Detection - Production Inference Pipeline")
        logger.info("With Explainable AI, Rich Context, and Anomaly Fingerprinting")
        logger.info("=" * 80)
    
    try:
        # Initialize pipeline and fingerprinting
        pipeline = SmartboxMLInferencePipeline(verbose=args.verbose)
        fingerprinter = _initialize_fingerprinting(args.fingerprint_db, args.verbose)

        # Health check VictoriaMetrics before proceeding
        if not _check_victoria_metrics_health(pipeline, args.verbose):
            return

        # Get and display system status
        status = pipeline.get_system_status()
        _display_system_status(status, fingerprinter, args.verbose)

        if status['status'] == 'no_models':
            _handle_no_models_error(args.verbose)
            return

        # Run inference with fingerprinting
        if args.verbose:
            logger.info("Starting Enhanced ML Inference...")
            logger.info("Using explainable time-aware anomaly detection with fingerprinting")
        
        results = pipeline.run_enhanced_time_aware_inference()
        fingerprinted_results, fp_stats = _apply_fingerprinting(results, fingerprinter, args.verbose)
        
        # Process and output results (now includes resolved incidents)
        _update_results_processor(pipeline, fingerprinted_results, fp_stats)
        _display_execution_summary(fingerprinted_results, fp_stats, args.verbose)
        
        pipeline.results_processor.output_anomalies()
        _send_to_observability_service(pipeline.results_processor.detected_anomalies, fp_stats, args.verbose)
        
        if args.verbose:
            logger.info("Enhanced inference completed!")
            _display_tips(status, fp_stats)
        
    except Exception as e:
        _handle_pipeline_error(e, args.verbose)


def _initialize_fingerprinting(db_path: str, verbose: bool) -> Optional['AnomalyFingerprinter']:
    """Initialize anomaly fingerprinting with error handling"""
    try:
        from anomaly_fingerprinter import AnomalyFingerprinter
        import os

        # Ensure the directory exists
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            # If Docker data directory doesn't exist, fall back to current directory
            if db_dir == "/app/data":
                db_path = "./anomaly_state.db"
                if verbose:
                    logger.info(f"Docker data directory not found, using local path: {db_path}")
            else:
                os.makedirs(db_dir, exist_ok=True)

        fingerprinter = AnomalyFingerprinter(db_path=db_path)

        if verbose:
            logger.info(f"Anomaly fingerprinting enabled (db: {db_path})")
            _display_fingerprinting_stats(fingerprinter)

        return fingerprinter

    except ImportError:
        if verbose:
            logger.warning("Fingerprinting module not available - continuing without fingerprinting")
        return None
    except Exception as e:
        if verbose:
            logger.warning(f"Failed to initialize fingerprinting: {e}")
        return None


def _check_victoria_metrics_health(
    pipeline: 'SmartboxMLInferencePipeline',
    verbose: bool,
    max_latency_ms: float = 5000
) -> bool:
    """Check VictoriaMetrics health before running inference.

    Args:
        pipeline: The inference pipeline with vm_client.
        verbose: Whether to log detailed output.
        max_latency_ms: Maximum acceptable response time in milliseconds.

    Returns:
        True if healthy and inference should proceed, False to skip inference.
    """
    if not hasattr(pipeline, 'vm_client') or pipeline.vm_client is None:
        if verbose:
            logger.warning("VictoriaMetrics client not available - skipping health check")
        return True  # Proceed anyway if no client

    try:
        is_healthy, details = pipeline.vm_client.health_check(max_latency_ms)

        if is_healthy:
            if verbose:
                logger.info(f"VictoriaMetrics health check passed (latency: {details['latency_ms']}ms)")
            return True

        # Health check failed
        error_msg = details.get('error', 'Unknown error')

        if details.get('circuit_breaker_open'):
            logger.warning(f"VictoriaMetrics circuit breaker is open - skipping inference")
            logger.warning("Too many recent failures. Will retry after circuit breaker timeout.")
        elif details.get('latency_ms') and details['latency_ms'] > max_latency_ms:
            logger.warning(
                f"VictoriaMetrics responding slowly ({details['latency_ms']:.0f}ms > {max_latency_ms:.0f}ms threshold) - skipping inference"
            )
            logger.warning("High latency may cause timeouts during metrics collection.")
        else:
            logger.warning(f"VictoriaMetrics health check failed: {error_msg} - skipping inference")

        if verbose:
            logger.info("Inference skipped to avoid incomplete or stale results.")
            logger.info("Next scheduled run will retry automatically.")

        return False

    except Exception as e:
        logger.error(f"VictoriaMetrics health check error: {e} - skipping inference")
        return False


def _display_fingerprinting_stats(fingerprinter: 'AnomalyFingerprinter') -> None:
    """Display current fingerprinting statistics"""
    fp_stats = fingerprinter.get_statistics()
    if fp_stats['total_open_incidents'] > 0:
        logger.info(f"Active incidents: {fp_stats['total_open_incidents']}")
        logger.info(f"Services with incidents: {len(fp_stats['open_incidents_by_service'])}")
        
        # Show severity breakdown
        by_severity = fp_stats['open_incidents_by_severity']
        if by_severity:
            severity_summary = ", ".join([f"{sev}: {count}" for sev, count in by_severity.items()])
            logger.info(f"By severity: {severity_summary}")
    else:
        logger.info("No active incidents in database")


def _display_system_status(status: Dict[str, Any], fingerprinter: Optional['AnomalyFingerprinter'], verbose: bool) -> None:
    """Display comprehensive system status"""
    if not verbose:
        return
        
    logger.info(f"System Status: {status['status']}")
    logger.info(f"Available Services: {status['available_services']}")
    logger.info(f"Explainable Services: {status['explainable_services']} ({status['explainability_coverage']})")
    logger.info(f"Services: {', '.join(status['services'][:5])}{'...' if len(status['services']) > 5 else ''}")
    logger.info(f"VictoriaMetrics: {status['vm_status']}")
    
    # Show feature status
    features = status['features']
    logger.info("Enhanced Features:")
    for feature, enabled in features.items():
        status_icon = "enabled" if enabled else "disabled"
        feature_name = feature.replace('_', ' ').title()
        logger.info(f"  {feature_name}: {status_icon}")
    
    # Add fingerprinting feature status
    fingerprinting_status = "enabled" if fingerprinter else "disabled"
    logger.info(f"  Anomaly Fingerprinting: {fingerprinting_status}")


def _handle_no_models_error(verbose: bool) -> None:
    """Handle the case when no models are found"""
    if verbose:
        logger.error("No trained models found. Please run the training pipeline first.")
        logger.info("   uv run main.py")
    else:
        error_result = {
            "error": "no_models",
            "message": "No trained models found. Please run the training pipeline first.",
            "timestamp": datetime.now().isoformat()
        }
        print(json.dumps(error_result, indent=2))


def _apply_fingerprinting(results: Dict, fingerprinter: Optional['AnomalyFingerprinter'], verbose: bool) -> Tuple[Dict, Dict]:
    """Apply fingerprinting to inference results"""
    if not fingerprinter:
        # Return results unchanged with empty stats
        return results, {
            'enhanced_services': 0, 
            'creates': 0, 
            'updates': 0, 
            'resolves': 0, 
            'fingerprinting_errors': 0,
            'resolved_incidents': []
        }
    
    fingerprinted_results = {}
    fp_stats = {
        'enhanced_services': 0, 
        'creates': 0, 
        'updates': 0, 
        'resolves': 0, 
        'fingerprinting_errors': 0
    }
    
    # Collect all resolved incidents across all services
    all_resolved_incidents = []
    resolution_summary = {}  # Track by service
    
    for service_name, result in results.items():
        if isinstance(result, dict) and not result.get('error'):
            try:
                full_service_name = _determine_full_service_name(service_name, result)
                
                if verbose:
                    logger.info(f"Processing {service_name} as {full_service_name}")
                
                enhanced_result = fingerprinter.process_anomalies(
                    full_service_name=full_service_name,
                    anomaly_result=result
                )
                
                fingerprinted_results[service_name] = enhanced_result
                fp_stats['enhanced_services'] += 1
                
                # Collect fingerprinting statistics
                fingerprinting_data = enhanced_result.get('fingerprinting', {})
                action_summary = fingerprinting_data.get('action_summary', {})
                
                fp_stats['creates'] += action_summary.get('incident_creates', 0)
                fp_stats['updates'] += action_summary.get('incident_continues', 0)
                fp_stats['resolves'] += action_summary.get('incident_closes', 0)
                
                # NEW: Collect resolved incidents for API notification
                resolved_incidents = fingerprinting_data.get('resolved_incidents', [])
                all_resolved_incidents.extend(resolved_incidents)
                
                # Track resolutions by service for summary
                if resolved_incidents:
                    resolution_summary[service_name] = len(resolved_incidents)
                
                if verbose and fingerprinting_data.get('overall_action') != 'NO_CHANGE':
                    overall_action = fingerprinting_data.get('overall_action', 'UNKNOWN')
                    logger.info(f"{service_name}: {overall_action} fingerprinting action")
                
            except Exception as e:
                if verbose:
                    logger.warning(f"Fingerprinting failed for {service_name}: {e}")
                fingerprinted_results[service_name] = result
                fp_stats['fingerprinting_errors'] += 1
        else:
            fingerprinted_results[service_name] = result
    
    # Store resolved incidents and summary for API notification
    fp_stats['resolved_incidents'] = all_resolved_incidents
    fp_stats['resolution_summary'] = resolution_summary
    
    # Enhanced logging for multiple resolutions
    if verbose and all_resolved_incidents:
        total_resolved = len(all_resolved_incidents)
        services_count = len(resolution_summary)
        logger.info(f"Batch resolution: {total_resolved} incidents across {services_count} services")
        
        for service, count in resolution_summary.items():
            logger.info(f"  {service}: {count} incidents resolved")
    
    return fingerprinted_results, fp_stats


def _determine_full_service_name(service_name: str, result: Dict) -> str:
    """Determine the full service name including model period"""
    # Check if service_name already includes model info
    model_periods = ['business_hours', 'evening_hours', 'night_hours', 'weekend_day', 'weekend_night']
    if '_' in service_name and any(model in service_name for model in model_periods):
        return service_name
    
    # Determine model from result or use default
    model_name = 'evening_hours'  # Default
    
    if 'time_period' in result and result['time_period']:
        model_name = result['time_period']
    elif 'model_type' in result and 'time_aware' in result.get('model_type', ''):
        # Infer from current time
        current_hour = datetime.now().hour
        if 8 <= current_hour < 18:
            model_name = 'business_hours'
        elif 22 <= current_hour or current_hour < 6:
            model_name = 'night_hours'
        else:
            model_name = 'evening_hours'
    
    return f"{service_name}_{model_name}"


def _update_results_processor(pipeline: 'SmartboxMLInferencePipeline', results: Dict, fp_stats: Dict) -> None:
    """Update the results processor with both active anomalies and resolved incidents.

    Formats results using the API payload specification before storing.
    """
    formatted_payloads = []

    # Add ACTIVE anomalies (ongoing + new) - format them properly
    for service_name, result in results.items():
        if isinstance(result, dict) and not result.get('error'):
            anomalies = result.get('anomalies', [])
            if isinstance(anomalies, dict):
                has_anomalies = len(anomalies) > 0
            else:
                has_anomalies = len(anomalies) > 0 if anomalies else False

            if has_anomalies:  # This service has active incidents
                # Format using the API spec formatter
                formatted_payload = pipeline.results_processor._format_time_aware_alert_json(result)
                formatted_payloads.append(formatted_payload)

    # Add RESOLVED incidents as special payloads
    resolved_incidents = fp_stats.get('resolved_incidents', [])
    for resolved_incident in resolved_incidents:
        resolution_payload = {
            "alert_type": "incident_resolved",
            "service_name": resolved_incident['service_name'],
            "timestamp": resolved_incident['resolved_at'],
            "incident_id": resolved_incident['incident_id'],
            "fingerprint_id": resolved_incident['fingerprint_id'],
            "anomaly_name": resolved_incident['anomaly_name'],
            "resolution_details": {
                "final_severity": resolved_incident['final_severity'],
                "total_occurrences": resolved_incident['total_occurrences'],
                "incident_duration_minutes": resolved_incident['incident_duration_minutes'],
                "first_seen": resolved_incident['first_seen'],
                "last_detected_by_model": resolved_incident.get('last_detected_by_model', 'unknown')
            },
            "model_type": "incident_resolution"
        }
        formatted_payloads.append(resolution_payload)

    pipeline.results_processor.detected_anomalies = formatted_payloads


def _display_execution_summary(results: Dict, fp_stats: Dict, verbose: bool) -> None:
    """Display comprehensive execution summary"""
    if not verbose:
        return
        
    logger.info("Execution Summary:")
    logger.info(f"Services Evaluated: {len(results)}")
    
    # Analyze results
    total_anomalies = 0
    services_with_anomalies = 0
    successful_services = 0
    explainable_alerts = 0
    
    for result in results.values():
        if isinstance(result, dict) and not result.get('error'):
            successful_services += 1
            
            # Count anomalies (handle both original and fingerprinted formats)
            anomalies = result.get('anomalies', [])
            if isinstance(anomalies, dict):
                anomaly_count = len(anomalies)
            else:
                anomaly_count = len(anomalies) if hasattr(anomalies, '__len__') else 0
            
            total_anomalies += anomaly_count
            
            if anomaly_count > 0:
                services_with_anomalies += 1
                
                # Check if alert has explainability features
                if (result.get('explainable', False) or 
                    'historical_context' in result or 
                    'explanation' in result):
                    explainable_alerts += 1
    
    logger.info(f"Total Anomalies: {total_anomalies}")
    logger.info(f"Services with Anomalies: {services_with_anomalies}")
    logger.info(f"Explainable Alerts: {explainable_alerts}/{services_with_anomalies}")
    logger.info(f"Successful Services: {successful_services}/{len(results)}")
    
    # Fingerprinting summary
    if fp_stats['enhanced_services'] > 0:
        logger.info("Fingerprinting Summary:")
        logger.info(f"Enhanced Services: {fp_stats['enhanced_services']}/{len(results)}")
        logger.info(f"Lifecycle Actions: {fp_stats['creates']} creates, "
              f"{fp_stats['updates']} updates, "
              f"{fp_stats['resolves']} resolves")
        
        # Show resolution details if any
        resolved_incidents = fp_stats.get('resolved_incidents', [])
        if resolved_incidents:
            resolution_summary = fp_stats.get('resolution_summary', {})
            logger.info(f"Resolved Incidents: {len(resolved_incidents)} total")
            for service, count in resolution_summary.items():
                logger.info(f"  {service}: {count} resolved")
        
        if fp_stats['fingerprinting_errors'] > 0:
            logger.info(f"Fingerprinting Errors: {fp_stats['fingerprinting_errors']}")


def _send_to_observability_service(detected_anomalies: List[Dict], fp_stats: Dict, verbose: bool) -> None:
    """Send anomalies and resolutions to appropriate observability service endpoints"""

    # Load observability config
    config = get_config()
    obs_config = config.observability

    # Check if API integration is enabled
    if not obs_config.enabled:
        if verbose:
            logger.info("Observability API integration is disabled in config")
        return

    # Separate current anomalies from resolved incidents
    active_anomalies = [item for item in detected_anomalies
                       if item.get('alert_type') != 'incident_resolved']

    resolved_incidents = [item for item in detected_anomalies
                         if item.get('alert_type') == 'incident_resolved']

    # Send active anomalies to the main anomalies endpoint
    if active_anomalies:
        _send_active_anomalies(active_anomalies, obs_config, verbose)

    # Send resolved incidents to a dedicated resolutions endpoint
    if resolved_incidents:
        _send_resolved_incidents(resolved_incidents, obs_config, verbose)

    # Summary logging
    if active_anomalies or resolved_incidents:
        if verbose:
            logger.info(f"Mixed batch sent: {len(active_anomalies)} active anomalies, {len(resolved_incidents)} incident resolutions")
        else:
            logger.info(f"Sent {len(active_anomalies)} active anomalies and {len(resolved_incidents)} resolutions")
    else:
        logger.info("No anomalies or resolutions to send to observability service")


def _send_active_anomalies(anomalies: List[Dict], obs_config: ObservabilityConfig, verbose: bool) -> None:
    """Send active anomalies to the main anomalies endpoint"""
    try:
        r = requests.post(
            obs_config.anomalies_url,
            json=anomalies,
            timeout=obs_config.request_timeout_seconds
        )
        r.raise_for_status()

        if verbose:
            fingerprinted_count = len([a for a in anomalies if 'fingerprinting' in a])
            logger.info(f"Sent {len(anomalies)} active anomalies to {obs_config.anomalies_endpoint} — status {r.status_code}")
            if fingerprinted_count > 0:
                logger.info(f"  {fingerprinted_count} with fingerprinting data")
        else:
            logger.info(f"Sent {len(anomalies)} active anomalies — status {r.status_code}")

    except Exception as e:
        logger.error(f"Failed to send active anomalies to {obs_config.anomalies_url}: {e}")


def _send_resolved_incidents(resolutions: List[Dict], obs_config: ObservabilityConfig, verbose: bool) -> None:
    """Send resolved incidents to the dedicated resolutions endpoint"""
    try:
        r = requests.post(
            obs_config.resolutions_url,
            json=resolutions,
            timeout=obs_config.request_timeout_seconds
        )
        r.raise_for_status()

        if verbose:
            logger.info(f"Sent {len(resolutions)} incident resolutions to {obs_config.resolutions_endpoint} — status {r.status_code}")
            # Log some details about what was resolved
            for resolution in resolutions[:3]:  # Show first 3
                service = resolution['service']
                incident_id = resolution['incident_id']
                duration = resolution['resolution_details']['incident_duration_minutes']
                logger.info(f"  Resolved {service}/{incident_id} after {duration}min")
            if len(resolutions) > 3:
                logger.info(f"  ... and {len(resolutions) - 3} more")
        else:
            logger.info(f"Sent {len(resolutions)} incident resolutions — status {r.status_code}")

    except Exception as e:
        logger.error(f"Failed to send incident resolutions to {obs_config.resolutions_url}: {e}")


def _display_tips(status: Dict, fp_stats: Dict) -> None:
    """Display helpful tips based on system status"""
    tips = []
    
    if status['explainable_services'] < status['available_services']:
        missing_explainable = status['available_services'] - status['explainable_services']
        tips.append(f"{missing_explainable} services could benefit from retraining with explainability features")
    
    if fp_stats['fingerprinting_errors'] > 0:
        tips.append("Some fingerprinting errors occurred - check service name formats")
    
    if tips:
        logger.info("Tips:")
        for tip in tips:
            logger.info(f"  • {tip}")


def _handle_pipeline_error(e: Exception, verbose: bool) -> None:
    """Handle pipeline execution errors"""
    logger.error(f"Enhanced pipeline execution failed: {e}")
    if verbose:
        logger.error(f"Pipeline failed: {e}")
    else:
        error_result = {
            "error": "pipeline_failed",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }
        print(json.dumps(error_result, indent=2))


if __name__ == "__main__":
    main()
