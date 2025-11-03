"""
Enhanced Production-Grade ML Inference Pipeline for Smartbox Anomaly Detection
Now supports explainable anomaly detection with rich context and recommendations
"""

import json
import logging
import argparse
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import time
import urllib3

# Import shared model class with explainability features
from anomaly_models import SmartboxAnomalyDetector
from vmclient import VictoriaMetricsClient, InferenceMetrics, MetricsCollectionError
from time_aware_anomaly_detection import TimeAwareAnomalyDetector
from anomaly_fingerprinter import AnomalyFingerprinter 

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


class AnomalySeverity(Enum):
    """Anomaly severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


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


class ModelLoadError(Exception):
    """Custom exception for model loading failures"""
    pass


class EnhancedModelManager:
    """Enhanced model management - UPDATED to not interfere with lazy loading"""
    
    def __init__(self, models_directory: str = "./smartbox_models/"):
        self.models_directory = Path(models_directory)
        self._model_cache = {}
        self._model_metadata = {}
        self._load_times = {}
        self._model_validators = {}
    
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
        """Format time-aware anomaly as structured JSON"""
        anomalies = result.get('anomalies', {})
        
        # Determine overall severity
        severity_order = ['low', 'medium', 'high', 'critical']
        max_severity = 'low'
        for anomaly_data in anomalies.values():
            if isinstance(anomaly_data, dict):
                anomaly_severity = anomaly_data.get('severity', 'medium')
                if severity_order.index(anomaly_severity) > severity_order.index(max_severity):
                    max_severity = anomaly_severity
        
        # Format anomalies array
        formatted_anomalies = []
        for anomaly_name, anomaly_data in anomalies.items():
            if isinstance(anomaly_data, dict):
                formatted_anomalies.append({
                    "type": anomaly_name,
                    "severity": anomaly_data.get('severity', 'medium'),
                    "confidence_score": anomaly_data.get('score', 0.0),
                    "description": anomaly_data.get('description', anomaly_name.replace('_', ' ').title()),
                    "detection_method": anomaly_data.get('type', 'unknown'),
                    "threshold_value": anomaly_data.get('threshold'),
                    "actual_value": anomaly_data.get('value'),
                    "metadata": {
                        key: value for key, value in anomaly_data.items() 
                        if key not in ['severity', 'score', 'description', 'type', 'threshold', 'value']
                    }
                })
        
        return {
            "alert_type": "anomaly_detected",
            "service": result['service'],
            "time_period": result.get('time_period', 'unknown'),
            "model_type": result.get('model_type', 'time_aware'),
            "timestamp": result.get('timestamp', datetime.now().isoformat()),
            "overall_severity": max_severity,
            "anomaly_count": len(formatted_anomalies),
            "current_metrics": result.get('metrics', {}),
            "anomalies": formatted_anomalies
        }
    
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
                from time_aware_anomaly_detection import TimeAwareAnomalyDetector
                
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
    
    def detect_with_explainability(self, service_name: str, metrics: Dict[str, Any], 
                                 timestamp: datetime, verbose: bool = False) -> Dict[str, Any]:
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
                        metrics, timestamp, self.models_directory, verbose
                    )
                    
                    # Add efficiency info
                    enhanced_result['performance_info'] = {
                        'lazy_loaded': True,
                        'models_loaded': list(detector.models.keys()),
                        'period_used': current_period,
                        'total_available': len(detector._available_periods)
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
                metrics, timestamp, self.models_directory, verbose
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
                    'total_available': len(detector._available_periods)
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
                 vm_endpoint: str = "https://otel-metrics.production.smartbox.com",
                 models_directory: str = "./smartbox_models/",
                 alerts_directory: str = "./alerts/",
                 max_workers: int = 3,
                 verbose: bool = False):
        
        self.vm_client = VictoriaMetricsClient(vm_endpoint)
        self.model_manager = EnhancedModelManager(models_directory)
        self.detection_engine = EnhancedAnomalyDetectionEngine(self.model_manager)
        self.results_processor = EnhancedResultsProcessor(alerts_directory, verbose)
        self.time_aware_detector = EnhancedTimeAwareDetector(models_directory)
        self.max_workers = max_workers
        self.verbose = verbose
        
        logger.info("Enhanced Smartbox ML Inference Pipeline initialized with explainability")
    
    def run_enhanced_time_aware_inference(self, service_names: Optional[List[str]] = None) -> Dict[str, Dict]:
        """Run inference with enhanced time-aware anomaly detection and explainability"""
        start_time = datetime.now()
    
        # Get base service names (without time period suffixes)
        if service_names is None:
            service_names = self.model_manager.get_base_services()
        
        if not service_names:
            logger.error("No services available for inference")
            return {}
    
        logger.info(f"Running enhanced time-aware inference for {len(service_names)} services: {service_names}")
    
        results = {}
    
        for service_name in service_names:
            if self.verbose:
                logger.info(f"Analyzing {service_name}")
                
            try:
                # Collect current metrics
                if self.verbose:
                    logger.info(f"Collecting metrics from VictoriaMetrics for {service_name}")
                metrics = self.vm_client.collect_service_metrics(service_name)
            
                # Try enhanced time-aware detection first
                try:
                    if self.verbose:
                        logger.info(f"Loading enhanced time-aware models for {service_name}")
                    
                    enhanced_result = self.time_aware_detector.detect_with_explainability(
                        service_name, metrics.to_dict(), start_time, self.verbose
                    )
                    
                    results[service_name] = enhanced_result
                    
                except Exception as model_error:
                    # Fallback to standard time-aware detection
                    if self.verbose:
                        logger.warning(f"Enhanced detection failed for {service_name}: {str(model_error)[:50]}...")
                        logger.info("Falling back to standard time-aware detection")
                    
                    logger.warning(f"Enhanced time-aware detection failed for {service_name}: {model_error}")
                    
                    try:
                        detector = TimeAwareAnomalyDetector.load_models(
                            str(self.model_manager.models_directory), service_name, False
                        )
                        
                        anomalies = detector.detect_anomalies(metrics.to_dict(), start_time)
                        period = detector.get_time_period(start_time)
                        
                        # Handle both dict and list formats
                        if isinstance(anomalies, dict):
                            anomaly_count = len(anomalies)
                        else:
                            anomaly_count = len(anomalies) if hasattr(anomalies, '__len__') else 0
                            anomalies = {}
                        
                        results[service_name] = {
                            'service': service_name,
                            'time_period': period,
                            'model_type': 'time_aware_fallback',
                            'anomaly_count': anomaly_count,
                            'anomalies': anomalies,
                            'metrics': metrics.to_dict(),
                            'timestamp': start_time.isoformat(),
                            'explainable': False
                        }
                        
                    except Exception as fallback_error:
                        if self.verbose:
                            logger.error(f"Both enhanced and standard models failed for {service_name}")
                        
                        logger.error(f"Both enhanced and standard time-aware models failed for {service_name}: {fallback_error}")
                        results[service_name] = {'service': service_name, 'error': str(fallback_error)}
                        continue
            
                # Process results for storage and batching
                self.results_processor.process_result(results[service_name])
            
                # Enhanced logging with explainability info
                result = results[service_name]
                if 'error' not in result:
                    anomaly_count = result.get('anomaly_count', len(result.get('anomalies', {})))
                    period = result.get('time_period', 'unknown')
                    model_type = result.get('model_type', 'unknown')
                    is_explainable = result.get('explainable', model_type.endswith('explainable'))
                    
                    if anomaly_count > 0:
                        explainable_indicator = "explainable" if is_explainable else "standard"
                        if self.verbose:
                            logger.info(f"ANOMALIES DETECTED for {service_name}: {anomaly_count} anomalies found ({explainable_indicator})")
                            
                            # Show explanation summary if available
                            if 'explanation' in result:
                                explanation = result['explanation']
                                if 'detailed_explanations' in explanation:
                                    logger.info(f"Explanation: {explanation['detailed_explanations'][0]}")
                                    
                            # Show top recommendation if available
                            if 'recommended_actions' in result and result['recommended_actions']:
                                logger.info(f"Recommendation: {result['recommended_actions'][0]}")
                        
                        logger.info(f"ANOMALY_DETECTED - {service_name} [{period}] {explainable_indicator}: {anomaly_count} anomalies detected")
                    else:
                        if self.verbose:
                            logger.info(f"No anomalies detected for {service_name}")
                        logger.info(f"NO_ANOMALIES - {service_name} [{period}]: No anomalies detected")
            
                time.sleep(0.2)  # Small delay
            
            except Exception as e:
                if self.verbose:
                    logger.error(f"Error processing {service_name}: {str(e)}")
                logger.error(f"Enhanced time-aware inference failed for {service_name}: {e}")
                results[service_name] = {'service': service_name, 'error': str(e)}
    
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
                        recommended_actions=enhanced_result.get('recommended_actions')
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
        default="./anomaly_state.db",
        help='Path to fingerprinting database (default: ./anomaly_state.db)'
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
        fingerprinter = AnomalyFingerprinter(db_path=db_path)
        
        if verbose:
            logger.info("Anomaly fingerprinting enabled")
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
        if isinstance(result, dict) and 'error' not in result:
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
    """Update the results processor with both active anomalies and resolved incidents"""
    enhanced_anomalies = []
    
    # Add ACTIVE anomalies (ongoing + new)
    for service_name, result in results.items():
        if isinstance(result, dict) and 'error' not in result:
            anomalies = result.get('anomalies', [])
            if isinstance(anomalies, dict):
                anomalies = list(anomalies.values())
            
            if anomalies:  # This service has active incidents
                enhanced_anomalies.append(result)
    
    # Add RESOLVED incidents as special payloads
    resolved_incidents = fp_stats.get('resolved_incidents', [])
    for resolved_incident in resolved_incidents:
        resolution_payload = {
            "alert_type": "incident_resolved",
            "service": resolved_incident['service_name'],
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
        enhanced_anomalies.append(resolution_payload)
    
    pipeline.results_processor.detected_anomalies = enhanced_anomalies


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
        if isinstance(result, dict) and 'error' not in result:
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
    
    # Separate current anomalies from resolved incidents
    active_anomalies = [item for item in detected_anomalies 
                       if item.get('alert_type') != 'incident_resolved']
    
    resolved_incidents = [item for item in detected_anomalies 
                         if item.get('alert_type') == 'incident_resolved']
    
    # Send active anomalies to the main anomalies endpoint
    if active_anomalies:
        _send_active_anomalies(active_anomalies, verbose)
    
    # Send resolved incidents to a dedicated resolutions endpoint
    if resolved_incidents:
        _send_resolved_incidents(resolved_incidents, verbose)
    
    # Summary logging
    if active_anomalies or resolved_incidents:
        if verbose:
            logger.info(f"Mixed batch sent: {len(active_anomalies)} active anomalies, {len(resolved_incidents)} incident resolutions")
        else:
            logger.info(f"Sent {len(active_anomalies)} active anomalies and {len(resolved_incidents)} resolutions")
    else:
        logger.info("No anomalies or resolutions to send to observability service")


def _send_active_anomalies(anomalies: List[Dict], verbose: bool) -> None:
    """Send active anomalies to the main anomalies endpoint"""
    try:
        r = requests.post(
            "http://localhost:8000/api/anomalies/batch",  # Original endpoint
            json=anomalies,
            timeout=5
        )
        r.raise_for_status()
        
        if verbose:
            fingerprinted_count = len([a for a in anomalies if 'fingerprinting' in a])
            logger.info(f"Sent {len(anomalies)} active anomalies to /api/anomalies/batch — status {r.status_code}")
            if fingerprinted_count > 0:
                logger.info(f"  {fingerprinted_count} with fingerprinting data")
        else:
            logger.info(f"Sent {len(anomalies)} active anomalies — status {r.status_code}")
            
    except Exception as e:
        logger.error(f"Failed to send active anomalies to /api/anomalies/batch: {e}")


def _send_resolved_incidents(resolutions: List[Dict], verbose: bool) -> None:
    """Send resolved incidents to the dedicated resolutions endpoint"""
    try:
        r = requests.post(
            "http://localhost:8000/api/incidents/resolve",  # NEW dedicated endpoint
            json=resolutions,
            timeout=5
        )
        r.raise_for_status()
        
        if verbose:
            logger.info(f"Sent {len(resolutions)} incident resolutions to /api/incidents/resolve — status {r.status_code}")
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
        logger.error(f"Failed to send incident resolutions to /api/incidents/resolve: {e}")


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
