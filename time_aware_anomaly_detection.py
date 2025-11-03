from datetime import datetime
from typing import Dict, Optional, List
import pandas as pd
from pathlib import Path
import warnings
from anomaly_models import SmartboxAnomalyDetector
import json

warnings.filterwarnings('ignore')

class TimeAwareAnomalyDetector:
    """Enhanced Time-aware anomaly detector with lazy loading for efficiency"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.models = {}  # Only loaded models will be here
        self._available_periods = set()  # Track which periods are available
        self._models_directory = None  # Will be set when loading
        
        # Enhanced time periods with weekend day/night split
        self.time_periods = {
            'business_hours': {'start': 8, 'end': 18, 'weekdays_only': True},
            'night_hours': {'start': 22, 'end': 6, 'weekdays_only': True},
            'evening_hours': {'start': 18, 'end': 22, 'weekdays_only': True},
            'weekend_day': {'start': 8, 'end': 22, 'weekends_only': True},
            'weekend_night': {'start': 22, 'end': 8, 'weekends_only': True}
        }
        
        # Enhanced validation thresholds for 5-period approach
        self.validation_thresholds = self._get_service_specific_thresholds()
    
    def train_time_aware_models(self, features_df: pd.DataFrame):
        """
        Train time-aware models for different time periods.
        
        This method splits the training data by time periods and trains
        a separate SmartboxAnomalyDetector model for each period.
        """
        if features_df.empty:
            raise ValueError("No training data provided")
        
        # Add time period column to features
        features_df = features_df.copy()
        features_df['time_period'] = features_df.index.map(self.get_time_period)
        
        # Train a model for each time period
        for period in self.time_periods.keys():
            period_data = features_df[features_df['time_period'] == period]
            
            min_samples = self._get_min_samples_for_period(period)
            if len(period_data) < min_samples:
                print(f"   ⚠️ Insufficient data for {period}: {len(period_data)} samples (need {min_samples})")
                continue
            
            print(f"   🚀 Training model for {period} ({len(period_data)} samples)")
            
            # Create and train a model for this period
            model = SmartboxAnomalyDetector(f"{self.service_name}_{period}")
            model.train(period_data.drop(columns=['time_period']))
            
            # Store the model
            self.models[period] = model
            print(f"   ✅ Trained model for {period}")
        
        if not self.models:
            raise ValueError("No models were trained. Insufficient data for all time periods.")
        
        print(f"   🎯 Trained {len(self.models)} time-aware models for {self.service_name}")
        return self.models
    
    def save_models(self, model_directory: str, metadata: Dict = None):
        """
        Save all time-aware models to the specified directory.
        
        Args:
            model_directory: Directory to save models
            metadata: Additional metadata to save with models
            
        Returns:
            Dictionary with paths to saved models
        """
        model_dir = Path(model_directory)
        model_dir.mkdir(exist_ok=True)
        
        saved_paths = {}
        
        for period, model in self.models.items():
            service_name = f"{self.service_name}_{period}"
            model_path = model.save_model_secure(str(model_dir), metadata)
            saved_paths[period] = str(model_path)
        
        return saved_paths
    
    def _get_min_samples_for_period(self, period: str) -> int:
        """Get minimum samples required for a time period"""
        service_type = self._get_service_type()
        
        # Weekend periods need fewer samples due to natural variability
        if period.startswith('weekend_'):
            if service_type in ['micro_service', 'admin_service']:
                return 50  # Even fewer for micro/admin services on weekends
            else:
                return 100  # Standard minimum for weekends
        else:
            # Weekday periods
            if service_type == 'micro_service':
                return 100
            elif service_type == 'admin_service':
                return 150
            else:
                return 200  # Standard minimum for critical services
    
    def _get_service_type(self) -> str:
        """Determine service type based on service name"""
        service_lower = self.service_name.lower()
        if any(pattern in service_lower for pattern in ['booking', 'search', 'mobile-api', 'shire-api']):
            return 'critical_service'
        elif any(pattern in service_lower for pattern in ['adm', 'admin']):
            return 'admin_service'
        elif any(pattern in service_lower for pattern in ['fa5', 'micro', 'util']):
            return 'micro_service'
        elif any(pattern in service_lower for pattern in ['m2-']):
            return 'core_service'
        else:
            return 'standard_service'
    
    def _get_service_specific_thresholds(self) -> Dict[str, float]:
        """Get service-specific validation thresholds for 5-period approach"""
        service_lower = self.service_name.lower()
        
        if any(pattern in service_lower for pattern in ['booking', 'search', 'mobile-api', 'shire-api']):
            return {
                'business_hours': 0.12, 'night_hours': 0.08, 'evening_hours': 0.15,
                'weekend_day': 0.20, 'weekend_night': 0.25
            }
        elif any(pattern in service_lower for pattern in ['adm', 'admin', 'management']):
            return {
                'business_hours': 0.20, 'night_hours': 0.15, 'evening_hours': 0.22,
                'weekend_day': 0.30, 'weekend_night': 0.35
            }
        elif any(pattern in service_lower for pattern in ['fa5', 'micro', 'internal', 'util', 'worker', 'job', 'task']):
            return {
                'business_hours': 0.25, 'night_hours': 0.30, 'evening_hours': 0.28,
                'weekend_day': 0.35, 'weekend_night': 0.40
            }
        elif any(pattern in service_lower for pattern in ['m2-', 'core', 'platform']):
            return {
                'business_hours': 0.15, 'night_hours': 0.10, 'evening_hours': 0.18,
                'weekend_day': 0.25, 'weekend_night': 0.28
            }
        else:
            return {
                'business_hours': 0.18, 'night_hours': 0.12, 'evening_hours': 0.20,
                'weekend_day': 0.28, 'weekend_night': 0.32
            }
    
    def get_time_period(self, timestamp: datetime) -> str:
        """Determine which of the 5 time periods a timestamp falls into"""
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        is_weekend = day_of_week >= 5
        
        if is_weekend:
            if 8 <= hour < 22:
                return 'weekend_day'
            else:
                return 'weekend_night'
        else:
            if 8 <= hour < 18:
                return 'business_hours'
            elif 22 <= hour or hour < 6:
                return 'night_hours'
            else:
                return 'evening_hours'
    
    def _get_period_type(self, period: str) -> str:
        """Get the type classification for a time period"""
        return {
            'business_hours': 'peak_activity', 
            'night_hours': 'minimal_activity',
            'evening_hours': 'transition_activity', 
            'weekend_day': 'weekend_moderate_activity',
            'weekend_night': 'weekend_minimal_activity'
        }.get(period, 'unknown_activity')
    
    # Include all the other methods from the original implementation
    # (lazy loading methods, detection methods, etc.)
    
    def _discover_available_periods(self, models_directory: str) -> set:
        """Discover which periods are available for this service without loading models"""
        models_path = Path(models_directory)
        available_periods = set()
        
        # Check for each possible period
        all_periods = ['business_hours', 'night_hours', 'evening_hours', 'weekend_day', 'weekend_night']
        
        for period in all_periods:
            period_dir = models_path / f"{self.service_name}_{period}"
            if period_dir.exists() and (period_dir / "model_data.json").exists():
                available_periods.add(period)
        
        return available_periods
    
    def _load_period_model_lazy(self, period: str, models_directory: str, verbose: bool = False) -> bool:
        """Load a specific period model on-demand"""
        if period in self.models:
            return True  # Already loaded
        
        if period not in self._available_periods:
            if verbose:
                print(f"   ❌ Period {period} not available for {self.service_name}")
            return False
        
        try:
            if verbose:
                period_emoji = "🌅" if period == 'weekend_day' else "🌙" if period == 'weekend_night' else "🕒"
                print(f"   {period_emoji} Loading {period} model for {self.service_name}...")
            
            from anomaly_models import SmartboxAnomalyDetector
            model = SmartboxAnomalyDetector.load_model_secure(
                models_directory, f"{self.service_name}_{period}", auto_tune=True
            )
            
            self.models[period] = model
            
            if verbose:
                explainability_status = "🧠" if hasattr(model, 'training_statistics') and model.training_statistics else "❌"
                performance_status = "🚀" if getattr(model, 'auto_tune', False) else "⚠️"
                threshold = self.validation_thresholds.get(period, 0.15)
                print(f"   ✅ Loaded {period} model {explainability_status}{performance_status} (threshold: {threshold:.1%})")
            
            return True
            
        except Exception as e:
            if verbose:
                print(f"   ❌ Failed to load {period} model: {e}")
            return False
    
    def detect_anomalies(self, current_metrics: Dict, timestamp: datetime, 
                        models_directory: str = None, verbose: bool = False) -> Dict:
        """Enhanced anomaly detection with lazy loading - only loads the model needed"""
        
        # Determine current period
        current_period = self.get_time_period(timestamp)
        
        if verbose:
            period_emoji = "🌅" if current_period == 'weekend_day' else "🌙" if current_period == 'weekend_night' else "🕒"
            print(f"  {period_emoji} Current period: {current_period}")
        
        # Set models directory if provided
        if models_directory:
            self._models_directory = models_directory
            if not self._available_periods:  # Discover on first use
                self._available_periods = self._discover_available_periods(models_directory)
                if verbose:
                    print(f"  📋 Available periods: {sorted(list(self._available_periods))}")
        
        # Lazy load only the current period model
        if current_period not in self.models:
            if not self._load_period_model_lazy(current_period, self._models_directory, verbose):
                # Fallback to similar period
                fallback_period = self._find_fallback_period(current_period)
                if fallback_period and self._load_period_model_lazy(fallback_period, self._models_directory, verbose):
                    if verbose:
                        print(f"  🔄 Using fallback period: {fallback_period}")
                    current_period = fallback_period
                else:
                    return {}  # No suitable model available
        
        # Use the loaded model for detection
        model = self.models[current_period]
        anomalies = model.detect_anomalies(current_metrics)
        
        # Add time context to all anomalies
        for anomaly_name, anomaly_data in anomalies.items():
            if isinstance(anomaly_data, dict):
                anomaly_data['time_period'] = current_period
                anomaly_data['timestamp'] = timestamp.isoformat()
                anomaly_data['time_confidence'] = self._calculate_time_confidence(current_period, current_metrics)
                anomaly_data['detection_optimized'] = True
                anomaly_data['lazy_loaded'] = True  # Mark as efficiently loaded
        
        return anomalies
    
    def _find_fallback_period(self, target_period: str) -> Optional[str]:
        """Find the best fallback period if target period model is not available"""
        
        fallback_map = {
            'business_hours': ['evening_hours', 'weekend_day', 'night_hours', 'weekend_night'],
            'night_hours': ['weekend_night', 'evening_hours', 'business_hours', 'weekend_day'],
            'evening_hours': ['business_hours', 'weekend_day', 'night_hours', 'weekend_night'],
            'weekend_day': ['business_hours', 'evening_hours', 'weekend_night', 'night_hours'],
            'weekend_night': ['night_hours', 'weekend_day', 'evening_hours', 'business_hours']
        }
        
        fallback_priorities = fallback_map.get(target_period, ['business_hours', 'night_hours', 'evening_hours'])
        
        # Return first available fallback period
        for fallback_period in fallback_priorities:
            if fallback_period in self._available_periods:
                return fallback_period
        
        return None
    
    def _calculate_time_confidence(self, period: str, metrics: Dict) -> float:
        """Calculate confidence score based on time period and metrics"""
        base_confidence = {
            'business_hours': 0.9, 'night_hours': 0.95, 'evening_hours': 0.8,
            'weekend_day': 0.7, 'weekend_night': 0.6
        }
        return base_confidence.get(period, 0.8)
    
    # Legacy compatibility method
    @classmethod
    def load_models(cls, model_directory: str, service_name: str, verbose: bool = False):
        """Create detector but don't load any models yet (lazy loading)"""
        detector = cls(service_name)
        detector._models_directory = model_directory
        detector._available_periods = detector._discover_available_periods(model_directory)
        
        if verbose:
            print(f"   📋 Discovered {len(detector._available_periods)} available periods for {service_name}")
            print(f"       Periods: {sorted(list(detector._available_periods))}")
            print(f"   ⚡ Lazy loading enabled - models will load on-demand")
        
        return detector
