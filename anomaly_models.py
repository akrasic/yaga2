
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler  
import joblib
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


class SmartboxAnomalyDetector:
    """Enhanced anomaly detection model with explainability, robust defaults, and adaptive hyperparameter tuning"""
    
    def __init__(self, service_name: str, auto_tune: bool = True):
        self.service_name = service_name
        self.auto_tune = auto_tune
        self.models = {}
        self.scalers = {}
        self.thresholds = {}
        self.feature_columns = []
        self.model_metadata = {}
        self.optimal_params = {}  # Store optimal parameters used
        
        # Explainability features
        self.training_statistics = {}
        self.time_period_stats = {}
        self.feature_importance = {}
        
        # Zero-normal metric statistics
        self.zero_statistics = {}
        
        # Suppress sklearn warnings during inference
        warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.utils.validation')
        
    def _ensure_consistent_feature_format(self, data, feature_names: List[str]) -> pd.DataFrame:
        """Ensure data is in consistent DataFrame format with proper feature names"""
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, (list, np.ndarray)):
            if hasattr(data, 'shape') and len(data.shape) == 1:
                data = [data]
            return pd.DataFrame(data, columns=feature_names)
        else:
            return pd.DataFrame([data], columns=feature_names)
    
    def _get_optimal_params(self, clean_data: pd.DataFrame) -> Dict:
        """
        Get optimal parameters based on service and data characteristics.
        Includes smart defaults and pattern-based detection for unknown services.
        """
        data_size = len(clean_data)
        variability = clean_data.std().mean()
        
        # Known service configurations
        service_params = {
            # High-traffic, critical services
            'booking': {'base_contamination': 0.02, 'complexity': 'high', 'category': 'critical'},
            'search': {'base_contamination': 0.04, 'complexity': 'high', 'category': 'critical'}, 
            'mobile-api': {'base_contamination': 0.03, 'complexity': 'high', 'category': 'critical'},
            'shire-api': {'base_contamination': 0.03, 'complexity': 'high', 'category': 'critical'},
            
            # Medium-traffic services
            'friday': {'base_contamination': 0.05, 'complexity': 'medium', 'category': 'standard'},
            'gambit': {'base_contamination': 0.05, 'complexity': 'medium', 'category': 'standard'},
            'titan': {'base_contamination': 0.05, 'complexity': 'medium', 'category': 'standard'},
            'r2d2': {'base_contamination': 0.05, 'complexity': 'medium', 'category': 'standard'},
            
            # Micro-services and internal services
            'fa5': {'base_contamination': 0.08, 'complexity': 'low', 'category': 'micro'},
            
            # Admin services (typically lower traffic)
            'm2-fr-adm': {'base_contamination': 0.06, 'complexity': 'low', 'category': 'admin'},
            'm2-it-adm': {'base_contamination': 0.06, 'complexity': 'low', 'category': 'admin'},
            'm2-bb-adm': {'base_contamination': 0.06, 'complexity': 'low', 'category': 'admin'},
            
            # Core services
            'm2-bb': {'base_contamination': 0.04, 'complexity': 'medium', 'category': 'core'},
            'm2-fr': {'base_contamination': 0.04, 'complexity': 'medium', 'category': 'core'},
            'm2-it': {'base_contamination': 0.04, 'complexity': 'medium', 'category': 'core'},
        }
        
        # Check if service is known
        if self.service_name in service_params:
            base_config = service_params[self.service_name]
            print(f"   🎯 Using known configuration for {self.service_name} ({base_config['category']})")
        else:
            # Smart pattern-based defaults for unknown services
            base_config = self._detect_service_pattern()
            print(f"   🔍 Auto-detected pattern for {self.service_name}: {base_config['category']}")
        
        # Adjust contamination based on data characteristics
        base_contamination = base_config['base_contamination']
        
        # Data-driven adjustments
        if variability > 3.0:  # Very high variability
            contamination = min(0.20, base_contamination * 2.0)
            adjustment_reason = "high variability"
        elif variability > 2.0:  # High variability
            contamination = min(0.15, base_contamination * 1.5)
            adjustment_reason = "moderate variability"
        elif variability < 0.5:  # Very stable data
            contamination = max(0.01, base_contamination * 0.7)
            adjustment_reason = "low variability"
        else:
            contamination = base_contamination
            adjustment_reason = "normal variability"
        
        # Adjust n_estimators based on complexity and data size
        complexity = base_config['complexity']
        
        if complexity == 'high' and data_size > 10000:
            n_estimators = 400
        elif complexity == 'high' and data_size > 5000:
            n_estimators = 300
        elif complexity == 'medium' and data_size > 8000:
            n_estimators = 250
        elif complexity == 'medium' and data_size > 3000:
            n_estimators = 200
        elif complexity == 'low' and data_size > 5000:
            n_estimators = 150
        else:
            # Default based on data size
            if data_size > 15000:
                n_estimators = 300
            elif data_size > 5000:
                n_estimators = 200
            elif data_size > 1000:
                n_estimators = 150
            else:
                n_estimators = 100
        
        # Adjust max_samples for large datasets
        if data_size > 5000:
            max_samples = min(1.0, 4000 / data_size)
        else:
            max_samples = 'auto'
        
        optimal_params = {
            'isolation_forest': {
                'contamination': round(contamination, 3),
                'n_estimators': n_estimators,
                'max_samples': max_samples,
                'bootstrap': data_size > 10000  # Use bootstrap for large datasets
            },
            'metadata': {
                'service_category': base_config['category'],
                'complexity': complexity,
                'data_size': data_size,
                'variability': round(variability, 3),
                'adjustment_reason': adjustment_reason,
                'auto_detected': self.service_name not in service_params
            }
        }
        
        # Log the parameter selection
        print(f"   📊 Parameters: contamination={contamination:.3f} ({adjustment_reason})")
        print(f"   🌳 Isolation Forest: {n_estimators} estimators, max_samples={max_samples}")
        
        return optimal_params
    
    def _detect_service_pattern(self) -> Dict:
        """
        Smart pattern detection for unknown services based on naming conventions.
        Returns appropriate default configuration.
        """
        service_lower = self.service_name.lower()
        
        # Pattern detection rules
        if any(pattern in service_lower for pattern in ['api', 'gateway', 'proxy']):
            return {
                'base_contamination': 0.03,
                'complexity': 'high', 
                'category': 'api_gateway'
            }
        
        elif any(pattern in service_lower for pattern in ['admin', 'adm', 'management', 'mgmt']):
            return {
                'base_contamination': 0.06,
                'complexity': 'low',
                'category': 'admin'
            }
        
        elif any(pattern in service_lower for pattern in ['worker', 'job', 'task', 'queue', 'processor']):
            return {
                'base_contamination': 0.08,
                'complexity': 'low',
                'category': 'background_service'
            }
        
        elif any(pattern in service_lower for pattern in ['micro', 'util', 'helper', 'tool']):
            return {
                'base_contamination': 0.07,
                'complexity': 'low',
                'category': 'micro_service'
            }
        
        elif any(pattern in service_lower for pattern in ['db', 'database', 'storage', 'cache']):
            return {
                'base_contamination': 0.04,
                'complexity': 'medium',
                'category': 'data_service'
            }
        
        elif any(pattern in service_lower for pattern in ['auth', 'login', 'security', 'oauth']):
            return {
                'base_contamination': 0.02,
                'complexity': 'high',
                'category': 'security_service'
            }
        
        elif any(pattern in service_lower for pattern in ['monitor', 'metric', 'log', 'trace']):
            return {
                'base_contamination': 0.05,
                'complexity': 'medium',
                'category': 'observability'
            }
        
        elif len(service_lower) <= 3:  # Short names like 'fa5', 'r2d2'
            return {
                'base_contamination': 0.07,
                'complexity': 'low',
                'category': 'legacy_or_internal'
            }
        
        elif any(pattern in service_lower for pattern in ['test', 'staging', 'dev']):
            return {
                'base_contamination': 0.10,
                'complexity': 'low',
                'category': 'development'
            }
        
        else:
            # Conservative default for unknown services
            return {
                'base_contamination': 0.05,
                'complexity': 'medium',
                'category': 'unknown_standard'
            }
    
    def train(self, features_df: pd.DataFrame):
        """Enhanced training with explainability data collection and adaptive hyperparameter tuning"""
        if features_df.empty:
            raise ValueError(f"No training data for {self.service_name}")
        
        self.feature_columns = features_df.columns.tolist()
        
        # Store training statistics for explainability
        self._calculate_training_statistics(features_df)
        
        # Core metrics for univariate analysis
        core_metrics = ['request_rate', 'application_latency', 'client_latency', 'database_latency', 'error_rate']
        
        # Train univariate models with adaptive parameters
        for metric in core_metrics:
            if metric in features_df.columns:
                self._train_univariate_model(metric, features_df[metric])
        
        # Train multivariate model on core metrics with PCA
        available_core_metrics = [m for m in core_metrics if m in features_df.columns]
        if len(available_core_metrics) >= 3:
            self._train_multivariate_model_improved(features_df[available_core_metrics])
        
        # Set pattern-based thresholds
        self._calculate_pattern_thresholds(features_df)
        
        # Calculate feature importance for explainability
        self._calculate_feature_importance()

        print(f"✅ Trained enhanced anomaly detector for {self.service_name}")
        print(f"   - Univariate models: {len([k for k in self.models.keys() if 'isolation' in k])}")
        print(f"   - Multivariate model: {'✅ Enhanced Isolation Forest' if 'multivariate_detector' in self.models else '❌'}")
        print(f"   - Statistical correlations: ✅ Fast correlation detection enabled")
        print(f"   - Feature dimensions: {len(self.feature_columns)}")
        print(f"   - Explainability: Training stats for {len(self.training_statistics)} metrics")
        print(f"   - Zero-normal handling: {len([k for k, v in self.zero_statistics.items() if v.get('zero_percentage', 0) > 0.1])} metrics")
        print(f"   - Adaptive tuning: {'✅' if self.auto_tune else '❌'}")
        print(f"   - Performance: 🚀 Optimized for production scale")
        
    def _calculate_training_statistics(self, features_df: pd.DataFrame):
        """Enhanced training statistics with zero-normal metric awareness"""
        core_metrics = ['request_rate', 'application_latency', 'client_latency', 
                       'database_latency', 'error_rate']
        
        self.training_statistics = {}
        zero_normal_metrics = ['client_latency', 'database_latency']
        
        for metric in core_metrics:
            if metric in features_df.columns:
                data = features_df[metric].dropna()
                if len(data) > 0:
                    # Standard statistics
                    stats = {
                        'mean': float(data.mean()),
                        'median': float(data.median()),
                        'std': float(data.std()),
                        'min': float(data.min()),
                        'max': float(data.max()),
                        'p25': float(data.quantile(0.25)),
                        'p50': float(data.quantile(0.50)),
                        'p75': float(data.quantile(0.75)),
                        'p90': float(data.quantile(0.90)),
                        'p95': float(data.quantile(0.95)),
                        'p99': float(data.quantile(0.99)),
                        'count': len(data),
                        'coefficient_of_variation': float(data.std() / (data.mean() + 1e-8))
                    }
                    
                    # Add zero-normal specific statistics
                    if metric in zero_normal_metrics:
                        zero_count = (data == 0).sum()
                        zero_percentage = zero_count / len(data)
                        non_zero_data = data[data > 0]
                        
                        stats.update({
                            'zero_count': int(zero_count),
                            'zero_percentage': float(zero_percentage),
                            'is_zero_dominant': zero_percentage > 0.5,
                            'non_zero_mean': float(non_zero_data.mean()) if len(non_zero_data) > 0 else 0.0,
                            'non_zero_p95': float(non_zero_data.quantile(0.95)) if len(non_zero_data) > 0 else 0.0,
                            'non_zero_count': len(non_zero_data)
                        })
                        
                        # Store in zero_statistics for training method use
                        self.zero_statistics[metric] = {
                            'zero_percentage': float(zero_percentage),
                            'non_zero_mean': stats['non_zero_mean'],
                            'non_zero_p95': stats['non_zero_p95'],
                            'is_zero_dominant': stats['is_zero_dominant']
                        }
                    
                    # Calculate ranges
                    stats.update({
                        'typical_range': {
                            'lower': float(data.quantile(0.25)),
                            'upper': float(data.quantile(0.75))
                        },
                        'normal_range': {
                            'lower': float(data.mean() - 2 * data.std()),
                            'upper': float(data.mean() + 2 * data.std())
                        },
                        'outlier_bounds': {
                            'lower': float(data.quantile(0.05)),
                            'upper': float(data.quantile(0.95))
                        }
                    })
                    
                    self.training_statistics[metric] = stats
    
    def _calculate_feature_importance(self):
        """Calculate feature importance scores for explainability"""
        self.feature_importance = {}
        
        for metric_name in self.training_statistics.keys():
            stats = self.training_statistics[metric_name]
            
            # Use coefficient of variation as importance proxy
            cv = stats['coefficient_of_variation']
            
            # Determine impact level based on variability and business criticality
            if metric_name == 'error_rate':
                impact_level = 'critical'  # Error rate is always critical
            elif 'latency' in metric_name and cv > 0.3:
                impact_level = 'high'
            elif metric_name == 'request_rate' and cv > 0.5:
                impact_level = 'high'
            elif cv > 0.5:
                impact_level = 'medium'
            else:
                impact_level = 'low'
            
            self.feature_importance[metric_name] = {
                'variability_score': float(cv),
                'impact_level': impact_level,
                'business_criticality': self._get_business_criticality(metric_name)
            }
    
    def _get_business_criticality(self, metric_name: str) -> str:
        """Determine business criticality of metrics"""
        criticality_map = {
            'error_rate': 'critical - affects user experience directly',
            'application_latency': 'high - impacts user satisfaction',
            'request_rate': 'high - indicates service utilization',
            'database_latency': 'medium - affects backend performance',
            'client_latency': 'medium - affects service interactions'
        }
        return criticality_map.get(metric_name, 'low - monitoring metric')
    
    def _train_univariate_model(self, metric_name: str, data: pd.Series):
        """Enhanced training with optimal parameters and RobustScaler"""
        clean_data = data.dropna()
        clean_data = clean_data.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Handle zero-normal metrics
        zero_normal_metrics = ['client_latency', 'database_latency']
        is_zero_normal = metric_name in zero_normal_metrics
        
        # For zero-normal metrics, calculate zero percentage
        if is_zero_normal:
            zero_percentage = (clean_data == 0).mean()
            print(f"     📊 {metric_name}: {zero_percentage:.1%} zero values")
            
            # If >50% of values are zero, this is a zero-dominant metric
            if zero_percentage > 0.5:
                print(f"     🎯 {metric_name} is zero-dominant - adjusting training")
                # Only train on non-zero values for isolation forest
                non_zero_data = clean_data[clean_data > 0]
                if len(non_zero_data) < 50:
                    print(f"     ⚠️ Insufficient non-zero data for {metric_name}: {len(non_zero_data)} points")
                    # Store threshold-only model for zero-dominant metrics
                    self.thresholds[f'{metric_name}_zero_dominant'] = True
                    self.thresholds[f'{metric_name}_non_zero_p95'] = self.zero_statistics[metric_name]['non_zero_p95']
                    return
                clean_data = non_zero_data
        
        # Remove extreme outliers before training
        if len(clean_data) > 100:
            p999 = clean_data.quantile(0.999)
            p001 = clean_data.quantile(0.001)
            clean_data = clean_data[(clean_data <= p999) & (clean_data >= p001)]
        
        clean_df = pd.DataFrame(clean_data.values, columns=[metric_name])
        
        if len(clean_df) < 50:
            print(f"     ⚠️ Insufficient clean data for {metric_name}: {len(clean_df)} points")
            return
        
        try:
            # Get optimal parameters for this service and data
            if self.auto_tune:
                optimal_params = self._get_optimal_params(clean_df)
                if_params = optimal_params['isolation_forest']
            else:
                # Use default parameters
                if_params = {
                    'contamination': 0.05,
                    'n_estimators': 200,
                    'max_samples': 'auto',
                    'bootstrap': False
                }
                optimal_params = {'isolation_forest': if_params, 'metadata': {'auto_detected': False}}
            
            # Use RobustScaler instead of StandardScaler for better outlier handling
            scaler = RobustScaler()
            scaled_data = scaler.fit_transform(clean_df)
            
            if np.std(scaled_data) < 1e-10:
                print(f"     ⚠️ Constant values detected for {metric_name}, skipping model training")
                return
            
            model = IsolationForest(
                contamination=if_params['contamination'],
                n_estimators=if_params['n_estimators'],
                max_samples=if_params['max_samples'],
                bootstrap=if_params.get('bootstrap', False),
                random_state=42,
                n_jobs=-1  # Use all CPU cores for faster training
            )
            model.fit(scaled_data)
            
            self.models[f'{metric_name}_isolation'] = model
            self.scalers[f'{metric_name}_scaler'] = scaler
            self.optimal_params[f'{metric_name}_params'] = optimal_params
            
            # Calculate thresholds from original clean data
            clean_values = clean_df[metric_name].values
            self.thresholds[f'{metric_name}_p95'] = np.percentile(clean_values, 95)
            self.thresholds[f'{metric_name}_p99'] = np.percentile(clean_values, 99)
            self.thresholds[f'{metric_name}_p90'] = np.percentile(clean_values, 90)
            self.thresholds[f'{metric_name}_median'] = np.percentile(clean_values, 50)
            
            print(f"     ✅ Trained {metric_name} model with {len(clean_df)} samples")
            if self.auto_tune:
                print(f"        📈 Optimized: contamination={if_params['contamination']:.3f}, estimators={if_params['n_estimators']}")
            
        except Exception as e:
            print(f"     ❌ Failed to train {metric_name} model: {e}")
            return
    
    def _train_multivariate_model_improved(self, core_metrics_df: pd.DataFrame):
        """Train enhanced multivariate anomaly detection using Isolation Forest ensemble"""
        clean_data = core_metrics_df.dropna()
        clean_data = clean_data.replace([np.inf, -np.inf], np.nan).dropna()
    
        if len(clean_data) < 100:
            print(f"     ⚠️ Insufficient clean data for multivariate model: {len(clean_data)} points")
            return
    
        try:
            original_features = clean_data.columns.tolist()
            constant_cols = clean_data.std() < 1e-10
        
            if constant_cols.any():
                constant_features = clean_data.columns[constant_cols].tolist()
                print(f"     ⚠️ Constant columns detected: {constant_features}")
            
                if 'error_rate' in constant_features:
                    error_mean = clean_data['error_rate'].mean()
                    if error_mean < 1e-6:
                        noise = np.random.normal(0, 1e-6, len(clean_data))
                        clean_data = clean_data.copy()
                        clean_data['error_rate'] = clean_data['error_rate'] + noise
                        print(f"     Added minimal noise to error_rate to preserve multivariate detection")
                        constant_cols = clean_data.std() < 1e-10
            
                non_constant_mask = ~constant_cols
                if not non_constant_mask.all():
                    remaining_constant = clean_data.columns[constant_cols].tolist()
                    print(f"     Removing truly constant columns: {remaining_constant}")
                    clean_data = clean_data.iloc[:, non_constant_mask]
        
            if clean_data.shape[1] < 2:
                print(f"     ⚠️ Insufficient features for multivariate model after cleaning")
                return
        
            # Get optimal parameters for multivariate model
            if self.auto_tune:
                optimal_params = self._get_optimal_params(clean_data)
                if_params = optimal_params['isolation_forest']
            else:
                if_params = {
                    'contamination': 0.05,
                    'n_estimators': 200,
                    'max_samples': 'auto',
                    'bootstrap': False
                }
                optimal_params = {'isolation_forest': if_params}
        
            # Use RobustScaler for multivariate model
            scaler = RobustScaler()
            scaled_data = scaler.fit_transform(clean_data)
        
            self.multivariate_feature_names = clean_data.columns.tolist()
        
            print(f"     Using {len(self.multivariate_feature_names)} features for multivariate model: {self.multivariate_feature_names}")
        
            # Enhanced Isolation Forest for multivariate detection
            # Use higher n_estimators for better multivariate performance
            multivariate_estimators = min(500, if_params['n_estimators'] * 2)
        
            detector = IsolationForest(
                contamination=if_params['contamination'],
                n_estimators=multivariate_estimators,
                max_samples=if_params['max_samples'],
                bootstrap=if_params.get('bootstrap', False),
                random_state=42,
                n_jobs=-1  # Use all CPU cores
            )
            detector.fit(scaled_data)
        
            # Store models (keeping same naming for compatibility)
            self.models['multivariate_detector'] = detector
            self.scalers['multivariate_scaler'] = scaler
            self.optimal_params['multivariate_params'] = optimal_params
        
            print(f"     ✅ Trained Enhanced Isolation Forest multivariate model on {len(scaled_data)} samples")
            print(f"        🌳 Using {multivariate_estimators} estimators for robust multivariate detection")
            if self.auto_tune:
                print(f"        📈 Optimized: contamination={if_params['contamination']:.3f}")
        
        except Exception as e:
            print(f"     ❌ Failed to train multivariate model: {e}")
            return
    
    # Step 3: Replace _detect_multivariate_anomalies method entirely

    def _detect_multivariate_anomalies(self, metrics: Dict) -> Dict:
        """Detect multivariate anomalies using enhanced Isolation Forest"""
        anomalies = {}
    
        if 'multivariate_detector' not in self.models:
            return anomalies
    
        try:
            core_metrics = ['request_rate', 'application_latency', 'client_latency', 'database_latency', 'error_rate']
        
            metric_values = []
            available_metrics = []
        
            if hasattr(self, 'multivariate_feature_names') and self.multivariate_feature_names:
                for feature_name in self.multivariate_feature_names:
                    if feature_name in core_metrics:
                        available_metrics.append(feature_name)
                        metric_values.append(metrics.get(feature_name, 0.0))
            else:
                for metric in core_metrics:
                    if f'{metric}_isolation' in self.models:
                        available_metrics.append(metric)
                        metric_values.append(metrics.get(metric, 0.0))
        
            if len(metric_values) < 2:
                return anomalies
        
            expected_features = self.scalers['multivariate_scaler'].n_features_in_
            actual_features = len(metric_values)
        
            if actual_features != expected_features:
                return anomalies
        
            scaler = self.scalers['multivariate_scaler']
            metrics_df = self._ensure_consistent_feature_format([metric_values], available_metrics)
        
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scaled_data = scaler.transform(metrics_df)
        
            # Direct Isolation Forest detection (no PCA needed)
            detector = self.models['multivariate_detector']
        
            prediction = detector.predict(scaled_data)[0]
            if prediction == -1:
                score = detector.decision_function(scaled_data)[0]
            
                # Enhanced severity calculation based on score
                if score < -0.6:
                    severity = 'critical'
                elif score < -0.4:
                    severity = 'high'
                else:
                    severity = 'medium'
            
                anomalies['multivariate_isolation'] = {
                    'type': 'multivariate',
                    'severity': severity,
                    'score': float(score),
                    'description': f'Unusual combination of metrics detected by Enhanced Isolation Forest ({len(available_metrics)} metrics: {", ".join(available_metrics)})',
                    'detection_method': 'enhanced_isolation_forest'
                }
        
        except Exception as e:
            pass
    
        return anomalies
        
    def _detect_service_patterns(self, metrics: Dict) -> Dict:
        """Detect problematic service patterns with operational thresholds"""
        patterns = {}
    
        if not hasattr(self, 'pattern_thresholds'):
            return patterns
    
        req_rate = metrics.get('request_rate', 0)
        app_latency = metrics.get('application_latency', 0)
        error_rate = metrics.get('error_rate', 0)
    
        # OPERATIONAL thresholds based on real-world SLA expectations
        CRITICAL_ERROR_RATE = 0.05    # 5% - truly problematic
        HIGH_ERROR_RATE = 0.02        # 2% - warning level
        CRITICAL_LATENCY_MS = 2000    # 2 seconds - unacceptable for most services
        HIGH_LATENCY_MS = 1000        # 1 second - concerning
    
        # Get statistical thresholds for traffic patterns
        req_p95 = self.pattern_thresholds.get('request_rate', {}).get('p95', float('inf'))
        lat_p95 = self.pattern_thresholds.get('application_latency', {}).get('p95', float('inf'))
        lat_median = self.pattern_thresholds.get('application_latency', {}).get('median', 0)
    
        # System overload: High traffic + High latency + HIGH errors (operational thresholds)
        if req_rate > req_p95 and app_latency > HIGH_LATENCY_MS and error_rate > CRITICAL_ERROR_RATE:
            patterns['system_overload'] = {
                'type': 'pattern',
                'severity': 'critical',
                'description': f'System overload: high traffic ({req_rate:.1f} req/s), latency ({app_latency:.0f}ms), and error rate ({error_rate:.2%})',
                'metrics': {
                    'request_rate': req_rate,
                    'application_latency': app_latency,
                    'error_rate': error_rate
                }
            }
    
        # Cascading failure: Only flag if error rate is operationally concerning
        elif error_rate > CRITICAL_ERROR_RATE and app_latency < lat_median:
            patterns['cascading_failure'] = {
                'type': 'pattern',
                'severity': 'critical',
                'description': f'Cascading failure: high error rate ({error_rate:.2%}) with fast failures',
                'metrics': {
                    'error_rate': error_rate,
                    'application_latency': app_latency
                }
            }
    
        # Performance degradation: Use operational latency thresholds
        elif app_latency > CRITICAL_LATENCY_MS and error_rate < HIGH_ERROR_RATE:
            patterns['performance_degradation'] = {
                'type': 'pattern',
                'severity': 'critical',
                'description': f'Critical performance degradation: {app_latency:.0f}ms latency (>{CRITICAL_LATENCY_MS}ms threshold)',
                'metrics': {
                    'application_latency': app_latency,
                    'error_rate': error_rate
                }
            }
        elif app_latency > HIGH_LATENCY_MS and error_rate < HIGH_ERROR_RATE:
            patterns['performance_degradation'] = {
                'type': 'pattern',
                'severity': 'high',
                'description': f'Performance degradation: {app_latency:.0f}ms latency (>{HIGH_LATENCY_MS}ms threshold)',
                'metrics': {
                    'application_latency': app_latency,
                    'error_rate': error_rate
                }
            }
    
        return patterns

    def _calculate_pattern_thresholds(self, features_df: pd.DataFrame):
        """Calculate thresholds for pattern-based detection"""
        required_metrics = ['request_rate', 'application_latency', 'error_rate']
    
        if not all(metric in features_df.columns for metric in required_metrics):
            return
    
        self.pattern_thresholds = {}
    
        for metric in required_metrics:
            data = features_df[metric].dropna()
            self.pattern_thresholds[metric] = {
                'p90': np.percentile(data, 90),
                'p95': np.percentile(data, 95),
                'p99': np.percentile(data, 99),
                'median': np.percentile(data, 50)
            }

    def _detect_statistical_correlations(self, metrics: Dict) -> Dict:
        """Fast statistical correlation detection for unusual metric combinations"""
        correlations = {}
    
        # Extract key metrics
        req_rate = metrics.get('request_rate', 0)
        app_latency = metrics.get('application_latency', 0)
        client_latency = metrics.get('client_latency', 0)
        db_latency = metrics.get('database_latency', 0)
        error_rate = metrics.get('error_rate', 0)
    
        # Get historical statistics for comparison
        if not hasattr(self, 'training_statistics') or not self.training_statistics:
            return correlations
    
        try:
            # Rule 1: High latency but low error rate (performance issue without failures)
            app_stats = self.training_statistics.get('application_latency', {})
            error_stats = self.training_statistics.get('error_rate', {})
        
            if app_stats and error_stats:
                high_latency_threshold = app_stats.get('p95', 1000) * 2
                low_error_threshold = error_stats.get('p75', 0.01)
            
                if app_latency > high_latency_threshold and error_rate <= low_error_threshold:
                    correlations['high_latency_low_errors'] = {
                        'type': 'correlation',
                        'severity': 'medium',
                        'description': f'High latency ({app_latency:.0f}ms) without corresponding errors - possible resource contention',
                        'metrics': {'application_latency': app_latency, 'error_rate': error_rate}
                    }
        
            # Rule 2: Low traffic but high resource usage (inefficiency)
            req_stats = self.training_statistics.get('request_rate', {})
            if req_stats and app_stats:
                low_traffic_threshold = req_stats.get('p25', 1.0)
                high_latency_threshold = app_stats.get('p75', 200)
            
                if req_rate < low_traffic_threshold and app_latency > high_latency_threshold:
                    correlations['low_traffic_high_latency'] = {
                        'type': 'correlation', 
                        'severity': 'medium',
                        'description': f'Low traffic ({req_rate:.1f} req/s) with high latency ({app_latency:.0f}ms) - inefficient resource usage',
                        'metrics': {'request_rate': req_rate, 'application_latency': app_latency}
                    }
        
            # Rule 3: External service dependency issues (high client latency affecting app latency)
            client_stats = self.training_statistics.get('client_latency', {})
            if client_stats and app_stats and client_latency > 0:
                high_client_threshold = client_stats.get('p90', 500)
            
                # If client latency is high and comprises >60% of total app latency
                if (client_latency > high_client_threshold and 
                    app_latency > 0 and 
                    (client_latency / app_latency) > 0.6):
                    correlations['external_service_impact'] = {
                        'type': 'correlation',
                        'severity': 'high',
                        'description': f'External service latency ({client_latency:.0f}ms) dominating total latency - dependency issue',
                        'metrics': {'client_latency': client_latency, 'application_latency': app_latency}
                    }
        
            # Rule 4: Database bottleneck (high DB latency affecting app performance)
            db_stats = self.training_statistics.get('database_latency', {})
            if db_stats and app_stats and db_latency > 0:
                high_db_threshold = db_stats.get('p90', 300)
            
                if (db_latency > high_db_threshold and 
                    app_latency > 0 and 
                    (db_latency / app_latency) > 0.5):
                    correlations['database_bottleneck'] = {
                        'type': 'correlation',
                        'severity': 'high', 
                        'description': f'Database latency ({db_latency:.0f}ms) causing application slowdown - DB performance issue',
                        'metrics': {'database_latency': db_latency, 'application_latency': app_latency}
                    }
        
            # Rule 5: Anomalous zero patterns (should have latency but don't)
            if req_rate > 1.0 and app_latency > 100:  # Active service
                if client_latency == 0 and db_latency == 0:
                    # Check if this is historically normal
                    client_zero_pct = self.zero_statistics.get('client_latency', {}).get('zero_percentage', 0)
                    db_zero_pct = self.zero_statistics.get('database_latency', {}).get('zero_percentage', 0)
                
                    # If historically these weren't often zero, this might be anomalous
                    if client_zero_pct < 0.8 and db_zero_pct < 0.8:
                        correlations['unexpected_zero_latencies'] = {
                            'type': 'correlation',
                            'severity': 'low',
                            'description': 'Active service with zero external/DB latencies - unusual caching pattern or monitoring issue',
                            'metrics': {'request_rate': req_rate, 'client_latency': client_latency, 'database_latency': db_latency}
                        }
    
        except Exception as e:
            # Fail silently for correlation detection
            pass
    
        return correlations

    
    def save_model_secure(self, model_directory: str, metadata: Dict = None):
        """Save model with enhanced metadata including explainability data and optimal parameters"""
        model_dir = Path(model_directory)
        model_dir.mkdir(exist_ok=True)
    
        service_dir = model_dir / self.service_name
        service_dir.mkdir(exist_ok=True)
    
        # Save model components
        if self.models:
            joblib.dump(self.models, service_dir / "models.joblib")
    
        if self.scalers:
            joblib.dump(self.scalers, service_dir / "scalers.joblib")
    
        # Enhanced model data with explainability, zero-normal handling, and optimal parameters
        model_data = {
            'service_name': self.service_name,
            'feature_columns': self.feature_columns,
            'thresholds': self.thresholds,
            'pattern_thresholds': getattr(self, 'pattern_thresholds', {}),
            'multivariate_feature_names': getattr(self, 'multivariate_feature_names', []),
            'training_statistics': self.training_statistics,  # Explainability
            'feature_importance': self.feature_importance,    # Explainability
            'zero_statistics': self.zero_statistics,         # Zero-normal handling
            'optimal_params': self.optimal_params,           # NEW: Adaptive hyperparameters
            'auto_tune': self.auto_tune,                     # NEW: Auto-tuning flag
            'metadata': metadata or {}
        }
    
        with open(service_dir / "model_data.json", 'w') as f:
            json.dump(model_data, f, indent=2, default=str)
    
        print(f"💾 Enhanced model saved for {self.service_name} with:")
        print(f"   🧠 Explainability and zero-normal handling")
        print(f"   🎯 Adaptive hyperparameter tuning: {'✅' if self.auto_tune else '❌'}")
        print(f"   📊 RobustScaler for better outlier handling")
        return service_dir
    
    @classmethod
    def load_model_secure(cls, model_directory: str, service_name: str, auto_tune: bool = True):
        """Load enhanced model with explainability, zero-normal data, and optimal parameters"""
        service_dir = Path(model_directory) / service_name
    
        if not service_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {service_dir}")
    
        # Create new instance with auto_tune setting
        detector = cls(service_name, auto_tune=auto_tune)
    
        # Load JSON data
        with open(service_dir / "model_data.json", 'r') as f:
            model_data = json.load(f)
    
        detector.feature_columns = model_data['feature_columns']
        detector.thresholds = model_data['thresholds']
        detector.pattern_thresholds = model_data.get('pattern_thresholds', {})
        detector.multivariate_feature_names = model_data.get('multivariate_feature_names', [])
        detector.model_metadata = model_data.get('metadata', {})
    
        # Load explainability data
        detector.training_statistics = model_data.get('training_statistics', {})
        detector.feature_importance = model_data.get('feature_importance', {})
    
        # Load zero-normal handling data
        detector.zero_statistics = model_data.get('zero_statistics', {})
    
        # NEW: Load optimal parameters and auto-tune setting
        detector.optimal_params = model_data.get('optimal_params', {})
        detector.auto_tune = model_data.get('auto_tune', auto_tune)
    
        # Load joblib components
        try:
            detector.models = joblib.load(service_dir / "models.joblib")
        except FileNotFoundError:
            detector.models = {}
    
        try:
            detector.scalers = joblib.load(service_dir / "scalers.joblib")
        except FileNotFoundError:
            detector.scalers = {}
    
        return detector
    
    # Helper methods for backward compatibility and service classification
    def _is_high_traffic_service(self) -> bool:
        """Helper method to identify high-traffic services"""
        high_traffic_patterns = [
            'booking', 'search', 'api', 'mobile', 'gateway', 'proxy'
        ]
        return any(pattern in self.service_name.lower() for pattern in high_traffic_patterns)

    def _is_micro_service(self) -> bool:
        """Helper method to identify micro-services"""
        micro_patterns = [
            'fa5', 'micro', 'util', 'helper', 'worker', 'job', 'task'
        ]
        return any(pattern in self.service_name.lower() for pattern in micro_patterns)

    def _is_admin_service(self) -> bool:
        """Helper method to identify admin services"""
        admin_patterns = [
            'admin', 'adm', 'management', 'mgmt', 'console'
        ]
        return any(pattern in self.service_name.lower() for pattern in admin_patterns)


    def get_model_summary(self) -> Dict:
        """Get a comprehensive summary of the model configuration and performance"""
        summary = {
            'service_name': self.service_name,
            'auto_tune_enabled': self.auto_tune,
            'models_trained': len(self.models),
            'scalers_available': len(self.scalers),
            'explainability_enabled': len(self.training_statistics) > 0,
            'zero_normal_handling': len(self.zero_statistics) > 0,
            'optimal_params_available': len(self.optimal_params) > 0,
            'feature_columns': self.feature_columns,
            'training_statistics_count': len(self.training_statistics),
            'feature_importance_count': len(self.feature_importance),
            'detection_methods': {
                'univariate_isolation_forest': len([k for k in self.models.keys() if 'isolation' in k and 'multivariate' not in k]),
                'enhanced_multivariate_isolation_forest': 'multivariate_detector' in self.models,
                'statistical_correlations': True,  # Always available
                'pattern_based_detection': hasattr(self, 'pattern_thresholds'),
                'zero_normal_specialized': len(self.zero_statistics) > 0
            },
            'performance_optimized': True,  # Now optimized for production
            'scalability': 'excellent'  # Linear scaling with Isolation Forest
        }
    
        # Add parameter summary if available
        if self.optimal_params:
            param_summary = {}
            for param_key, params in self.optimal_params.items():
                if isinstance(params, dict) and 'metadata' in params:
                    param_summary[param_key] = {
                        'service_category': params['metadata'].get('service_category', 'unknown'),
                        'complexity': params['metadata'].get('complexity', 'unknown'),
                        'auto_detected': params['metadata'].get('auto_detected', False)
                    }
            summary['parameter_summary'] = param_summary
    
        return summary
        
    def validate_configuration(self) -> Dict:
        """Validate the current model configuration and suggest improvements"""
        validation_results = {
            'status': 'valid',
            'warnings': [],
            'recommendations': [],
            'score': 0
        }
    
        # Check basic model availability
        if not self.models:
            validation_results['status'] = 'invalid'
            validation_results['warnings'].append("No trained models available")
            return validation_results
    
        validation_results['score'] += 20  # Base score for having models
    
        # Check explainability features
        if self.training_statistics:
            validation_results['score'] += 20
        else:
            validation_results['warnings'].append("Explainability features not available")
            validation_results['recommendations'].append("Retrain model to enable explainability features")
    
        # Check zero-normal handling
        if self.zero_statistics:
            validation_results['score'] += 15
        else:
            validation_results['recommendations'].append("Consider retraining to enable zero-normal metrics handling")
    
        # Check adaptive tuning
        if self.auto_tune and self.optimal_params:
            validation_results['score'] += 25
        elif self.auto_tune:
            validation_results['warnings'].append("Auto-tuning enabled but no optimal parameters found")
            validation_results['recommendations'].append("Retrain model to apply adaptive hyperparameter tuning")
        else:
            validation_results['recommendations'].append("Enable auto_tune=True for better performance")
    
        # Check scaler type (if we can detect it)
        robust_scalers = sum(1 for k in self.scalers.keys() if 'scaler' in k)
        if robust_scalers > 0:
            validation_results['score'] += 20  # Assume RobustScaler if recently trained
    
        # Final assessment
        if validation_results['score'] >= 80:
            validation_results['status'] = 'excellent'
        elif validation_results['score'] >= 60:
            validation_results['status'] = 'good'
        elif validation_results['score'] >= 40:
            validation_results['status'] = 'fair'
        else:
            validation_results['status'] = 'needs_improvement'
    
        return validation_results

    def detect_anomalies(self, current_metrics: Dict) -> Dict:
        """Enhanced anomaly detection with business-aware request rate rules"""
        anomalies = {}
    
        # Sanitize input metrics
        clean_metrics = {}
        for key, value in current_metrics.items():
            if pd.isna(value) or np.isinf(value):
                clean_metrics[key] = 0.0
            else:
                clean_metrics[key] = float(value)
    
        # 1. FIRST: Apply business-aware request rate detection
        request_rate_anomalies = self._enhance_request_rate_anomaly_detection(clean_metrics)
        anomalies.update(request_rate_anomalies)
    
        # 2. Univariate anomaly detection with zero-normal handling (for non-request_rate metrics)
        for metric_name, value in clean_metrics.items():
            if value is None:
                continue
        
            # Skip request_rate for ML detection if business rules already applied
            if metric_name == 'request_rate' and request_rate_anomalies:
                continue  # Business rules take precedence
        
            # Handle zero-normal metrics
            zero_normal_metrics = ['client_latency', 'database_latency']
            is_zero_normal = metric_name in zero_normal_metrics
        
            if is_zero_normal and value == 0:
                # For zero-normal metrics, 0 is always acceptable
                continue
        
            model_key = f'{metric_name}_isolation'
            scaler_key = f'{metric_name}_scaler'
        
            # Check if this is a zero-dominant metric
            if self.thresholds.get(f'{metric_name}_zero_dominant', False):
                if value == 0:
                    # Zero is expected for zero-dominant metrics
                    continue
                else:
                    # Non-zero value in zero-dominant metric - check against non-zero thresholds
                    non_zero_p95 = self.thresholds.get(f'{metric_name}_non_zero_p95', float('inf'))
                    if value > non_zero_p95 * 3:  # More lenient threshold
                        anomalies[f'{metric_name}_elevated'] = {
                            'type': 'zero_dominant_elevated',
                            'severity': 'medium',
                            'value': value,
                            'threshold': non_zero_p95 * 3,
                            'description': f'{metric_name} elevated ({value:.1f}ms) - unusual for typically zero metric'
                        }
                    continue
        
            if model_key in self.models and scaler_key in self.scalers:
            
                # Special handling for error_rate with operational thresholds
                if metric_name == 'error_rate':
                    if value > 0.05:  # 5% critical threshold
                        anomalies[f'{metric_name}_critical'] = {
                            'type': 'threshold',
                            'severity': 'critical',
                            'value': value,
                            'threshold': 0.05,
                            'description': f'Error rate critically high: {value:.2%} (>5%)'
                        }
                    elif value > 0.02:  # 2% warning threshold
                        anomalies[f'{metric_name}_elevated'] = {
                            'type': 'threshold',
                            'severity': 'medium',
                            'value': value,
                            'threshold': 0.02,
                            'description': f'Error rate elevated: {value:.2%} (>2%) - monitor closely'
                        }
            
                # For client/database latency, only flag high values
                elif is_zero_normal:
                    p95_threshold = self.thresholds.get(f'{metric_name}_p95')
                    if p95_threshold and value > p95_threshold * 4.0:  # More lenient for zero-normal
                        anomalies[f'{metric_name}_high'] = {
                            'type': 'threshold',
                            'severity': 'medium',
                            'value': value,
                            'threshold': p95_threshold * 4.0,
                            'description': f'{metric_name} unusually high ({value:.1f}ms) - external service delays?'
                        }
            
                # For request_rate, only apply ML if no business rules triggered
                elif metric_name == 'request_rate':
                    # Use statistical threshold-based detection, but be more lenient
                    p99_threshold = self.thresholds.get(f'{metric_name}_p99')
                
                    # Only flag as anomaly if it's extremely high (beyond business rule thresholds)
                    # and wasn't already caught by business rules
                    if p99_threshold and value > p99_threshold * 5.0:  # Very lenient
                        anomalies[f'{metric_name}_extreme_ml'] = {
                            'type': 'threshold_ml',
                            'severity': 'low',  # Lower severity since business rules didn't trigger
                            'value': value,
                            'threshold': p99_threshold * 5.0,
                            'description': f'{metric_name} extremely high by ML standards but within business tolerance'
                        }
            
                else:
                    # Use statistical threshold-based detection for other metrics
                    p99_threshold = self.thresholds.get(f'{metric_name}_p99')
                    p95_threshold = self.thresholds.get(f'{metric_name}_p95')
                
                    if p99_threshold and value > p99_threshold * 2.0:
                        anomalies[f'{metric_name}_extreme'] = {
                            'type': 'threshold',
                            'severity': 'high',
                            'value': value,
                            'threshold': p99_threshold * 2.0,
                            'description': f'{metric_name} significantly exceeds 99th percentile'
                        }
                    elif p95_threshold and value > p95_threshold * 3.0:
                        anomalies[f'{metric_name}_high'] = {
                            'type': 'threshold',
                            'severity': 'medium',
                            'value': value,
                            'threshold': p95_threshold * 3.0,
                            'description': f'{metric_name} significantly exceeds 95th percentile'
                        }
            
                # ML-based detection with zero-normal awareness
                if not (is_zero_normal and value == 0):  # Skip ML detection for zero-normal zeros
                    try:
                        scaler = self.scalers[scaler_key]
                        model = self.models[model_key]
                    
                        value_df = self._ensure_consistent_feature_format([value], [metric_name])
                    
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            scaled_value = scaler.transform(value_df)
                    
                        if np.isfinite(scaled_value).all():
                            anomaly_score = model.decision_function(scaled_value)[0]
                        
                            # Adjust thresholds for different metrics
                            if metric_name == 'request_rate':
                                # Much more lenient for request_rate ML detection
                                threshold = -2.0  # Only flag very strong ML signals
                            elif is_zero_normal:
                                threshold = -0.5  # Much more lenient for zero-normal metrics
                            elif metric_name == 'error_rate':
                                threshold = -1.5
                            else:
                                threshold = -1.0
                        
                            if anomaly_score < threshold:
                                severity = 'high' if anomaly_score < threshold - 0.5 else 'medium'
                            
                                # Additional checks
                                if metric_name == 'error_rate' and value < 0.01:
                                    continue
                            
                                if is_zero_normal and value < 10:  # Less than 10ms is probably fine
                                    continue
                            
                                # For request_rate, lower the severity since business rules take precedence
                                if metric_name == 'request_rate':
                                    severity = 'low'  # Downgrade ML-detected request rate anomalies
                                
                                anomalies[f'{metric_name}_isolation'] = {
                                    'type': 'ml_isolation',
                                    'severity': severity,
                                    'score': float(anomaly_score),
                                    'description': f'{metric_name} detected as anomaly by ML model',
                                    'business_rule_priority': metric_name == 'request_rate'  # Flag for lower priority
                                }
                    
                    except Exception as e:
                        pass
    
        # 3. Multivariate anomaly detection with zero-normal awareness
        try:
            if not self._should_skip_multivariate_detection(clean_metrics):
                multivariate_anomalies = self._detect_multivariate_anomalies(clean_metrics)
                anomalies.update(multivariate_anomalies)
        except Exception as e:
            pass
    
        # 4. Pattern-based detection (updated to respect business rules)
        try:
            pattern_anomalies = self._detect_service_patterns_with_business_rules(clean_metrics, request_rate_anomalies)
            anomalies.update(pattern_anomalies)
        except Exception as e:
            pass
    
        return anomalies
    
        
    def detect_anomalies_with_context(self, current_metrics: Dict, 
                                    timestamp: datetime = None) -> Dict:
        """NEW: Enhanced anomaly detection with full explainability context"""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Get basic anomaly detection results
        try:
            anomalies = self.detect_anomalies(current_metrics)
        except Exception as e:
            # If basic detection fails, return minimal result
            return {
                'alert_type': 'anomaly_detected',
                'service': self.service_name,
                'timestamp': timestamp.isoformat(),
                'overall_severity': 'normal',
                'anomaly_count': 0,
                'current_metrics': current_metrics.copy(),
                'anomalies': [],
                'error': f'Detection failed: {str(e)}'
            }
        
        # Ensure anomalies is always a dict
        if not isinstance(anomalies, dict):
            anomalies = {}
        
        # Build enhanced result with explainability
        enhanced_result = {
            'alert_type': 'anomaly_detected',
            'service': self.service_name,
            'timestamp': timestamp.isoformat(),
            'overall_severity': self._calculate_overall_severity(anomalies),
            'anomaly_count': len(anomalies),
            'current_metrics': current_metrics.copy(),
            'explainable': True  # Mark as explainable
        }
        
        # Add explainability features only if training statistics are available
        if hasattr(self, 'training_statistics') and self.training_statistics:
            enhanced_result.update({
                'historical_context': self._get_historical_context(current_metrics),
                'metric_analysis': self._analyze_individual_metrics(current_metrics),
                'explanation': self._generate_explanation(current_metrics, anomalies),
                'recommended_actions': self._generate_recommendations(current_metrics, anomalies),
                'model_metadata': {
                    'training_samples': sum(stats.get('count', 0) for stats in self.training_statistics.values()),
                    'feature_importance': getattr(self, 'feature_importance', {}),
                    'model_types': list(set(k.split('_')[-1] for k in self.models.keys()))
                }
            })
        
        # Always include anomalies (enhanced if explainable, basic if not)
        enhanced_result['anomalies'] = self._enhance_anomaly_descriptions(anomalies, current_metrics)
        
        return enhanced_result

    
    def _should_skip_multivariate_detection(self, metrics: Dict) -> bool:
            """Check if multivariate detection should be skipped due to zero-normal patterns"""
        
            # Skip if both client and database latency are zero (cached operation)
            client_latency = metrics.get('client_latency', -1)
            database_latency = metrics.get('database_latency', -1)
        
            if client_latency == 0 and database_latency == 0:
                # Additional checks to ensure this is normal operation
                request_rate = metrics.get('request_rate', 0)
                app_latency = metrics.get('application_latency', 0)
                error_rate = metrics.get('error_rate', 0)
            
                # If service is handling requests with low latency and no errors,
                # this is likely normal cached operation
                if request_rate > 0.1 and app_latency < 500 and error_rate < 0.02:
                    return True
        
            # Also skip if only client_latency is zero but it was historically zero-dominant
            if client_latency == 0:
                if hasattr(self, 'zero_statistics'):
                    client_stats = self.zero_statistics.get('client_latency', {})
                    zero_percentage = client_stats.get('zero_percentage', 0)
                    # If >50% of training data was zero, this is expected
                    if zero_percentage > 0.5:
                        return True
        
            return False

    
    def _calculate_overall_severity(self, anomalies: Dict) -> str:
            """NEW: Calculate overall severity of all anomalies"""
            if not anomalies:
                return 'normal'
        
            severities = []
            for anomaly_data in anomalies.values():
                if isinstance(anomaly_data, dict):
                    severities.append(anomaly_data.get('severity', 'low'))
        
            if 'critical' in severities:
                return 'critical'
            elif 'high' in severities:
                return 'high'
            elif 'medium' in severities:
                return 'medium'
            else:
                return 'low'


        
    def _get_historical_context(self, current_metrics: Dict) -> Dict:
            """NEW: Get historical context for current metrics"""
            context = {}
        
            # Check if we have training statistics
            if not hasattr(self, 'training_statistics') or not self.training_statistics:
                return context
        
            for metric_name, current_value in current_metrics.items():
                if metric_name in self.training_statistics:
                    try:
                        stats = self.training_statistics[metric_name]
                    
                        # Calculate percentile position
                        percentile = self._calculate_percentile_position(current_value, stats)
                    
                        # Determine position relative to ranges
                        in_typical_range = stats['typical_range']['lower'] <= current_value <= stats['typical_range']['upper']
                        in_normal_range = stats['normal_range']['lower'] <= current_value <= stats['normal_range']['upper']
                    
                        context[metric_name] = {
                            'current_value': current_value,
                            'historical_stats': {
                                'mean': stats['mean'],
                                'median': stats['median'],
                                'typical_range': stats['typical_range'],
                                'p95': stats['p95'],
                                'p99': stats['p99']
                            },
                            'percentile_position': round(percentile, 1),
                            'vs_mean_ratio': current_value / (stats['mean'] + 1e-8),
                            'vs_median_ratio': current_value / (stats['median'] + 1e-8),
                            'std_deviations_from_mean': (current_value - stats['mean']) / (stats['std'] + 1e-8),
                            'in_typical_range': in_typical_range,
                            'in_normal_range': in_normal_range,
                            'interpretation': self._interpret_metric_position(current_value, stats)
                        }
                    
                        # Add zero-normal specific context
                        zero_normal_metrics = ['client_latency', 'database_latency']
                        if metric_name in zero_normal_metrics:
                            zero_percentage = stats.get('zero_percentage', 0)
                            context[metric_name]['zero_context'] = {
                                'is_zero': current_value == 0,
                                'historical_zero_percentage': f"{zero_percentage:.1%}",
                                'interpretation': "Normal - no external calls" if current_value == 0 and metric_name == 'client_latency' else
                                                "Normal - no DB calls or cached" if current_value == 0 and metric_name == 'database_latency' else
                                                f"Active - {metric_name.replace('_', ' ')}"
                            }
                        
                    except Exception as e:
                        # Skip this metric if there's an error
                        continue
        
            return context

        
    def _analyze_individual_metrics(self, current_metrics: Dict) -> Dict:
        """Enhanced analysis of each metric with dynamic business impact assessment"""
        analysis = {}

        for metric_name, current_value in current_metrics.items():
            if metric_name in self.training_statistics:
                stats = self.training_statistics[metric_name]
            
                # Calculate percentile position using the new method
                percentile_position = self._calculate_percentile_position(current_value, stats)
            
                # Determine anomaly severity based on statistical boundaries
                z_score = (current_value - stats['mean']) / (stats['std'] + 1e-8)
            
                # Enhanced severity classification
                if abs(z_score) <= 2:
                    severity = 'normal'
                elif abs(z_score) <= 3:
                    severity = 'warning_statistical'
                else:
                    severity = 'critical_statistical'
            
                # Additional percentile-based severity
                if percentile_position <= 0.05 or percentile_position >= 0.95:
                    if severity == 'normal':
                        severity = 'warning_percentile'
                elif percentile_position <= 0.01 or percentile_position >= 0.99:
                    severity = 'critical_percentile'
            
                # Special handling for zero-normal metrics
                zero_normal_metrics = ['client_latency', 'database_latency']
                if metric_name in zero_normal_metrics and current_value == 0:
                    severity = 'normal_zero'
            
                # Calculate expected range display
                expected_range = f"{stats['p25']:.2f} - {stats['p75']:.2f}"
            
                # Use the new dynamic business impact assessment
                business_impact = self._assess_business_impact(metric_name, current_value, stats)
            
                analysis[metric_name] = {
                    'status': severity,
                    'expected_range': expected_range,
                    'current_value': current_value,
                    'percentile_position': percentile_position,
                    'z_score': z_score,
                    'deviation_magnitude': abs(z_score),
                    'business_impact': business_impact,  # Now dynamic and accurate!
                
                    # Additional context for explainability
                    'statistical_context': {
                        'mean': stats['mean'],
                        'median': stats.get('median', stats.get('p50', 0)),
                        'std_dev': stats['std'],
                        'p95': stats['p95'],
                        'p99': stats['p99']
                    },
                
                    # Comparison ratios
                    'vs_mean_ratio': current_value / (stats['mean'] + 1e-8),
                    'vs_median_ratio': current_value / (stats.get('median', stats.get('p50', 1)) + 1e-8),
                    'vs_p95_ratio': current_value / (stats['p95'] + 1e-8),
                
                    # Human-readable status
                    'human_readable_status': self._get_human_readable_status(
                        metric_name, current_value, stats, percentile_position, severity
                    )
                }

        return analysis

    def _get_human_readable_status(self, metric_name: str, current_value: float, 
                                  stats: Dict, percentile_position: float, severity: str) -> str:
        """Generate human-readable status description"""
    
        if severity == 'normal_zero':
            return f"{metric_name.replace('_', ' ').title()}: No activity (normal)"
    
        # Get appropriate comparison values
        p50 = stats.get('median', stats.get('p50', 0))
        p75 = stats.get('p75', 0)
        p95 = stats.get('p95', 0)
    
        # Format values appropriately
        if metric_name in ['error_rate']:
            current_display = f"{current_value:.3%}"
            p50_display = f"{p50:.3%}"
            p95_display = f"{p95:.3%}"
        elif 'latency' in metric_name:
            current_display = f"{current_value:.1f}ms"
            p50_display = f"{p50:.1f}ms"
            p95_display = f"{p95:.1f}ms"
        elif 'rate' in metric_name:
            current_display = f"{current_value:.1f}/s"
            p50_display = f"{p50:.1f}/s"
            p95_display = f"{p95:.1f}/s"
        else:
            current_display = f"{current_value:.2f}"
            p50_display = f"{p50:.2f}"
            p95_display = f"{p95:.2f}"
    
        metric_display = metric_name.replace('_', ' ').title()
    
        if percentile_position <= 0.25:
            return f"{metric_display}: Below normal ({current_display}, {percentile_position:.1%} percentile)"
        elif percentile_position <= 0.75:
            return f"{metric_display}: Normal ({current_display}, typical range)"
        elif percentile_position <= 0.95:
            return f"{metric_display}: Elevated ({current_display} vs typical {p75:.2f})"
        else:
            if percentile_position >= 0.99:
                comparison = f"vs 99th percentile {p95_display}"
            else:
                comparison = f"vs 95th percentile {p95_display}"
            return f"{metric_display}: Significantly elevated ({current_display} {comparison})"
    
    
    def _generate_explanation(self, current_metrics: Dict, anomalies: Dict) -> Dict:
            """NEW: Generate human-readable explanation of the anomaly"""
            explanations = []
        
            # Analyze metric patterns
            if 'request_rate' in current_metrics and 'application_latency' in current_metrics:
                rate = current_metrics['request_rate']
                latency = current_metrics['application_latency']
            
                if rate < 0.1 and latency > 500:
                    explanations.append("Very low traffic with high latency suggests resource constraints or blocking operations")
                elif rate > 10 and latency > 1000:
                    explanations.append("High traffic with high latency indicates system overload")
                elif current_metrics.get('error_rate', 0) > 0.05:
                    explanations.append("High error rate indicates service reliability issues")
        
            # NEW: Zero-normal specific explanations
            client_latency = current_metrics.get('client_latency', 0)
            database_latency = current_metrics.get('database_latency', 0)
        
            if client_latency == 0 and database_latency == 0:
                explanations.append("Client and database latencies are 0, no calls made")
            elif client_latency == 0:
                explanations.append("No external service calls detected - service running in cached/offline mode")
            elif database_latency == 0:
                explanations.append("No database queries detected")
        
            # Analyze multivariate patterns
            multivariate_anomalies = [k for k in anomalies.keys() if 'multivariate' in k]
            if multivariate_anomalies:
                explanations.append("Unusual combination of metrics detected - metrics are correlated in an unexpected way")
        
            return {
                'summary': "Anomalous behavior detected in service metrics",
                'detailed_explanations': explanations,
                'confidence_level': 'high' if len(anomalies) > 1 else 'medium'
            }

        
    def _generate_recommendations(self, current_metrics: Dict, anomalies: Dict) -> List[str]:
            """Enhanced recommendations with zero-normal metric awareness"""
            recommendations = []
        
            # Check for common patterns and suggest actions
            if current_metrics.get('application_latency', 0) > 1000:
                recommendations.append("Check application performance metrics, CPU usage, and memory utilization")
                recommendations.append("Review recent deployments or configuration changes")
        
            if current_metrics.get('request_rate', 0) < 0.1:
                recommendations.append("Verify upstream services are routing traffic correctly")
                recommendations.append("Check load balancer configuration and health checks")
        
            if current_metrics.get('error_rate', 0) > 0.02:
                recommendations.append("Review application logs for error patterns")
                recommendations.append("Check database connectivity and external service dependencies")
        
            # NEW: Zero-normal metric specific recommendations
            client_latency = current_metrics.get('client_latency', 0)
            if client_latency > 1000:
                recommendations.append("External service latency high - check third-party API status")
                recommendations.append("Consider implementing circuit breakers for external calls")
            elif client_latency == 0:
                recommendations.append("Service operating in cached/offline mode - verify expected behavior")
        
            database_latency = current_metrics.get('database_latency', 0)
            if database_latency > 500:
                recommendations.append("Database queries slow - check query performance and indexing")
                recommendations.append("Monitor database connection pool and resource utilization")
            elif database_latency == 0:
                recommendations.append("No database activity recorded.")
        
            # Add generic recommendations
            recommendations.extend([
                "Monitor service for next 15 minutes to see if pattern persists",
                "Check service dependencies and downstream impact",
                "Review infrastructure metrics (CPU, memory, disk, network)"
            ])
        
            return recommendations

    def _enhance_anomaly_descriptions(self, anomalies: Dict, current_metrics: Dict) -> List[Dict]:
            """NEW: Enhance anomaly descriptions with context"""
            enhanced_anomalies = []
        
            # Handle both dict and other formats
            if not isinstance(anomalies, dict):
                # If anomalies is not a dict, return empty list
                return []
        
            for anomaly_name, anomaly_data in anomalies.items():
                if isinstance(anomaly_data, dict):
                    enhanced_anomaly = anomaly_data.copy()
                
                    # Add comparison data for univariate anomalies
                    if 'isolation' in anomaly_name:
                        metric_name = anomaly_name.replace('_isolation', '')
                        if metric_name in self.training_statistics and metric_name in current_metrics:
                            stats = self.training_statistics[metric_name]
                            enhanced_anomaly['comparison_data'] = {
                                'current_value': current_metrics[metric_name],
                                'historical_mean': stats['mean'],
                                'historical_median': stats['median'],
                                'historical_p95': stats['p95'],
                                'typical_range': stats['typical_range']
                            }
                
                    # Add feature contributions for multivariate anomalies
                    elif 'multivariate' in anomaly_name:
                        enhanced_anomaly['feature_contributions'] = self._calculate_feature_contributions(current_metrics)
                
                    enhanced_anomalies.append(enhanced_anomaly)
        
            return enhanced_anomalies
    

    def _assess_business_impact(self, metric_name: str, current_value: float, stats: Dict) -> str:
        """
        Dynamic business impact assessment based on actual training statistics and service context
    
        Args:
            metric_name: Name of the metric (e.g., 'application_latency')
            current_value: Current observed value
            stats: Training statistics for this metric
    
        Returns:
            Human-readable business impact assessment
        """
        try:
            # Get service type for context-aware impact assessment
            service_type = self._get_service_type() if hasattr(self, '_get_service_type') else 'standard_service'
        
            # Extract statistical boundaries from training data
            mean = stats.get('mean', 0)
            median = stats.get('median', 0)
            std = stats.get('std', 0)
            p25 = stats.get('p25', 0)
            p75 = stats.get('p75', 0)
            p90 = stats.get('p90', 0)
            p95 = stats.get('p95', 0)
            p99 = stats.get('p99', 0)
        
            # Calculate dynamic thresholds based on actual data distribution
            normal_range_lower = max(0, mean - 2 * std)  # 2-sigma lower bound
            normal_range_upper = mean + 2 * std          # 2-sigma upper bound
        
            typical_range_lower = p25
            typical_range_upper = p75
        
            # Handle zero-normal metrics specially
            zero_normal_metrics = ['client_latency', 'database_latency']
            is_zero_normal = metric_name in zero_normal_metrics
        
            if is_zero_normal and current_value == 0:
                return "Impact: None - no external/database calls made (normal for this metric)"
        
            # Calculate deviation metrics
            if std > 0:
                z_score = (current_value - mean) / std
                deviation_from_median = abs(current_value - median) / (median + 1e-8)
            else:
                z_score = 0
                deviation_from_median = 0
        
            # Determine percentile position more accurately
            percentile_position = self._calculate_percentile_position(current_value, stats)
        
            # Business impact assessment logic by metric type
            if metric_name == 'application_latency':
                return self._assess_latency_impact(
                    current_value, stats, service_type, 'application', percentile_position, z_score
                )
            elif metric_name == 'client_latency':
                return self._assess_latency_impact(
                    current_value, stats, service_type, 'client', percentile_position, z_score
                )
            elif metric_name == 'database_latency':
                return self._assess_latency_impact(
                    current_value, stats, service_type, 'database', percentile_position, z_score
                )
            elif metric_name == 'error_rate':
                return self._assess_error_rate_impact(
                    current_value, stats, service_type, percentile_position, z_score
                )
            elif metric_name == 'request_rate':
                return self._assess_request_rate_impact(
                    current_value, stats, service_type, percentile_position, z_score
                )
            else:
                # Generic assessment for other metrics
                return self._assess_generic_impact(
                    metric_name, current_value, stats, percentile_position, z_score
                )
            
        except Exception as e:
            return f"Impact assessment unavailable: {str(e)}"

    def _assess_latency_impact(self, current_value: float, stats: Dict, service_type: str, 
                              latency_type: str, percentile_position: float, z_score: float) -> str:
        """Assess business impact for latency metrics using dynamic thresholds"""
    
        p50 = stats.get('p50', stats.get('median', 0))
        p75 = stats.get('p75', 0)
        p90 = stats.get('p90', 0)
        p95 = stats.get('p95', 0)
        p99 = stats.get('p99', 0)
        mean = stats.get('mean', 0)
    
        # Service-specific latency tolerance
        service_tolerances = {
            'critical_service': {'excellent': 0.75, 'good': 0.90, 'warning': 0.95, 'critical': 0.99},
            'admin_service': {'excellent': 0.80, 'good': 0.95, 'warning': 0.98, 'critical': 0.995},
            'micro_service': {'excellent': 0.70, 'good': 0.85, 'warning': 0.95, 'critical': 0.99},
            'core_service': {'excellent': 0.75, 'good': 0.90, 'warning': 0.95, 'critical': 0.99}
        }
    
        tolerance = service_tolerances.get(service_type, service_tolerances['critical_service'])
    
        # Determine impact based on percentile position and absolute values
        if percentile_position <= tolerance['excellent']:
            if latency_type == 'application':
                performance_desc = "excellent" if current_value < p50 else "good"
                return f"Impact: Minimal - {latency_type} performance {performance_desc} ({current_value:.1f}ms, {percentile_position:.1%} percentile)"
            else:
                return f"Impact: None - {latency_type} latency within normal range ({current_value:.1f}ms, {percentile_position:.1%} percentile)"
    
        elif percentile_position <= tolerance['good']:
            return f"Impact: Low - {latency_type} latency slightly elevated ({current_value:.1f}ms, {percentile_position:.1%} percentile, expected: ≤{p75:.1f}ms)"
    
        elif percentile_position <= tolerance['warning']:
            impact_level = "Medium" if latency_type == 'application' else "Low-Medium"
            expected_range = f"≤{p90:.1f}ms"
        
            if latency_type == 'application':
                user_impact = "may cause user frustration"
            elif latency_type == 'client':
                user_impact = "external service delays affecting performance"
            else:  # database
                user_impact = "database performance impacting response times"
            
            return f"Impact: {impact_level} - {latency_type} latency elevated ({current_value:.1f}ms vs expected {expected_range}), {user_impact}"
    
        elif percentile_position <= tolerance['critical']:
            impact_level = "High" if latency_type == 'application' else "Medium-High"
            expected_range = f"≤{p95:.1f}ms"
        
            if latency_type == 'application':
                user_impact = "likely causing user experience degradation"
            elif latency_type == 'client':
                user_impact = "significant external service delays"
            else:  # database
                user_impact = "database bottleneck affecting service performance"
            
            return f"Impact: {impact_level} - {latency_type} latency significantly elevated ({current_value:.1f}ms vs expected {expected_range}), {user_impact}"
    
        else:  # Above 99th percentile
            expected_range = f"≤{p99:.1f}ms"
            severity_multiplier = current_value / p99 if p99 > 0 else 1
        
            if latency_type == 'application':
                if severity_multiplier > 3:
                    user_impact = "severe user experience degradation, likely causing user abandonment"
                elif severity_multiplier > 2:
                    user_impact = "major user experience issues, high risk of user frustration"
                else:
                    user_impact = "significant user experience degradation"
            elif latency_type == 'client':
                user_impact = "critical external service performance issues"
            else:  # database
                user_impact = "critical database performance issues"
        
            return f"Impact: Critical - {latency_type} latency critically high ({current_value:.1f}ms vs expected {expected_range}, {severity_multiplier:.1f}x normal), {user_impact}"

    def _assess_error_rate_impact(self, current_value: float, stats: Dict, service_type: str, 
                                 percentile_position: float, z_score: float) -> str:
        """Assess business impact for error rate using dynamic thresholds and industry standards"""
    
        p75 = stats.get('p75', 0)
        p90 = stats.get('p90', 0)
        p95 = stats.get('p95', 0)
        p99 = stats.get('p99', 0)
        mean = stats.get('mean', 0)
    
        # Industry standard error rate thresholds (absolute)
        industry_thresholds = {
            'excellent': 0.001,     # 0.1% - excellent service
            'good': 0.005,          # 0.5% - good service
            'acceptable': 0.01,     # 1% - acceptable for most services
            'concerning': 0.02,     # 2% - concerning, needs attention
            'critical': 0.05        # 5% - critical, immediate action needed
        }
    
        # Service-specific tolerance adjustments
        if service_type == 'micro_service':
            # Micro-services can be more tolerant
            for key in industry_thresholds:
                industry_thresholds[key] *= 1.5
        elif service_type == 'critical_service':
            # Critical services should be stricter
            for key in industry_thresholds:
                industry_thresholds[key] *= 0.7
    
        # Compare against both historical patterns and industry standards
        historical_bad = current_value > max(p95, mean + 3 * stats.get('std', 0))
    
        if current_value <= industry_thresholds['excellent']:
            if percentile_position <= 0.75:
                return f"Impact: None - error rate excellent ({current_value:.3%}, industry standard: ≤{industry_thresholds['excellent']:.1%})"
            else:
                return f"Impact: Minimal - error rate within industry standards ({current_value:.3%}) but elevated for this service (typical: ≤{p75:.3%})"
    
        elif current_value <= industry_thresholds['good']:
            status = "elevated" if historical_bad else "acceptable"
            return f"Impact: Low - error rate {status} ({current_value:.3%}, industry standard: ≤{industry_thresholds['good']:.1%})"
    
        elif current_value <= industry_thresholds['acceptable']:
            return f"Impact: Medium - error rate concerning ({current_value:.3%}), approaching industry warning level ({industry_thresholds['acceptable']:.1%})"
    
        elif current_value <= industry_thresholds['concerning']:
            user_impact = "causing user experience degradation"
            return f"Impact: High - error rate elevated ({current_value:.3%} vs acceptable ≤{industry_thresholds['acceptable']:.1%}), {user_impact}"
    
        elif current_value <= industry_thresholds['critical']:
            user_impact = "causing significant user experience issues"
            return f"Impact: High-Critical - error rate high ({current_value:.3%} vs concerning ≤{industry_thresholds['concerning']:.1%}), {user_impact}"
    
        else:  # Above critical threshold
            severity_desc = "extremely high" if current_value > 0.1 else "critically high"
            return f"Impact: Critical - error rate {severity_desc} ({current_value:.3%} vs critical ≤{industry_thresholds['critical']:.1%}), severe service degradation"


    def _assess_request_rate_impact(self, current_value: float, stats: Dict, service_type: str, 
                               percentile_position: float, z_score: float) -> str:
        """
        Assess business impact for request rate with business-aware alerting rules:
        - High traffic (above max): Good until 2x max (Warning), 3x max (Critical)
        - Low traffic (below min but >0): Fine
        - Zero traffic: Critical
        """
    
        # Get statistical boundaries from actual training data
        minimum = stats.get('min', 0)
        maximum = stats.get('max', 0)
        p25 = stats.get('p25', 0)
        p75 = stats.get('p75', 0)
        p90 = stats.get('p90', 0)
        p95 = stats.get('p95', 0)
        mean = stats.get('mean', 0)
        median = stats.get('median', stats.get('p50', 0))
    
        # Handle zero traffic - always critical
        if current_value == 0:
            return "Impact: Critical - zero traffic detected, service may be down or unreachable"
    
        # Handle high traffic (above historical maximum)
        if current_value > maximum and maximum > 0:
            multiplier = current_value / maximum
        
            if multiplier >= 3.0:
                # 3x or more above maximum - Critical
                capacity_warning = "severe system overload risk, immediate capacity review needed"
                return f"Impact: Critical - traffic critically high ({current_value:.1f} req/s, {multiplier:.1f}x historical max {maximum:.1f}), {capacity_warning}"
        
            elif multiplier >= 2.0:
                # 2x above maximum - Warning  
                capacity_warning = "monitor system capacity and performance"
                return f"Impact: Warning - traffic significantly elevated ({current_value:.1f} req/s, {multiplier:.1f}x historical max {maximum:.1f}), {capacity_warning}"
        
            else:
                # Above max but less than 2x - Good (no issue)
                return f"Impact: None - high traffic within acceptable range ({current_value:.1f} req/s, {multiplier:.1f}x historical max {maximum:.1f}), positive load increase"
    
        # Handle low traffic (below minimum but above zero)
        elif current_value < minimum and minimum > 0:
            # Below minimum but not zero - Fine
            reduction_pct = ((minimum - current_value) / minimum * 100) if minimum > 0 else 0
            return f"Impact: None - reduced traffic but within operational range ({current_value:.1f} req/s, {reduction_pct:.0f}% below historical min {minimum:.1f})"
    
        # Handle normal traffic range (between min and max)
        else:
            # Determine position within normal range for context
            if current_value <= p25:
                traffic_level = "low-normal"
            elif current_value <= median:
                traffic_level = "moderate"
            elif current_value <= p75:
                traffic_level = "normal-high"
            elif current_value <= p90:
                traffic_level = "high-normal"
            else:
                traffic_level = "elevated-normal"
        
            return f"Impact: None - {traffic_level} traffic ({current_value:.1f} req/s, typical range: {minimum:.1f}-{maximum:.1f})"

    
    def _assess_generic_impact(self, metric_name: str, current_value: float, stats: Dict, 
                              percentile_position: float, z_score: float) -> str:
        """Generic impact assessment for any metric using statistical boundaries"""
    
        p25 = stats.get('p25', 0)
        p75 = stats.get('p75', 0)
        p95 = stats.get('p95', 0)
        mean = stats.get('mean', 0)
    
        if percentile_position <= 0.25:
            return f"Impact: Low - {metric_name} below normal ({current_value:.2f}, {percentile_position:.1%} percentile)"
    
        elif percentile_position <= 0.75:
            return f"Impact: None - {metric_name} within normal range ({current_value:.2f}, typical: {p25:.2f}-{p75:.2f})"
    
        elif percentile_position <= 0.95:
            return f"Impact: Low-Medium - {metric_name} elevated ({current_value:.2f} vs typical ≤{p75:.2f})"
    
        else:
            severity_multiplier = current_value / p95 if p95 > 0 else 1
            return f"Impact: High - {metric_name} significantly elevated ({current_value:.2f} vs expected ≤{p95:.2f}, {severity_multiplier:.1f}x typical)"

    def _calculate_percentile_position(self, value: float, stats: Dict) -> float:
        """
        Calculate approximate percentile position of a value based on training statistics
        Uses interpolation between known percentiles for better accuracy
        """
        try:
            # Get available percentiles
            percentiles = {}
            for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
                key = f'p{p}'
                if key in stats:
                    percentiles[p] = stats[key]
        
            # Add min/max as 0th and 100th percentiles
            if 'min' in stats:
                percentiles[0] = stats['min']
            if 'max' in stats:
                percentiles[100] = stats['max']
        
            if not percentiles:
                return 0.5  # Default to median if no percentiles available
        
            # Sort percentiles
            sorted_percentiles = sorted(percentiles.items())
        
            # Handle edge cases
            if value <= sorted_percentiles[0][1]:
                return sorted_percentiles[0][0] / 100.0
            if value >= sorted_percentiles[-1][1]:
                return sorted_percentiles[-1][0] / 100.0
        
            # Find bracketing percentiles and interpolate
            for i in range(len(sorted_percentiles) - 1):
                p1, v1 = sorted_percentiles[i]
                p2, v2 = sorted_percentiles[i + 1]
            
                if v1 <= value <= v2:
                    if v2 == v1:  # Avoid division by zero
                        return p1 / 100.0
                
                    # Linear interpolation
                    ratio = (value - v1) / (v2 - v1)
                    interpolated_percentile = p1 + ratio * (p2 - p1)
                    return interpolated_percentile / 100.0
        
            return 0.5  # Fallback
        
        except Exception:
            return 0.5  # Safe fallback

    def _get_service_type(self) -> str:
        """Determine service type for context-aware impact assessment"""
        if not hasattr(self, 'service_name'):
            return 'standard_service'
        
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

    def _assess_request_rate_anomaly_severity(self, current_value: float, stats: Dict) -> str:
        """
        Determine anomaly severity level for request rate based on business rules
        Returns: 'normal', 'warning', 'critical'
        """
    
        minimum = stats.get('min', 0)
        maximum = stats.get('max', 0)
    
        # Zero traffic = Critical
        if current_value == 0:
            return 'critical'
    
        # High traffic assessment
        if current_value > maximum and maximum > 0:
            multiplier = current_value / maximum
        
            if multiplier >= 3.0:
                return 'critical'  # 3x+ above max
            elif multiplier >= 2.0:
                return 'warning'   # 2x-3x above max
            else:
                return 'normal'    # Above max but <2x
    
        # Low traffic (below min but >0) = Normal
        elif current_value < minimum and minimum > 0:
            return 'normal'
    
        # Within normal range = Normal
        else:
            return 'normal'

    def _enhance_request_rate_anomaly_detection(self, metrics: Dict) -> Dict:
        """
        Enhanced request rate anomaly detection with business-aware rules
        This should be called from your main detect_anomalies method
        """
        enhanced_anomalies = {}
    
        if 'request_rate' not in metrics:
            return enhanced_anomalies
    
        current_rate = metrics['request_rate']
    
        # Get training statistics for request_rate
        if not hasattr(self, 'training_statistics') or 'request_rate' not in self.training_statistics:
            return enhanced_anomalies
    
        stats = self.training_statistics['request_rate']
    
        # Determine severity using business rules
        severity = self._assess_request_rate_anomaly_severity(current_rate, stats)
    
        # Only create anomaly if it's warning or critical
        if severity in ['warning', 'critical']:
            # Get business impact assessment
            business_impact = self._assess_request_rate_impact(
                current_rate, stats, self._get_service_type(), 0.0, 0.0
            )
        
            # Create anomaly entry
            anomaly_key = f'request_rate_business_rule'
            enhanced_anomalies[anomaly_key] = {
                'type': 'business_rule',
                'severity': severity,
                'score': -3.0 if severity == 'critical' else -2.0,  # High confidence scores
                'description': self._get_request_rate_anomaly_description(current_rate, stats, severity),
                'value': current_rate,
                'threshold': stats.get('max', 0),
                'detection_method': 'business_aware_request_rate',
                'business_impact': business_impact,
                'business_rule': True,
                'rule_triggered': self._get_triggered_rule(current_rate, stats)
            }
    
        return enhanced_anomalies

    def _get_request_rate_anomaly_description(self, current_value: float, stats: Dict, severity: str) -> str:
        """Generate descriptive text for request rate anomalies"""
    
        maximum = stats.get('max', 0)
    
        if current_value == 0:
            return "Request rate dropped to zero - potential service outage or routing issue"
    
        elif current_value > maximum and maximum > 0:
            multiplier = current_value / maximum
        
            if severity == 'critical':
                return f"Request rate critically high ({multiplier:.1f}x historical maximum) - severe overload risk"
            elif severity == 'warning':
                return f"Request rate significantly elevated ({multiplier:.1f}x historical maximum) - monitor capacity"
    
        return f"Request rate anomaly detected: {current_value:.1f} req/s"

    def _get_triggered_rule(self, current_value: float, stats: Dict) -> str:
        """Get description of which business rule was triggered"""
    
        maximum = stats.get('max', 0)
    
        if current_value == 0:
            return "Zero traffic rule: request_rate == 0 → Critical"
    
        elif current_value > maximum and maximum > 0:
            multiplier = current_value / maximum
        
            if multiplier >= 3.0:
                return f"High traffic rule: request_rate >= 3x max ({multiplier:.1f}x) → Critical"
            elif multiplier >= 2.0:
                return f"High traffic rule: request_rate >= 2x max ({multiplier:.1f}x) → Warning"
    
        return "Business rule evaluation"


    def _detect_service_patterns_with_business_rules(self, metrics: Dict, request_rate_anomalies: Dict) -> Dict:
        """Detect problematic service patterns while respecting business rules for request rate"""
        patterns = {}

        if not hasattr(self, 'pattern_thresholds'):
            return patterns

        req_rate = metrics.get('request_rate', 0)
        app_latency = metrics.get('application_latency', 0)
        error_rate = metrics.get('error_rate', 0)

        # Get request rate business rule status
        request_rate_severity = None
        if request_rate_anomalies:
            for anomaly_data in request_rate_anomalies.values():
                if anomaly_data.get('business_rule', False):
                    request_rate_severity = anomaly_data.get('severity')
                    break

        # OPERATIONAL thresholds based on real-world SLA expectations
        CRITICAL_ERROR_RATE = 0.05    # 5% - truly problematic
        HIGH_ERROR_RATE = 0.02        # 2% - warning level
        CRITICAL_LATENCY_MS = 2000    # 2 seconds - unacceptable for most services
        HIGH_LATENCY_MS = 1000        # 1 second - concerning

        # Get statistical thresholds for traffic patterns
        req_p95 = self.pattern_thresholds.get('request_rate', {}).get('p95', float('inf'))
        lat_p95 = self.pattern_thresholds.get('application_latency', {}).get('p95', float('inf'))
        lat_median = self.pattern_thresholds.get('application_latency', {}).get('median', 0)

        # System overload: High traffic + High latency + HIGH errors
        # BUT: only flag if request rate business rules indicate a problem
        if app_latency > HIGH_LATENCY_MS and error_rate > CRITICAL_ERROR_RATE:
            if request_rate_severity == 'critical':
                # Business rules say traffic is critical AND we have latency + errors
                patterns['system_overload_critical'] = {
                    'type': 'pattern',
                    'severity': 'critical',
                    'description': f'System overload: critical traffic load ({req_rate:.1f} req/s), high latency ({app_latency:.0f}ms), and high error rate ({error_rate:.2%})',
                    'metrics': {
                        'request_rate': req_rate,
                        'application_latency': app_latency,
                        'error_rate': error_rate
                    },
                    'business_rule_informed': True
                }
            elif request_rate_severity == 'warning':
                # Business rules say traffic is elevated AND we have latency + errors
                patterns['system_stress'] = {
                    'type': 'pattern',
                    'severity': 'high',
                    'description': f'System stress: elevated traffic ({req_rate:.1f} req/s), high latency ({app_latency:.0f}ms), and error rate ({error_rate:.2%})',
                    'metrics': {
                        'request_rate': req_rate,
                        'application_latency': app_latency,
                        'error_rate': error_rate
                    }
                }
            # If request rate is not flagged by business rules, don't consider it overload

        # Cascading failure: Only flag if error rate is operationally concerning
        elif error_rate > CRITICAL_ERROR_RATE and app_latency < lat_median:
            # Don't factor request rate into cascading failure - it's about errors + fast failures
            patterns['cascading_failure'] = {
                'type': 'pattern',
                'severity': 'critical',
                'description': f'Cascading failure: high error rate ({error_rate:.2%}) with fast failures',
                'metrics': {
                    'error_rate': error_rate,
                    'application_latency': app_latency
                }
            }

        # Performance degradation: Use operational latency thresholds
        elif app_latency > CRITICAL_LATENCY_MS and error_rate < HIGH_ERROR_RATE:
            patterns['performance_degradation'] = {
                'type': 'pattern',
                'severity': 'critical',
                'description': f'Critical performance degradation: {app_latency:.0f}ms latency (>{CRITICAL_LATENCY_MS}ms threshold)',
                'metrics': {
                    'application_latency': app_latency,
                    'error_rate': error_rate
                }
            }
        elif app_latency > HIGH_LATENCY_MS and error_rate < HIGH_ERROR_RATE:
            patterns['performance_degradation'] = {
                'type': 'pattern',
                'severity': 'high',
                'description': f'Performance degradation: {app_latency:.0f}ms latency (>{HIGH_LATENCY_MS}ms threshold)',
                'metrics': {
                    'application_latency': app_latency,
                    'error_rate': error_rate
                }
            }

        # Service unavailability: Zero traffic (business rules will catch this)
        if req_rate == 0 and request_rate_severity == 'critical':
            patterns['service_unavailable'] = {
                'type': 'pattern', 
                'severity': 'critical',
                'description': 'Service appears unavailable: zero traffic detected',
                'metrics': {
                    'request_rate': req_rate
                },
                'business_rule_informed': True
            }

        return patterns
