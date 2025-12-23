# Enhanced Training Pipeline with Explainability Support
# Now uses the enhanced SmartboxAnomalyDetector automatically

from smartbox_anomaly.detection.detector import SmartboxAnomalyDetector
from smartbox_anomaly.detection.time_aware import TimeAwareAnomalyDetector
from vmclient import VictoriaMetricsClient
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class SmartboxMetricsExtractor:
    def __init__(self, vm_client: VictoriaMetricsClient):
        self.vm_client = vm_client
        
        # Your actual Smartbox queries (single line format for VictoriaMetrics)
        self.queries = {
            'request_rate': 'http_requests:count:rate_5m',
            
            'application_latency': 'sum(rate(duration_milliseconds_sum{span_kind="SPAN_KIND_SERVER", deployment_environment_name=~"production"}[1m])) by (service_name) / sum(rate(duration_milliseconds_count{span_kind="SPAN_KIND_SERVER", deployment_environment_name=~"production"}[1m])) by (service_name)',
            
            'client_latency': 'sum(rate(duration_milliseconds_sum{span_kind="SPAN_KIND_CLIENT", deployment_environment_name=~"production", db_system="", db_system_name=""}[1m])) by (service_name) / sum(rate(duration_milliseconds_count{span_kind="SPAN_KIND_CLIENT", deployment_environment_name=~"production", db_system="", db_system_name=""}[1m])) by (service_name)',
            
            'database_latency': 'sum(rate(duration_milliseconds_sum{span_kind="SPAN_KIND_CLIENT", deployment_environment_name=~"production", db_system_name!=""}[1m])) by (service_name) / sum(rate(duration_milliseconds_count{span_kind="SPAN_KIND_CLIENT", deployment_environment_name=~"production", db_system_name!=""}[1m])) by (service_name)',
            
            'error_rate': 'sum(rate(duration_milliseconds_count{span_kind="SPAN_KIND_SERVER", deployment_environment_name=~"production", http_response_status_code=~"5.*|"}[1m])) by (service_name) / sum(rate(duration_milliseconds_count{span_kind="SPAN_KIND_SERVER", deployment_environment_name=~"production"}[1m])) by (service_name)'
        }
    
    def get_available_services(self) -> List[str]:
        """Get list of services from VictoriaMetrics"""
        query = 'group by (service_name) (http_requests:count:rate_5m)'
        result = self.vm_client.query(query)
        
        services = []
        if result.get('data', {}).get('result'):
            for item in result['data']['result']:
                service_name = item.get('metric', {}).get('service_name')
                if service_name:
                    services.append(service_name)
        
        return sorted(list(set(services)))
    
    def test_queries(self, service_name: str) -> Dict[str, bool]:
        """Test all queries for a service to see which ones work"""
        print(f"🧪 Testing queries for {service_name}")
        
        query_results = {}
        
        for metric_name, base_query in self.queries.items():
            # Clean up query and add service filter
            clean_query = ' '.join(base_query.split())
            
            if 'service_name' in clean_query and 'by (service_name)' in clean_query:
                query = clean_query.replace('deployment_environment_name=~"production"', f'deployment_environment_name=~"production", service_name="{service_name}"')
            else:
                query = f'{clean_query}{{service_name="{service_name}"}}'
            
            print(f"   Testing {metric_name}...")
            print(f"   Query: {query}")
            
            # Test with a simple instant query first
            result = self.vm_client.query(query)
            
            if result.get('data', {}).get('result'):
                query_results[metric_name] = True
                print(f"   ✅ {metric_name}: Working")
            else:
                query_results[metric_name] = False
                print(f"   ❌ {metric_name}: No data or error")
                
                # Try without service filter to see if the base query works
                base_result = self.vm_client.query(clean_query)
                if base_result.get('data', {}).get('result'):
                    print(f"      (Base query works, but no data for {service_name})")
                else:
                    print()
        
        return query_results
    
    def extract_service_metrics(self, service_name: str, lookback_days: int = 30) -> pd.DataFrame:
        """Extract all metrics for a specific service"""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=lookback_days)
        
        print(f"📊 Extracting metrics for {service_name} from {start_time.date()} to {end_time.date()}")
        
        all_metrics = {}
        
        for metric_name, base_query in self.queries.items():
            # Clean up query and add service filter
            clean_query = ' '.join(base_query.split())  # Remove extra whitespace
            
            # Add service filter to queries that support it
            if 'service_name' in clean_query and 'by (service_name)' in clean_query:
                # For aggregated queries, filter before aggregation
                query = clean_query.replace('deployment_environment_name=~"production"', f'deployment_environment_name=~"production", service_name="{service_name}"')
            else:
                # For simple metrics like request_rate, add service filter
                query = f'{clean_query}{{service_name="{service_name}"}}'
            
            print(f"   Querying {metric_name}...")
            print(f"   Query: {query[:100]}{'...' if len(query) > 100 else ''}")
            
            result = self.vm_client.query_range(
                query=query,
                start_time=start_time,
                end_time=end_time,
                step='5m'
            )
            
            # Parse the result
            metric_data = self._parse_metric_result(result, metric_name)
            if not metric_data.empty:
                all_metrics[metric_name] = metric_data
                print(f"   ✅ {metric_name}: {len(metric_data)} data points")
            else:
                print(f"   ⚠️ {metric_name}: No data found")
        
        # Combine all metrics
        if all_metrics:
            combined_df = self._combine_metrics(all_metrics)
            print(f"✅ Combined dataset: {len(combined_df)} rows, {len(combined_df.columns)} columns")
            return combined_df
        else:
            print(f"❌ No metrics found for {service_name}")
            return pd.DataFrame()
    
    def _parse_metric_result(self, result: Dict, metric_name: str) -> pd.DataFrame:
        """Parse VictoriaMetrics result into DataFrame"""
        data_points = []
        
        if result.get('data', {}).get('result'):
            for series in result['data']['result']:
                values = series.get('values', [])
                
                for timestamp, value in values:
                    try:
                        data_points.append({
                            'timestamp': pd.to_datetime(float(timestamp), unit='s'),
                            metric_name: float(value) if value != 'NaN' else np.nan
                        })
                    except (ValueError, TypeError):
                        continue
        
        if data_points:
            df = pd.DataFrame(data_points)
            df = df.groupby('timestamp').mean().reset_index()  # Handle duplicates
            return df.set_index('timestamp')
        else:
            return pd.DataFrame()
    
    def _combine_metrics(self, metrics_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Combine multiple metric DataFrames with robust data cleaning"""
        combined = None
        
        for metric_name, df in metrics_dict.items():
            if combined is None:
                combined = df
            else:
                combined = combined.join(df, how='outer')
        
        # Robust data cleaning
        print(f"   🧹 Cleaning data...")
        
        # Replace infinite values with NaN
        combined = combined.replace([np.inf, -np.inf], np.nan)
        
        # Log data quality issues
        inf_count = np.isinf(combined.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            print(f"   ⚠️ Replaced {inf_count} infinite values with NaN")
        
        # Handle extremely large values (likely errors)
        numeric_cols = combined.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            # Define reasonable upper bounds for each metric type
            if 'latency' in col.lower():
                max_reasonable = 60000  # 60 seconds in milliseconds
            elif 'rate' in col.lower() and 'error' not in col.lower():
                max_reasonable = 100000  # 100k requests per second
            elif 'error' in col.lower():
                max_reasonable = 1.0  # Error rate should be <= 1
            else:
                max_reasonable = 1e6  # Generic large number
            
            # Replace unreasonable values
            mask = combined[col] > max_reasonable
            outlier_count = mask.sum()
            if outlier_count > 0:
                print(f"   ⚠️ Capped {outlier_count} outlier values in {col} (>{max_reasonable})")
                combined.loc[mask, col] = np.nan
        
        # Forward fill missing values (common in time series)
        combined = combined.fillna(method='ffill')
        
        # For remaining NaNs, use column median or 0
        for col in combined.columns:
            remaining_nans = combined[col].isna().sum()
            if remaining_nans > 0:
                if combined[col].dtype in ['float64', 'int64']:
                    median_val = combined[col].median()
                    fill_value = median_val if not pd.isna(median_val) else 0.0
                    combined[col] = combined[col].fillna(fill_value)
                    print(f"   🔧 Filled {remaining_nans} missing values in {col} with {fill_value:.3f}")
        
        # Final validation - ensure no inf/nan values remain
        final_inf_count = np.isinf(combined.select_dtypes(include=[np.number])).sum().sum()
        final_nan_count = combined.isna().sum().sum()
        
        if final_inf_count > 0 or final_nan_count > 0:
            print(f"   ❌ Data cleaning incomplete: {final_inf_count} inf, {final_nan_count} NaN values remain")
            # As last resort, fill with zeros
            combined = combined.replace([np.inf, -np.inf], 0).fillna(0)
            print(f"   🚑 Emergency cleanup: replaced remaining invalid values with 0")
        
        print(f"   ✅ Data cleaning completed")
        
        return combined

class SmartboxFeatureEngineer:
    def __init__(self):
        self.feature_windows = ['5T', '15T', '1H']  # 5min, 15min, 1hour
        self.base_metrics = ['request_rate', 'application_latency', 'client_latency', 'database_latency', 'error_rate']

    def engineer_features(self, metrics_df: pd.DataFrame, include_rolling: bool = True) -> pd.DataFrame:
        """Create ML features from raw metrics with robust data handling.

        Args:
            metrics_df: Raw metrics DataFrame with timestamp index.
            include_rolling: If True, compute rolling features (may cause leakage if used
                           before train/validation split). Set to False and use
                           engineer_features_split() for proper temporal splitting.

        Returns:
            DataFrame with engineered features.
        """
        if metrics_df.empty:
            return metrics_df

        print(f"   🔧 Engineering features...")

        features = metrics_df.copy()

        # Clean infinite and extreme values before feature engineering
        features = features.replace([np.inf, -np.inf], np.nan)

        # Time-based features (no leakage risk)
        features['hour'] = features.index.hour
        features['day_of_week'] = features.index.dayofweek
        features['is_business_hours'] = features['hour'].between(9, 17)
        features['is_weekend'] = features['day_of_week'] >= 5

        # Derived correlation features with safe division (no leakage risk)
        if all(col in features.columns for col in ['request_rate', 'application_latency', 'error_rate']):
            features['latency_error_ratio'] = np.where(
                features['application_latency'] > 0.001,
                features['error_rate'] / features['application_latency'],
                0
            )

            features['throughput_latency_efficiency'] = np.where(
                features['application_latency'] > 0.001,
                features['request_rate'] / features['application_latency'],
                0
            )

        # Database vs Application latency comparison (no leakage risk)
        if all(col in features.columns for col in ['database_latency', 'application_latency']):
            features['db_app_latency_ratio'] = np.where(
                features['application_latency'] > 0.001,
                features['database_latency'] / features['application_latency'],
                0
            )
            features['db_latency_dominance'] = features['database_latency'] > features['application_latency']

        # Only add rolling features if explicitly requested
        # WARNING: Rolling features computed on full dataset cause leakage
        if include_rolling:
            print(f"   ⚠️ Computing rolling features on full dataset (potential leakage)")
            features = self._add_rolling_features(features)

        # Final cleanup
        features = self._cleanup_features(features)

        print(f"   ✅ Feature engineering completed: {len(features)} rows, {len(features.columns)} features")

        return features

    def engineer_features_split(
        self,
        metrics_df: pd.DataFrame,
        validation_fraction: float = 0.2
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Engineer features with proper temporal split to prevent leakage.

        Rolling features are computed separately on train and validation sets,
        preventing information from validation leaking into training statistics.

        Args:
            metrics_df: Raw metrics DataFrame with timestamp index.
            validation_fraction: Fraction of data to use for validation (from end).

        Returns:
            Tuple of (train_features, validation_features) with rolling features
            computed separately on each split.
        """
        if metrics_df.empty:
            return pd.DataFrame(), pd.DataFrame()

        print(f"   🔧 Engineering features with temporal split (no leakage)...")

        # First, compute base features (no leakage risk)
        base_features = self.engineer_features(metrics_df, include_rolling=False)

        # Temporal split
        split_idx = int(len(base_features) * (1 - validation_fraction))
        train_base = base_features.iloc[:split_idx].copy()
        validation_base = base_features.iloc[split_idx:].copy()

        print(f"   📊 Split: {len(train_base)} train, {len(validation_base)} validation samples")

        # Add rolling features separately to each split
        print(f"   🔄 Computing rolling features on train set...")
        train_features = self._add_rolling_features(train_base)
        train_features = self._cleanup_features(train_features)

        print(f"   🔄 Computing rolling features on validation set...")
        validation_features = self._add_rolling_features(validation_base)
        validation_features = self._cleanup_features(validation_features)

        # For validation, we need some warm-up period for rolling features
        # Drop the first few rows that have NaN from rolling windows
        warmup_periods = 12  # 1 hour of 5-min data
        if len(validation_features) > warmup_periods:
            validation_features = validation_features.iloc[warmup_periods:]
            print(f"   ⏭️ Dropped {warmup_periods} warm-up rows from validation")

        print(f"   ✅ Leakage-free features: {len(train_features)} train, {len(validation_features)} validation")

        return train_features, validation_features

    def _add_rolling_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Add rolling statistical features to a DataFrame.

        Should be called separately on train and validation sets to prevent leakage.
        """
        for metric in self.base_metrics:
            if metric in features.columns:
                for window in self.feature_windows:
                    try:
                        min_periods = max(1, self._window_to_periods(window) // 3)

                        features[f'{metric}_mean_{window}'] = features[metric].rolling(window, min_periods=min_periods).mean()
                        features[f'{metric}_std_{window}'] = features[metric].rolling(window, min_periods=min_periods).std()
                        features[f'{metric}_max_{window}'] = features[metric].rolling(window, min_periods=min_periods).max()
                        features[f'{metric}_min_{window}'] = features[metric].rolling(window, min_periods=min_periods).min()

                        # Rate of change
                        periods = self._window_to_periods(window)
                        pct_change = features[metric].pct_change(periods=periods)
                        pct_change = np.clip(pct_change, -10, 10)  # Cap at ±1000%
                        features[f'{metric}_pct_change_{window}'] = pct_change

                    except Exception as e:
                        print(f"     ⚠️ Failed to create rolling features for {metric}_{window}: {e}")

        # Anomaly indicators using rolling percentiles (within this split only)
        if all(col in features.columns for col in ['request_rate', 'application_latency', 'error_rate']):
            try:
                features['high_error_rate'] = features['error_rate'] > features['error_rate'].rolling('1H', min_periods=1).quantile(0.95)
                features['high_latency'] = features['application_latency'] > features['application_latency'].rolling('1H', min_periods=1).quantile(0.95)
                features['traffic_spike'] = features['request_rate'] > features['request_rate'].rolling('1H', min_periods=1).quantile(0.95)
            except Exception as e:
                print(f"     ⚠️ Failed to create anomaly indicators: {e}")

        return features

    def _cleanup_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Clean up features by handling NaN and inf values."""
        features = features.replace([np.inf, -np.inf], np.nan)

        # Remove rows with too many NaNs (from rolling windows)
        valid_threshold = int(len(features.columns) * 0.5)  # At least 50% valid values
        features = features.dropna(thresh=valid_threshold)

        # Fill any remaining NaNs with 0
        features = features.fillna(0)

        return features

    def _window_to_periods(self, window: str) -> int:
        """Convert window string to number of periods (5min intervals)"""
        if window == '5T':
            return 1
        elif window == '15T':
            return 3
        elif window == '1H':
            return 12
        else:
            return 1

class ParquetTrainingDataStorage:
    def __init__(self, base_path: str = "./smartbox_training_data/"):
        self.base_path = base_path
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create directory structure"""
        os.makedirs(self.base_path, exist_ok=True)
        
    def save_training_data(self, service_name: str, raw_data: pd.DataFrame, 
                          features: pd.DataFrame, metadata: Dict) -> Dict[str, str]:
        """Save training data in parquet format"""
        date_str = datetime.now().strftime('%Y-%m-%d')
        service_path = f"{self.base_path}/{service_name}/"
        
        # Create directories
        for subdir in ['raw_metrics', 'features', 'metadata']:
            os.makedirs(f"{service_path}/{subdir}/date={date_str}/", exist_ok=True)
        
        # Save raw data
        raw_data_path = f"{service_path}/raw_metrics/date={date_str}/data.parquet"
        raw_data.to_parquet(raw_data_path, compression='snappy', engine='pyarrow')
        
        # Save features
        features_path = f"{service_path}/features/date={date_str}/features.parquet"
        features.to_parquet(features_path, compression='snappy', engine='pyarrow')
        
        # Save metadata
        metadata_path = f"{service_path}/metadata/date={date_str}/metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, default=str, indent=2)
        
        print(f"💾 Saved training data for {service_name} at {date_str}")
        
        return {
            'raw_data_path': raw_data_path,
            'features_path': features_path,
            'metadata_path': metadata_path
        }

class EnhancedSmartboxTrainingPipeline:
    """Enhanced training pipeline that automatically uses explainability features"""

    def __init__(self, vm_endpoint: str = "https://otel-metrics.production.smartbox.com", config_path: str = "./config.json"):
        self.vm_client = VictoriaMetricsClient(vm_endpoint)
        self.metrics_extractor = SmartboxMetricsExtractor(self.vm_client)
        self.feature_engineer = SmartboxFeatureEngineer()
        self.storage = ParquetTrainingDataStorage()

        # Load training configuration from config.json
        self.config = self._load_training_config(config_path)

        print("🚀 Enhanced Smartbox Training Pipeline initialized")
        print("   ✨ Explainable anomaly detection enabled by default")
        print(f"   📊 Validation fraction: {self.config.get('validation_fraction', 0.2)}")
        print(f"   📊 Threshold calibration: {'enabled' if self.config.get('threshold_calibration', {}).get('enabled', True) else 'disabled'}")

    def _load_training_config(self, config_path: str) -> Dict:
        """Load training configuration from config.json."""
        default_config = {
            'lookback_days': 30,
            'min_data_points': 1000,
            'validation_fraction': 0.2,
            'validation_split': 0.8,  # backward compatibility
            'threshold_calibration': {'enabled': True},
            'drift_detection': {'enabled': False}
        }

        try:
            with open(config_path) as f:
                full_config = json.load(f)

            training_config = full_config.get("training", {})

            # Merge with defaults
            for key, value in default_config.items():
                if key not in training_config:
                    training_config[key] = value

            # Ensure backward compatibility
            training_config['validation_split'] = 1 - training_config.get('validation_fraction', 0.2)

            return training_config

        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"⚠️ Could not load training config: {e}, using defaults")
            return default_config

    def load_services_from_config(self, config_path: str = "./config.json") -> List[str]:
        """Load services list from config.json.

        Combines all service categories: critical, standard, micro, admin, core.
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

            if unique_services:
                print(f"📋 Loaded {len(unique_services)} services from {config_path}")
                return unique_services
            else:
                print(f"⚠️ No services found in {config_path}, will discover from VictoriaMetrics")
                return []

        except FileNotFoundError:
            print(f"⚠️ Config file not found: {config_path}, will discover from VictoriaMetrics")
            return []
        except json.JSONDecodeError as e:
            print(f"⚠️ Invalid JSON in {config_path}: {e}, will discover from VictoriaMetrics")
            return []
        except Exception as e:
            print(f"⚠️ Error loading config: {e}, will discover from VictoriaMetrics")
            return []
    
    def test_service_queries(self, service_name: str) -> Dict[str, bool]:
        """Test queries for a specific service"""
        return self.metrics_extractor.test_queries(service_name)
    
    def discover_services(self) -> List[str]:
        """Discover available services from VictoriaMetrics"""
        services = self.metrics_extractor.get_available_services()
        print(f"🔍 Discovered {len(services)} services: {services[:10]}{'...' if len(services) > 10 else ''}")
        return services
    
    def train_service_model(self, service_name: str, lookback_days: Optional[int] = None) -> Dict:
        """Train enhanced anomaly detection model with explainability for a single service"""
        # Use custom lookback_days if provided, otherwise use config default
        training_days = lookback_days or self.config['lookback_days']
        
        print(f"\n🚀 Training enhanced model for service: {service_name}")
        print(f"📅 Using {training_days} days of training data")
        print(f"✨ Explainability features will be automatically included")
        
        try:
            # 1. Extract raw metrics
            raw_data = self.metrics_extractor.extract_service_metrics(
                service_name, 
                lookback_days=training_days
            )
            
            if raw_data.empty:
                return {'status': 'no_data', 'message': 'No metrics data found'}
            
            if len(raw_data) < self.config['min_data_points']:
                return {
                    'status': 'insufficient_data',
                    'message': f'Only {len(raw_data)} data points, need {self.config["min_data_points"]}'
                }

            # 2. Engineer features with proper temporal split (no leakage)
            validation_fraction = self.config.get('validation_fraction', 0.2)
            train_features, validation_features = self.feature_engineer.engineer_features_split(
                raw_data, validation_fraction=validation_fraction
            )

            if train_features.empty:
                return {'status': 'feature_engineering_failed', 'message': 'Feature engineering produced no features'}

            # Also create combined features for storage (with leakage warning acknowledged)
            features = self.feature_engineer.engineer_features(raw_data, include_rolling=True)

            # 3. Train enhanced model with leakage-free validation split
            model = SmartboxAnomalyDetector(service_name)

            # Train with properly split validation data for threshold calibration
            min_validation_samples = 100  # Need enough for reliable calibration
            train_result = model.train(
                train_features,
                validation_df=validation_features if len(validation_features) >= min_validation_samples else None
            )

            # Log calibration results
            if train_result.get("validation_results"):
                print(f"   📊 Thresholds calibrated on {len(validation_features)} validation samples")

            # 4. Validate model using properly split validation data (no leakage)
            validation_results = self._validate_enhanced_model_split(model, train_features, validation_features)
            
            if not validation_results['passed']:
                return {
                    'status': 'validation_failed',
                    'validation_results': validation_results
                }
            
            # 5. Save training data
            metadata = {
                'service': service_name,
                'model_type': 'enhanced_explainable',
                'training_start': str(raw_data.index.min()),
                'training_end': str(raw_data.index.max()),
                'data_points': len(raw_data),
                'feature_count': len(features.columns),
                'training_samples': len(train_features),
                'validation_samples': len(validation_features),
                'explainability_features': len(model.training_statistics),
                'calibrated_thresholds': len(model.calibrated_thresholds),
                'validation_calibrated': bool(train_result.get("validation_results")),
                'validation_results': validation_results,
                'model_version': f"enhanced_v{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'training_config': self.config
            }
            
            storage_result = self.storage.save_training_data(
                service_name, raw_data, features, metadata
            )
            
            # 6. Save enhanced model (now includes explainability data)
            model_path = self.save_enhanced_model(service_name, model, metadata)
            
            # Log explainability status
            explainability_status = "✅" if model.training_statistics else "❌"
            print(f"   🧠 Explainability features: {explainability_status} ({len(model.training_statistics)} metrics)")
            
            return {
                'status': 'success',
                'model_path': model_path,
                'storage_result': storage_result,
                'metadata': metadata,
                'validation_results': validation_results,
                'explainability_metrics': len(model.training_statistics)
            }
            
        except Exception as e:
            print(f"❌ Enhanced training failed for {service_name}: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def train_service_model_time_aware(self, service_name: str, lookback_days: Optional[int] = None) -> Dict:
        """Train enhanced time-aware anomaly detection model with explainability"""
        training_days = lookback_days or self.config['lookback_days']
    
        print(f"\n🚀 Training enhanced time-aware model for service: {service_name}")
        print(f"📅 Using {training_days} days of training data")
        print(f"✨ Time-aware + explainability features will be included")
    
        try:
            # Extract raw metrics (same as before)
            raw_data = self.metrics_extractor.extract_service_metrics(
                service_name, lookback_days=training_days
            )
        
            if raw_data.empty:
                return {'status': 'no_data', 'message': 'No metrics data found'}
        
            if len(raw_data) < self.config['min_data_points']:
                return {
                    'status': 'insufficient_data', 
                    'message': f'Only {len(raw_data)} data points, need {self.config["min_data_points"]}'
                }
        
            # Engineer features
            features = self.feature_engineer.engineer_features(raw_data)
        
            if features.empty:
                return {'status': 'feature_engineering_failed', 'message': 'Feature engineering produced no features'}
        
            # Train enhanced time-aware models with temporal validation split per period
            time_aware_detector = TimeAwareAnomalyDetector(service_name)
            validation_fraction = self.config.get('validation_fraction', 0.2)
            time_aware_detector.train_time_aware_models(
                features,
                validation_fraction=validation_fraction
            )
        
            # Enhanced validation for time-aware models
            validation_results = self._validate_enhanced_time_aware_models(time_aware_detector, features)
        
            # Count explainability features across all time periods
            total_explainability_metrics = 0
            for period, model in time_aware_detector.models.items():
                if hasattr(model, 'training_statistics'):
                    total_explainability_metrics += len(model.training_statistics)
        
            # Save models with enhanced metadata
            metadata = {
                'service': service_name,
                'model_type': 'enhanced_time_aware_explainable',
                'time_periods': list(time_aware_detector.models.keys()),
                'training_start': str(raw_data.index.min()),
                'training_end': str(raw_data.index.max()),
                'data_points': len(raw_data),
                'feature_count': len(features.columns),
                'total_explainability_metrics': total_explainability_metrics,  # NEW
                'validation_results': validation_results,
                'model_version': f"enhanced_time_aware_v{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'training_config': self.config
            }
        
            saved_paths = time_aware_detector.save_models("./smartbox_models/", metadata)
        
            # Log explainability status for each time period
            print(f"   🧠 Explainability features across {len(time_aware_detector.models)} time periods:")
            for period, model in time_aware_detector.models.items():
                explainability_count = len(model.training_statistics) if hasattr(model, 'training_statistics') else 0
                status = "✅" if explainability_count > 0 else "❌"
                print(f"      {status} {period}: {explainability_count} metrics")
        
            return {
                'status': 'success',
                'model_paths': saved_paths,
                'metadata': metadata,
                'validation_results': validation_results,
                'total_explainability_metrics': total_explainability_metrics
            }
        
        except Exception as e:
            print(f"❌ Enhanced time-aware training failed for {service_name}: {e}")
            return {'status': 'error', 'error': str(e)}

    
    def _validate_enhanced_model_split(
        self,
        model: SmartboxAnomalyDetector,
        train_features: pd.DataFrame,
        validation_features: pd.DataFrame
    ) -> Dict:
        """Enhanced validation using properly split data (no leakage).

        Args:
            model: Trained model to validate.
            train_features: Training features (for reference).
            validation_features: Held-out validation features (for testing).

        Returns:
            Validation results dictionary.
        """
        print("   🔍 Validating enhanced model (leakage-free)...")

        # Use the split validation method
        validation_result = self._validate_model_split(model, train_features, validation_features)

        # Additional explainability validation
        explainability_checks = {
            'has_training_statistics': hasattr(model, 'training_statistics') and len(model.training_statistics) > 0,
            'has_feature_importance': hasattr(model, 'feature_importance') and len(model.feature_importance) > 0,
            'has_context_method': hasattr(model, 'detect_anomalies_with_context')
        }

        validation_result['explainability_checks'] = explainability_checks
        validation_result['explainability_passed'] = all(explainability_checks.values())
        validation_result['explainability_metrics'] = len(model.training_statistics) if hasattr(model, 'training_statistics') else 0
        validation_result['enhanced_passed'] = validation_result['passed'] and validation_result['explainability_passed']
        validation_result['leakage_free'] = True

        if validation_result['explainability_passed']:
            print(f"     ✅ Explainability validation passed! ({validation_result['explainability_metrics']} metrics)")
        else:
            print(f"     ❌ Explainability validation failed:")
            for check, passed in explainability_checks.items():
                status = "✅" if passed else "❌"
                print(f"       {status} {check}")

        return validation_result

    def _validate_model_split(
        self,
        model: SmartboxAnomalyDetector,
        train_features: pd.DataFrame,
        validation_features: pd.DataFrame
    ) -> Dict:
        """Validate trained model using properly split data (no leakage).

        Args:
            model: Trained model to validate.
            train_features: Training features (for synthetic anomaly baseline).
            validation_features: Held-out validation features (for testing).

        Returns:
            Validation results dictionary.
        """
        print("   🔍 Validating model with split data (no leakage)...")

        if len(validation_features) < 50:
            return {
                'passed': False,
                'reason': 'Insufficient validation data',
                'validation_samples': len(validation_features),
                'leakage_free': True
            }

        total_tests = min(100, len(validation_features))
        test_errors = []

        print(f"     Testing on {total_tests} validation samples (properly split)...")

        # Test 1: Normal validation data (should have low anomaly rate)
        normal_anomalies = 0
        for i in range(min(total_tests // 2, len(validation_features))):
            try:
                row = validation_features.iloc[i]
                current_metrics = {
                    'request_rate': row.get('request_rate', 0),
                    'application_latency': row.get('application_latency', 0),
                    'client_latency': row.get('client_latency', 0),
                    'database_latency': row.get('database_latency', 0),
                    'error_rate': row.get('error_rate', 0)
                }

                for key, value in current_metrics.items():
                    if pd.isna(value) or np.isinf(value):
                        current_metrics[key] = 0.0

                anomalies = model.detect_anomalies(current_metrics)
                if anomalies:
                    normal_anomalies += 1

            except Exception as e:
                test_errors.append(str(e))
                if len(test_errors) <= 3:
                    print(f"     ⚠️ Validation test {i} failed: {e}")

        # Test 2: Synthetic anomalies based on TRAINING statistics (no leakage)
        # Use training data statistics to create realistic synthetic anomalies
        synthetic_anomalies = 0
        synthetic_tests = total_tests // 2

        for i in range(synthetic_tests):
            try:
                # Use training data for baseline (this is correct - no leakage)
                row = train_features.iloc[i % len(train_features)]

                current_metrics = {
                    'request_rate': row.get('request_rate', 0) * 3,
                    'application_latency': row.get('application_latency', 0) * 5,
                    'client_latency': row.get('client_latency', 0) * 2,
                    'database_latency': row.get('database_latency', 0) * 2,
                    'error_rate': min(1.0, row.get('error_rate', 0) * 10)
                }

                for key, value in current_metrics.items():
                    if pd.isna(value) or np.isinf(value):
                        current_metrics[key] = 0.0

                anomalies = model.detect_anomalies(current_metrics)
                if anomalies:
                    synthetic_anomalies += 1

            except Exception as e:
                test_errors.append(str(e))

        total_anomalies = normal_anomalies + synthetic_anomalies
        total_tests_run = (total_tests // 2) + synthetic_tests
        anomaly_rate = total_anomalies / total_tests_run if total_tests_run > 0 else 0
        synthetic_detection_rate = synthetic_anomalies / synthetic_tests if synthetic_tests > 0 else 0
        normal_anomaly_rate = normal_anomalies / (total_tests // 2) if total_tests // 2 > 0 else 0

        validation_checks = {
            'has_models': len(model.models) > 0,
            'has_thresholds': len(model.thresholds) > 0,
            'reasonable_anomaly_rate': 0.0 <= anomaly_rate <= 0.6,
            'normal_data_reasonable': normal_anomaly_rate <= 0.15,  # Max 15% on normal data
            'few_test_errors': len(test_errors) < total_tests_run * 0.1,
            'model_responds': total_tests_run > 0,
            'detects_synthetic': synthetic_detection_rate > 0.2
        }

        validation_passed = all(validation_checks.values())

        validation_result = {
            'passed': validation_passed,
            'anomaly_rate': anomaly_rate,
            'normal_anomalies': normal_anomalies,
            'normal_anomaly_rate': normal_anomaly_rate,
            'synthetic_anomalies': synthetic_anomalies,
            'synthetic_detection_rate': synthetic_detection_rate,
            'models_trained': len(model.models),
            'thresholds_set': len(model.thresholds),
            'validation_samples': total_tests_run,
            'test_errors': len(test_errors),
            'checks': validation_checks,
            'leakage_free': True
        }

        if not validation_passed:
            print(f"     ❌ Validation failed:")
            for check, passed in validation_checks.items():
                status = "✅" if passed else "❌"
                print(f"       {status} {check}: {passed}")
            print(f"     Normal data anomaly rate: {normal_anomaly_rate:.1%}")
            print(f"     Synthetic detection rate: {synthetic_detection_rate:.1%}")
        else:
            print(f"     ✅ Validation passed! (leakage-free)")
            print(f"       Normal data anomaly rate: {normal_anomaly_rate:.1%}")
            print(f"       Synthetic detection rate: {synthetic_detection_rate:.1%}")

        return validation_result

    def _validate_enhanced_model(self, model: SmartboxAnomalyDetector, features: pd.DataFrame) -> Dict:
        """Enhanced validation including explainability features (legacy method)."""
        print("   🔍 Validating enhanced model...")

        # Standard validation
        validation_result = self._validate_model(model, features)
        
        # Additional explainability validation
        explainability_checks = {
            'has_training_statistics': hasattr(model, 'training_statistics') and len(model.training_statistics) > 0,
            'has_feature_importance': hasattr(model, 'feature_importance') and len(model.feature_importance) > 0,
            'has_context_method': hasattr(model, 'detect_anomalies_with_context')
        }
        
        # Update validation result
        validation_result['explainability_checks'] = explainability_checks
        validation_result['explainability_passed'] = all(explainability_checks.values())
        validation_result['explainability_metrics'] = len(model.training_statistics) if hasattr(model, 'training_statistics') else 0
        
        # Overall pass includes explainability
        validation_result['enhanced_passed'] = validation_result['passed'] and validation_result['explainability_passed']
        
        if validation_result['explainability_passed']:
            print(f"     ✅ Explainability validation passed! ({validation_result['explainability_metrics']} metrics)")
        else:
            print(f"     ❌ Explainability validation failed:")
            for check, passed in explainability_checks.items():
                status = "✅" if passed else "❌"
                print(f"       {status} {check}")
        
        return validation_result
    
    def _validate_enhanced_time_aware_models(self, detector: TimeAwareAnomalyDetector, features_df: pd.DataFrame) -> Dict:
        """Enhanced validation for time-aware models with 5-period approach"""
        print("   🔍 Validating enhanced time-aware models with 5-period approach...")
        validation_results = {}
        
        # Add time period column
        features_df = features_df.copy()
        features_df['time_period'] = features_df.index.map(detector.get_time_period)
        
        # Use the detector's service-specific thresholds
        thresholds = detector.validation_thresholds
        service_type = detector._get_service_type()
        
        print(f"     Using {service_type} thresholds for 5 periods:")
        for period, threshold in thresholds.items():
            period_emoji = "🌅" if period == 'weekend_day' else "🌙" if period == 'weekend_night' else "🕒"
            print(f"       {period_emoji} {period}: {threshold:.1%} max anomaly rate")
        
        # All 5 periods
        all_periods = ['business_hours', 'night_hours', 'evening_hours', 'weekend_day', 'weekend_night']
        
        for period in detector.models.keys():
            period_data = features_df[features_df['time_period'] == period]
            
            # Use detector's minimum sample requirements
            min_samples = detector._get_min_samples_for_period(period)
            
            if len(period_data) < min_samples:
                validation_results[period] = {
                    'status': 'insufficient_data', 
                    'samples': len(period_data),
                    'min_required': min_samples,
                    'passed': False,
                    'explainability_passed': False
                }
                continue
            
            print(f"     Testing enhanced {period} model...")
            
            # Enhanced validation with realistic expectations
            model = detector.models[period]
            
            # Use smaller test sample for micro-services, admin services, and weekend periods
            if service_type in ['micro_service', 'admin_service'] or period.startswith('weekend_'):
                test_samples = min(25, len(period_data) // 3)  # Smaller samples for variable periods
            else:
                test_samples = min(40, len(period_data) // 2)  # Standard samples for predictable periods
            
            normal_anomaly_count = 0
            test_errors = 0
            
            # Test on validation data with improved error handling
            for i in range(test_samples):
                try:
                    row_idx = -(i+1) if i < len(period_data) else i % len(period_data)
                    row = period_data.iloc[row_idx]
                    test_metrics = {
                        'request_rate': max(0, row.get('request_rate', 0)),
                        'application_latency': max(0, row.get('application_latency', 0)),
                        'client_latency': max(0, row.get('client_latency', 0)),
                        'database_latency': max(0, row.get('database_latency', 0)),
                        'error_rate': max(0, min(1.0, row.get('error_rate', 0)))
                    }
                    
                    # Ensure no invalid values
                    for key, value in test_metrics.items():
                        if pd.isna(value) or np.isinf(value):
                            test_metrics[key] = 0.0
                    
                    anomalies = model.detect_anomalies(test_metrics)
                    if anomalies and len(anomalies) > 0:
                        normal_anomaly_count += 1
                        
                except Exception as e:
                    test_errors += 1
                    if test_errors <= 2:  # Only log first few errors
                        print(f"       ⚠️ Test error for {period} sample {i}: {str(e)[:50]}...")
            
            normal_anomaly_rate = normal_anomaly_count / test_samples if test_samples > 0 else 0
            
            # Enhanced explainability validation
            explainability_checks = {
                'has_training_statistics': hasattr(model, 'training_statistics') and len(model.training_statistics) > 0,
                'has_feature_importance': hasattr(model, 'feature_importance') and len(model.feature_importance) > 0,
                'has_context_method': hasattr(model, 'detect_anomalies_with_context'),
                'has_zero_statistics': hasattr(model, 'zero_statistics') and len(model.zero_statistics) > 0
            }
            
            # Use service-specific threshold for this period
            period_threshold = thresholds.get(period, 0.15)
            
            # Enhanced validation criteria with period-specific adjustments
            error_tolerance = 0.3 if period.startswith('weekend_') else 0.2  # More lenient for weekends
            
            validation_checks = {
                'anomaly_rate_acceptable': normal_anomaly_rate <= period_threshold,
                'sufficient_tests': test_samples >= 10,  # Minimum tests
                'low_error_rate': test_errors < test_samples * error_tolerance,
                'model_functional': normal_anomaly_count >= 0  # Basic functionality check
            }
            
            # Overall validation status
            validation_passed = all(validation_checks.values())
            explainability_passed = all(explainability_checks.values())
            
            validation_results[period] = {
                'samples_tested': test_samples,
                'normal_anomaly_rate': normal_anomaly_rate,
                'threshold_used': period_threshold,
                'test_errors': test_errors,
                'validation_checks': validation_checks,
                'passed': validation_passed,
                'explainability_checks': explainability_checks,
                'explainability_passed': explainability_passed,
                'explainability_metrics': len(model.training_statistics) if hasattr(model, 'training_statistics') else 0,
                'enhanced_passed': validation_passed and explainability_passed,
                'period_type': detector._get_period_type(period)
            }
            
            # Enhanced status reporting with period-specific emojis
            period_emoji = "🌅" if period == 'weekend_day' else "🌙" if period == 'weekend_night' else "🕒"
            
            if validation_passed:
                status_icon = "✅"
                status_msg = f"{normal_anomaly_rate:.1%} anomaly rate (≤{period_threshold:.1%})"
            else:
                status_icon = "❌"
                failed_checks = [check for check, passed in validation_checks.items() if not passed]
                if not validation_checks['anomaly_rate_acceptable']:
                    status_msg = f"{normal_anomaly_rate:.1%} anomaly rate (>{period_threshold:.1%} threshold)"
                else:
                    status_msg = f"Failed: {', '.join(failed_checks)}"
            
            explainability_icon = "🧠" if explainability_passed else "❌"
            print(f"     {status_icon} {period_emoji} {period}: {status_msg} {explainability_icon}")
            
            # Additional context for failures with 5-period awareness
            if not validation_passed:
                if period == 'weekend_night':
                    print(f"       💡 Weekend nights have the highest natural variability")
                elif period == 'weekend_day':
                    print(f"       💡 Weekend days have moderate variability compared to weekdays")
                elif period.startswith('weekend_') and service_type == 'micro_service':
                    print(f"       💡 Micro-services often have irregular weekend patterns")
                elif period.startswith('weekend_') and service_type == 'admin_service':
                    print(f"       💡 Admin services typically have minimal weekend activity")
        
        # Enhanced overall assessment with 5-period logic
        total_periods = len(validation_results)
        passed_periods = sum(1 for result in validation_results.values() if result.get('enhanced_passed', False))
        
        # Critical periods: business_hours and night_hours (most important for weekday operations)
        critical_periods_passed = sum(1 for period in ['business_hours', 'night_hours'] 
                                    if period in validation_results and validation_results[period].get('enhanced_passed', False))
        
        # Weekend periods: weekend_day and weekend_night (important for continuous monitoring)
        weekend_periods_passed = sum(1 for period in ['weekend_day', 'weekend_night'] 
                                   if period in validation_results and validation_results[period].get('enhanced_passed', False))
        
        # Enhanced validation logic for 5-period approach
        if total_periods == 0:
            overall_passed = False
            overall_message = "No periods validated"
        elif critical_periods_passed >= 1:  # At least one critical period must pass
            if total_periods <= 3:
                # If we have 3 or fewer periods, at least half must pass
                overall_passed = passed_periods >= max(1, total_periods // 2)
                overall_message = f"Validation passed: {passed_periods}/{total_periods} periods validated"
            else:
                # For 4+ periods, we need at least 2 periods AND either weekend coverage or strong weekday coverage
                weekday_coverage = critical_periods_passed >= 1 and passed_periods >= 2
                weekend_coverage = weekend_periods_passed >= 1
                
                if weekday_coverage and (weekend_coverage or passed_periods >= 3):
                    overall_passed = True
                    overall_message = f"Validation passed: {passed_periods}/{total_periods} periods validated"
                else:
                    overall_passed = False
                    overall_message = f"Insufficient coverage: {passed_periods}/{total_periods} periods (need weekday + weekend or ≥3 total)"
        else:
            overall_passed = False
            overall_message = "No critical periods (business/night hours) passed validation"
        
        explainability_periods = sum(1 for result in validation_results.values() if result.get('explainability_passed', False))
        explainability_overall = explainability_periods > 0
        
        print(f"   {'✅' if overall_passed else '❌'} Enhanced 5-period validation: {overall_message}")
        print(f"   {'🧠' if explainability_overall else '❌'} Explainability: {explainability_periods}/{total_periods} periods enabled")
        
        # Show period type breakdown
        if validation_results:
            print(f"   📊 Period breakdown:")
            weekday_passed = sum(1 for period in ['business_hours', 'night_hours', 'evening_hours'] 
                               if period in validation_results and validation_results[period].get('enhanced_passed', False))
            weekday_total = sum(1 for period in ['business_hours', 'night_hours', 'evening_hours'] 
                              if period in validation_results)
            
            weekend_passed = sum(1 for period in ['weekend_day', 'weekend_night'] 
                               if period in validation_results and validation_results[period].get('enhanced_passed', False))
            weekend_total = sum(1 for period in ['weekend_day', 'weekend_night'] 
                              if period in validation_results)
            
            if weekday_total > 0:
                print(f"     🕒 Weekday periods: {weekday_passed}/{weekday_total} passed")
            if weekend_total > 0:
                print(f"     🌅 Weekend periods: {weekend_passed}/{weekend_total} passed")
        
        # Add service-specific recommendations for 5-period approach
        if not overall_passed:
            print(f"   💡 Recommendations for {service_type} with 5-period approach:")
            if service_type == 'micro_service':
                print(f"      - Consider increasing training data collection period")
                print(f"      - Focus on getting business_hours and night_hours models working first")
                print(f"      - Weekend models may need higher tolerance for variability")
            elif service_type == 'admin_service':
                print(f"      - Admin services often have minimal weekend activity")
                print(f"      - Consider focusing on weekday period models (business/night/evening hours)")
                print(f"      - Weekend periods may not be reliable for admin services")
            else:
                print(f"      - Ensure adequate data collection across all time periods")
                print(f"      - Weekend periods require different expectations than weekdays")
                print(f"      - Consider adjusting anomaly detection sensitivity for weekend periods")
        
        # Add summary to validation results
        validation_results['_summary'] = {
            'overall_passed': overall_passed,
            'overall_message': overall_message,
            'explainability_overall': explainability_overall,
            'service_type': service_type,
            'total_periods': total_periods,
            'passed_periods': passed_periods,
            'critical_periods_passed': critical_periods_passed,
            'weekend_periods_passed': weekend_periods_passed,
            'thresholds_used': thresholds,
            'approach': '5_period_weekend_split'
        }
        
        return validation_results
        
    def train_all_services(self, service_list: Optional[List[str]] = None) -> Dict:
        """Train enhanced models for all services"""
        if service_list is None:
            service_list = self.discover_services()
        
        print(f"\n🎯 Training enhanced models for {len(service_list)} services")
        print(f"✨ All models will include explainability features")
        
        results = {}
        successful_trainings = 0
        explainable_models = 0
        
        for i, service in enumerate(service_list, 1):
            print(f"\n[{i}/{len(service_list)}] Processing {service}")
            
            result = self.train_service_model(service)
            results[service] = result
            
            if result['status'] == 'success':
                successful_trainings += 1
                if result.get('explainability_metrics', 0) > 0:
                    explainable_models += 1
                print(f"✅ {service}: Enhanced training successful")
                print(f"   🧠 Explainability: {result.get('explainability_metrics', 0)} metrics")
            else:
                print(f"❌ {service}: {result['status']} - {result.get('message', '')}")
        
        print("\n📊 Enhanced Training Summary:")
        print(f"   Total services: {len(service_list)}")
        print(f"   Successful: {successful_trainings}")
        print(f"   With explainability: {explainable_models}")
        print(f"   Failed: {len(service_list) - successful_trainings}")
        
        return results

    def train_all_services_time_aware(self, service_list: Optional[List[str]] = None) -> Dict:
        """Train enhanced time-aware models for all services with 5-period approach"""
        if service_list is None:
            service_list = self.discover_services()
        
        print(f"\n🎯 Training enhanced time-aware models for {len(service_list)} services")
        print(f"🕒 Using 5-period approach: business_hours, night_hours, evening_hours, weekend_day, weekend_night")
        print(f"✨ All models will include time-awareness + explainability features")
        print(f"🎚️ Using service-specific validation thresholds for realistic assessment")
        
        results = {}
        successful_trainings = 0
        total_explainability_metrics = 0
        validation_passed_services = 0
        period_stats = {
            'business_hours': {'trained': 0, 'passed': 0},
            'night_hours': {'trained': 0, 'passed': 0},
            'evening_hours': {'trained': 0, 'passed': 0},
            'weekend_day': {'trained': 0, 'passed': 0},
            'weekend_night': {'trained': 0, 'passed': 0}
        }
        
        for i, service in enumerate(service_list, 1):
            print(f"\n[{i}/{len(service_list)}] Processing {service}")
            
            result = self.train_service_model_time_aware(service)
            results[service] = result
            
            if result['status'] == 'success':
                successful_trainings += 1
                explainability_count = result.get('total_explainability_metrics', 0)
                total_explainability_metrics += explainability_count
                
                # Check if validation passed with new 5-period logic
                validation_results = result.get('validation_results', {})
                summary = validation_results.get('_summary', {})
                validation_passed = summary.get('overall_passed', False)
                service_type = summary.get('service_type', 'unknown')
                approach = summary.get('approach', 'unknown')
                
                # Track period-specific statistics
                for period in period_stats.keys():
                    if period in validation_results and validation_results[period].get('status') != 'insufficient_data':
                        period_stats[period]['trained'] += 1
                        if validation_results[period].get('enhanced_passed', False):
                            period_stats[period]['passed'] += 1
                
                if validation_passed:
                    validation_passed_services += 1
                    print(f"✅ {service} ({service_type}): Enhanced 5-period training successful")
                    
                    # Show period breakdown
                    weekday_periods = summary.get('critical_periods_passed', 0)
                    weekend_periods = summary.get('weekend_periods_passed', 0)
                    total_periods = summary.get('passed_periods', 0)
                    print(f"   🕒 Weekday coverage: {weekday_periods}/3, 🌅 Weekend coverage: {weekend_periods}/2")
                else:
                    print(f"⚠️ {service} ({service_type}): Training successful but validation concerns")
                    overall_message = summary.get('overall_message', 'Validation issues')
                    print(f"   📋 {overall_message}")
                
                print(f"   🧠 Total explainability metrics: {explainability_count}")
            else:
                print(f"❌ {service}: {result['status']} - {result.get('message', '')}")
        
        print("\n📊 Enhanced 5-Period Time-aware Training Summary:")
        print(f"   Total services: {len(service_list)}")
        print(f"   Successful trainings: {successful_trainings}")
        print(f"   Validation passed: {validation_passed_services}/{successful_trainings}")
        print(f"   Total explainability metrics: {total_explainability_metrics}")
        print(f"   Failed trainings: {len(service_list) - successful_trainings}")
        
        # Enhanced summary with 5-period validation insights
        if successful_trainings > 0:
            validation_rate = validation_passed_services / successful_trainings
            print(f"\n🎯 5-Period Validation Summary:")
            print(f"   Validation success rate: {validation_rate:.1%}")
            
            if validation_rate >= 0.8:
                print("   🟢 Excellent validation performance")
            elif validation_rate >= 0.6:
                print("   🟡 Good validation performance - some services need attention")
            else:
                print("   🔴 Many services have validation concerns - review thresholds")
                print("   💡 Consider:")
                print("      - Increasing training data collection period")
                print("      - Reviewing service-specific patterns")
                print("      - Adjusting anomaly detection sensitivity for weekend periods")
        
        # Show detailed period-by-period breakdown
        print(f"\n📈 Period-by-Period Training Success:")
        for period, stats in period_stats.items():
            if stats['trained'] > 0:
                success_rate = stats['passed'] / stats['trained']
                period_emoji = "🌅" if period == 'weekend_day' else "🌙" if period == 'weekend_night' else "🕒"
                status_color = "🟢" if success_rate >= 0.8 else "🟡" if success_rate >= 0.6 else "🔴"
                print(f"   {period_emoji} {period}: {stats['passed']}/{stats['trained']} ({success_rate:.1%}) {status_color}")
        
        # Show service type breakdown
        service_types = {}
        for service, result in results.items():
            if result['status'] == 'success':
                validation_results = result.get('validation_results', {})
                summary = validation_results.get('_summary', {})
                service_type = summary.get('service_type', 'unknown')
                
                if service_type not in service_types:
                    service_types[service_type] = {
                        'total': 0, 
                        'passed': 0,
                        'weekday_coverage': 0,
                        'weekend_coverage': 0
                    }
                
                service_types[service_type]['total'] += 1
                if summary.get('overall_passed', False):
                    service_types[service_type]['passed'] += 1
                
                # Track coverage types
                if summary.get('critical_periods_passed', 0) >= 1:
                    service_types[service_type]['weekday_coverage'] += 1
                if summary.get('weekend_periods_passed', 0) >= 1:
                    service_types[service_type]['weekend_coverage'] += 1
        
        if service_types:
            print(f"\n📈 Validation by Service Type (5-Period Approach):")
            for service_type, stats in service_types.items():
                total = stats['total']
                passed = stats['passed']
                weekday_cov = stats['weekday_coverage']
                weekend_cov = stats['weekend_coverage']
                
                rate = passed / total if total > 0 else 0
                weekday_rate = weekday_cov / total if total > 0 else 0
                weekend_rate = weekend_cov / total if total > 0 else 0
                
                print(f"   {service_type}: {passed}/{total} ({rate:.1%}) overall")
                print(f"     🕒 Weekday coverage: {weekday_cov}/{total} ({weekday_rate:.1%})")
                print(f"     🌅 Weekend coverage: {weekend_cov}/{total} ({weekend_rate:.1%})")
        
        # 5-period specific insights
        total_weekend_models = period_stats['weekend_day']['trained'] + period_stats['weekend_night']['trained']
        total_weekday_models = (period_stats['business_hours']['trained'] + 
                               period_stats['night_hours']['trained'] + 
                               period_stats['evening_hours']['trained'])
        
        if total_weekend_models > 0 and total_weekday_models > 0:
            weekend_vs_weekday_ratio = total_weekend_models / total_weekday_models
            print(f"\n🌅 Weekend vs Weekday Model Distribution:")
            print(f"   Weekend models: {total_weekend_models}")
            print(f"   Weekday models: {total_weekday_models}")
            print(f"   Ratio: {weekend_vs_weekday_ratio:.2f}")
            
            if weekend_vs_weekday_ratio < 0.5:
                print("   💡 Many services lack sufficient weekend data")
            elif weekend_vs_weekday_ratio > 0.8:
                print("   ✅ Good weekend data coverage across services")
        
        return results
        
    def _validate_model(self, model: SmartboxAnomalyDetector, features: pd.DataFrame) -> Dict:
        """Validate trained model with detailed diagnostics"""
        print("   🔍 Validating model...")
        
        validation_split = int(len(features) * self.config['validation_split'])
        validation_data = features[validation_split:]
        
        if len(validation_data) < 50:
            return {
                'passed': False, 
                'reason': 'Insufficient validation data',
                'validation_samples': len(validation_data)
            }
        
        # Test anomaly detection on validation data + synthetic anomalies
        total_tests = min(100, len(validation_data))
        test_errors = []
        
        print(f"     Testing on {total_tests} validation samples...")
        
        # Test 1: Normal validation data
        normal_anomalies = 0
        for i in range(total_tests // 2):
            try:
                row = validation_data.iloc[i]
                current_metrics = {
                    'request_rate': row.get('request_rate', 0),
                    'application_latency': row.get('application_latency', 0),
                    'client_latency': row.get('client_latency', 0),
                    'database_latency': row.get('database_latency', 0),
                    'error_rate': row.get('error_rate', 0)
                }
                
                # Ensure no invalid values
                for key, value in current_metrics.items():
                    if pd.isna(value) or np.isinf(value):
                        current_metrics[key] = 0.0
                
                anomalies = model.detect_anomalies(current_metrics)
                if anomalies:
                    normal_anomalies += 1
                    
            except Exception as e:
                test_errors.append(str(e))
                if len(test_errors) <= 3:
                    print(f"     ⚠️ Validation test {i} failed: {e}")
        
        # Test 2: Synthetic anomalies (should be detected)
        synthetic_anomalies = 0
        synthetic_tests = total_tests // 2
        
        for i in range(synthetic_tests):
            try:
                row = validation_data.iloc[i % len(validation_data)]
                
                # Create synthetic anomaly by amplifying values
                current_metrics = {
                    'request_rate': row.get('request_rate', 0) * 3,  # 3x normal traffic
                    'application_latency': row.get('application_latency', 0) * 5,  # 5x normal latency
                    'client_latency': row.get('client_latency', 0) * 2,
                    'database_latency': row.get('database_latency', 0) * 2,
                    'error_rate': min(1.0, row.get('error_rate', 0) * 10)  # 10x errors, capped at 100%
                }
                
                # Ensure no invalid values
                for key, value in current_metrics.items():
                    if pd.isna(value) or np.isinf(value):
                        current_metrics[key] = 0.0
                
                anomalies = model.detect_anomalies(current_metrics)
                if anomalies:
                    synthetic_anomalies += 1
                    
            except Exception as e:
                test_errors.append(str(e))
        
        total_anomalies = normal_anomalies + synthetic_anomalies
        total_tests_run = (total_tests // 2) + synthetic_tests
        anomaly_rate = total_anomalies / total_tests_run if total_tests_run > 0 else 0
        
        # Calculate synthetic detection rate
        synthetic_detection_rate = synthetic_anomalies / synthetic_tests if synthetic_tests > 0 else 0
        
        # Balanced validation criteria
        validation_checks = {
            'has_models': len(model.models) > 0,
            'has_thresholds': len(model.thresholds) > 0,
            'reasonable_anomaly_rate': 0.0 <= anomaly_rate <= 0.6,  # Allow up to 60% overall (includes synthetic)
            'normal_data_reasonable': normal_anomalies <= (total_tests//2) * 0.15,  # Max 15% anomalies in normal data (more strict)
            'few_test_errors': len(test_errors) < total_tests_run * 0.1,
            'model_responds': total_tests_run > 0,
            'detects_synthetic': synthetic_detection_rate > 0.2  # Should detect at least 20% of obvious anomalies
        }
        
        validation_passed = all(validation_checks.values())
        
        validation_result = {
            'passed': validation_passed,
            'anomaly_rate': anomaly_rate,
            'normal_anomalies': normal_anomalies,
            'normal_anomaly_rate': normal_anomalies / (total_tests//2) if total_tests//2 > 0 else 0,
            'synthetic_anomalies': synthetic_anomalies,
            'synthetic_detection_rate': synthetic_detection_rate,
            'models_trained': len(model.models),
            'thresholds_set': len(model.thresholds),
            'validation_samples': total_tests_run,
            'test_errors': len(test_errors),
            'checks': validation_checks
        }
        
        normal_rate = validation_result['normal_anomaly_rate']
        
        if not validation_passed:
            print(f"     ❌ Validation failed:")
            for check, passed in validation_checks.items():
                status = "✅" if passed else "❌"
                print(f"       {status} {check}: {passed}")
            print(f"     Overall anomaly rate: {anomaly_rate:.3f}")
            print(f"     Normal data anomaly rate: {normal_rate:.1%} ({normal_anomalies}/{total_tests//2})")
            print(f"     Synthetic anomaly detection: {synthetic_anomalies}/{synthetic_tests} ({synthetic_detection_rate:.1%})")
            print(f"     Test errors: {len(test_errors)}/{total_tests_run}")
        else:
            print(f"     ✅ Validation passed!")
            print(f"       Overall anomaly rate: {anomaly_rate:.3f}")
            print(f"       Normal data anomaly rate: {normal_rate:.1%}")
            print(f"       Synthetic detection rate: {synthetic_detection_rate:.1%}")
            print(f"       Models: {len(model.models)}, Thresholds: {len(model.thresholds)}")
        
        return validation_result
    
    def save_enhanced_model(self, service_name: str, model: SmartboxAnomalyDetector, metadata: Dict) -> str:
        """Save enhanced model with explainability data"""
        models_dir = "./smartbox_models/"
        os.makedirs(models_dir, exist_ok=True)
        
        # The enhanced SmartboxAnomalyDetector automatically saves explainability data
        model_dir = model.save_model_secure(models_dir, metadata)
        
        print(f"💾 Enhanced model saved: {model_dir}")
        if hasattr(model, 'training_statistics'):
            print(f"   🧠 Explainability data included for {len(model.training_statistics)} metrics")
        
        return str(model_dir)


# Create alias for backward compatibility
SmartboxTrainingPipeline = EnhancedSmartboxTrainingPipeline

# Example usage
if __name__ == "__main__":
    # Initialize enhanced training pipeline
    training_pipeline = EnhancedSmartboxTrainingPipeline()

    # Load services from config.json (falls back to discovery if not found)
    services = training_pipeline.load_services_from_config()
    if not services:
        print("🔍 Discovering services from VictoriaMetrics...")
        services = training_pipeline.discover_services()

    if not services:
        print("❌ No services found to train. Please check config.json or VictoriaMetrics connection.")
        exit(1)

    # Choose training type
    use_time_aware = True  # Set to True to use enhanced time-aware models

    if use_time_aware:
        print("🕐 Using enhanced time-aware anomaly detection with explainability")

        results = training_pipeline.train_all_services_time_aware(services)

        # Print enhanced results summary
        print(f"\n📊 Final Results Summary:")
        for service, result in results.items():
            status_icon = "✅" if result['status'] == 'success' else "❌"
            explainability_count = result.get('total_explainability_metrics', 0)
            explainability_status = "🧠" if explainability_count > 0 else "❌"
            print(f"{status_icon} {service}: {result['status']} {explainability_status} ({explainability_count} metrics)")
    else:
        # Enhanced regular training
        results = training_pipeline.train_all_services(services)

        # Print enhanced results summary
        print(f"\n📊 Final Results Summary:")
        for service, result in results.items():
            status_icon = "✅" if result['status'] == 'success' else "❌"
            explainability_count = result.get('explainability_metrics', 0)
            explainability_status = "🧠" if explainability_count > 0 else "❌"
            print(f"{status_icon} {service}: {result['status']} {explainability_status} ({explainability_count} metrics)")
