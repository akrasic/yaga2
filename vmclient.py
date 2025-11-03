import logging
import urllib3
import requests
import time
from datetime import datetime
from typing import Dict, Optional
from dataclasses import dataclass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Reduce urllib3 logging level to avoid connection pool warnings
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)


@dataclass
class InferenceMetrics:
    """Input metrics for inference"""
    service_name: str
    timestamp: datetime
    request_rate: float
    application_latency: Optional[float] = None
    client_latency: Optional[float] = None
    database_latency: Optional[float] = None
    error_rate: Optional[float] = None
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to model input format"""
        return {
            'request_rate': self.request_rate,
            'application_latency': self.application_latency or 0.0,
            'client_latency': self.client_latency or 0.0,
            'database_latency': self.database_latency or 0.0,
            'error_rate': self.error_rate or 0.0
        }
    
    def validate(self) -> bool:
        """Validate input data quality"""
        try:
            # Check for reasonable values
            if self.request_rate < 0 or self.request_rate > 1000000:  # Max 1M req/sec
                return False
            if self.application_latency and (self.application_latency < 0 or self.application_latency > 300000):  # Max 5min
                return False
            if self.error_rate and (self.error_rate < 0 or self.error_rate > 1):  # Max 100%
                return False
            return True
        except (TypeError, ValueError):
            return False


class MetricsCollectionError(Exception):
    """Custom exception for metrics collection failures"""
    pass


class VictoriaMetricsClient:
    """Robust VictoriaMetrics client with retry logic and circuit breaker"""
    
    def __init__(self, endpoint: str, timeout: int = 10, max_retries: int = 3):
        self.endpoint = endpoint.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Enhanced session configuration to prevent connection pool issues
        self.session = requests.Session()
        
        # Configure connection pool with more connections and better settings
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=20,  # Increased pool size
            pool_maxsize=20,      # Increased max pool size
            max_retries=urllib3.util.retry.Retry(
                total=2,
                backoff_factor=0.3,
                status_forcelist=[500, 502, 503, 504]
            ),
            pool_block=False      # Don't block when pool is full
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        
        # Circuit breaker state
        self.failure_count = 0
        self.last_failure_time = None
        self.circuit_breaker_threshold = 5
        self.circuit_breaker_timeout = 300  # 5 minutes
        
        # Queries for current metrics
        self.queries = {
            'request_rate': 'http_requests:count:rate_5m',
            'application_latency': 'sum(rate(duration_milliseconds_sum{span_kind="SPAN_KIND_SERVER", deployment_environment_name=~"production"}[5m])) by (service_name) / sum(rate(duration_milliseconds_count{span_kind="SPAN_KIND_SERVER", deployment_environment_name=~"production"}[5m])) by (service_name)',
            'client_latency': 'sum(rate(duration_milliseconds_sum{span_kind="SPAN_KIND_CLIENT", deployment_environment_name=~"production", db_system="", db_system_name=""}[5m])) by (service_name) / sum(rate(duration_milliseconds_count{span_kind="SPAN_KIND_CLIENT", deployment_environment_name=~"production", db_system="", db_system_name=""}[5m])) by (service_name)',
            'database_latency': 'sum(rate(duration_milliseconds_sum{span_kind="SPAN_KIND_CLIENT", deployment_environment_name=~"production", db_system_name!=""}[5m])) by (service_name) / sum(rate(duration_milliseconds_count{span_kind="SPAN_KIND_CLIENT", deployment_environment_name=~"production", db_system_name!=""}[5m])) by (service_name)',
            'error_rate': 'sum(rate(duration_milliseconds_count{span_kind="SPAN_KIND_SERVER", deployment_environment_name=~"production", http_response_status_code=~"5.*|"}[5m])) by (service_name) / sum(rate(duration_milliseconds_count{span_kind="SPAN_KIND_SERVER", deployment_environment_name=~"production"}[5m])) by (service_name)'
        }
    
    def is_circuit_open(self) -> bool:
        """Check if circuit breaker is open"""
        if self.failure_count < self.circuit_breaker_threshold:
            return False
        
        if self.last_failure_time is None:
            return False
        
        time_since_failure = time.time() - self.last_failure_time
        return time_since_failure < self.circuit_breaker_timeout
    
    def record_success(self):
        """Record successful request"""
        self.failure_count = 0
        self.last_failure_time = None
    
    def record_failure(self):
        """Record failed request"""
        self.failure_count += 1
        self.last_failure_time = time.time()
    
    def collect_service_metrics(self, service_name: str) -> InferenceMetrics:
        """Collect current metrics for a service with robust error handling"""
        if self.is_circuit_open():
            raise MetricsCollectionError(f"Circuit breaker open for VictoriaMetrics")
        
        current_time = datetime.now()
        metrics_data = {'service_name': service_name, 'timestamp': current_time}
        
        try:
            # Collect all metrics with sequential requests to reduce connection pool pressure
            for metric_name, query in self.queries.items():
                try:
                    value = self._query_metric_with_retry(query, service_name)
                    metrics_data[metric_name] = value
                except Exception as e:
                    logger.warning(f"Failed to collect {metric_name} for {service_name}: {e}")
                    metrics_data[metric_name] = 0.0
                
                # Small delay to prevent overwhelming the connection pool
                time.sleep(0.1)
            
            metrics = InferenceMetrics(**metrics_data)
            
            if not metrics.validate():
                raise MetricsCollectionError(f"Invalid metrics for {service_name}: {metrics}")
            
            self.record_success()
            return metrics
            
        except Exception as e:
            self.record_failure()
            raise MetricsCollectionError(f"Failed to collect metrics for {service_name}: {e}")
    
    def query(self, query: str) -> Dict:
        """Query VictoriaMetrics for current value"""
        params = {'query': query}
        
        try:
            response = self.session.get(f"{self.endpoint}/api/v1/query", params=params, timeout=self.timeout)
            
            if response.status_code != 200:
                logger.error(f"Query failed with status {response.status_code}")
                logger.error(f"Query: {query}")
                logger.error(f"Response: {response.text[:500]}")
                
            response.raise_for_status()
            result = response.json()
            
            # Check for VictoriaMetrics errors in response
            if result.get('status') == 'error':
                logger.error(f"VictoriaMetrics error: {result.get('error', 'Unknown error')}")
                logger.error(f"Query: {query}")
                return {'data': {'result': []}}
                
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            logger.error(f"Query: {query}")
            return {'data': {'result': []}}
    
    def query_range(self, query: str, start_time: datetime, end_time: datetime, step: str = '5m') -> Dict:
        """Query VictoriaMetrics for a time range"""
        params = {
            'query': query,
            'start': start_time.isoformat() + 'Z',
            'end': end_time.isoformat() + 'Z', 
            'step': step
        }
        
        try:
            response = self.session.get(f"{self.endpoint}/api/v1/query_range", params=params, timeout=30)
            
            if response.status_code != 200:
                logger.error(f"Query range failed with status {response.status_code}")
                logger.error(f"Query: {query}")
                logger.error(f"Response: {response.text[:500]}")
                
            response.raise_for_status()
            result = response.json()
            
            # Check for VictoriaMetrics errors in response
            if result.get('status') == 'error':
                logger.error(f"VictoriaMetrics error: {result.get('error', 'Unknown error')}")
                logger.error(f"Query: {query}")
                return {'data': {'result': []}}
                
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            logger.error(f"Query: {query}")
            return {'data': {'result': []}}
    
    def _query_metric_with_retry(self, query: str, service_name: str) -> float:
        """Query single metric with retry logic"""
        # Add service filter to query
        if 'by (service_name)' in query:
            filtered_query = query.replace(
                'deployment_environment_name=~"production"',
                f'deployment_environment_name=~"production", service_name="{service_name}"'
            )
        else:
            filtered_query = f'{query}{{service_name="{service_name}"}}'
        
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(
                    f"{self.endpoint}/api/v1/query",
                    params={'query': filtered_query},
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                result = response.json()
                if result.get('status') == 'error':
                    raise MetricsCollectionError(f"VictoriaMetrics error: {result.get('error')}")
                
                # Extract value
                if result.get('data', {}).get('result'):
                    for series in result['data']['result']:
                        if series.get('value') and len(series['value']) > 1:
                            try:
                                return float(series['value'][1])
                            except (ValueError, TypeError, IndexError):
                                pass
                
                return 0.0
                
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    time.sleep(0.5 * (attempt + 1))  # Exponential backoff
                    continue
                break
        
        raise last_exception or MetricsCollectionError("Max retries exceeded")
