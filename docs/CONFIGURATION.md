# Configuration Guide

The Smartbox Anomaly Detection system uses a layered configuration approach with support for JSON configuration files and environment variable overrides.

## Configuration Priority

Configuration values are resolved in the following priority order (highest to lowest):

1. **Environment Variables** - Override all other settings
2. **JSON Configuration File** - Primary configuration source
3. **Dataclass Defaults** - Built-in fallback values

## Configuration File Locations

The system searches for configuration files in this order:

1. Path specified in `CONFIG_FILE` environment variable
2. `./config.json` (current working directory)
3. `./config/config.json` (config subdirectory)
4. `~/.smartbox/config.json` (user home directory)
5. `/etc/smartbox/config.json` (system-wide)

## Environment Variables

Key configuration values can be overridden via environment variables:

| Environment Variable | Config Path | Description |
|---------------------|-------------|-------------|
| `CONFIG_FILE` | - | Path to JSON config file |
| `VM_ENDPOINT` | `victoria_metrics.endpoint` | VictoriaMetrics server URL |
| `VM_TIMEOUT` | `victoria_metrics.timeout_seconds` | Request timeout (seconds) |
| `VM_MAX_RETRIES` | `victoria_metrics.max_retries` | Maximum retry attempts |
| `MODELS_DIR` | `model.models_directory` | Directory for trained models |
| `ALERTS_DIR` | `inference.alerts_directory` | Directory for alert output |
| `MAX_WORKERS` | `inference.max_workers` | Parallel inference workers |
| `FINGERPRINT_DB` | `fingerprinting.db_path` | SQLite database path |
| `OBSERVABILITY_URL` | `observability_api.base_url` | Observability API server URL |
| `OBSERVABILITY_ENABLED` | `observability_api.enabled` | Enable/disable API calls |

## Configuration Sections

### Victoria Metrics (`victoria_metrics`)

Connection settings for the VictoriaMetrics time-series database.

```json
{
  "victoria_metrics": {
    "endpoint": "https://otel-metrics.production.smartbox.com",
    "timeout_seconds": 10,
    "max_retries": 3,
    "pool_connections": 20,
    "pool_maxsize": 20,
    "circuit_breaker_threshold": 5,
    "circuit_breaker_timeout_seconds": 300,
    "retry_backoff_factor": 0.3,
    "retry_status_forcelist": [500, 502, 503, 504]
  }
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `endpoint` | string | `https://otel-metrics.production.smartbox.com` | VictoriaMetrics server URL |
| `timeout_seconds` | int | 10 | Request timeout |
| `max_retries` | int | 3 | Maximum retry attempts |
| `pool_connections` | int | 20 | HTTP connection pool size |
| `pool_maxsize` | int | 20 | Maximum pool size |
| `circuit_breaker_threshold` | int | 5 | Failures before circuit opens |
| `circuit_breaker_timeout_seconds` | int | 300 | Circuit breaker reset timeout |
| `retry_backoff_factor` | float | 0.3 | Exponential backoff factor |
| `retry_status_forcelist` | array | [500, 502, 503, 504] | HTTP codes to retry |

### Observability API (`observability_api`)

Settings for the observability platform integration.

```json
{
  "observability_api": {
    "base_url": "http://localhost:8000",
    "anomalies_endpoint": "/api/anomalies/batch",
    "resolutions_endpoint": "/api/incidents/resolve",
    "request_timeout_seconds": 5,
    "enabled": true
  }
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `base_url` | string | `http://localhost:8000` | API server base URL |
| `anomalies_endpoint` | string | `/api/anomalies/batch` | Endpoint for posting anomalies |
| `resolutions_endpoint` | string | `/api/incidents/resolve` | Endpoint for incident resolutions |
| `request_timeout_seconds` | int | 5 | API request timeout |
| `enabled` | bool | true | Enable/disable API integration |

### Model Configuration (`model`)

Machine learning model training parameters.

```json
{
  "model": {
    "models_directory": "./smartbox_models/",
    "min_training_samples": 500,
    "min_multivariate_samples": 1000,
    "default_contamination": 0.05,
    "default_n_estimators": 200,
    "random_state": 42,
    "contamination_by_category": {
      "critical": 0.03,
      "standard": 0.05,
      "micro": 0.08,
      "admin": 0.06,
      "core": 0.04,
      "background": 0.08
    },
    "contamination_by_service": {
      "booking": 0.02,
      "search": 0.04
    },
    "n_estimators_by_complexity": {
      "high": {"default": 250, "above_5k": 300, "above_10k": 400},
      "medium": {"default": 150, "above_3k": 200, "above_8k": 250},
      "low": {"default": 100, "above_5k": 150}
    }
  }
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `models_directory` | string | `./smartbox_models/` | Directory for model storage |
| `min_training_samples` | int | 500 | Minimum samples for univariate model training |
| `min_multivariate_samples` | int | 1000 | Minimum samples for multivariate detection |
| `default_contamination` | float | 0.05 | Default anomaly contamination rate |
| `default_n_estimators` | int | 200 | Default Isolation Forest estimators |
| `random_state` | int | 42 | Random seed for reproducibility |
| `contamination_by_category` | object | See defaults | Contamination rates per category |
| `contamination_by_service` | object | {} | Service-specific overrides |
| `n_estimators_by_complexity` | object | See defaults | Estimators by data complexity |

**Note**: The minimum sample requirements (500/1000) ensure statistical significance. Models trained with fewer samples may produce unreliable results.

**Contamination Rates**: Lower values = fewer expected anomalies = more sensitive detection.
- `critical` (0.03): High-traffic, business-critical services
- `standard` (0.05): Normal production services
- `micro` (0.08): Low-traffic microservices
- `admin` (0.06): Administrative interfaces
- `core` (0.04): Core platform services
- `background` (0.08): Background workers/jobs

### Training Configuration (`training`)

Settings for the model training pipeline.

```json
{
  "training": {
    "lookback_days": 30,
    "min_data_points": 2000,
    "validation_fraction": 0.2,
    "contamination_estimation": {
      "method": "knee",
      "min_samples": 100,
      "fallback": 0.05
    },
    "threshold_calibration": {
      "enabled": true,
      "percentiles": {
        "critical": 0.1,
        "high": 1,
        "medium": 5,
        "low": 10
      }
    },
    "drift_detection": {
      "enabled": false,
      "z_score_warning": 3,
      "z_score_critical": 5
    }
  }
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `lookback_days` | int | 30 | Days of historical data for training |
| `min_data_points` | int | 2000 | Minimum data points required |
| `validation_fraction` | float | 0.2 | Fraction of data for validation (temporal split) |
| `contamination_estimation.method` | string | `knee` | Method for estimating contamination (`knee` or `gap`) |
| `contamination_estimation.min_samples` | int | 100 | Minimum samples for estimation |
| `contamination_estimation.fallback` | float | 0.05 | Fallback contamination if estimation fails |
| `threshold_calibration.enabled` | bool | true | Enable per-model threshold calibration |
| `threshold_calibration.percentiles` | object | See defaults | Percentiles for each severity level |
| `drift_detection.enabled` | bool | false | Enable drift baseline computation at training |
| `drift_detection.z_score_warning` | float | 3 | Z-score threshold for drift warning |
| `drift_detection.z_score_critical` | float | 5 | Z-score threshold for critical drift |

**Validation Fraction**: The temporal train/validation split ensures no future data leakage. The last 20% of data (chronologically) is used for threshold calibration.

**Contamination Estimation**: When enabled, the system automatically estimates optimal contamination from the data distribution using the knee detection method.

### Inference Configuration (`inference`)

Settings for the inference pipeline execution.

```json
{
  "inference": {
    "alerts_directory": "./alerts/",
    "max_workers": 3,
    "inter_service_delay_seconds": 0.2,
    "check_drift": false
  }
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `alerts_directory` | string | `./alerts/` | Output directory for alerts |
| `max_workers` | int | 3 | Parallel processing workers |
| `inter_service_delay_seconds` | float | 0.2 | Delay between service processing |
| `check_drift` | bool | false | Enable drift detection at inference time |

**Drift Detection**: When `check_drift` is enabled, each inference run compares current metrics against training baselines. If significant drift is detected, confidence scores are reduced and a `drift_warning` is included in the output.

### Fingerprinting Configuration (`fingerprinting`)

Settings for anomaly tracking and deduplication.

```json
{
  "fingerprinting": {
    "db_path": "./anomaly_state.db",
    "cleanup_max_age_hours": 72,
    "incident_separation_minutes": 30
  }
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `db_path` | string | `./anomaly_state.db` | SQLite database path |
| `cleanup_max_age_hours` | int | 72 | Hours before old incidents are cleaned |
| `incident_separation_minutes` | int | 30 | Minutes between separate incidents |

### Time Periods (`time_periods`)

Configuration for time-aware anomaly detection.

```json
{
  "time_periods": {
    "business_hours": {"start": 8, "end": 18, "weekdays_only": true},
    "evening_hours": {"start": 18, "end": 22, "weekdays_only": true},
    "night_hours": {"start": 22, "end": 6, "weekdays_only": true},
    "weekend_day": {"start": 8, "end": 22, "weekends_only": true},
    "weekend_night": {"start": 22, "end": 8, "weekends_only": true},
    "min_samples": {
      "weekday": {"critical": 200, "standard": 200, "admin": 150, "micro": 100},
      "weekend": {"critical": 100, "standard": 100, "admin": 75, "micro": 50}
    },
    "confidence_scores": {
      "business_hours": 0.9,
      "evening_hours": 0.8,
      "night_hours": 0.95,
      "weekend_day": 0.7,
      "weekend_night": 0.6
    },
    "validation_thresholds": {
      "business_hours": 0.18,
      "evening_hours": 0.20,
      "night_hours": 0.12,
      "weekend_day": 0.28,
      "weekend_night": 0.32
    }
  }
}
```

**Time Period Definitions**: Each period specifies start/end hours (24-hour format).

**Confidence Scores**: Higher values indicate more reliable models for that time period.
- Business hours (0.9): Most training data, highest confidence
- Night hours (0.95): Consistent low-traffic patterns
- Weekend (0.6-0.7): Less data, more variability

**Validation Thresholds**: Maximum acceptable anomaly rate during model validation.
Lower thresholds = stricter validation = higher quality models required.

### Detection Thresholds (`detection_thresholds`)

Thresholds for anomaly severity classification.

```json
{
  "detection_thresholds": {
    "severity_scores": {
      "critical": -0.6,
      "high": -0.3,
      "medium": -0.1
    },
    "error_rates": {
      "critical": 0.05,
      "high": 0.02,
      "very_high": 0.20
    },
    "latency_ms": {
      "critical": 2000,
      "high": 1000
    },
    "ratios": {
      "client_latency_bottleneck": 0.6,
      "database_latency_bottleneck": 0.7,
      "traffic_cliff_threshold": 0.3
    },
    "outlier_percentiles": {
      "lower": 0.001,
      "upper": 0.999
    },
    "validation": {
      "max_request_rate": 1000000.0,
      "max_latency_ms": 300000.0,
      "constant_value_threshold": 1e-10
    }
  }
}
```

**Severity Scores** (Isolation Forest scores, more negative = more anomalous):
- `critical` (-0.6): Severe anomalies requiring immediate attention
- `high` (-0.3): Significant anomalies
- `medium` (-0.1): Minor anomalies

**Error Rate Thresholds**:
- `critical` (5%): Critical error rate threshold
- `high` (2%): High error rate threshold
- `very_high` (20%): Extreme error rate

**Ratios**:
- `client_latency_bottleneck` (0.6): Client vs server latency ratio for bottleneck detection
- `database_latency_bottleneck` (0.7): Database latency as fraction of total
- `traffic_cliff_threshold` (0.3): Sudden traffic drop threshold

### Services (`services`)

Service classification configuration.

```json
{
  "services": {
    "critical": ["booking", "search", "mobile-api", "shire-api"],
    "standard": ["friday", "gambit", "titan", "r2d2"],
    "micro": ["fa5"],
    "admin": ["m2-fr-adm", "m2-it-adm", "m2-bb-adm"],
    "core": ["m2-bb", "m2-fr", "m2-it"],
    "pattern_detection": {
      "api_patterns": ["api", "gateway", "proxy"],
      "admin_patterns": ["admin", "adm", "mgmt"],
      "background_patterns": ["worker", "job", "task", "queue"],
      "micro_patterns": ["micro", "util", "helper"],
      "core_patterns": ["m2-", "core", "platform"]
    }
  }
}
```

Services not explicitly listed are classified using pattern matching against `pattern_detection` rules.

### Dependencies (`dependencies`)

Service dependency graph for cascade detection. When a service has a latency anomaly, the system checks if upstream dependencies also have anomalies to identify cascade failures.

```json
{
  "dependencies": {
    "graph": {
      "mobile-api": ["booking", "search", "shire-api"],
      "booking": ["search", "vms", "r2d2"],
      "search": ["catalog", "r2d2"],
      "shire-api": ["vms", "gambit"],
      "vms": ["titan"],
      "gambit": ["titan", "r2d2"],
      "friday": ["fa5", "gambit"],
      "titan": [],
      "r2d2": ["catalog"],
      "catalog": [],
      "fa5": [],
      "tc14": ["r2d2"]
    },
    "cascade_detection": {
      "enabled": true,
      "max_depth": 5,
      "latency_propagation_threshold": 0.6,
      "require_temporal_correlation": true,
      "correlation_window_minutes": 5
    }
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `graph` | object | Service dependency map. Key = service name, Value = list of services it calls |
| `cascade_detection.enabled` | bool | Enable/disable cascade detection |
| `cascade_detection.max_depth` | int | Maximum depth for dependency chain traversal |
| `cascade_detection.latency_propagation_threshold` | float | Minimum latency ratio to consider propagation |
| `cascade_detection.require_temporal_correlation` | bool | Require anomalies to occur within time window |
| `cascade_detection.correlation_window_minutes` | int | Time window for temporal correlation |

**How it works:**
1. When a service has a latency anomaly, the system looks up its dependencies from the graph
2. If any dependency also has an active anomaly, it traces the chain to find the root cause
3. The anomaly is then labeled as `upstream_cascade` or `dependency_chain_degradation`
4. The `cascade_analysis` field in the output identifies the root cause service

**Example:** If `booking` is slow and `vms` (which booking depends on) also has an anomaly, the system identifies this as a cascade from `vms`.

## Adding New Services

This section explains how to add new services to the anomaly detection system.

### Step 1: Choose the Appropriate Category

Each service must be assigned to a category that determines its default anomaly detection sensitivity (contamination rate). Choose based on the service's criticality and traffic patterns:

| Category | Contamination | Use For |
|----------|---------------|---------|
| `critical` | 0.03 (3%) | High-traffic, revenue-impacting services (e.g., checkout, search, API gateways). Lower contamination = stricter detection, fewer false positives. |
| `standard` | 0.05 (5%) | Normal production services with moderate traffic and business importance. |
| `core` | 0.04 (4%) | Platform infrastructure services that other services depend on. |
| `admin` | 0.06 (6%) | Administrative interfaces and back-office tools with lower traffic. |
| `micro` | 0.08 (8%) | Small utility services, helpers, or low-traffic microservices. Higher contamination = more tolerant of variance. |
| `background` | 0.08 (8%) | Background workers, job processors, queue consumers. |

### Step 2: Add the Service to config.json

Edit the `services` section in `config.json` and add your service to the appropriate category array:

```json
{
  "services": {
    "critical": ["booking", "search", "mobile-api", "shire-api", "new-payment-service"],
    "standard": ["friday", "gambit", "titan", "r2d2", "my-new-service"],
    "micro": ["fa5", "utility-helper"],
    "admin": ["m2-fr-adm", "m2-it-adm", "m2-bb-adm"],
    "core": ["m2-bb", "m2-fr", "m2-it"]
  }
}
```

**Important**: The service name must match exactly how it appears in VictoriaMetrics metrics (the `service` label value).

### Step 3: (Optional) Set Custom Contamination

If the default category contamination doesn't fit your service, you can override it with a service-specific value:

```json
{
  "model": {
    "contamination_by_service": {
      "booking": 0.02,
      "search": 0.04,
      "my-new-service": 0.03
    }
  }
}
```

Service-specific contamination takes precedence over category defaults.

### Step 4: Train Models for the New Service

After adding the service, you need to train models before inference will work:

```bash
# If running locally
python main.py

# If running in Docker
docker compose run --rm yaga train
```

The training pipeline will:
1. Fetch 30 days of historical metrics from VictoriaMetrics
2. Train time-aware Isolation Forest models for each time period
3. Save models to `./smartbox_models/<service-name>/`

### Step 5: Verify the Service

After training completes, verify the service is detected:

```bash
# Run inference once to test
docker compose run --rm yaga inference --verbose

# Check that models exist
ls -la ./smartbox_models/my-new-service/
```

### Understanding Contamination

**Contamination** is the expected proportion of anomalies in the data. It directly affects detection sensitivity:

| Contamination | Sensitivity | False Positives | Best For |
|---------------|-------------|-----------------|----------|
| 0.01 (1%) | Very High | Very Low | Mission-critical services where any alert matters |
| 0.03 (3%) | High | Low | Critical business services |
| 0.05 (5%) | Medium | Medium | Standard production services |
| 0.08 (8%) | Low | Higher | Variable/noisy services, background jobs |
| 0.10 (10%) | Very Low | High | Experimental or highly variable services |

**Guidelines for choosing contamination:**
- Start with the category default
- If you get too many false positives, increase contamination (e.g., 0.05 → 0.07)
- If you're missing real anomalies, decrease contamination (e.g., 0.05 → 0.03)
- Monitor for 1-2 weeks and adjust based on alert quality

### Pattern-Based Auto-Classification

Services not explicitly listed in any category are automatically classified using pattern matching:

```json
{
  "services": {
    "pattern_detection": {
      "api_patterns": ["api", "gateway", "proxy"],
      "admin_patterns": ["admin", "adm", "mgmt"],
      "background_patterns": ["worker", "job", "task", "queue"],
      "micro_patterns": ["micro", "util", "helper"],
      "core_patterns": ["m2-", "core", "platform"]
    }
  }
}
```

For example:
- `user-api` → matches "api" → classified as API service
- `order-worker` → matches "worker" → classified as background service
- `admin-dashboard` → matches "admin" → classified as admin service

**Recommendation**: Explicitly list services in categories rather than relying on pattern matching for better control.

### Example: Adding Multiple Services

Here's a complete example of adding three new services:

```json
{
  "services": {
    "critical": ["booking", "search", "mobile-api", "shire-api", "checkout-api", "inventory-service"],
    "standard": ["friday", "gambit", "titan", "r2d2", "user-profile"],
    "micro": ["fa5"],
    "admin": ["m2-fr-adm", "m2-it-adm", "m2-bb-adm"],
    "core": ["m2-bb", "m2-fr", "m2-it"]
  },
  "model": {
    "contamination_by_service": {
      "checkout-api": 0.02,
      "inventory-service": 0.03,
      "user-profile": 0.05
    }
  }
}
```

After editing, rebuild and restart the container:

```bash
docker compose build
docker compose up -d
```

### Logging (`logging`)

Logging configuration.

```json
{
  "logging": {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  }
}
```

## Usage Examples

### Loading Configuration in Code

```python
from smartbox_anomaly.core.config import get_config, PipelineConfig

# Use global config (auto-loads from file)
config = get_config()
print(config.observability.base_url)

# Load from specific file
config = PipelineConfig.from_file("./my-config.json")

# Use defaults only
config = PipelineConfig.default()
```

### Environment Variable Overrides

```bash
# Override API server URL
export OBSERVABILITY_URL=http://production-api:8000

# Override VictoriaMetrics endpoint
export VM_ENDPOINT=https://vm.internal:8428

# Disable API integration
export OBSERVABILITY_ENABLED=false

# Use custom config file
export CONFIG_FILE=/opt/smartbox/production.json
```

### Docker/Kubernetes Usage

```yaml
# docker-compose.yml
services:
  anomaly-detector:
    environment:
      - CONFIG_FILE=/config/config.json
      - OBSERVABILITY_URL=http://api-server:8000
      - VM_ENDPOINT=http://victoria-metrics:8428
    volumes:
      - ./config.json:/config/config.json:ro
```

```yaml
# kubernetes ConfigMap
apiVersion: v1
kind: ConfigMap
metadata:
  name: anomaly-config
data:
  config.json: |
    {
      "victoria_metrics": {
        "endpoint": "http://vm-service:8428"
      },
      "observability_api": {
        "base_url": "http://observability-api:8000"
      }
    }
```

## Schema Validation

A JSON Schema file `config.schema.json` can be used for validation. Reference it in your config:

```json
{
  "$schema": "./config.schema.json",
  "victoria_metrics": { ... }
}
```

This enables IDE autocompletion and validation in editors like VS Code.
