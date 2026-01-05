# Smartbox ML Anomaly Detection Pipeline

A production-grade machine learning pipeline for real-time anomaly detection in microservices, using Isolation Forest models with time-aware period separation, calibrated thresholds, drift detection, and enhanced explainability.

## Key Features

- **Time-Aware Detection**: 5 behavioral periods (business hours, evening, night, weekend day/night)
- **SLO-Aware Severity**: Adjusts ML severity based on operational SLO thresholds (latency, error rates)
- **Calibrated Thresholds**: Per-model severity thresholds from validation data percentiles
- **Empirical Contamination**: Automatic contamination estimation using knee/gap detection
- **Drift Detection**: Z-score and Mahalanobis distance for distribution drift monitoring
- **Robust Statistics**: Trimmed mean and IQR-based standard deviation
- **Input Validation**: Metrics sanitization at inference boundary
- **Anomaly Fingerprinting**: Incident lifecycle tracking with two-level identity system
- **Explainable AI**: Feature contributions, business impact analysis, and recommendations
- **Lazy Loading**: Memory-efficient model loading (only loads needed time period)
- **Cascade Detection**: Dependency-aware anomaly correlation
- **Admin Dashboard**: Web UI for configuration management and service visualization
- **Docker Ready**: Production container with cron scheduling for training and inference

## Quick Start

```bash
# Install
uv pip install -e .

# Train models (requires VictoriaMetrics access)
python main.py

# Run inference
python inference.py --verbose
```

## Docker Deployment

```bash
# Build and start all services
docker compose up -d

# View logs
docker compose logs -f

# Run training manually
docker compose run --rm yaga train

# Run inference manually
docker compose run --rm yaga inference --verbose
```

The Docker deployment includes:
- **yaga**: Main service running cron-scheduled training (daily 2 AM) and inference (every 2 min)
- **admin-dashboard**: Web UI for configuration at http://localhost:8050

See [DEPLOYMENT.md](docs/DEPLOYMENT.md) for Kubernetes and advanced deployment options.

## Documentation

Comprehensive documentation is available in the `docs/` directory:

| Document | Description |
|----------|-------------|
| [MACHINE_LEARNING.md](docs/MACHINE_LEARNING.md) | ML concepts, algorithms, and how detection works |
| [ML_TRAINING.md](docs/ML_TRAINING.md) | Technical deep-dive for ML engineers |
| [CONFIGURATION.md](docs/CONFIGURATION.md) | All configuration options and tuning |
| [INFERENCE_API_PAYLOAD.md](docs/INFERENCE_API_PAYLOAD.md) | JSON output format specification |
| [OPERATIONS.md](docs/OPERATIONS.md) | Day-to-day operations and troubleshooting |
| [FINGERPRINTING.md](docs/FINGERPRINTING.md) | Incident tracking system |
| [DEPLOYMENT.md](docs/DEPLOYMENT.md) | Docker and Kubernetes deployment |
| [API_SPECIFICATION.md](docs/API_SPECIFICATION.md) | Observability API integration |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         TRAINING                                 │
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────┐
│  VictoriaMetrics (30 days)   │  → Historical metrics
└──────────────────────────────┘
     │
     ▼
┌──────────────────────────────┐
│  Feature Engineering         │  → Rolling stats, temporal split (80/20)
└──────────────────────────────┘
     │
     ▼
┌──────────────────────────────┐
│  Contamination Estimation    │  → Knee/gap detection
└──────────────────────────────┘
     │
     ▼
┌──────────────────────────────┐
│  Isolation Forest Training   │  → Per metric + multivariate
└──────────────────────────────┘
     │
     ▼
┌──────────────────────────────┐
│  Threshold Calibration       │  → Percentile-based severity
└──────────────────────────────┘
     │
     ▼
┌──────────────────────────────┐
│  Drift Baseline Storage      │  → Mean, covariance for Mahalanobis
└──────────────────────────────┘


┌─────────────────────────────────────────────────────────────────┐
│                         INFERENCE                                │
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────┐
│  Input Validation            │  → Sanitize NaN, inf, outliers
└──────────────────────────────┘
     │
     ▼
┌──────────────────────────────┐
│  Time Period Routing         │  → Select appropriate model
└──────────────────────────────┘
     │
     ▼
┌──────────────────────────────┐
│  Anomaly Detection           │  → IF scoring with calibrated thresholds
└──────────────────────────────┘
     │
     ▼
┌──────────────────────────────┐
│  Drift Detection             │  → Z-score + Mahalanobis (optional)
└──────────────────────────────┘
     │
     ▼
┌──────────────────────────────┐
│  Fingerprinting              │  → Incident lifecycle management
└──────────────────────────────┘
     │
     ▼
┌──────────────────────────────┐
│  API Output                  │  → JSON with explanations
└──────────────────────────────┘
```

## Time Periods

| Period | Hours | Days | Confidence |
|--------|-------|------|------------|
| `business_hours` | 08:00-18:00 | Mon-Fri | 0.9 |
| `evening_hours` | 18:00-22:00 | Mon-Fri | 0.8 |
| `night_hours` | 22:00-06:00 | Mon-Fri | 0.95 |
| `weekend_day` | 08:00-22:00 | Sat-Sun | 0.7 |
| `weekend_night` | 22:00-08:00 | Sat-Sun | 0.6 |

## Service Categories

Services are classified into categories that determine default detection sensitivity:

| Category | Contamination | Description |
|----------|---------------|-------------|
| `critical` | 0.03 (3%) | High-traffic, revenue-impacting services (booking, search, APIs) |
| `core` | 0.04 (4%) | Platform infrastructure services |
| `standard` | 0.05 (5%) | Normal production services |
| `admin` | 0.06 (6%) | Administrative interfaces |
| `micro` | 0.08 (8%) | Low-traffic microservices |
| `background` | 0.08 (8%) | Background workers and jobs |

Lower contamination = stricter detection = fewer false positives. Configure services in `config.json`:

```json
{
  "services": {
    "critical": ["booking", "search", "mobile-api"],
    "standard": ["friday", "gambit", "titan"]
  }
}
```

## SLO-Aware Severity Evaluation

The SLO layer adjusts ML-detected severity based on operational thresholds. This ensures alerts reflect business impact, not just statistical deviation.

**Severity Matrix:**

| | Within Acceptable | Approaching SLO | Breaching SLO |
|---|---|---|---|
| **Anomaly Detected** | informational | warning/high | critical |
| **No Anomaly** | none | warning | critical |

Configure per-service SLO thresholds:

```json
{
  "slos": {
    "enabled": true,
    "defaults": {
      "latency_acceptable_ms": 500,
      "latency_warning_ms": 800,
      "latency_critical_ms": 1000,
      "error_rate_acceptable": 0.005,
      "error_rate_warning": 0.01,
      "error_rate_critical": 0.02
    },
    "services": {
      "booking": {
        "latency_acceptable_ms": 300,
        "latency_critical_ms": 500,
        "error_rate_critical": 0.01
      }
    },
    "busy_periods": [
      {"start": "2024-12-20T00:00:00", "end": "2025-01-05T23:59:59"}
    ]
  }
}
```

During busy periods, thresholds are relaxed by `busy_period_factor` (default 1.5x).

See [CONFIGURATION.md](docs/CONFIGURATION.md) for full SLO configuration options.

## ML Improvements

### Temporal Train/Validation Split

Data is split chronologically (80% train, 20% validation) to prevent data leakage. Rolling features are computed separately for each split.

### Calibrated Severity Thresholds

Instead of hardcoded thresholds, severity levels are calibrated per model from validation data:

| Severity | Percentile | Description |
|----------|------------|-------------|
| Critical | Bottom 0.1% | Extreme outliers |
| High | Bottom 1% | Significant anomalies |
| Medium | Bottom 5% | Moderate deviations |
| Low | Bottom 10% | Minor anomalies |

### Contamination Estimation

Optimal contamination is estimated automatically using knee detection in the IF score distribution, bounded by service category limits.

### Drift Detection

When enabled (`check_drift: true`), the system monitors for distribution drift:

- **Univariate**: Z-score comparison against training statistics
- **Multivariate**: Mahalanobis distance using inverse covariance matrix

| Drift Score | Confidence Penalty |
|-------------|-------------------|
| < 3 | 0% (normal) |
| 3-5 | 15% (moderate) |
| > 5 | 30% (severe) |

### Minimum Sample Requirements

| Model Type | Minimum Samples |
|------------|-----------------|
| Univariate | 500 |
| Multivariate | 1000 |

## Installation

```bash
# Clone and enter directory
cd yaga2

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install with uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .

# Dev dependencies
uv pip install -e ".[dev]"
```

## Usage

### Training

```bash
# Train all configured services
python main.py

# Training fetches 30 days of data, splits 80/20, and:
# - Estimates contamination empirically
# - Calibrates severity thresholds
# - Stores drift detection baselines
```

### Inference

```bash
# Basic inference
python inference.py

# Verbose mode
python inference.py --verbose

# With drift detection
# (requires check_drift: true in config.json)
python inference.py --verbose
```

### Programmatic Usage

```python
from inference import SmartboxMLInferencePipeline

pipeline = SmartboxMLInferencePipeline(
    vm_endpoint="https://otel-metrics.production.smartbox.com",
    models_directory="./smartbox_models/",
    verbose=True
)

# Run inference with drift detection
results = pipeline.run_enhanced_time_aware_inference(
    service_names=["booking", "search"]
)

for service, result in results.items():
    # Check for anomalies
    if result.get('anomaly_count', 0) > 0:
        print(f"Anomalies in {service}: {result['anomalies']}")

    # Check for drift warnings
    if 'drift_warning' in result:
        print(f"Drift detected: {result['drift_warning']}")

    # Check for validation issues
    if result.get('validation_warnings'):
        print(f"Validation issues: {result['validation_warnings']}")
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VM_ENDPOINT` | VictoriaMetrics endpoint | `https://otel-metrics.production.smartbox.com` |
| `MODELS_DIR` | Models directory | `./smartbox_models/` |
| `ALERTS_DIR` | Alerts output | `./alerts/` |
| `FINGERPRINT_DB` | SQLite database | `./anomaly_state.db` |
| `OBSERVABILITY_URL` | API server URL | `http://localhost:8000` |

### config.json

```json
{
  "model": {
    "min_training_samples": 500,
    "min_multivariate_samples": 1000
  },
  "training": {
    "validation_fraction": 0.2,
    "contamination_estimation": {
      "method": "knee"
    }
  },
  "inference": {
    "check_drift": false
  }
}
```

See [CONFIGURATION.md](docs/CONFIGURATION.md) for full options.

## API Output

### Anomaly Detection Response

```json
{
  "alert_type": "anomaly_detected",
  "service_name": "booking",
  "timestamp": "2025-12-23T10:30:00",
  "time_period": "business_hours",
  "overall_severity": "high",
  "anomaly_count": 1,

  "anomalies": {
    "recent_degradation": {
      "type": "consolidated",
      "severity": "high",
      "confidence": 0.85,
      "description": "Latency degradation: 636ms (92nd percentile)",
      "recommended_actions": ["Check recent deployments", "Review dependencies"]
    }
  },

  "drift_warning": {
    "type": "model_drift",
    "overall_drift_score": 3.5,
    "confidence_penalty_applied": 0.15,
    "affected_metrics": ["request_rate"]
  },

  "validation_warnings": [],

  "performance_info": {
    "drift_check_enabled": true,
    "drift_penalty_applied": 0.15
  }
}
```

See [INFERENCE_API_PAYLOAD.md](docs/INFERENCE_API_PAYLOAD.md) for full specification.

## Admin Dashboard

A web-based configuration interface is available for managing services and SLOs:

```bash
# Run standalone
python admin_dashboard.py
# Access at http://localhost:8050

# Or via Docker
docker compose up -d admin-dashboard
```

Features:
- Service management with search/filter
- SLO threshold configuration with validation
- Service dependency graph visualization
- Configuration export/import
- Audit logging for all changes

## Project Structure

```
yaga2/
├── main.py                         # Training pipeline
├── inference.py                    # Inference pipeline
├── admin_dashboard.py              # Configuration web UI
├── config.json                     # Configuration
│
├── smartbox_anomaly/               # Core package
│   ├── core/                       # Config, logging, constants
│   ├── detection/                  # ML detector, time-aware, service config
│   ├── metrics/                    # VictoriaMetrics client
│   ├── fingerprinting/             # Incident tracking
│   ├── slo/                        # SLO evaluation
│   └── api/                        # Pydantic models
│
├── docs/                           # Documentation
│   ├── MACHINE_LEARNING.md
│   ├── ML_TRAINING.md
│   ├── CONFIGURATION.md
│   ├── INFERENCE_API_PAYLOAD.md
│   ├── OPERATIONS.md
│   └── ...
│
├── tests/                          # Test suite
│
├── Dockerfile                      # Production container
├── docker-compose.yml              # Multi-service deployment
│
├── smartbox_models/                # Trained models (generated)
└── alerts/                         # Alert output (generated)
```

## Testing

```bash
# Run all tests
uv run pytest tests/

# With coverage
uv run pytest tests/ --cov=smartbox_anomaly

# Specific tests
uv run pytest tests/test_detector.py -v
```

## Operations

### Enabling Drift Detection

1. Set `check_drift: true` in config.json
2. Retrain models: `python main.py`
3. Run inference: `python inference.py --verbose`

### Troubleshooting

| Issue | Solution |
|-------|----------|
| Too many false positives | Increase contamination for service |
| Missing anomalies | Decrease contamination |
| Drift warnings | Retrain models with recent data |
| Validation warnings | Check upstream data quality |

See [OPERATIONS.md](docs/OPERATIONS.md) for detailed runbooks.

## License

Proprietary - Smartbox Group

## Contributing

1. Follow existing code style
2. Add tests for new functionality
3. Update documentation
4. Run `pytest` and `ruff check` before submitting
