# Smartbox ML Anomaly Detection Pipeline

A production-grade machine learning pipeline for real-time anomaly detection in microservices, using Isolation Forest models with time-aware period separation and enhanced explainability features.

## Features

- **Time-Aware Detection**: 5 behavioral periods (business hours, evening, night, weekend day/night)
- **Anomaly Fingerprinting**: Incident lifecycle tracking with two-level identity system
- **Explainable AI**: Feature contributions, business impact analysis, and recommendations
- **Lazy Loading**: Memory-efficient model loading (only loads needed time period)
- **Circuit Breaker**: Fault-tolerant metrics collection from VictoriaMetrics
- **SQLite Persistence**: Stateful incident tracking across runs

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    INFERENCE EXECUTION                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────┐
│   VictoriaMetrics Client     │  ← Collects real-time metrics
│   (vmclient.py)              │
└──────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────┐
│  TimeAwareAnomalyDetector    │  ← Routes to period-specific model
│  (time_aware_anomaly_        │
│   detection.py)              │
└──────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────┐
│  SmartboxAnomalyDetector     │  ← Isolation Forest detection
│  (anomaly_models.py)         │
└──────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────┐
│  AnomalyFingerprinter        │  ← Incident lifecycle management
│  (anomaly_fingerprinter.py)  │
└──────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────┐
│  Observability Service       │  ← API output
└──────────────────────────────┘
```

## Project Structure

```
yaga2/
├── main.py                         # Training pipeline entry point
├── inference.py                    # Inference pipeline entry point
├── anomaly_models.py               # ML model implementation
├── anomaly_fingerprinter.py        # Incident tracking
├── time_aware_anomaly_detection.py # Time-period routing
├── vmclient.py                     # VictoriaMetrics client
├── pyproject.toml                  # Project configuration
│
├── smartbox_anomaly/               # Core package
│   ├── __init__.py                 # Package exports
│   ├── compat.py                   # Backward compatibility
│   ├── core/                       # Core infrastructure
│   │   ├── config.py               # Centralized configuration
│   │   ├── constants.py            # Enums and constants
│   │   ├── exceptions.py           # Custom exception hierarchy
│   │   ├── logging.py              # Structured logging
│   │   └── protocols.py            # Interfaces/protocols
│   ├── metrics/                    # Metrics collection
│   │   ├── client.py               # VictoriaMetrics client
│   │   └── validation.py           # Input validation
│   ├── detection/                  # Anomaly detection
│   │   └── detector.py             # ML detector implementation
│   ├── fingerprinting/             # Incident tracking
│   │   └── fingerprinter.py        # Anomaly fingerprinter
│   └── api/                        # API models
│       └── models.py               # Pydantic response models
│
├── tests/                          # Test suite (210+ tests)
│   ├── conftest.py                 # Pytest fixtures
│   ├── fixtures.py                 # Shared test fixtures
│   ├── test_*.py                   # Unit tests
│   └── test_integration.py         # Integration tests
│
├── smartbox_models/                # Trained models (generated)
├── smartbox_training_data/         # Training data (generated)
└── alerts/                         # Alert output storage (generated)
```

## Installation

```bash
# Clone the repository
cd yaga2

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies (using uv for faster installs)
uv pip install -e .

# Or with pip
pip install -e .

# Install dev dependencies for testing
uv pip install -e ".[dev]"
```

## Quick Start Guide

The pipeline has three main phases: **Data Collection** → **Training** → **Inference**

### Step 1: Verify VictoriaMetrics Connectivity

Before training, ensure you have access to VictoriaMetrics:

```bash
# Test connectivity (replace with your endpoint)
curl -s "https://otel-metrics.production.smartbox.com/api/v1/query?query=up" | head -c 200
```

### Step 2: Train Models

Training collects historical metrics from VictoriaMetrics and builds time-aware Isolation Forest models:

```bash
# Train models for all configured services (30 days of data)
python main.py

# Or use the installed entry point
smartbox-train
```

**What happens during training:**
1. Connects to VictoriaMetrics and discovers available services
2. Extracts 30 days of metrics (request_rate, latencies, error_rate)
3. Engineers features (rolling statistics, time-based features)
4. Trains 5 time-period models per service:
   - `business_hours` (Mon-Fri 08:00-18:00)
   - `evening_hours` (Mon-Fri 18:00-22:00)
   - `night_hours` (Mon-Fri 22:00-06:00)
   - `weekend_day` (Sat-Sun 08:00-22:00)
   - `weekend_night` (Sat-Sun 22:00-08:00)
5. Validates models and saves to `./smartbox_models/`
6. Saves training data to `./smartbox_training_data/`

**Training output structure:**
```
smartbox_models/
├── booking_business_hours/
│   └── model_data.json
├── booking_evening_hours/
│   └── model_data.json
├── booking_night_hours/
│   └── model_data.json
├── booking_weekend_day/
│   └── model_data.json
└── booking_weekend_night/
    └── model_data.json
```

### Step 3: Run Inference

Once models are trained, run real-time anomaly detection:

```bash
# Production mode (JSON output only)
python inference.py

# Verbose mode with detailed logging and explanations
python inference.py --verbose

# Custom fingerprinting database location
python inference.py --fingerprint-db ./custom_state.db

# Or use the installed entry point
smartbox-inference
smartbox-inference --verbose
```

**Inference process:**
1. Loads trained models (lazy loading - only loads needed time period)
2. Collects current metrics from VictoriaMetrics
3. Routes to appropriate time-period model based on current time
4. Runs Isolation Forest anomaly detection
5. Applies fingerprinting for incident lifecycle tracking
6. Outputs JSON alerts and sends to observability service

## Usage Examples

### Basic Production Usage

```bash
# Run inference and pipe output
python inference.py | jq '.[] | select(.alert_type == "anomaly_detected")'

# Run on a schedule (e.g., every 5 minutes via cron)
*/5 * * * * cd /path/to/yaga2 && .venv/bin/python inference.py >> /var/log/smartbox-anomaly.log 2>&1
```

### Training Specific Services

Edit `main.py` to customize the service list:

```python
# In main.py, modify the specific_services list:
specific_services = ["booking", "friday", "search", "fa5", "gambit"]
results = training_pipeline.train_all_services_time_aware(specific_services)
```

### Custom Training Configuration

```python
from main import EnhancedSmartboxTrainingPipeline

# Custom VictoriaMetrics endpoint
pipeline = EnhancedSmartboxTrainingPipeline(
    vm_endpoint="https://your-vm-endpoint.com"
)

# Train with custom lookback period (default is 30 days)
result = pipeline.train_service_model_time_aware("booking", lookback_days=60)
```

### Programmatic Inference

```python
from inference import SmartboxMLInferencePipeline

# Initialize pipeline
pipeline = SmartboxMLInferencePipeline(
    vm_endpoint="https://otel-metrics.production.smartbox.com",
    models_directory="./smartbox_models/",
    alerts_directory="./alerts/",
    verbose=True
)

# Check system status
status = pipeline.get_system_status()
print(f"Available services: {status['available_services']}")

# Run inference for specific services
results = pipeline.run_enhanced_time_aware_inference(
    service_names=["booking", "search"]
)

# Process results
for service, result in results.items():
    if result.get('anomaly_count', 0) > 0:
        print(f"Anomalies detected in {service}: {result['anomalies']}")
```

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=smartbox_anomaly --cov-report=html

# Run specific test file
pytest tests/test_validation.py -v

# Run only unit tests
pytest tests/ -m unit

# Run integration tests
pytest tests/ -m integration
```

## Code Quality

```bash
# Run linter
ruff check .

# Run type checker
mypy smartbox_anomaly/

# Format code
ruff format .
```

## Configuration

Configuration can be set via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `VM_ENDPOINT` | VictoriaMetrics endpoint | `https://otel-metrics.production.smartbox.com` |
| `VM_TIMEOUT` | Query timeout (seconds) | `10` |
| `VM_MAX_RETRIES` | Max retry attempts | `3` |
| `MODELS_DIR` | Models directory path | `./smartbox_models/` |
| `ALERTS_DIR` | Alerts output directory | `./alerts/` |
| `FINGERPRINT_DB` | SQLite database path | `./anomaly_state.db` |
| `OBSERVABILITY_URL` | Observability service URL | `http://localhost:8000` |

## Time Periods

The pipeline uses 5 behavioral periods:

| Period | Hours | Days |
|--------|-------|------|
| `business_hours` | 08:00-18:00 | Mon-Fri |
| `evening_hours` | 18:00-22:00 | Mon-Fri |
| `night_hours` | 22:00-06:00 | Mon-Fri |
| `weekend_day` | 08:00-22:00 | Sat-Sun |
| `weekend_night` | 22:00-08:00 | Sat-Sun |

## Anomaly Severity Levels

| Severity | IF Score | Description |
|----------|----------|-------------|
| Critical | < -0.6 | Immediate attention required |
| High | -0.6 to -0.3 | Significant deviation |
| Medium | -0.3 to -0.1 | Moderate deviation |
| Low | ≥ -0.1 | Minor deviation |

## Fingerprinting

The fingerprinting system provides:

- **Fingerprint ID**: Deterministic hash for pattern tracking
- **Incident ID**: Unique identifier for each occurrence
- **State Machine**: CREATE → CONTINUE → RESOLVE → CLOSED

See `docs/FINGERPRINTING.md` for detailed documentation.

## API Output

### Anomaly Detection Response

```json
{
  "alert_type": "anomaly_detected",
  "service": "booking",
  "timestamp": "2024-01-15T10:30:00",
  "overall_severity": "high",
  "anomaly_count": 1,
  "anomalies": [
    {
      "type": "multivariate_enhanced_isolation_forest",
      "severity": "high",
      "confidence_score": 0.85,
      "description": "Application latency significantly elevated",
      "metadata": {
        "fingerprint_id": "anomaly_abc123",
        "incident_id": "incident_xyz789",
        "occurrence_count": 3
      }
    }
  ]
}
```

### Incident Resolution Response

```json
{
  "alert_type": "incident_resolved",
  "service": "booking",
  "incident_id": "incident_xyz789",
  "fingerprint_id": "anomaly_abc123",
  "resolution_details": {
    "final_severity": "high",
    "total_occurrences": 5,
    "incident_duration_minutes": 45
  }
}
```

## Development

### Code Quality Standards

- Type hints throughout the codebase
- Centralized configuration management
- Custom exception hierarchy
- Protocol-based interfaces for testability
- Input validation at system boundaries

### Adding New Services

1. Add service to known services in `config.py` → `ServiceConfig`
2. Or let the auto-detection categorize based on naming patterns

### Adding New Metrics

1. Add metric name to `constants.py` → `MetricName`
2. Add PromQL query to `vmclient.py` → `VictoriaMetricsClient.QUERIES`
3. Update validation in `validation.py` if needed

## Contributing

1. Follow the existing code style
2. Add tests for new functionality
3. Update documentation as needed
4. Run `pytest` before submitting changes
