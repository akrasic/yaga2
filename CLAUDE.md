# Yaga2 - Anomaly Detection System

## Overview

ML-based anomaly detection for Smartbox services. Detects latency, error rate, and traffic anomalies using time-aware Isolation Forest models with SLO-aware severity evaluation.

## Tech Stack

- **Python 3.13** with `uv` for dependency management
- **FastAPI** for admin dashboard
- **VictoriaMetrics** for metrics (endpoint: `otel-metrics.production.smartbox.com`)
- **SQLite** for fingerprinting state persistence
- **Pydantic** for API payload validation

## Commands

```bash
# Run tests
uv run pytest

# Run specific test file
uv run pytest tests/test_slo_evaluator.py -v

# Lint
ruff check

# Run inference pipeline
uv run python inference.py --verbose

# Run admin dashboard
uv run python admin_dashboard.py
```

## Project Structure

```
yaga2/
├── inference.py              # Main batch inference pipeline
├── admin_dashboard.py        # FastAPI admin UI (Yaga2 Control Center)
├── config.json               # All configuration (SLOs, services, thresholds)
├── smartbox_anomaly/
│   ├── detection/            # ML models, pattern matching, interpretations
│   ├── slo/                  # SLO-aware severity evaluation
│   ├── enrichment/           # Exception and service graph enrichment
│   ├── metrics/              # VictoriaMetrics client
│   ├── api/                  # Pydantic models for API payloads
│   └── core/                 # Config, logging, constants
├── smartbox_models/          # Trained ML models per service/time-period
├── tests/                    # Pytest test suite
└── docs/                     # Project documentation
```

## Key Architecture Concepts

### Inference Pipeline Flow
```
Detection (ML) → SLO Evaluation → Exception Enrichment → Service Graph Enrichment → Alert
```

1. **Detection**: Time-aware Isolation Forest + named pattern matching
2. **SLO Evaluation**: Adjusts/suppresses anomalies based on operational thresholds
3. **Exception Enrichment**: Queries `events_total` by exception_type on error SLO breach
4. **Service Graph Enrichment**: Queries `traces_service_graph_request_total` on latency SLO breach
5. **Fingerprinting**: Tracks incident lifecycle (CREATE → CONTINUE → RESOLVE)

### SLO Evaluation

- **Latency thresholds**: acceptable/warning/critical (ms)
- **Error rate thresholds**: acceptable/warning/critical (0.0-1.0)
- **Database latency**: Ratio-based (1.5x/2x/3x/5x baseline) with noise floor (default: 1ms)
- **Request rate**: Surge (3x baseline) and cliff (0.1x baseline) detection
- Anomalies below operational floors are suppressed (no alert sent)

### Important Patterns

- `lower_is_better` metrics: `error_rate`, `database_latency`, `client_latency` - low values are improvements, not anomalies
- Database latency < floor (1ms) → anomaly suppressed as operationally insignificant
- SLO config is in `config.json` under `slos` section

## Docker

```bash
# Build and run
docker-compose up -d

# Copy existing DB to persistent volume
docker cp smartbox-anomaly:/app/anomaly_state.db ./data/anomaly_state.db
```

- Volume mount: `./data:/app/data` for persistent state
- Fingerprint DB path: `/app/data/anomaly_state.db`

## Documentation

Detailed documentation in `docs/` directory:

@docs/INFERENCE_API_PAYLOAD.md - API payload specification with all fields
@docs/CONFIGURATION.md - Configuration options and SLO setup
@docs/FINGERPRINTING.md - Incident lifecycle and fingerprinting
@docs/DETECTION_SIGNALS.md - Detection methods and signals
@docs/MACHINE_LEARNING.md - ML model details
@docs/ML_TRAINING.md - Model training process
@docs/OPERATIONS.md - Operational runbook
@docs/DEPLOYMENT.md - Deployment guide
@docs/API_SPECIFICATION.md - Full API spec
@docs/KNOWN_ISSUES.md - Known issues and workarounds

## Recent Changes

- **Service Graph Enrichment**: Queries downstream service calls when client_latency SLO breached
- **Database Latency Floor**: Values below 1ms are suppressed as operationally insignificant
- **SLO-based Alert Suppression**: Anomalies with metrics below operational thresholds are filtered out entirely
