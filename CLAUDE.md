# Yaga2 - Anomaly Detection System

## Overview

ML-based anomaly detection for Smartbox services. Detects latency, error rate, and traffic anomalies using time-aware Isolation Forest models with SLO-aware severity evaluation.

## Tech Stack

- **Python 3.13** with `uv` for dependency management
- **FastAPI** for admin dashboard
- **VictoriaMetrics** for metrics (endpoint: `otel-metrics.production.smartbox.com`)
- **SQLite** for fingerprinting state persistence
- **Pydantic** for API payload validation
- **mdBook** for user-facing documentation

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

# Build mdbook documentation
cd docs/book && mdbook build

# Serve mdbook locally
cd docs/book && mdbook serve
```

## Project Structure

```
yaga2/
├── inference.py              # Main batch inference entry point
├── admin_dashboard.py        # FastAPI admin UI (Yaga2 Control Center)
├── config.json               # All configuration (SLOs, services, thresholds)
├── smartbox_anomaly/
│   ├── inference/            # Inference module (see INFERENCE_MODULE_ARCHITECTURE.md)
│   │   ├── pipeline.py       # Main orchestrator (SmartboxMLInferencePipeline)
│   │   ├── detection_runner.py # Two-pass detection logic
│   │   ├── enrichment_runner.py # SLO/exception/service graph enrichment
│   │   ├── detection_engine.py  # ML detection engine
│   │   ├── model_manager.py     # Model loading and caching
│   │   ├── time_aware.py        # Time-aware detection
│   │   └── results_processor.py # Alert formatting and API posting
│   ├── detection/            # ML models, pattern matching, interpretations
│   ├── slo/                  # SLO-aware severity evaluation
│   ├── enrichment/           # Exception and service graph enrichment
│   ├── fingerprinting/       # Incident lifecycle tracking
│   ├── metrics/              # VictoriaMetrics client
│   ├── api/                  # Pydantic models for API payloads
│   └── core/                 # Config, logging, constants
├── smartbox_models/          # Trained ML models per service/time-period
├── tests/                    # Pytest test suite (439 tests)
├── docs/                     # Technical documentation
│   ├── book/                 # mdBook user documentation
│   │   └── src/              # mdBook source files
│   └── *.md                  # Technical reference docs
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
5. **Fingerprinting**: Tracks incident lifecycle (SUSPECTED → OPEN → RECOVERING → CLOSED)

### SLO Evaluation

- **Latency thresholds**: acceptable/warning/critical (ms)
- **Error rate thresholds**: acceptable/warning/critical (0.0-1.0)
- **Database latency**: Ratio-based (1.5x/2x/3x/5x baseline) with noise floor (default: 5ms)
- **Request rate**: Surge (3x baseline) and cliff (0.1x baseline) detection
- Anomalies below operational floors are suppressed (no alert sent)

### Important Patterns

- `lower_is_better` metrics: `error_rate`, `database_latency`, `dependency_latency` - low values are improvements, not anomalies
- Database latency < floor (5ms) → anomaly suppressed as operationally insignificant
- SLO config is in `config.json` under `slos` section

### Severity Adjustment Logic

When SLO evaluation is enabled:
- **SLO ok** (all metrics within acceptable) → severity adjusted to `low`
- **SLO warning** → severity stays `high`
- **SLO breached** → severity stays `critical`

The root `overall_severity` field reflects the SLO-adjusted value.

### Named Pattern Naming Convention

Patterns follow the standardized `{metric}_{state}_{modifier}` format:

**Metric Prefixes:**
- `request_rate_` - Traffic/request volume patterns
- `error_rate_` - Error-related patterns
- `application_latency_` - Internal processing latency
- `dependency_latency_` - External service call latency
- `database_latency_` - Database operation latency

**Common Patterns:**

| Pattern | Severity | Description |
|---------|----------|-------------|
| `request_rate_surge_failing` | critical | High traffic + high latency + high errors |
| `request_rate_surge_degrading` | high | High traffic causing latency increase |
| `request_rate_surge_healthy` | low | High traffic handled well (non-alerting) |
| `request_rate_cliff` | critical | Sudden traffic drop |
| `error_rate_critical` | critical | Very high error rate (>20%) |
| `error_rate_elevated` | high | Elevated error rate above baseline |
| `error_rate_fast_rejection` | critical | Fast failures with very high errors |
| `error_rate_partial_rejection` | high | Some requests being rejected |
| `dependency_latency_cascade` | high | External dependency causing slowdown |
| `application_latency_bottleneck` | high | Internal processing constraint |
| `database_latency_bottleneck` | high | Database is primary constraint |
| `database_latency_degraded` | medium | Database slow but app compensating |
| `latency_spike_recent` | high | Recent latency increase |
| `internal_latency_issue` | high | High latency with healthy deps |

**Cascade Naming with Root Cause:**

When cascade analysis identifies a root cause with high confidence (≥80%), the pattern name includes the root service:
- `dependency_latency_cascade:titan` - Cascade caused by titan service
- `dependency_latency_high:vms` - Latency issue traced to vms service

**Backward Compatibility:**

Old pattern names (e.g., `traffic_surge_healthy`, `downstream_cascade`) are automatically normalized to new names via `PATTERN_ALIASES` in `interpretations.py`. Config files using old names will work without modification.

### Incident Lifecycle (Fingerprinting)

Incidents follow a four-state lifecycle:

```
SUSPECTED → OPEN → RECOVERING → CLOSED
```

- **SUSPECTED**: First detection, waiting for confirmation (2 cycles default)
- **OPEN**: Confirmed incident, alerts being sent
- **RECOVERING**: Not detected for 1-2 cycles, grace period
- **CLOSED**: Resolved after grace period (3 cycles default)

Key behaviors:
- Only **confirmed** incidents (OPEN/RECOVERING) are sent to web API (v1.3.2+)
- Staleness check: gaps > 30 minutes create new incidents (`auto_stale`)

## Docker

```bash
# Build and run
docker compose up -d

# Copy existing DB to persistent volume
docker cp smartbox-anomaly:/app/anomaly_state.db ./data/anomaly_state.db
```

- Volume mount: `./data:/app/data` for persistent state
- Fingerprint DB path: `/app/data/anomaly_state.db`

## Documentation

### Technical Documentation (docs/)

Reference documentation for developers and operators:

- @docs/INFERENCE_MODULE_ARCHITECTURE.md - Inference module structure and design
- @docs/INFERENCE_API_PAYLOAD.md - API payload specification with all fields
- @docs/CONFIGURATION.md - Configuration options and SLO setup
- @docs/FINGERPRINTING.md - Incident lifecycle and fingerprinting
- @docs/DETECTION_SIGNALS.md - Detection methods and signals
- @docs/MACHINE_LEARNING.md - ML model details
- @docs/ML_TRAINING.md - Model training process
- @docs/OPERATIONS.md - Operational runbook
- @docs/DEPLOYMENT.md - Deployment guide
- @docs/API_SPECIFICATION.md - Full API spec
- @docs/KNOWN_ISSUES.md - Known issues and workarounds
- @docs/API_CHANGELOG.md - API version history and breaking changes

### User Documentation (docs/book/)

Comprehensive mdBook documentation for end-users. Located in `docs/book/src/`:

| Section | Key Files | Content |
|---------|-----------|---------|
| Introduction | `introduction.md` | System overview, core concepts (SLO, Isolation Forest, incidents) |
| Detection | `detection/` | Isolation Forest explanation, pattern matching, detection pipeline |
| SLO Evaluation | `slo/` | SLO concepts, severity adjustment, threshold configuration |
| Incidents | `incidents/` | State machine, confirmation logic, fingerprinting, resolution |
| Reference | `reference/` | Troubleshooting guide, decision matrix |

**Key mdBook files with comprehensive content:**

| File | Lines | Coverage |
|------|-------|----------|
| `incidents/README.md` | ~475 | Incident lifecycle overview, why tracking matters |
| `incidents/state-machine.md` | ~636 | Four-state lifecycle, transitions, grace periods |
| `incidents/confirmation.md` | ~490 | Confirmation logic, why 2-cycle default, tuning |
| `incidents/fingerprinting.md` | ~658 | Dual-ID system, database schema, staleness |
| `reference/troubleshooting.md` | ~1393 | Comprehensive diagnostic guide with decision trees |
| `detection/isolation-forest.md` | ~417 | ML algorithm explanation, training, inference |

Build and serve:
```bash
cd docs/book && mdbook serve
```

## Key Files to Know

| File | Purpose |
|------|---------|
| `inference.py` | Main entry point - orchestrates batch inference |
| `smartbox_anomaly/inference/pipeline.py` | Pipeline orchestrator - composes detection and enrichment |
| `smartbox_anomaly/inference/detection_runner.py` | Two-pass detection logic with dependency context |
| `smartbox_anomaly/inference/enrichment_runner.py` | SLO evaluation, exception, and service graph enrichment |
| `smartbox_anomaly/slo/evaluator.py` | SLO evaluation and severity adjustment |
| `smartbox_anomaly/detection/detector.py` | ML detection and pattern matching |
| `smartbox_anomaly/detection/interpretations.py` | Pattern definitions and recommendations |
| `smartbox_anomaly/fingerprinting/fingerprinter.py` | Incident lifecycle state machine |
| `config.json` | All configuration (SLOs, services, thresholds) |

## Recent Changes

### v1.4.0 - Enhanced Resolution Payload (January 2026)

Resolution payloads now include comprehensive context about service state at resolution time:
- `metrics_at_resolution` - Current metric values when resolved
- `slo_evaluation` - Full SLO evaluation showing operational status
- `comparison_to_baseline` - Statistical comparison to training data
- `health_summary` - Human-readable summary for UI display

Key implementation:
- `fingerprinter.py`: Added `_build_resolution_context()` method
- `evaluator.py`: Added `evaluate_metrics()` helper for standalone evaluation

### v1.3.3 - Inference Module Refactoring (January 2026)

Split `pipeline.py` (1,114 lines) into focused modules using composition pattern:

| Component | Lines | Purpose |
|-----------|-------|---------|
| `pipeline.py` | 452 | Orchestrator - composes runners |
| `detection_runner.py` | 516 | Two-pass detection with dependency context |
| `enrichment_runner.py` | 314 | SLO evaluation and context enrichment |

### v1.3.2 - Confirmed-Only Alerts (January 2026)

- Only OPEN/RECOVERING incidents sent to web API (fixes orphaned incidents)
- SUSPECTED incidents filtered before API calls

### v1.3.1 - SLO Improvements

- SLO severity adjustment: `ok` status now consistently → `low` severity
- Root `overall_severity` reflects SLO-adjusted value
- Pattern naming cleanup (e.g., `partial_outage` → `error_rate_critical`)
- Database latency noise floor (default 5ms)
- Error rate suppression via `error_rate_floor` config
