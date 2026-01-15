# Configuration Reference

Complete configuration reference for `config.json`.

## Configuration File Location

The system searches for configuration in this order:

1. `CONFIG_FILE` environment variable
2. `./config.json` (current directory)
3. `./config/config.json`
4. `~/.smartbox/config.json`
5. `/etc/smartbox/config.json`

## Core Sections

### VictoriaMetrics

```json
{
  "victoria_metrics": {
    "endpoint": "https://otel-metrics.production.smartbox.com",
    "timeout_seconds": 10,
    "max_retries": 3,
    "pool_connections": 20,
    "circuit_breaker_threshold": 5,
    "circuit_breaker_timeout_seconds": 300
  }
}
```

| Field | Default | Description |
|-------|---------|-------------|
| `endpoint` | required | VictoriaMetrics server URL |
| `timeout_seconds` | 10 | Request timeout |
| `max_retries` | 3 | Retry attempts |
| `circuit_breaker_threshold` | 5 | Failures before circuit opens |

### SLO Configuration

```json
{
  "slos": {
    "enabled": true,
    "allow_downgrade_to_informational": true,
    "require_slo_breach_for_critical": true,
    "defaults": {
      "latency_acceptable_ms": 500,
      "latency_warning_ms": 800,
      "latency_critical_ms": 1000,
      "error_rate_acceptable": 0.005,
      "error_rate_warning": 0.01,
      "error_rate_critical": 0.02,
      "error_rate_floor": 0,
      "database_latency_floor_ms": 5.0,
      "database_latency_ratios": {
        "info": 1.5,
        "warning": 2.0,
        "high": 3.0,
        "critical": 5.0
      },
      "busy_period_factor": 1.5
    },
    "services": {
      "booking": {
        "latency_acceptable_ms": 300,
        "latency_critical_ms": 500,
        "error_rate_acceptable": 0.002,
        "error_rate_floor": 0.002
      }
    },
    "busy_periods": [
      {
        "start": "2024-12-20T00:00:00",
        "end": "2025-01-05T23:59:59"
      }
    ]
  }
}
```

| Field | Default | Description |
|-------|---------|-------------|
| `enabled` | true | Enable SLO evaluation |
| `allow_downgrade_to_informational` | true | Allow severity reduction when SLO ok |
| `require_slo_breach_for_critical` | true | Only critical if SLO breached |
| `latency_acceptable_ms` | 500 | Latency SLO acceptable threshold |
| `latency_critical_ms` | 1000 | Latency SLO critical threshold |
| `error_rate_acceptable` | 0.005 | Error rate acceptable (0.5%) |
| `error_rate_floor` | 0 | Error rate suppression floor |
| `database_latency_floor_ms` | 5.0 | DB latency noise floor |

### Fingerprinting

```json
{
  "fingerprinting": {
    "db_path": "./anomaly_state.db",
    "cleanup_max_age_hours": 72,
    "incident_separation_minutes": 30,
    "confirmation_cycles": 2,
    "resolution_grace_cycles": 3
  }
}
```

| Field | Default | Description |
|-------|---------|-------------|
| `db_path` | ./anomaly_state.db | SQLite database path |
| `cleanup_max_age_hours` | 72 | Hours before cleanup |
| `incident_separation_minutes` | 30 | Gap triggering new incident |
| `confirmation_cycles` | 2 | Cycles to confirm (send alert) |
| `resolution_grace_cycles` | 3 | Cycles before closing |

### Services

```json
{
  "services": {
    "critical": ["booking", "search", "mobile-api"],
    "standard": ["friday", "gambit", "titan"],
    "micro": ["fa5"],
    "admin": ["m2-fr-adm", "m2-it-adm"],
    "core": ["m2-bb", "m2-fr"]
  }
}
```

Service categories affect default contamination rates:

| Category | Contamination | Description |
|----------|---------------|-------------|
| critical | 0.03 | Revenue-critical services |
| standard | 0.05 | Normal production |
| core | 0.04 | Platform infrastructure |
| admin | 0.06 | Administrative tools |
| micro | 0.08 | Low-traffic services |

### Dependencies

```json
{
  "dependencies": {
    "graph": {
      "booking": ["search", "vms", "r2d2"],
      "vms": ["titan"],
      "search": ["catalog", "r2d2"]
    },
    "cascade_detection": {
      "enabled": true,
      "max_depth": 5
    }
  }
}
```

### Model Configuration

```json
{
  "model": {
    "models_directory": "./smartbox_models/",
    "min_training_samples": 500,
    "min_multivariate_samples": 1000,
    "default_contamination": 0.05,
    "default_n_estimators": 200,
    "contamination_by_service": {
      "booking": 0.02,
      "search": 0.04
    }
  }
}
```

### Time Periods

```json
{
  "time_periods": {
    "business_hours": {"start": 8, "end": 18, "weekdays_only": true},
    "evening_hours": {"start": 18, "end": 22, "weekdays_only": true},
    "night_hours": {"start": 22, "end": 6, "weekdays_only": true},
    "weekend_day": {"start": 8, "end": 22, "weekends_only": true},
    "weekend_night": {"start": 22, "end": 8, "weekends_only": true}
  }
}
```

### Inference

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

## Environment Variables

| Variable | Config Path | Description |
|----------|-------------|-------------|
| `CONFIG_FILE` | - | Path to config file |
| `VM_ENDPOINT` | victoria_metrics.endpoint | VictoriaMetrics URL |
| `FINGERPRINT_DB` | fingerprinting.db_path | SQLite path |
| `OBSERVABILITY_URL` | observability_api.base_url | API server URL |
| `OBSERVABILITY_ENABLED` | observability_api.enabled | Enable API |

## Docker Configuration

```yaml
environment:
  - TZ=UTC
  - CONFIG_PATH=/app/config.json
  - TRAIN_SCHEDULE=0 2 * * *
  - INFERENCE_SCHEDULE=*/10 * * * *
volumes:
  - ./smartbox_models:/app/smartbox_models
  - ./data:/app/data
  - ./config.json:/app/config.json
```

## Adding New Services

1. Add to appropriate category in `services` section
2. Optionally add per-service SLO thresholds
3. Optionally add contamination override
4. Run training: `docker compose run --rm yaga train`
5. Verify: `docker compose run --rm yaga inference --verbose`
