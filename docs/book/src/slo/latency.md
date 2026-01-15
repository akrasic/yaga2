# Latency Evaluation

Latency evaluation compares current response time against SLO thresholds.

## Thresholds

| Level | Default | Meaning |
|-------|---------|---------|
| `acceptable` | 500ms | Latency below this is fine |
| `warning` | 800ms | Approaching SLO limit |
| `critical` | 1000ms | SLO breach |

## Evaluation Logic

```
Current Latency
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│  Is latency < acceptable?                               │
│  ────────────────────────                               │
│  YES → status: "ok", proximity: latency/acceptable      │
│  NO  → Continue...                                      │
└─────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│  Is latency < warning?                                  │
│  ─────────────────────                                  │
│  YES → status: "elevated", minor concern                │
│  NO  → Continue...                                      │
└─────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│  Is latency < critical?                                 │
│  ──────────────────────                                 │
│  YES → status: "warning", investigate soon              │
│  NO  → status: "breached", immediate action             │
└─────────────────────────────────────────────────────────┘
```

## Proximity Score

The `proximity` value indicates how close to breach (0.0 - 1.0+):

| Proximity | Interpretation |
|-----------|----------------|
| 0.0 - 0.5 | Comfortable margin |
| 0.5 - 0.8 | Getting close |
| 0.8 - 1.0 | Near threshold |
| > 1.0 | Threshold exceeded |

## Output Example

```json
{
  "latency_evaluation": {
    "status": "warning",
    "value": 750.0,
    "threshold_acceptable": 500,
    "threshold_warning": 800,
    "threshold_critical": 1000,
    "proximity": 0.94
  }
}
```

## Per-Service Configuration

Critical services can have stricter thresholds:

```json
{
  "slos": {
    "services": {
      "booking": {
        "latency_acceptable_ms": 300,
        "latency_warning_ms": 400,
        "latency_critical_ms": 500
      },
      "search": {
        "latency_acceptable_ms": 200,
        "latency_critical_ms": 400
      }
    }
  }
}
```

## Busy Period Handling

During configured busy periods, thresholds are relaxed by `busy_period_factor`:

```json
{
  "slos": {
    "defaults": {
      "busy_period_factor": 1.5
    },
    "busy_periods": [
      {"start": "2024-12-20T00:00:00", "end": "2025-01-05T23:59:59"}
    ]
  }
}
```

During busy periods:
- 500ms acceptable → 750ms acceptable
- 1000ms critical → 1500ms critical
