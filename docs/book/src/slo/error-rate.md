# Error Rate Evaluation

Error rate evaluation compares current error percentage against SLO thresholds.

## Thresholds

| Level | Default | Meaning |
|-------|---------|---------|
| `acceptable` | 0.5% | Error rate below this is fine |
| `warning` | 1.0% | Approaching SLO limit |
| `critical` | 2.0% | SLO breach |

## Error Rate Floor (Suppression)

To prevent alert noise from tiny error rate deviations, an optional `error_rate_floor` can suppress anomalies when errors are operationally insignificant.

```json
{
  "slos": {
    "services": {
      "booking": {
        "error_rate_acceptable": 0.002,
        "error_rate_floor": 0.002
      }
    }
  }
}
```

| Error Rate | Floor (0.2%) | Result |
|------------|--------------|--------|
| 0.01% | Below floor | **Suppressed** (no alert) |
| 0.15% | Below floor | **Suppressed** (no alert) |
| 0.25% | Above floor | Alert fires |

## Evaluation Logic

```
Current Error Rate
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│  Is error_rate < floor?                                 │
│  ──────────────────────                                 │
│  YES → Suppress anomaly entirely (no alert)             │
│  NO  → Continue...                                      │
└─────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│  Is error_rate < acceptable?                            │
│  ───────────────────────────                            │
│  YES → status: "ok", within_acceptable: true            │
│  NO  → Continue...                                      │
└─────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│  Is error_rate < warning?                               │
│  ────────────────────────                               │
│  YES → status: "elevated"                               │
│  NO  → Continue...                                      │
└─────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│  Is error_rate < critical?                              │
│  ─────────────────────────                              │
│  YES → status: "warning"                                │
│  NO  → status: "breached"                               │
└─────────────────────────────────────────────────────────┘
```

## Output Example

```json
{
  "error_rate_evaluation": {
    "status": "ok",
    "value": 0.001,
    "value_percent": "0.10%",
    "threshold_acceptable": 0.005,
    "threshold_warning": 0.01,
    "threshold_critical": 0.02,
    "within_acceptable": true
  }
}
```

## When Floor is Active

```json
{
  "slo_context": {
    "current_value": 0.005,
    "current_value_percent": "0.50%",
    "acceptable_threshold": 0.002,
    "critical_threshold": 0.01,
    "suppression_threshold": 0.002,
    "within_acceptable": false
  }
}
```

## Per-Service Configuration

```json
{
  "slos": {
    "services": {
      "booking": {
        "error_rate_acceptable": 0.002,
        "error_rate_warning": 0.005,
        "error_rate_critical": 0.01,
        "error_rate_floor": 0.002
      },
      "admin-api": {
        "error_rate_acceptable": 0.01,
        "error_rate_critical": 0.05
      }
    }
  }
}
```

## Exception Enrichment

When error SLO is breached (HIGH or CRITICAL severity), exception context is automatically queried and added to the alert:

```json
{
  "exception_context": {
    "service_name": "search",
    "total_exception_rate": 0.35,
    "top_exceptions": [
      {"type": "R2D2Exception", "rate": 0.217, "percentage": 62.0},
      {"type": "UserInputException", "rate": 0.083, "percentage": 23.7}
    ]
  }
}
```

This helps identify which exception types are causing the error spike.
