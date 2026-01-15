# Request Rate Evaluation (Surge/Cliff)

Request rate evaluation detects significant traffic changes: **surges** (spikes) and **cliffs** (drops).

## Key Insight

Traffic anomalies use **correlation-based severity**:
- A surge alone is often benign (marketing campaign, organic growth)
- A surge becomes problematic when causing SLO issues
- A cliff is inherently concerning (may indicate upstream failure)

## Thresholds

| Type | Threshold | Default |
|------|-----------|---------|
| Surge | ≥ 200% of baseline | 2x normal traffic |
| Cliff | ≤ 50% of baseline | Half normal traffic |

## Surge Severity Logic

| Condition | Severity | Rationale |
|-----------|----------|-----------|
| Surge only | `informational` | Normal growth or campaign |
| Surge + latency SLO breach | `warning` | Capacity stress |
| Surge + error SLO breach | `high` | Active incident |

## Cliff Severity Logic

| Condition | Severity | Rationale |
|-----------|----------|-----------|
| Cliff off-peak | `warning` | May be expected |
| Cliff peak hours | `high` | Likely incident |
| Cliff + errors | `critical` | Confirmed incident |

## Evaluation Flow

```
                    ┌───────────────────────────────────────────┐
                    │         Calculate Traffic Ratio           │
                    │    ratio = current / baseline_mean        │
                    └───────────────────────────────────────────┘
                                        │
               ┌────────────────────────┼────────────────────────┐
               │                        │                        │
               ▼                        ▼                        ▼
        ratio ≥ 2.0              0.5 < ratio < 2.0          ratio ≤ 0.5
         (SURGE)                    (NORMAL)                  (CLIFF)
               │                        │                        │
               ▼                        │                        ▼
    ┌──────────────────────┐           │          ┌──────────────────────┐
    │ Check SLO correlation │           │          │ Check time of day    │
    │ - Latency breach?     │           │          │ - Peak hours?        │
    │ - Error breach?       │           │          │ - Errors elevated?   │
    └──────────────────────┘           │          └──────────────────────┘
               │                        │                        │
               ▼                        ▼                        ▼
       severity based               "ok"                  severity based
       on correlation               status                on peak/errors
```

## Output Examples

### Surge (Standalone)

```json
{
  "request_rate_evaluation": {
    "status": "info",
    "type": "surge",
    "severity": "informational",
    "value_rps": 250.0,
    "baseline_mean_rps": 100.0,
    "ratio": 2.5,
    "threshold_percent": 200.0,
    "correlated_with_latency": false,
    "correlated_with_errors": false,
    "explanation": "Traffic surge (2.5x baseline) without SLO impact. Normal growth or campaign traffic."
  }
}
```

### Surge with Latency Impact

```json
{
  "request_rate_evaluation": {
    "status": "warning",
    "type": "surge",
    "severity": "warning",
    "value_rps": 350.0,
    "baseline_mean_rps": 100.0,
    "ratio": 3.5,
    "correlated_with_latency": true,
    "correlated_with_errors": false,
    "explanation": "Traffic surge (3.5x baseline) correlating with latency SLO breach - capacity issue."
  }
}
```

### Cliff (Peak Hours)

```json
{
  "request_rate_evaluation": {
    "status": "high",
    "type": "cliff",
    "severity": "high",
    "value_rps": 30.0,
    "baseline_mean_rps": 100.0,
    "ratio": 0.3,
    "threshold_percent": 50.0,
    "is_peak_hours": true,
    "correlated_with_errors": false,
    "explanation": "Traffic cliff (0.3x baseline) during peak hours - investigate potential incident."
  }
}
```

### Cliff with Errors

```json
{
  "request_rate_evaluation": {
    "status": "critical",
    "type": "cliff",
    "severity": "critical",
    "value_rps": 15.0,
    "baseline_mean_rps": 100.0,
    "ratio": 0.15,
    "is_peak_hours": true,
    "correlated_with_errors": true,
    "explanation": "Traffic cliff (0.15x baseline) with errors - likely upstream failure or routing issue."
  }
}
```

## Configuration

```json
{
  "slos": {
    "defaults": {
      "request_rate_surge_threshold": 2.0,
      "request_rate_cliff_threshold": 0.5
    },
    "services": {
      "booking": {
        "request_rate_evaluation": {
          "surge": {
            "threshold": 3.0
          },
          "cliff": {
            "standalone_severity": "high",
            "peak_hours_severity": "critical"
          }
        }
      }
    }
  }
}
```

## Relationship to Pattern Matching

| Pattern | Request Rate Evaluation |
|---------|------------------------|
| `traffic_surge_healthy` | Surge, no SLO correlation |
| `traffic_surge_degrading` | Surge + latency correlation |
| `traffic_surge_failing` | Surge + error correlation |
| `traffic_cliff` | Cliff detected |
