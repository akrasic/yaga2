# Database Latency Evaluation

Database latency uses a **hybrid approach**: noise floor filtering + ratio-based thresholds against training baseline.

## Why Ratio-Based?

Unlike application latency with fixed SLO targets, database latency varies significantly:
- A fast service might have 5ms DB latency normally
- A reporting service might have 500ms DB latency normally

**Ratio-based thresholds** adapt to each service's baseline.

## Noise Floor

Very low database latency values are filtered as operationally meaningless:

| Latency | Floor (5ms) | Result |
|---------|-------------|--------|
| 2ms | Below floor | Always `ok` |
| 0.3ms → 0.4ms | Below floor | Ignored |
| 10ms | Above floor | Evaluate ratio |

This prevents alerts for sub-millisecond changes that the ML might flag as anomalous.

## Ratio Thresholds

| Status | Ratio | Meaning |
|--------|-------|---------|
| `ok` | < 1.5x | Within normal variance |
| `info` | 1.5x - 2x | Slightly elevated |
| `warning` | 2x - 3x | Elevated, investigate |
| `high` | 3x - 5x | Significantly elevated |
| `critical` | ≥ 5x | SLO breach |

## Evaluation Logic

```
Current DB Latency
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│  Is latency < floor_ms?                                 │
│  ──────────────────────                                 │
│  YES → status: "ok", below_floor: true                  │
│  NO  → Continue...                                      │
└─────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│  Calculate ratio = current / baseline_mean              │
│  ──────────────────────────────────────                 │
│  Example: 25ms / 10ms = 2.5x                            │
└─────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│  Map ratio to status:                                   │
│  ────────────────────                                   │
│  < 1.5 → "ok"                                           │
│  < 2.0 → "info"                                         │
│  < 3.0 → "warning"                                      │
│  < 5.0 → "high"                                         │
│  ≥ 5.0 → "critical"                                     │
└─────────────────────────────────────────────────────────┘
```

## Output Examples

### Below Floor

```json
{
  "database_latency_evaluation": {
    "status": "ok",
    "value_ms": 2.0,
    "baseline_mean_ms": 1.0,
    "ratio": 0.0,
    "below_floor": true,
    "floor_ms": 5.0,
    "explanation": "Below noise floor (2.0ms < 5.0ms)"
  }
}
```

### Elevated Ratio

```json
{
  "database_latency_evaluation": {
    "status": "warning",
    "value_ms": 25.0,
    "baseline_mean_ms": 10.0,
    "ratio": 2.5,
    "below_floor": false,
    "floor_ms": 5.0,
    "thresholds": {
      "info": 1.5,
      "warning": 2.0,
      "high": 3.0,
      "critical": 5.0
    },
    "explanation": "DB latency elevated: 25.0ms is 2.5x baseline (10.0ms)"
  }
}
```

## Per-Service Configuration

Services with different DB performance characteristics can have custom thresholds:

```json
{
  "slos": {
    "services": {
      "search": {
        "database_latency_floor_ms": 2.0,
        "database_latency_ratios": {
          "info": 1.3,
          "warning": 1.5,
          "high": 2.0,
          "critical": 3.0
        }
      },
      "reporting": {
        "database_latency_floor_ms": 50.0,
        "database_latency_ratios": {
          "info": 2.0,
          "warning": 3.0,
          "high": 5.0,
          "critical": 10.0
        }
      }
    }
  }
}
```

## Relationship to Pattern Matching

Database latency evaluation complements pattern matching:

| Pattern | Database Latency Evaluation |
|---------|---------------------------|
| `database_degradation` | DB ratio 2-3x, compensating |
| `database_bottleneck` | DB ratio ≥3x, dominant latency |

The SLO evaluation provides the **ratio context** that pattern matching uses for severity.
