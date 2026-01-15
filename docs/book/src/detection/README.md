# Detection Layer

The detection layer identifies anomalies using a **sequential pipeline** where ML detection triggers pattern interpretation.

## Pipeline Flow

```
Current Metrics
      │
      ▼
┌─────────────────────────────────────┐
│   Phase 1: Isolation Forest (IF)    │
│   - Per-metric anomaly scoring      │
│   - Multivariate relationship check │
│                                     │
│   Output: List of AnomalySignals    │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│   Phase 2: Pattern Interpretation   │
│   - Convert signals to levels       │
│   - Match against named patterns    │
│   - Build interpreted anomaly       │
└──────────────────┬──────────────────┘
                   │
                   ▼
           Single Alert
     (with pattern interpretation)
```

## Key Concepts

### Anomaly Signal

When IF detects an anomaly, it produces a signal:

```json
{
  "metric": "application_latency",
  "score": -0.35,
  "direction": "high",
  "percentile": 92.0
}
```

### Metric Levels

Signals are converted to semantic levels for pattern matching:

| Percentile | Level |
|------------|-------|
| > 95 | `very_high` |
| 90 - 95 | `high` |
| 80 - 90 | `elevated` |
| 70 - 80 | `moderate` |
| 10 - 70 | `normal` |
| 5 - 10 | `low` |
| < 5 | `very_low` |

### Lower-is-Better Metrics

Some metrics treat low values as improvements, not anomalies:

- `database_latency` - Faster queries are good
- `dependency_latency` - Faster dependencies are good
- `error_rate` - Fewer errors are good

For these metrics, when IF flags them as "low", they are treated as `normal`.

## Detection Methods

| Method | Type | Best For |
|--------|------|----------|
| [Isolation Forest](./isolation-forest.md) | ML | Novel/unknown anomalies |
| [Pattern Matching](./pattern-matching.md) | Rule-based | Known operational scenarios |

## Sections

- [Isolation Forest (ML)](./isolation-forest.md) - How ML detection works
- [Pattern Matching](./pattern-matching.md) - Named patterns and interpretations
- [Detection Pipeline](./pipeline.md) - End-to-end detection flow
