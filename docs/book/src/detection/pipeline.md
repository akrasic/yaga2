# Detection Pipeline

The complete detection pipeline from metrics collection to anomaly output.

## Pipeline Phases

### Phase 1: Metrics Collection

```
VictoriaMetrics
      │
      ▼
┌─────────────────────────────┐
│ Collect 5 core metrics      │
│ - request_rate              │
│ - application_latency       │
│ - dependency_latency       │
│ - database_latency         │
│ - error_rate               │
└─────────────────────────────┘
```

### Phase 2: Input Validation

Metrics are validated before processing:

| Check | Action |
|-------|--------|
| NaN/Inf values | Replaced with 0.0 |
| Negative rates | Capped at 0.0 |
| Negative latencies | Capped at 0.0 |
| Extreme latencies (>5 min) | Capped at 300,000ms |
| Extreme request rates (>1M/s) | Capped at 1,000,000 |
| Error rates > 100% | Capped at 1.0 |

Validation warnings are included in the output.

### Phase 3: Pass 1 Detection

First pass detects anomalies without dependency context:

```
For each service:
    1. Load time-aware model (based on current time period)
    2. Run univariate IF on each metric
    3. Run multivariate IF on combined metrics
    4. Collect anomaly signals
    5. Match signals against patterns
    6. Store result
```

### Phase 4: Pass 2 Detection (Cascade Analysis)

Second pass re-analyzes services with latency anomalies:

```
For each service with latency anomaly:
    1. Build dependency context from Pass 1 results
    2. Check upstream dependencies for anomalies
    3. Re-run pattern matching with dependency info
    4. Add cascade_analysis if root cause found
```

**Dependency Context**:

```json
{
  "upstream_status": "anomaly",
  "dependencies": {
    "vms": {"has_anomaly": true, "type": "database_bottleneck"},
    "titan": {"has_anomaly": true, "type": "latency_elevated"}
  },
  "root_cause_service": "titan"
}
```

### Phase 5: SLO Evaluation

Each anomaly is evaluated against SLO thresholds:

```
For each detected anomaly:
    1. Evaluate latency vs SLO
    2. Evaluate error rate vs SLO
    3. Evaluate database latency vs baseline
    4. Check request rate (surge/cliff)
    5. Determine combined SLO status
    6. Adjust severity if needed
```

See [SLO Evaluation](../slo/README.md) for details.

### Phase 6: Incident Lifecycle

Anomalies enter the incident lifecycle:

```
anomaly detected
      │
      ▼
┌─────────────────────────────┐
│ Check existing incidents    │
│ - Same fingerprint exists?  │
│ - Status of that incident?  │
└─────────────────────────────┘
      │
      ├─── No existing ──▶ CREATE (SUSPECTED)
      │
      ├─── SUSPECTED ──▶ Increment consecutive_detections
      │                   If >= 2: transition to OPEN
      │
      ├─── OPEN ──▶ CONTINUE (update occurrence_count)
      │
      └─── RECOVERING ──▶ Return to OPEN
```

See [Incident Lifecycle](../incidents/README.md) for details.

## Output Structure

Final output for each service:

```json
{
  "alert_type": "anomaly_detected",
  "service_name": "booking",
  "timestamp": "2024-01-15T10:30:00",
  "time_period": "business_hours",
  "overall_severity": "high",
  "anomaly_count": 1,

  "anomalies": {
    "latency_spike_recent": {
      "type": "consolidated",
      "severity": "high",
      "detection_signals": [...],
      "cascade_analysis": {...}
    }
  },

  "current_metrics": {...},
  "slo_evaluation": {...},
  "fingerprinting": {...}
}
```

## Error Handling

### Metrics Unavailable

When VictoriaMetrics is unreachable:

```json
{
  "alert_type": "metrics_unavailable",
  "service_name": "booking",
  "error": "Metrics collection failed",
  "failed_metrics": ["request_rate", "application_latency"],
  "skipped_reason": "critical_metrics_unavailable"
}
```

Detection is skipped to prevent false alerts from missing data.

### Model Not Found

When no trained model exists:

```json
{
  "alert_type": "error",
  "service_name": "new-service",
  "error_message": "No trained model found for time period"
}
```

Run training to create models for new services.
