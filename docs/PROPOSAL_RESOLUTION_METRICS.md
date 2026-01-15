# Proposal: Enhanced Resolution Payload with Metrics and SLO Context

**Date:** 2026-01-13
**Status:** Ready for Web UI Evaluation
**Schema Version Change:** 1.3.2 → 1.4.0

---

## Problem Statement

When an incident resolves, the current payload only includes metadata about the incident lifecycle:

```json
{
  "fingerprint_id": "anomaly_061598e9ca91",
  "incident_id": "incident_abc123def456",
  "anomaly_name": "database_degradation",
  "final_severity": "medium",
  "resolved_at": "2025-12-17T14:30:00.000000",
  "total_occurrences": 5,
  "incident_duration_minutes": 45,
  "first_seen": "2025-12-17T13:45:00.000000",
  "service_name": "booking",
  "resolution_reason": "resolved"
}
```

**What's missing:**
1. The metric values at resolution time that demonstrate the service returned to normal
2. SLO evaluation context showing WHY the alert was eligible to close
3. Statistical context for comparison against baseline

**Why it matters:**
1. Web UI can show "what healthy looks like" alongside incident history
2. Enables "Alert Closed Summary" view with full context
3. SRE teams can verify the resolution was genuine, not just a detection gap
4. Post-incident analysis can compare incident peak vs resolved values

---

## Technical Context

The `current_metrics` and SLO evaluation data **are available** at resolution time but not currently included:

```python
# In fingerprinter.py process_anomalies():
current_metrics = anomaly_result.get("current_metrics", {})  # Line 259

# But _process_resolutions doesn't receive it:
resolved_incidents = self._process_resolutions(
    active_incidents, processed_fingerprints, timestamp  # No metrics passed!
)
```

The SLO evaluator produces comprehensive evaluation data via `SLOEvaluationResult.to_dict()` which is already in the anomaly payloads but not in resolutions.

---

## Proposed Solution: Comprehensive Resolution Context

### New `resolution_context` Object

Add a `resolution_context` field to the resolution payload containing:

1. **`metrics_at_resolution`** - Current metric values when resolved
2. **`slo_evaluation`** - Full SLO evaluation showing operational status
3. **`comparison_to_baseline`** - Statistical comparison to training data
4. **`health_summary`** - Human-readable summary for UI display

### Enhanced Resolution Payload Schema

```json
{
  "fingerprint_id": "anomaly_061598e9ca91",
  "incident_id": "incident_abc123def456",
  "anomaly_name": "database_degradation",
  "fingerprint_action": "RESOLVE",
  "incident_action": "CLOSE",
  "final_severity": "medium",
  "resolved_at": "2025-12-17T14:30:00.000000",
  "total_occurrences": 5,
  "incident_duration_minutes": 45,
  "first_seen": "2025-12-17T13:45:00.000000",
  "service_name": "booking",
  "last_detected_by_model": "business_hours",
  "resolution_reason": "resolved",

  "resolution_context": {
    "metrics_at_resolution": {
      "request_rate": 52.7,
      "application_latency": 110.5,
      "dependency_latency": 1.4,
      "database_latency": 0.8,
      "error_rate": 0.0001
    },

    "slo_evaluation": {
      "slo_status": "ok",
      "operational_impact": "none",
      "latency_evaluation": {
        "status": "ok",
        "value": 110.5,
        "threshold_acceptable": 300,
        "threshold_warning": 400,
        "threshold_critical": 500,
        "proximity": 0.37
      },
      "error_rate_evaluation": {
        "status": "ok",
        "value": 0.0001,
        "value_percent": "0.01%",
        "threshold_acceptable": 0.005,
        "threshold_warning": 0.01,
        "within_acceptable": true
      },
      "database_latency_evaluation": {
        "status": "ok",
        "value_ms": 0.8,
        "baseline_mean_ms": 2.5,
        "ratio": 0.32,
        "below_floor": true
      }
    },

    "comparison_to_baseline": {
      "request_rate": {
        "current": 52.7,
        "training_mean": 42.5,
        "deviation_sigma": 0.4,
        "percentile_estimate": 69.0,
        "status": "normal"
      },
      "application_latency": {
        "current": 110.5,
        "training_mean": 110.3,
        "deviation_sigma": 0.03,
        "percentile_estimate": 56.0,
        "status": "normal"
      },
      "error_rate": {
        "current": 0.0001,
        "training_mean": 0.00012,
        "deviation_sigma": -0.05,
        "percentile_estimate": 45.0,
        "status": "normal"
      }
    },

    "health_summary": {
      "all_metrics_normal": true,
      "slo_compliant": true,
      "summary": "All metrics within normal operating ranges. Latency 110ms (acceptable < 300ms), errors 0.01% (acceptable < 0.5%)."
    }
  }
}
```

---

## Field Definitions

### `resolution_context` Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `metrics_at_resolution` | object | Yes | Raw metric values at resolution time |
| `slo_evaluation` | object | Yes | Full SLO evaluation result |
| `comparison_to_baseline` | object | No | Statistical comparison to training data |
| `health_summary` | object | Yes | Summary for UI display |

### `metrics_at_resolution` Object

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `request_rate` | float | req/s | Requests per second |
| `application_latency` | float | ms | Server processing time |
| `dependency_latency` | float | ms | External dependency latency |
| `database_latency` | float | ms | Database query time |
| `error_rate` | float | ratio | Error rate (0.05 = 5%) |

### `slo_evaluation` Object

| Field | Type | Description |
|-------|------|-------------|
| `slo_status` | string | Overall status: `ok`, `warning`, `breached` |
| `operational_impact` | string | Impact level: `none`, `informational`, `actionable`, `critical` |
| `latency_evaluation` | object | Latency vs SLO thresholds |
| `error_rate_evaluation` | object | Error rate vs SLO thresholds |
| `database_latency_evaluation` | object | Database latency ratio evaluation (optional) |
| `request_rate_evaluation` | object | Traffic surge/cliff evaluation (optional) |

### `latency_evaluation` Object

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | `ok`, `warning`, `critical` |
| `value` | float | Current latency in ms |
| `threshold_acceptable` | float | Acceptable threshold (ms) |
| `threshold_warning` | float | Warning threshold (ms) |
| `threshold_critical` | float | Critical threshold (ms) |
| `proximity` | float | How close to breach (0.0 = far, 1.0 = at threshold) |

### `error_rate_evaluation` Object

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | `ok`, `warning`, `critical` |
| `value` | float | Current error rate (ratio) |
| `value_percent` | string | Human-readable percentage |
| `threshold_acceptable` | float | Acceptable threshold |
| `threshold_warning` | float | Warning threshold |
| `within_acceptable` | boolean | Is value within acceptable range |

### `comparison_to_baseline` Object (Per Metric)

| Field | Type | Description |
|-------|------|-------------|
| `current` | float | Current value |
| `training_mean` | float | Mean from training data |
| `deviation_sigma` | float | Standard deviations from mean |
| `percentile_estimate` | float | Estimated percentile (0-100) |
| `status` | string | `normal`, `elevated`, `low` |

### `health_summary` Object

| Field | Type | Description |
|-------|------|-------------|
| `all_metrics_normal` | boolean | True if all metrics within normal range |
| `slo_compliant` | boolean | True if all SLO thresholds are met |
| `summary` | string | Human-readable summary for display |

---

## Resolution Eligibility

An incident is eligible for resolution when:

1. **Grace period exceeded**: Anomaly not detected for N consecutive cycles (default: 3)
2. **SLO status is OK**: All metrics within acceptable SLO thresholds

The `resolution_context` captures WHY the incident was closed by showing:
- Which metrics returned to normal
- How they compare to SLO thresholds
- How they compare to historical baseline

---

## Web UI Integration

### Alert Closed Summary View

The Web UI can render an "Alert Closed" summary using the `resolution_context`:

```
┌────────────────────────────────────────────────────────────────────┐
│  ✅ INCIDENT RESOLVED                                              │
│  database_degradation • booking service                            │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  Duration: 45 minutes (13:45 - 14:30)                             │
│  Detections: 5 occurrences                                        │
│  Final Severity: medium                                           │
│                                                                    │
├────────────────────────────────────────────────────────────────────┤
│  METRICS AT RESOLUTION                                             │
│                                                                    │
│  ┌────────────────────┬──────────┬──────────┬──────────┐          │
│  │ Metric             │ Value    │ SLO      │ Baseline │          │
│  ├────────────────────┼──────────┼──────────┼──────────┤          │
│  │ Latency            │ 110 ms   │ ✓ < 300  │ 56th %   │          │
│  │ Error Rate         │ 0.01%    │ ✓ < 0.5% │ 45th %   │          │
│  │ Request Rate       │ 52.7/s   │ ✓ normal │ 69th %   │          │
│  │ Database Latency   │ 0.8 ms   │ ✓ ok     │ normal   │          │
│  └────────────────────┴──────────┴──────────┴──────────┘          │
│                                                                    │
│  All metrics within normal operating ranges.                       │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### Resolution Card (Compact View)

```
┌──────────────────────────────────────────┐
│ ✅ Resolved                              │
├──────────────────────────────────────────┤
│ Duration: 45 min • Detections: 5         │
│                                          │
│ State at Resolution:                     │
│   Latency: 110ms ✓   Error: 0.01% ✓     │
│   Traffic: 52.7/s    DB: 0.8ms          │
│                                          │
│ [View Details]                           │
└──────────────────────────────────────────┘
```

### Timeline View Integration

```
┌─────────────────────────────────────────────────────────────────────┐
│ INCIDENT TIMELINE: database_degradation                             │
├──────────┬──────────────────────────────────────────────────────────┤
│ 13:45:00 │ 🔴 CREATED - DB latency 45ms (18x baseline)             │
│ 13:48:00 │ 🟡 CONTINUED - severity elevated to HIGH                 │
│ 14:00:00 │ 🟡 CONTINUED - DB latency peaked at 180ms               │
│ 14:15:00 │ 🟢 RECOVERING - latency decreasing                       │
│ 14:30:00 │ ✅ RESOLVED - All metrics normal                         │
│          │    └─ Latency: 110ms (✓ SLO)                            │
│          │    └─ Errors: 0.01% (✓ SLO)                             │
│          │    └─ DB: 0.8ms (✓ baseline)                            │
└──────────┴──────────────────────────────────────────────────────────┘
```

---

## Backward Compatibility

### API Contract

This change is **backward compatible**:

1. **New field is additive**: `resolution_context` is a new optional field
2. **Existing fields unchanged**: All current resolution fields remain
3. **Graceful degradation**: Web UI can check for presence of `resolution_context`

### Web UI Handling

```javascript
// Example handling in Web UI
function renderResolution(resolution) {
  const context = resolution.resolution_context;

  if (context) {
    // Enhanced view with metrics and SLO context
    renderEnhancedResolutionCard(resolution, context);
  } else {
    // Fallback for resolutions without context (older data)
    renderBasicResolutionCard(resolution);
  }
}
```

---

## Implementation Plan

### Files to Modify

| File | Change |
|------|--------|
| `smartbox_anomaly/fingerprinting/fingerprinter.py` | Pass metrics and SLO context to `_process_resolutions()`, build `resolution_context` |
| `smartbox_anomaly/slo/evaluator.py` | Add helper method `evaluate_metrics()` for standalone evaluation |
| `inference.py` | Ensure SLO evaluation is available to fingerprinter |
| `docs/INFERENCE_API_PAYLOAD.md` | Document new `resolution_context` field |
| `docs/API_CHANGELOG.md` | Add v1.4.0 changelog entry |

### Code Changes Overview

**1. Update `_process_resolutions()` signature:**

```python
def _process_resolutions(
    self,
    active_incidents: dict[str, dict],
    processed_fps: set[str],
    timestamp: datetime,
    current_metrics: dict[str, Any] | None = None,
    slo_evaluator: SLOEvaluator | None = None,
    training_statistics: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
```

**2. Build resolution context in resolution dict:**

```python
resolved.append({
    "fingerprint_id": fp_id,
    "incident_id": incident["incident_id"],
    # ... existing fields ...

    # NEW: Resolution context
    "resolution_context": self._build_resolution_context(
        current_metrics,
        slo_evaluator,
        training_statistics,
        incident["service_name"],
    ),
})
```

**3. Add helper method for building context:**

```python
def _build_resolution_context(
    self,
    metrics: dict[str, Any] | None,
    slo_evaluator: SLOEvaluator | None,
    training_stats: dict[str, Any] | None,
    service_name: str,
) -> dict[str, Any] | None:
    """Build resolution context with metrics, SLO, and baseline comparison."""
    if not metrics:
        return None

    context = {
        "metrics_at_resolution": metrics,
        "health_summary": {
            "all_metrics_normal": True,
            "slo_compliant": True,
            "summary": "",
        },
    }

    # Add SLO evaluation if available
    if slo_evaluator:
        slo_result = slo_evaluator.evaluate_metrics(metrics, service_name)
        context["slo_evaluation"] = slo_result.to_dict()
        context["health_summary"]["slo_compliant"] = slo_result.slo_status == "ok"

    # Add baseline comparison if available
    if training_stats:
        context["comparison_to_baseline"] = self._build_comparison(
            metrics, training_stats
        )

    # Build summary message
    context["health_summary"]["summary"] = self._build_health_summary(context)

    return context
```

---

## Example Complete Resolution Payload

```json
{
  "fingerprint_id": "anomaly_061598e9ca91",
  "incident_id": "incident_abc123def456",
  "anomaly_name": "database_degradation",
  "fingerprint_action": "RESOLVE",
  "incident_action": "CLOSE",
  "final_severity": "medium",
  "resolved_at": "2025-12-17T14:30:00.000000",
  "total_occurrences": 5,
  "incident_duration_minutes": 45,
  "first_seen": "2025-12-17T13:45:00.000000",
  "service_name": "booking",
  "last_detected_by_model": "business_hours",
  "missed_cycles_before_close": 3,
  "resolution_reason": "resolved",

  "resolution_context": {
    "metrics_at_resolution": {
      "request_rate": 52.7,
      "application_latency": 110.5,
      "dependency_latency": 1.4,
      "database_latency": 0.8,
      "error_rate": 0.0001
    },

    "slo_evaluation": {
      "slo_status": "ok",
      "operational_impact": "none",
      "is_busy_period": false,
      "latency_evaluation": {
        "status": "ok",
        "value": 110.5,
        "threshold_acceptable": 300,
        "threshold_warning": 400,
        "threshold_critical": 500,
        "proximity": 0.368
      },
      "error_rate_evaluation": {
        "status": "ok",
        "value": 0.0001,
        "value_percent": "0.01%",
        "threshold_acceptable": 0.005,
        "threshold_warning": 0.01,
        "threshold_critical": 0.02,
        "within_acceptable": true
      },
      "database_latency_evaluation": {
        "status": "ok",
        "value_ms": 0.8,
        "baseline_mean_ms": 2.5,
        "ratio": 0.32,
        "below_floor": true,
        "floor_ms": 1.0
      }
    },

    "comparison_to_baseline": {
      "request_rate": {
        "current": 52.7,
        "training_mean": 42.5,
        "training_std": 25.4,
        "deviation_sigma": 0.40,
        "percentile_estimate": 69.0,
        "status": "normal"
      },
      "application_latency": {
        "current": 110.5,
        "training_mean": 110.3,
        "training_std": 45.2,
        "deviation_sigma": 0.004,
        "percentile_estimate": 56.0,
        "status": "normal"
      },
      "database_latency": {
        "current": 0.8,
        "training_mean": 2.5,
        "training_std": 1.2,
        "deviation_sigma": -1.42,
        "percentile_estimate": 8.0,
        "status": "low"
      },
      "error_rate": {
        "current": 0.0001,
        "training_mean": 0.00012,
        "training_std": 0.0004,
        "deviation_sigma": -0.05,
        "percentile_estimate": 45.0,
        "status": "normal"
      }
    },

    "health_summary": {
      "all_metrics_normal": true,
      "slo_compliant": true,
      "summary": "All metrics within normal operating ranges. Latency 110ms (acceptable < 300ms), errors 0.01% (acceptable < 0.5%)."
    }
  }
}
```

---

## Questions for Web UI Team

1. **Data granularity**: Is the proposed level of detail appropriate, or should we simplify/expand certain sections?

2. **Baseline comparison**: Is the `comparison_to_baseline` section valuable, or is SLO evaluation sufficient?

3. **Historical resolutions**: Should we backfill resolution_context for existing closed incidents, or only apply to new resolutions?

4. **Peak metrics**: Would you like to also include `peak_metrics_during_incident` to show "before vs after"? (Requires additional tracking)

5. **Format preferences**: Any specific formatting requirements for the `health_summary.summary` text?

---

## Decision

Pending evaluation by Web UI team.
