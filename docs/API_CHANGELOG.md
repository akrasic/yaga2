# API Changelog

This document tracks changes to the API payload schema.

---

## Version 1.3.1

**Release Date**: 2026-01-07

### Behavioral Changes

#### SLO Severity Adjustment Logic Updated

The SLO evaluation layer now consistently adjusts severity to `low` when all metrics are within acceptable SLO thresholds (`slo_status: "ok"`), regardless of the original ML-assigned severity.

**Previous Behavior:**
| Original Severity | SLO Status | Adjusted Severity |
|-------------------|------------|-------------------|
| critical | ok | medium |
| high | ok | medium |
| medium | ok | low |

**New Behavior:**
| Original Severity | SLO Status | Adjusted Severity |
|-------------------|------------|-------------------|
| critical | ok | **low** |
| high | ok | **low** |
| medium | ok | **low** |

**Rationale:** When SLO status is `ok`, all metrics are within acceptable operational thresholds. The anomaly is statistically real but operationally insignificant. Using `low` severity consistently reflects this - the alert is logged for awareness but doesn't require action.

**Impact:** Alerts that previously showed `medium` severity when metrics were within acceptable SLO will now show `low` severity. This reduces alert noise for operationally acceptable situations.

**Configuration:** This behavior requires `slos.enabled: true` and `slos.allow_downgrade_to_informational: true` in config.

#### Root `overall_severity` Now Reflects SLO Adjustment

Fixed an issue where the root-level `overall_severity` field was not updated to reflect the SLO-adjusted severity. Previously, the root level could show `critical` while `slo_evaluation.adjusted_severity` showed `medium`.

**Previous Behavior:**
```json
{
  "overall_severity": "critical",
  "slo_evaluation": {
    "adjusted_severity": "medium",
    "severity_changed": true
  }
}
```

**New Behavior:**
```json
{
  "overall_severity": "low",
  "slo_evaluation": {
    "original_severity": "critical",
    "adjusted_severity": "low",
    "severity_changed": true
  }
}
```

The root `overall_severity` now correctly matches `slo_evaluation.adjusted_severity` when SLO evaluation adjusts the severity.

---

## Version 1.3.0

**Release Date**: 2026-01-06

### New Fields

#### `SLOEvaluationResult.request_rate_evaluation`

Added `request_rate_evaluation` field to the SLO evaluation result. This field provides correlation-based severity evaluation for traffic anomalies (surges and cliffs), following Google SRE and NewRelic/Datadog best practices.

**Field Type**: `object | null`

**When Populated**: When request rate metrics (`request_rate`) are present in the inference result.

**Key Concepts**:
- **Traffic Surge**: Current traffic significantly exceeds baseline (default: ≥200% of baseline)
- **Traffic Cliff**: Current traffic significantly below baseline (default: ≤50% of baseline)
- **Correlation-based Severity**: Severity depends on whether the traffic anomaly correlates with other SLO breaches

**Schema (Surge)**:

```json
{
  "request_rate_evaluation": {
    "status": "info | warning | high",
    "type": "surge",
    "severity": "informational | warning | high",
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

**Schema (Cliff)**:

```json
{
  "request_rate_evaluation": {
    "status": "warning | high | critical",
    "type": "cliff",
    "severity": "warning | high | critical",
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

**Severity Logic**:

| Type  | Condition                    | Severity        |
|-------|------------------------------|-----------------|
| Surge | Standalone (no SLO issues)   | informational   |
| Surge | With latency SLO breach      | warning         |
| Surge | With error SLO breach        | high            |
| Cliff | Off-peak hours               | warning         |
| Cliff | Peak/business hours          | high            |
| Cliff | With errors                  | critical        |

**Why Correlation-based?**

Traffic surges alone are often benign (marketing campaigns, organic growth). They only become problematic when causing capacity issues (latency/errors). Traffic cliffs are more concerning as they often indicate upstream failures or routing issues.

**Configuration**:

Per-service override is supported. Example for booking service (stricter cliff handling):

```json
{
  "slos": {
    "services": {
      "booking": {
        "request_rate_evaluation": {
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

---

## Version 1.2.0

**Release Date**: 2025-01-06

### New Fields

#### `SLOEvaluationResult.database_latency_evaluation`

Added `database_latency_evaluation` field to the SLO evaluation result. This field provides detailed information about database latency SLO evaluation using a hybrid noise floor + ratio-based threshold approach.

**Field Type**: `object | null`

**When Populated**: When database latency metrics (`db_latency_avg`) are present in the inference result.

**Schema**:

```json
{
  "database_latency_evaluation": {
    "status": "ok | info | warning | high | critical",
    "value_ms": 10.5,
    "baseline_mean_ms": 5.0,
    "ratio": 2.1,
    "below_floor": false,
    "floor_ms": 5.0,
    "thresholds": {
      "info": 1.5,
      "warning": 2.0,
      "high": 3.0,
      "critical": 5.0
    },
    "explanation": "DB latency elevated: 10.5ms is 2.1x baseline (5.0ms)"
  }
}
```

**Status Values**:
- `ok` - Database latency within acceptable range (below floor or ratio < info threshold)
- `info` - Informational: ratio between info and warning thresholds (1.5x - 2x baseline)
- `warning` - Warning: ratio between warning and high thresholds (2x - 3x baseline)
- `high` - High: ratio between high and critical thresholds (3x - 5x baseline)
- `critical` - Critical: ratio at or above critical threshold (≥5x baseline)

**Key Fields**:
- `below_floor`: `true` if latency is below the noise floor (default 1ms), always considered OK
- `ratio`: Current latency divided by training baseline mean
- `floor_ms`: Configured noise floor for this service

**Example - Below Floor (Noise Filtered)**:

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

This filters out operationally meaningless changes like 0.3ms → 0.4ms that the ML might detect as anomalous.

### Configuration Changes

#### New SLO Configuration Fields

Added database latency SLO configuration to `config.json`:

```json
{
  "slos": {
    "defaults": {
      "database_latency_floor_ms": 5.0,
      "database_latency_ratios": {
        "info": 1.5,
        "warning": 2.0,
        "high": 3.0,
        "critical": 5.0
      }
    },
    "services": {
      "search": {
        "database_latency_floor_ms": 2.0,
        "database_latency_ratios": {
          "info": 1.3,
          "warning": 1.5,
          "high": 2.0,
          "critical": 3.0
        }
      }
    }
  }
}
```

**Per-Service Overrides**: Services can have custom floor and ratio thresholds based on their database performance characteristics.

### Backward Compatibility

This is a **backward-compatible** change:
- The `database_latency_evaluation` field is only included when database latency data is present
- Existing API consumers can ignore the field
- No changes required to existing integrations

---

## Version 1.1.0

**Release Date**: 2024-01-15

### New Fields

#### `AnomalyDetectedPayload.exception_context`

Added optional `exception_context` field to the anomaly detection payload. This field is populated when error-related anomalies are detected and provides a breakdown of exception types from OpenTelemetry metrics.

#### `ResolutionDetails.resolution_reason`

Added `resolution_reason` field to incident resolution payloads. This field indicates why an incident was closed.

**Field Type**: `string`

**Possible Values**:
- `"resolved"` - Normal resolution after grace period (anomaly stopped being detected)
- `"auto_stale"` - Auto-closed because the time gap exceeded `incident_separation_minutes` threshold (default: 30 min)

**When `auto_stale` Occurs**:

When an anomaly reappears after a gap longer than the configured `incident_separation_minutes`:
1. The old incident is automatically closed with `resolution_reason: "auto_stale"`
2. A new incident is created for the reappearing anomaly
3. Both actions are included in the same API payload

**Example**:

```json
{
  "fingerprinting": {
    "resolved_incidents": [
      {
        "incident_id": "incident_abc123",
        "fingerprint_id": "anomaly_xyz789",
        "anomaly_name": "database_degradation",
        "incident_action": "CLOSE",
        "resolution_reason": "auto_stale",
        "final_severity": "medium",
        "total_occurrences": 3,
        "incident_duration_minutes": 45
      }
    ]
  },
  "anomalies": [
    {
      "incident_id": "incident_newone",
      "fingerprint_id": "anomaly_xyz789",
      "incident_action": "CREATE",
      "status": "SUSPECTED"
    }
  ]
}
```

**Backend Changes Required**:

The web API should:
1. **Accept the new field** - Add `resolution_reason` to the resolution payload validation
2. **Store the reason** - Save the resolution reason with the incident record
3. **Display in UI** - Optionally show "Auto-closed (stale)" vs "Resolved" in the incident history

**Field Type**: `object | null`

**When Populated**:
- Severity is HIGH or CRITICAL
- Error-related anomaly detected (pattern contains "error", "failure", "outage")
- Error rate > 1%

**Schema**:

```json
{
  "exception_context": {
    "service_name": "string",
    "timestamp": "string (ISO 8601)",
    "total_exception_rate": "number (exceptions per second)",
    "exception_count": "integer",
    "top_exceptions": [
      {
        "type": "string (full exception class name)",
        "short_name": "string (class name only)",
        "rate": "number (exceptions per second)",
        "percentage": "number (0-100)"
      }
    ],
    "query_successful": "boolean",
    "error_message": "string | null"
  }
}
```

**Example**:

```json
{
  "alert_type": "anomaly_detected",
  "service_name": "search",
  "timestamp": "2024-01-15T10:30:00",
  "time_period": "business_hours",
  "model_name": "business_hours",
  "model_type": "time_aware_5period",
  "anomalies": {
    "elevated_errors": {
      "type": "consolidated",
      "severity": "high",
      "confidence": 0.85,
      "score": -0.45,
      "description": "Error rate elevated above normal threshold",
      "root_metric": "error_rate",
      "pattern_name": "elevated_errors",
      "interpretation": "Error rate is significantly above the normal range...",
      "recommended_actions": [
        "INVESTIGATE: Top exception is R2D2Exception (62% of errors)",
        "CHECK: Application logs for error details"
      ]
    }
  },
  "anomaly_count": 1,
  "overall_severity": "high",
  "current_metrics": {
    "request_rate": 150.5,
    "application_latency": 45.2,
    "error_rate": 0.05
  },
  "exception_context": {
    "service_name": "search",
    "timestamp": "2024-01-15T10:30:00",
    "total_exception_rate": 0.35,
    "exception_count": 3,
    "top_exceptions": [
      {
        "type": "Smartbox\\Search\\R2D2\\Exception\\R2D2Exception",
        "short_name": "R2D2Exception",
        "rate": 0.217,
        "percentage": 62.0
      },
      {
        "type": "Smartbox\\Search\\Exception\\UserInputException",
        "short_name": "UserInputException",
        "rate": 0.083,
        "percentage": 23.7
      },
      {
        "type": "Smartbox\\Search\\SearchMiddleware\\Exception\\ClientSideMiddlewareException",
        "short_name": "ClientSideMiddlewareException",
        "rate": 0.050,
        "percentage": 14.3
      }
    ],
    "query_successful": true,
    "error_message": null
  },
  "fingerprinting": { ... },
  "metadata": { ... }
}
```

### Backend Changes Required

The web API should:

1. **Accept the new field** - Add `exception_context` to the payload validation schema
2. **Store the data** - Store exception context with the anomaly for display
3. **Display in UI** - Show exception breakdown when viewing error-related anomalies

**Suggested UI Display**:

When `exception_context` is present and `query_successful` is true:

```
Exception Breakdown (0.35/s total)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
R2D2Exception           0.217/s  (62.0%)  ████████████░░░░░░░░
UserInputException      0.083/s  (23.7%)  █████░░░░░░░░░░░░░░░
ClientSideMiddleware... 0.050/s  (14.3%)  ███░░░░░░░░░░░░░░░░░
```

### Backward Compatibility

This is a **backward-compatible** change:
- The field is optional (`null` by default)
- Existing API consumers can ignore the field
- No changes required to existing integrations

---

## Version 1.0.0

**Release Date**: Initial release

Initial API schema with:
- `AnomalyDetectedPayload`
- `IncidentResolvedPayload`
- `ErrorPayload`
- `HeartbeatPayload`
