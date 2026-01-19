# API Changelog

This document tracks changes to the API payload schema.

---

## Version 2.0.1

**Release Date**: 2026-01-15

### Bug Fixes

#### SLO Now Caps Pattern-Assigned Critical Severity to High When SLO Status is Warning

Fixed a bug where pattern-assigned `critical` severity was not being capped by SLO evaluation when the SLO status was `warning`.

**Previous Behavior (Bug):**
```
Pattern Detection: CRITICAL (e.g., error_rate_critical pattern)
SLO Evaluation: WARNING (latency in warning zone, error rate below warning threshold)
Final Severity: CRITICAL ← Pattern severity passed through unchanged
```

**New Behavior (Fixed):**
```
Pattern Detection: CRITICAL
SLO Evaluation: WARNING
Final Severity: HIGH ← SLO caps severity at high
```

**Rationale:** SLO thresholds represent actual operational impact. If SLO says "warning" (no critical thresholds breached), the alert should not be `critical` regardless of what the pattern detection assigns. The SLO evaluation is now authoritative for operational severity.

**Severity Adjustment Logic (Updated):**

| Pattern Severity | SLO Status | Final Severity |
|------------------|------------|----------------|
| any | breached | critical |
| critical | **warning** | **high** ← Fixed (was critical) |
| high | warning | high |
| any | ok | low |

**Impact:** Alerts that previously showed `critical` severity when SLO status was `warning` will now show `high` severity. This reduces alert fatigue by ensuring severity accurately reflects operational impact.

**Bug Report Reference:** `BUG_REPORT_SLO_SEVERITY_OVERRIDE.md`

---

## Version 2.0.0

**Release Date**: 2026-01-15

### Breaking Changes

#### Metric Renamed: `client_latency` → `dependency_latency`

The `client_latency` metric has been renamed to `dependency_latency` throughout the API payload.

**Rationale:** "Client" was ambiguous - could be interpreted as end-user latency. "Dependency" clearly indicates latency spent waiting on external services/dependencies.

**Fields Changed:**

| Location | Old Field | New Field |
|----------|-----------|-----------|
| `current_metrics` | `client_latency` | `dependency_latency` |
| `comparison_data` | `client_latency` | `dependency_latency` |
| `metrics_at_resolution` | `client_latency` | `dependency_latency` |
| Pattern conditions | `"client_latency": "high"` | `"dependency_latency": "high"` |
| Ratio calculations | `client_latency_ratio` | `dependency_latency_ratio` |

**Configuration Changed:**
- `detection_thresholds.ratios.client_latency_bottleneck` → `dependency_latency_bottleneck`

**Migration Guide:**

Search and replace in all integrations consuming the API payload:
```
client_latency → dependency_latency
client_latency_ratio → dependency_latency_ratio
```

**Example - Before (v1.x):**
```json
{
  "current_metrics": {
    "application_latency": 250.0,
    "client_latency": 120.0,
    "database_latency": 45.0
  }
}
```

**Example - After (v2.0.0):**
```json
{
  "current_metrics": {
    "application_latency": 250.0,
    "dependency_latency": 120.0,
    "database_latency": 45.0
  }
}
```

**Impact:**
- All API consumers must update field references
- Dashboards and alerts referencing `client_latency` must be updated
- Historical data stored with old field names will need migration or dual-read logic

---

## Version 1.5.0

**Release Date**: 2026-01-15

### New Features

#### Envoy Edge/Ingress Metrics Enrichment

Added `envoy_context` field to anomaly detection payloads. This field provides edge-level metrics from Envoy proxy to correlate with OpenTelemetry application metrics, giving operators a complete view of service health from both edge and application perspectives.

**Problem Solved:**
- Anomaly alerts only contained OTel application metrics
- No visibility into how the service appears from the edge/ingress layer
- Difficult to correlate application anomalies with edge-level symptoms (5xx rates, connection issues)
- Had to manually query Mimir for edge metrics during incident investigation

**New Behavior:**
- When anomalies are detected for services with Envoy cluster mappings, the `envoy_context` field is populated
- Queries Mimir for Envoy metrics aligned with the anomaly detection window
- Provides request rates by response class (2xx, 4xx, 5xx), latency percentiles, and active connections

**Schema (Envoy Context):**

```json
{
  "envoy_context": {
    "service_name": "booking",
    "cluster_name": "booking",
    "timestamp": "2025-01-15T10:30:00",
    "request_rates": {
      "total": 150.5,
      "rate_2xx": 147.2,
      "rate_4xx": 2.1,
      "rate_5xx": 1.2,
      "success_rate": 0.978,
      "error_rate_4xx": 0.014,
      "error_rate_5xx": 0.008
    },
    "latency_percentiles": {
      "p50_ms": 45.2,
      "p90_ms": 120.5,
      "p99_ms": 250.0
    },
    "connections": {
      "active": 42
    },
    "query_successful": true,
    "has_data": true
  }
}
```

**When Envoy Context is Populated:**
- Service has a configured cluster mapping in `envoy_enrichment.cluster_mapping`
- Envoy enrichment is enabled (`envoy_enrichment.enabled: true`)
- Mimir query succeeds and returns data

**Use Cases:**
1. Correlate OTel error rate spikes with Envoy 5xx rates
2. Identify edge-level issues (connection limits, circuit breakers)
3. Compare application latency with edge-observed latency
4. Detect discrepancies between edge and application views

**Configuration:**

New `envoy_enrichment` section in `config.json`:

```json
{
  "envoy_enrichment": {
    "enabled": true,
    "mimir_endpoint": "https://mimir.sbxtest.net/prometheus",
    "lookback_minutes": 5,
    "timeout_seconds": 10,
    "cluster_mapping": {
      "booking": "booking",
      "search": "search_k8s",
      "mobile-api": "mobile-api",
      "shire-api": "shireapi_cluster"
    }
  }
}
```

**Files Added/Modified:**
- `smartbox_anomaly/enrichment/envoy.py` - New EnvoyEnrichmentService class
- `smartbox_anomaly/enrichment/__init__.py` - Export Envoy enrichment classes
- `smartbox_anomaly/inference/enrichment_runner.py` - Added `apply_envoy_enrichment()` method
- `smartbox_anomaly/inference/pipeline.py` - Initialize EnvoyEnrichmentService
- `smartbox_anomaly/core/config.py` - Added EnvoyEnrichmentConfig dataclass
- `config.json` - Added envoy_enrichment configuration section

**Backward Compatibility:**
This is a **backward-compatible** change:
- The `envoy_context` field is optional (`null` for unsupported services)
- Existing API consumers can ignore the field
- Services without cluster mappings continue to work normally
- No changes required to existing integrations

---

## Version 1.4.0

**Release Date**: 2026-01-14

### New Features

#### Enhanced Resolution Payload with Context

Added comprehensive `resolution_context` field to incident resolution payloads. This provides detailed context about the service state at resolution time, enabling the Web UI to show WHY an incident was closed and what "healthy" looks like.

**Problem Solved:**
- Resolution payloads only contained metadata (duration, occurrences) but not the actual metric values
- No visibility into SLO status at resolution time
- No way to verify the resolution was genuine vs. a detection gap

**New Behavior:**
- When incidents resolve, the `resolution_context` field includes:
  - `metrics_at_resolution` - Current metric values when resolved
  - `slo_evaluation` - Full SLO evaluation showing operational status
  - `comparison_to_baseline` - Statistical comparison to training data
  - `health_summary` - Human-readable summary for UI display

**Schema (Resolution Context):**

```json
{
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
      "latency_evaluation": {...},
      "error_rate_evaluation": {...}
    },
    "comparison_to_baseline": {
      "application_latency": {
        "current": 110.5,
        "training_mean": 110.3,
        "deviation_sigma": 0.03,
        "percentile_estimate": 56.0,
        "status": "normal"
      }
    },
    "health_summary": {
      "all_metrics_normal": true,
      "slo_compliant": true,
      "summary": "All metrics within acceptable SLO thresholds..."
    }
  }
}
```

**Use Cases:**
1. Web UI can render "Alert Closed Summary" with full context
2. SRE teams can verify resolutions were genuine (not detection gaps)
3. Post-incident analysis can compare incident peak vs resolved values
4. Historical views can show "what healthy looks like" for each service

**Files Modified:**
- `smartbox_anomaly/fingerprinting/fingerprinter.py` - Added `_build_resolution_context()` method
- `smartbox_anomaly/slo/evaluator.py` - Added `evaluate_metrics()` public method for standalone evaluation
- `inference.py` - Pass SLO evaluator to fingerprinter for resolution context

**Backward Compatibility:**
This is a **backward-compatible** change:
- The `resolution_context` field is optional (may be `null` if metrics unavailable)
- Existing API consumers can ignore the field
- No changes required to existing integrations

---

## Version 1.3.3

**Release Date**: 2026-01-13

### New Features

#### Error Rate Suppression

Added `error_rate_floor` configuration option to suppress error-related anomalies when the error rate is below operational thresholds. This eliminates alert noise from statistically valid but operationally insignificant error rate deviations.

**Problem Solved:**
- Services with very low baseline error rates (e.g., 0.01%) would generate many alerts for tiny deviations
- These alerts were statistically valid (ML detected real deviation) but operationally meaningless
- Example: booking service generated 256 incidents in 7 days, with 84% having error rates below the SLO acceptable threshold

**New Behavior:**
- When `error_rate_floor` is set, anomalies with error rates below this threshold are **suppressed entirely**
- If `error_rate_floor` is 0 (default), uses `error_rate_acceptable` as the suppression threshold
- Suppressed anomalies are logged at DEBUG level but not included in the output

**Configuration:**
```json
{
  "slos": {
    "defaults": {
      "error_rate_floor": 0
    },
    "services": {
      "booking": {
        "error_rate_acceptable": 0.002,
        "error_rate_floor": 0.002
      }
    }
  }
}
```

**Impact:**
- Reduces false positive error alerts for services with low baseline error rates
- No changes required for API consumers - suppressed anomalies simply don't appear in the payload
- Existing anomalies that exceed the floor will include a new `suppression_threshold` field in `slo_context`

**New Field in `slo_context` for Error Anomalies:**
```json
{
  "slo_context": {
    "current_value": 0.005,
    "current_value_percent": "0.50%",
    "acceptable_threshold": 0.002,
    "critical_threshold": 0.01,
    "suppression_threshold": 0.002,
    "within_acceptable": false,
    "is_busy_period": false
  }
}
```

---

## Version 1.3.2

**Release Date**: 2026-01-13

### Behavioral Changes

#### Confirmed-Only Web API Alerts

The inference engine now filters out **SUSPECTED** anomalies before sending alerts to the web API. Only confirmed anomalies (status = OPEN or RECOVERING) are included in the `anomaly_detected` payload.

**Previous Behavior:**
- All detected anomalies were sent to the web API immediately
- First detection created incident in web API as OPEN
- If anomaly expired without confirmation (`suspected_expired`), no resolution was sent
- Result: Orphaned OPEN incidents in web API

**New Behavior:**
- First detection creates SUSPECTED incident (not sent to web API)
- After 2+ consecutive detections, incident is confirmed (OPEN) and sent to web API
- If anomaly expires before confirmation, it's silently closed (no orphan created)

**Impact:**
- Reduces orphaned incidents in the web API
- Web API only receives confirmed, actionable anomalies
- No changes required for API consumers - all received anomalies are now guaranteed to be confirmed

**Related Fix:** `suspected_expired` incidents no longer create orphaned OPEN incidents in the web API. See `BUG_REPORT_SUSPECTED_ALERTS.md` for details.

### Bug Fixes

#### KeyError in Verbose Logging

Fixed a KeyError when logging resolution details in verbose mode. The code was accessing `resolution['service']` but the payload uses `service_name`.

**Before:**
```python
service = resolution['service']  # KeyError
```

**After:**
```python
service = resolution.get('service_name', resolution.get('service', 'unknown'))
```

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

#### Pattern Naming Improvements

Renamed several anomaly patterns to be more descriptive and consistent:

| Old Name | New Name | Reason |
|----------|----------|--------|
| `partial_outage` | `error_rate_critical` | "Outage" was misleading - service responds normally, just high errors |
| `recent_degradation` | `latency_spike_recent` | More specific about what degraded |
| `isolated_service_issue` | `internal_latency_issue` | Clearer that it's a latency problem internal to service |
| `partial_fast_fail` | `partial_rejection` | "Rejection" better describes requests failing before processing |

Renamed recommendation keys for consistency:

| Old Key | New Key |
|---------|---------|
| `error_rate_high` | `error_rate_elevated` |
| `application_latency_high` | `latency_elevated` |
| `request_rate_high` | `traffic_surge` |
| `request_rate_low` | `traffic_cliff` |

**Naming Convention:**
```
{metric}_{state}_{modifier}

Examples:
- error_rate_critical
- error_rate_elevated
- latency_spike_recent
- latency_elevated
- internal_latency_issue
```

**Impact:** Alerts will now use the new pattern names in the `pattern_name` and anomaly key fields. Downstream consumers should update any hardcoded pattern name references.

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
