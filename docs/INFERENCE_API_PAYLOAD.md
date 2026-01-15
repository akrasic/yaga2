# Inference Engine API Payload Specification

**Schema Version**: 1.4.0

This document describes the JSON payload format sent by the inference engine to the API server after anomaly detection.

---

## Overview

The inference engine performs anomaly detection using multiple methods (Isolation Forest, Pattern Matching, Statistical Correlation) and returns a consolidated view where related anomalies are grouped into single actionable alerts.

### Confirmed-Only Alerts (v1.3.2)

**Important**: Only **confirmed** anomalies are sent to the web API. Anomalies in SUSPECTED state (first detection, not yet confirmed) are filtered out before the API call.

This behavior prevents orphaned incidents in the web API:
- First detection creates a SUSPECTED incident (not sent to API)
- After 2+ consecutive detections, incident is confirmed (OPEN) and sent to API
- If anomaly expires before confirmation (`suspected_expired`), no alert is sent

For API consumers: All anomalies in `alert_type: "anomaly_detected"` payloads are confirmed and should be displayed/alerted on.

---

## Top-Level Response Structure

```json
{
  "alert_type": "anomaly_detected | no_anomaly | metrics_unavailable",
  "service_name": "string",
  "timestamp": "ISO8601 datetime",
  "time_period": "business_hours | evening_hours | night_hours | weekend_day | weekend_night",
  "model_name": "string (e.g., 'business_hours')",
  "model_type": "time_aware_5period | single",

  "anomalies": { ... },
  "anomaly_count": "integer",
  "overall_severity": "critical | high | medium | low | none",

  "current_metrics": { ... },
  "exception_context": { ... },
  "service_graph_context": { ... },
  "fingerprinting": { ... },
  "performance_info": { ... },
  "metadata": { ... },

  "drift_warning": { ... },
  "validation_warnings": [ ... ],
  "drift_analysis": { ... },
  "slo_evaluation": { ... }
}
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `alert_type` | string | `"anomaly_detected"`, `"no_anomaly"`, or `"metrics_unavailable"` |
| `service_name` | string | Name of the service being monitored |
| `timestamp` | string | ISO8601 timestamp of the detection |
| `time_period` | string | Current time period used for detection |
| `model_name` | string | Name of the model used (matches time_period for time-aware) |
| `model_type` | string | Type of model: `"time_aware_5period"` or `"single"` |
| `anomaly_count` | integer | Number of distinct anomalies (after consolidation) |
| `overall_severity` | string | Highest severity among all anomalies (may be adjusted by SLO evaluation) |
| `original_severity` | string | Original ML-assigned severity before SLO adjustment (only present if adjusted) |
| `drift_warning` | object | Present when model drift is detected (see Drift Warning section) |
| `validation_warnings` | array | List of input validation issues (see Validation Warnings section) |
| `drift_analysis` | object | Detailed drift analysis results (when `check_drift` is enabled) |
| `slo_evaluation` | object | SLO-aware severity evaluation results (when SLO config enabled) |
| `skipped_reason` | string | Present when `alert_type` is `"metrics_unavailable"` - explains why detection was skipped |
| `failed_metrics` | array | Present when metrics collection failed - list of metric names that couldn't be collected |
| `partial_metrics_failure` | object | Present when some non-critical metrics failed but detection proceeded |
| `exception_context` | object \| null | Exception breakdown when error-related anomalies detected (see Exception Context section) |
| `service_graph_context` | object \| null | Downstream service call breakdown when latency anomalies detected (see Service Graph Context section) |

### Alert Types

| Alert Type | Description |
|------------|-------------|
| `anomaly_detected` | Anomalies were detected for this service |
| `no_anomaly` | No anomalies detected, service is healthy |
| `metrics_unavailable` | Metrics collection failed, detection was skipped to avoid false alerts |

---

## Anomalies Object

The `anomalies` object is a dictionary where keys are anomaly names and values are anomaly details.

```json
{
  "anomalies": {
    "anomaly_name": { ... anomaly details ... },
    "another_anomaly": { ... }
  }
}
```

### Anomaly Naming Convention

| Anomaly Type | Name Format | Examples |
|--------------|-------------|----------|
| Named pattern | `{pattern_name}` | `latency_spike_recent`, `fast_rejection`, `traffic_cliff` |
| Latency anomaly | `latency_anomaly` | When consolidated from latency-related detections |
| Error anomaly | `error_rate_anomaly` | When consolidated from error-related detections |
| Traffic anomaly | `traffic_anomaly` | When consolidated from request_rate detections |
| Single detection | `{metric}_{direction}` | `latency_high`, `error_rate_elevated` |

---

## Consolidated Anomaly Structure

When multiple detection methods identify the same underlying issue, they are consolidated into a single anomaly:

```json
{
  "latency_spike_recent": {
    "type": "consolidated",
    "root_metric": "application_latency",
    "severity": "high",
    "confidence": 0.80,
    "score": -0.5,
    "signal_count": 2,

    "description": "Latency degradation: 636ms (92nd percentile). (confirmed by 2 detection methods)",
    "interpretation": "Latency recently increased without traffic change - something changed...",
    "pattern_name": "latency_spike_recent",

    "value": 636.1,
    "detection_signals": [ ... ],

    "possible_causes": [ ... ],
    "recommended_actions": [ ... ],
    "checks": [ ... ],

    "comparison_data": { ... },
    "business_impact": "string"
  }
}
```

### Consolidated Anomaly Fields

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | `"consolidated"` for grouped anomalies, or original type for single detections |
| `root_metric` | string | Primary metric that triggered this anomaly |
| `severity` | string | Aggregated severity (maximum of all signals) |
| `confidence` | float | Confidence score 0.0-1.0 (higher when multiple methods agree) |
| `score` | float | Best (most negative) anomaly score from detection methods |
| `signal_count` | integer | Number of detection methods that identified this anomaly |
| `description` | string | Human-readable description of the anomaly |
| `interpretation` | string | Semantic interpretation of what this anomaly means |
| `pattern_name` | string | Name of matched pattern (if pattern-based) |
| `value` | float | Current value of the root metric |
| `detection_signals` | array | Individual signals from each detection method |
| `possible_causes` | array | List of possible root causes |
| `recommended_actions` | array | Prioritized list of actions to take |
| `checks` | array | Diagnostic checks to perform |
| `comparison_data` | object | Statistical comparison with training data |
| `business_impact` | string | Description of potential business impact |
| `cascade_analysis` | object | Cascade info when anomaly is caused by upstream dependency |

---

## Cascade Analysis Object

When an anomaly is identified as part of a dependency cascade, the `cascade_analysis` field provides details:

```json
{
  "cascade_analysis": {
    "is_cascade": true,
    "root_cause_service": "titan",
    "affected_chain": ["titan", "vms"],
    "cascade_type": "upstream_cascade",
    "confidence": 0.85,
    "propagation_path": [
      {"service": "titan", "has_anomaly": true, "anomaly_type": "database_bottleneck"},
      {"service": "vms", "has_anomaly": true, "anomaly_type": "downstream_cascade"}
    ]
  }
}
```

### Cascade Analysis Fields

| Field | Type | Description |
|-------|------|-------------|
| `is_cascade` | boolean | Whether this anomaly is part of a cascade failure |
| `root_cause_service` | string | Service identified as the root cause of the cascade |
| `affected_chain` | array | List of services in the cascade chain, from root to current |
| `cascade_type` | string | One of: `upstream_cascade`, `chain_degraded`, `dependencies_healthy`, `none` |
| `confidence` | float | Confidence score for the cascade analysis (0.0-1.0) |
| `propagation_path` | array | Detailed path showing each service's status |

### Cascade Types

| Type | Description |
|------|-------------|
| `upstream_cascade` | Single upstream dependency has an anomaly causing this service's issue |
| `chain_degraded` | Multiple services in the dependency chain are affected |
| `dependencies_healthy` | All dependencies are healthy, issue is internal to this service |
| `none` | No cascade analysis performed or no dependencies configured |

---

## Detection Signals Array

Each detection method that contributed to the anomaly is recorded:

```json
{
  "detection_signals": [
    {
      "method": "isolation_forest",
      "type": "ml_isolation",
      "severity": "low",
      "score": -0.01,
      "direction": "high",
      "percentile": 91.7
    },
    {
      "method": "named_pattern_matching",
      "type": "multivariate_pattern",
      "severity": "high",
      "score": -0.5,
      "pattern": "latency_spike_recent"
    }
  ]
}
```

### Detection Signal Fields

| Field | Type | Description |
|-------|------|-------------|
| `method` | string | Detection method used |
| `type` | string | Type of detection |
| `severity` | string | Severity from this individual method |
| `score` | float | Anomaly score from this method |
| `direction` | string | Direction of anomaly: `"high"`, `"low"`, `"activated"` |
| `percentile` | float | Percentile position (for statistical methods) |
| `pattern` | string | Pattern name (for pattern-based methods) |

### Detection Methods

| Method | Type | Description |
|--------|------|-------------|
| `isolation_forest` | `ml_isolation` | Statistical outlier detection per metric |
| `named_pattern_matching` | `multivariate_pattern` | Semantic pattern recognition across metrics |
| `zero_normal_threshold` | `threshold` | Threshold detection for normally-zero metrics |
| `correlation` | `correlation` | Cross-metric correlation analysis |
| `fast_fail` | `pattern` | Fast failure pattern detection |

---

## Single (Non-Consolidated) Anomaly Structure

When only one detection method fires, the anomaly is not consolidated:

```json
{
  "latency_high": {
    "type": "ml_isolation",
    "severity": "medium",
    "score": -0.15,
    "description": "Latency degradation: 450ms (85th percentile, normally 320ms)",
    "detection_method": "isolation_forest",
    "direction": "high",
    "value": 450.0,
    "threshold": 520.0,
    "percentile": 85.2,
    "deviation_sigma": 1.2,
    "possible_causes": [ ... ],
    "checks": [ ... ],
    "comparison_data": { ... },
    "business_impact": "..."
  }
}
```

---

## Comparison Data Structure

Provides statistical context for each core metric:

```json
{
  "comparison_data": {
    "application_latency": {
      "current": 636.1,
      "training_mean": 421.9,
      "training_std": 368.7,
      "training_p95": 668.9,
      "deviation_sigma": 0.58,
      "percentile_estimate": 91.7
    },
    "error_rate": {
      "current": 0.0,
      "training_mean": 0.005,
      "training_std": 0.037,
      "training_p95": 0.0,
      "deviation_sigma": -0.15,
      "percentile_estimate": 0.0
    },
    "request_rate": { ... },
    "dependency_latency": { ... }
  }
}
```

### Comparison Data Fields

| Field | Type | Description |
|-------|------|-------------|
| `current` | float | Current metric value |
| `training_mean` | float | Mean from training data |
| `training_std` | float | Standard deviation from training |
| `training_p95` | float | 95th percentile from training |
| `deviation_sigma` | float | Standard deviations from mean (z-score) |
| `percentile_estimate` | float | Estimated percentile of current value (0-100) |

---

## Current Metrics Object

Raw metric values at detection time:

```json
{
  "current_metrics": {
    "application_latency": 636.1,
    "dependency_latency": 183.3,
    "database_latency": 0.0,
    "error_rate": 0.0,
    "request_rate": 0.039
  }
}
```

### Core Metrics

| Metric | Unit | Description |
|--------|------|-------------|
| `application_latency` | ms | Application response time |
| `dependency_latency` | ms | External client/dependency latency |
| `database_latency` | ms | Database query latency |
| `error_rate` | ratio (0-1) | Error rate as decimal (0.05 = 5%) |
| `request_rate` | req/s | Requests per second |

---

## Exception Context Object

Present when error-related anomalies are detected with HIGH or CRITICAL severity. Provides a breakdown of exception types from OpenTelemetry metrics to help identify root causes.

```json
{
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
  }
}
```

### Exception Context Fields

| Field | Type | Description |
|-------|------|-------------|
| `service_name` | string | Service for which exceptions were queried |
| `timestamp` | string | ISO8601 timestamp of the query (aligned with anomaly detection) |
| `total_exception_rate` | float | Total exceptions per second across all types |
| `exception_count` | integer | Number of distinct exception types found |
| `top_exceptions` | array | Top exception types by rate (max 10) |
| `query_successful` | boolean | Whether the exception query succeeded |
| `error_message` | string \| null | Error message if query failed |

### Top Exceptions Array

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | Full exception class name (including namespace) |
| `short_name` | string | Short class name for display |
| `rate` | float | Exceptions per second for this type |
| `percentage` | float | Percentage of total exceptions (0-100) |

### When Exception Context is Populated

The `exception_context` field is populated when ALL of the following conditions are met:

1. **Severity is HIGH or CRITICAL** - Low/medium severity anomalies don't include exception context
2. **Error-related anomaly detected** - Pattern name contains "error", "failure", or "outage"
3. **Error rate > 1%** - Current `error_rate` metric exceeds 0.01

If any condition is not met, `exception_context` will be `null`.

### Time Alignment

Exception queries are time-aligned with anomaly detection:
- Query window: `[anomaly_timestamp - 5min, anomaly_timestamp]`
- This ensures exception data matches the anomaly detection window
- Prevents querying "current" exceptions when viewing historical anomalies

---

## Service Graph Context Object

Present when client latency anomalies are detected and SLO evaluation confirms latency is above threshold. Provides a breakdown of downstream service calls from OpenTelemetry service graph metrics to help identify which dependencies are contributing to latency issues.

```json
{
  "service_graph_context": {
    "service_name": "cmhub",
    "timestamp": "2024-01-15T10:30:00",
    "total_request_rate": 2.1,
    "route_count": 5,
    "unique_servers": ["r2d2", "eai.production.smartbox.com", "database"],
    "routes": [
      {
        "server": "r2d2",
        "route": "app_broadcastlistener_roomavailabilitylistener",
        "request_rate": 0.117,
        "avg_latency_ms": 29.0,
        "percentage": 5.6
      },
      {
        "server": "eai.production.smartbox.com",
        "route": null,
        "request_rate": 0.087,
        "avg_latency_ms": null,
        "percentage": 4.1
      },
      {
        "server": "database",
        "route": "query",
        "request_rate": 0.050,
        "avg_latency_ms": 150.0,
        "percentage": 2.4
      }
    ],
    "top_route": {
      "server": "r2d2",
      "route": "app_broadcastlistener_roomavailabilitylistener",
      "request_rate": 0.117
    },
    "slowest_route": {
      "server": "database",
      "route": "query",
      "avg_latency_ms": 150.0
    },
    "summary": "Service graph for cmhub (2.10 req/s total):\n  Downstream services: r2d2, eai.production.smartbox.com, database\n  Top routes by traffic:\n    - r2d2/roomavailabilitylistener: 0.117/s (5.6%), 29ms\n    - eai.production.smartbox.com: 0.087/s (4.1%)",
    "query_successful": true,
    "error_message": null
  }
}
```

### Service Graph Context Fields

| Field | Type | Description |
|-------|------|-------------|
| `service_name` | string | Client service for which downstream calls were queried |
| `timestamp` | string | ISO8601 timestamp of the query (aligned with anomaly detection) |
| `total_request_rate` | float | Total requests per second to all downstream services |
| `route_count` | integer | Number of distinct server/route combinations |
| `unique_servers` | array | List of unique downstream server names |
| `routes` | array | All routes sorted by request rate descending |
| `top_route` | object \| null | Route with highest request rate |
| `slowest_route` | object \| null | Route with highest average latency |
| `summary` | string | Human-readable summary of the service graph |
| `query_successful` | boolean | Whether the service graph query succeeded |
| `error_message` | string \| null | Error message if query failed |

### Routes Array

| Field | Type | Description |
|-------|------|-------------|
| `server` | string | Downstream service/server name |
| `route` | string \| null | HTTP route being called (may be null for external services) |
| `request_rate` | float | Requests per second to this route |
| `avg_latency_ms` | float \| null | Average latency in milliseconds (may be null if no latency data) |
| `percentage` | float | Percentage of total downstream traffic (0-100) |

### When Service Graph Context is Populated

The `service_graph_context` field is populated when ALL of the following conditions are met:

1. **SLO latency breach** - `slo_evaluation.latency_evaluation.status` is NOT `"ok"`
2. **Client latency elevated** - The service has elevated dependency_latency metric
3. **Service graph data exists** - VictoriaMetrics has OpenTelemetry service graph metrics for this service

If any condition is not met, `service_graph_context` will be `null`.

### VictoriaMetrics Queries Used

**Request Rate Query:**
```promql
sum(rate(traces_service_graph_request_total{client="<SERVICE>"}[5m]))
    by (client, server, server_http_route)
```

**Latency Query:**
```promql
sum(rate(traces_service_graph_request_server_seconds_sum{client="<SERVICE>"}[5m]))
    by (client, server, server_http_route)
/ sum(rate(traces_service_graph_request_server_seconds_count{client="<SERVICE>"}[5m]))
    by (client, server, server_http_route)
```

### Use Cases

1. **Identify slow dependencies** - The `slowest_route` field immediately highlights which downstream service is causing latency
2. **Traffic distribution** - The `percentage` field shows where most traffic is going
3. **Root cause analysis** - Compare `top_route` vs `slowest_route` to see if high-traffic routes are also slow
4. **External vs internal** - Routes without HTTP route data (null) are typically external services

---

## Fingerprinting Object

Incident tracking and lifecycle information with cycle-based state management:

```json
{
  "fingerprinting": {
    "service_name": "titan",
    "model_name": "business_hours",
    "timestamp": "2025-12-17T13:56:06.028585",
    "overall_action": "CREATE | CONFIRMED | UPDATE | MIXED | RESOLVE | NO_CHANGE",
    "total_active_incidents": 1,
    "total_alerting_incidents": 1,

    "action_summary": {
      "incident_creates": 1,
      "incident_continues": 0,
      "incident_closes": 0,
      "newly_confirmed": 0
    },

    "status_summary": {
      "suspected": 0,
      "confirmed": 1,
      "recovering": 0
    },

    "detection_context": {
      "inference_timestamp": "2025-12-17T13:56:06.028585",
      "model_used": "business_hours",
      "confirmation_cycles": 2,
      "resolution_grace_cycles": 3
    },

    "resolved_incidents": [ ... ],
    "newly_confirmed_incidents": [ ... ]
  }
}
```

### Fingerprinting Fields

| Field | Type | Description |
|-------|------|-------------|
| `overall_action` | string | Summary action: `CREATE`, `CONFIRMED`, `UPDATE`, `MIXED`, `RESOLVE`, `NO_CHANGE` |
| `total_active_incidents` | integer | Total incidents (SUSPECTED + OPEN + RECOVERING) |
| `total_alerting_incidents` | integer | Confirmed incidents only (OPEN status) |
| `action_summary` | object | Counts of each action type |
| `status_summary` | object | Counts by incident status |
| `resolved_incidents` | array | Details of incidents closed in this detection |
| `newly_confirmed_incidents` | array | Incidents that just transitioned to OPEN (ready to alert) |
| `detection_context` | object | Cycle configuration and model info |

### Incident Status Values

| Status | Description |
|--------|-------------|
| `SUSPECTED` | First detection, waiting for confirmation (no alert yet) |
| `OPEN` | Confirmed incident, alerts being sent |
| `RECOVERING` | Not detected for 1-2 cycles (grace period, no resolution yet) |
| `CLOSED` | Incident resolved |

### Overall Action Values

| Action | Description |
|--------|-------------|
| `CREATE` | New SUSPECTED incident created |
| `CONFIRMED` | Incident transitioned from SUSPECTED to OPEN (alert triggered) |
| `UPDATE` | Existing incident continued |
| `RESOLVE` | Incident closed after grace period |
| `MIXED` | Multiple different actions in one cycle |
| `NO_CHANGE` | No significant changes (e.g., still in grace period) |

### Resolved Incidents Array

The `resolved_incidents` array contains details of incidents closed in this detection cycle. Each item has the following structure:

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

| Field | Type | Description |
|-------|------|-------------|
| `fingerprint_id` | string | Pattern identifier that was resolved |
| `incident_id` | string | Unique incident ID being closed |
| `anomaly_name` | string | Name of the resolved anomaly pattern |
| `fingerprint_action` | string | Always `"RESOLVE"` for closed incidents |
| `incident_action` | string | Always `"CLOSE"` for closed incidents |
| `final_severity` | string | Severity level at time of resolution |
| `resolved_at` | string | ISO8601 timestamp of resolution |
| `total_occurrences` | integer | Total times the anomaly was detected |
| `incident_duration_minutes` | integer | Total duration from first_seen to resolved_at |
| `first_seen` | string | When the incident was first detected |
| `service_name` | string | Service name |
| `last_detected_by_model` | string | Which time-period model last detected it |
| `resolution_reason` | string | Why closed: `"resolved"` or `"auto_stale"` |
| `resolution_context` | object \| null | Context about service state at resolution (v1.4.0+) |

### Resolution Context Object

The `resolution_context` field (added in v1.4.0) provides comprehensive context about the service state at resolution time. This enables the Web UI to show WHY an incident was closed and what "healthy" looks like.

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
      }
    },
    "health_summary": {
      "all_metrics_normal": true,
      "slo_compliant": true,
      "summary": "All metrics within acceptable SLO thresholds. Latency 110ms (acceptable < 300ms). Errors 0.01% (acceptable < 0.5%)."
    }
  }
}
```

#### Resolution Context Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `metrics_at_resolution` | object | Yes | Raw metric values at resolution time |
| `slo_evaluation` | object | No | Full SLO evaluation showing operational status (if SLO enabled) |
| `comparison_to_baseline` | object | No | Statistical comparison to training data (if available) |
| `health_summary` | object | Yes | Summary for UI display |

#### Metrics at Resolution

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `request_rate` | float | req/s | Requests per second |
| `application_latency` | float | ms | Server processing time |
| `dependency_latency` | float | ms | External dependency latency |
| `database_latency` | float | ms | Database query time |
| `error_rate` | float | ratio | Error rate (0.05 = 5%) |

#### Comparison to Baseline (Per Metric)

| Field | Type | Description |
|-------|------|-------------|
| `current` | float | Current value at resolution |
| `training_mean` | float | Mean from training data |
| `training_std` | float | Standard deviation from training |
| `deviation_sigma` | float | Standard deviations from mean (z-score) |
| `percentile_estimate` | float | Estimated percentile position (0-100) |
| `status` | string | `"normal"`, `"elevated"`, `"low"`, `"high"`, or `"very_low"` |

#### Health Summary

| Field | Type | Description |
|-------|------|-------------|
| `all_metrics_normal` | boolean | True if all metrics within normal range |
| `slo_compliant` | boolean | True if all SLO thresholds are met |
| `summary` | string | Human-readable summary for display |

#### When Resolution Context is Available

The `resolution_context` field is populated when:
1. **Current metrics are available** at resolution time
2. **SLO evaluation** is enabled (for `slo_evaluation` section)
3. **Training statistics** are available (for `comparison_to_baseline` section)

If metrics are not available at resolution time, `resolution_context` will be `null`.

### Resolution Reasons

| Reason | Description |
|--------|-------------|
| `resolved` | Normal resolution - anomaly stopped being detected for `resolution_grace_cycles` (default: 3 cycles) |
| `auto_stale` | Auto-closed because anomaly reappeared after gap > `incident_separation_minutes` (default: 30 min) |

When `auto_stale` occurs, the old incident is closed and a new incident is created in the same detection cycle. Both the resolution and the new anomaly will appear in the same payload.

---

## Anomaly-Level Fingerprinting Fields

Each anomaly includes fingerprinting metadata with cycle-based state information:

```json
{
  "fingerprint_id": "anomaly_d18f6ae2bf62",
  "fingerprint_action": "CREATE | UPDATE | RESOLVE",
  "incident_id": "incident_31e9e23d4b2b",
  "incident_action": "CREATE | CONTINUE | CLOSE",
  "status": "SUSPECTED | OPEN | RECOVERING",
  "previous_status": "SUSPECTED",
  "incident_duration_minutes": 0,
  "first_seen": "2025-12-17T13:56:06.028585",
  "last_updated": "2025-12-17T13:56:06.028585",
  "occurrence_count": 1,
  "consecutive_detections": 1,
  "confirmation_pending": true,
  "cycles_to_confirm": 1,
  "is_confirmed": false,
  "newly_confirmed": false
}
```

| Field | Type | Description |
|-------|------|-------------|
| `fingerprint_id` | string | Deterministic pattern identifier (content-based hash) |
| `fingerprint_action` | string | Pattern action: `CREATE`, `UPDATE`, `RESOLVE` |
| `incident_id` | string | Unique incident occurrence identifier |
| `incident_action` | string | Incident lifecycle action: `CREATE`, `CONTINUE`, `CLOSE` |
| `status` | string | Current incident status: `SUSPECTED`, `OPEN`, `RECOVERING` |
| `previous_status` | string | Status before this cycle (for tracking transitions) |
| `incident_duration_minutes` | integer | Time since first_seen |
| `first_seen` | string | When incident was first detected |
| `last_updated` | string | Last detection timestamp |
| `occurrence_count` | integer | Total detections during incident |
| `consecutive_detections` | integer | Cycles detected in a row (for confirmation) |
| `confirmation_pending` | boolean | True if still in SUSPECTED state |
| `cycles_to_confirm` | integer | Remaining cycles needed for confirmation |
| `is_confirmed` | boolean | True if status is OPEN |
| `newly_confirmed` | boolean | True if just transitioned to OPEN this cycle |

### Cycle-Based Alerting Logic

**For API consumers:**

1. **Only alert on OPEN incidents**: Check `status == "OPEN"` or `is_confirmed == true`
2. **New alerts**: Check `newly_confirmed == true` for fresh confirmations
3. **Ignore SUSPECTED**: These are not confirmed yet, don't alert
4. **Resolution**: Wait for `incident_action == "CLOSE"` (after grace period)

---

## Severity Levels

| Severity | Priority | Typical Triggers |
|----------|----------|------------------|
| `critical` | 1 (highest) | Multiple methods agree on high severity, system overload patterns |
| `high` | 2 | Pattern match with operational impact, >p95 metrics |
| `medium` | 3 | Statistical anomaly with moderate deviation |
| `low` | 4 | Minor statistical outlier, single method detection |
| `none` | 5 | No anomaly detected |

### Severity Aggregation Rules

For consolidated anomalies:
1. Take maximum severity from all detection signals
2. Boost confidence when multiple methods agree
3. Pattern-based detections typically have higher severity than pure statistical

---

## Named Patterns

Common pattern names and their meanings:

| Pattern Name | Severity | Description |
|--------------|----------|-------------|
| `error_rate_critical` | critical | Very high error rate with normal traffic/latency |
| `error_rate_elevated` | high | Elevated error rate above baseline |
| `latency_spike_recent` | high | Latency increased without traffic change |
| `latency_elevated` | high | Latency above normal threshold |
| `internal_latency_issue` | high | High latency with healthy dependencies (internal problem) |
| `fast_rejection` | critical | Requests rejected rapidly (circuit breaker, rate limit, auth) |
| `partial_rejection` | high | Some requests failing before processing |
| `fast_failure` | critical | Fast failure mode without full processing |
| `traffic_cliff` | critical | Sudden traffic drop (upstream issue) |
| `traffic_surge` | high | Traffic significantly above baseline |
| `traffic_surge_failing` | critical | High traffic + high latency + high errors |
| `traffic_surge_degrading` | high | High traffic causing latency degradation |
| `traffic_surge_healthy` | low | High traffic absorbed successfully |
| `database_bottleneck` | high | Database latency dominating response time |
| `database_degradation` | medium | Database slow but application compensating |
| `downstream_cascade` | high | External dependency causing slowdown |
| `upstream_cascade` | high | Upstream dependency failure affecting service |
| `internal_bottleneck` | high | Internal processing constraint |
| `gradual_degradation` | medium | Slowly worsening performance |
| `recovery_in_progress` | low | Metrics returning to normal |
| `flapping_service` | high | Unstable, oscillating behavior |

---

## Drift Warning Object

Present when model drift is detected (requires `check_drift: true` in config):

```json
{
  "drift_warning": {
    "type": "model_drift",
    "overall_drift_score": 4.2,
    "recommendation": "WARNING: Moderate drift detected. Monitor closely.",
    "affected_metrics": ["request_rate", "error_rate"],
    "confidence_penalty_applied": 0.15,
    "multivariate_drift": false
  }
}
```

### Drift Warning Fields

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | Always `"model_drift"` |
| `overall_drift_score` | float | Maximum z-score across all metrics |
| `recommendation` | string | Action recommendation based on drift severity |
| `affected_metrics` | array | Metrics that have drifted significantly |
| `confidence_penalty_applied` | float | Confidence reduction applied (0.0-0.3) |
| `multivariate_drift` | boolean | Whether multivariate (Mahalanobis) drift detected |

### Confidence Penalties

| Drift Score | Penalty | Meaning |
|-------------|---------|---------|
| < 3 | 0% | Normal variance |
| 3-5 | 15% | Moderate drift, monitor model performance |
| > 5 | 30% | Severe drift, consider retraining |

---

## Drift Analysis Object

Detailed drift analysis (when `check_drift: true`):

```json
{
  "drift_analysis": {
    "has_drift": true,
    "overall_drift_score": 4.2,
    "recommendation": "WARNING: Moderate drift detected. Monitor closely.",
    "drift_metrics": {
      "request_rate": {
        "z_score": 4.2,
        "training_mean": 150.5,
        "training_std": 45.2,
        "current_value": 340.0,
        "severity": "medium"
      }
    },
    "multivariate_drift": false,
    "mahalanobis_distance": 2.1,
    "multivariate_threshold": 8.5
  }
}
```

### Drift Analysis Fields

| Field | Type | Description |
|-------|------|-------------|
| `has_drift` | boolean | Whether any significant drift detected |
| `overall_drift_score` | float | Maximum z-score across metrics |
| `recommendation` | string | Action recommendation |
| `drift_metrics` | object | Per-metric drift details |
| `multivariate_drift` | boolean | Mahalanobis distance exceeds threshold |
| `mahalanobis_distance` | float | Computed Mahalanobis distance |
| `multivariate_threshold` | float | Threshold for multivariate drift |

---

## Validation Warnings Array

Present when input metrics had validation issues:

```json
{
  "validation_warnings": [
    "error_rate: value 1.5 > 1.0, capping at 1.0",
    "application_latency: negative latency -50, using 0.0"
  ]
}
```

### Validation Checks

| Check | Action |
|-------|--------|
| NaN/Inf values | Replaced with 0.0 |
| Negative rates | Capped at 0.0 |
| Negative latencies | Capped at 0.0 |
| Extreme latencies (>5 min) | Capped at 300,000ms |
| Extreme request rates (>1M/s) | Capped at 1,000,000 |
| Error rates > 100% | Capped at 1.0 |

**Note**: When validation warnings are present, the anomaly detection ran on sanitized values. Review the warnings to understand if the original data had quality issues.

---

## Performance Info Object

Model loading and performance metadata:

```json
{
  "performance_info": {
    "lazy_loaded": true,
    "models_loaded": ["business_hours"],
    "period_used": "business_hours",
    "total_available": 5,
    "drift_check_enabled": true,
    "drift_penalty_applied": 0.15
  }
}
```

### Performance Info Fields

| Field | Type | Description |
|-------|------|-------------|
| `lazy_loaded` | boolean | Whether models were lazily loaded |
| `models_loaded` | array | List of time periods with loaded models |
| `period_used` | string | Time period used for this detection |
| `total_available` | integer | Total number of available time periods |
| `drift_check_enabled` | boolean | Whether drift checking was enabled |
| `drift_penalty_applied` | float | Confidence penalty applied due to drift (0.0-0.3) |

---

## Metadata Object

Detection feature flags and context:

```json
{
  "metadata": {
    "service_name": "titan",
    "detection_timestamp": "2025-12-17T13:56:06.028585",
    "models_used": ["application_latency_isolation", "multivariate_detector"],
    "enhanced_messaging": true,
    "features": {
      "contextual_severity": true,
      "named_patterns": true,
      "recommendations": true,
      "interpretations": true,
      "anomaly_correlation": true
    }
  }
}
```

---

## SLO Evaluation Object

Present when SLO-aware severity evaluation is enabled (see `slos.enabled` in config). This layer adjusts ML-detected severity based on operational SLO thresholds.

```json
{
  "slo_evaluation": {
    "original_severity": "high",
    "adjusted_severity": "low",
    "severity_changed": true,
    "slo_status": "ok",
    "slo_proximity": 0.45,
    "operational_impact": "informational",
    "is_busy_period": false,
    "latency_evaluation": {
      "status": "ok",
      "proximity": 0.45,
      "value": 225.0,
      "threshold_acceptable": 300,
      "threshold_warning": 400,
      "threshold_critical": 500
    },
    "error_rate_evaluation": {
      "status": "ok",
      "proximity": 0.1,
      "value": 0.001,
      "value_percent": "0.10%",
      "threshold_acceptable": 0.005,
      "threshold_warning": 0.01,
      "threshold_critical": 0.02
    },
    "database_latency_evaluation": {
      "status": "warning",
      "value_ms": 45.2,
      "baseline_mean_ms": 15.0,
      "ratio": 3.01,
      "floor_ms": 5.0,
      "thresholds": {
        "info": 1.5,
        "warning": 2.0,
        "high": 3.0,
        "critical": 5.0
      },
      "explanation": "DB latency elevated: 45.2ms is 3.0x baseline (15.0ms)"
    },
    "request_rate_evaluation": {
      "status": "ok",
      "value": 150.5,
      "baseline_mean": 145.0,
      "ratio": 1.04,
      "surge_threshold": 3.0,
      "cliff_threshold": 0.1,
      "explanation": "Request rate within normal range"
    },
    "explanation": "Severity adjusted from high to low based on SLO evaluation. Anomaly detected but metrics within acceptable SLO thresholds (latency: 225ms < 300ms, errors: 0.10% < 0.50%)."
  }
}
```

### SLO Evaluation Fields

| Field | Type | Description |
|-------|------|-------------|
| `original_severity` | string | ML-assigned severity before SLO adjustment |
| `adjusted_severity` | string | Final severity after SLO evaluation |
| `severity_changed` | boolean | Whether severity was adjusted |
| `slo_status` | string | Overall SLO status: `ok`, `elevated`, `warning`, `breached` |
| `slo_proximity` | float | How close to SLO breach (0.0 = far, 1.0 = at threshold, >1.0 = breached) |
| `operational_impact` | string | Impact level: `none`, `informational`, `actionable`, `critical` |
| `is_busy_period` | boolean | Whether detection occurred during configured busy period |
| `latency_evaluation` | object | Latency metrics vs SLO thresholds |
| `error_rate_evaluation` | object | Error rate vs SLO thresholds |
| `database_latency_evaluation` | object | Database latency ratio evaluation (optional) |
| `request_rate_evaluation` | object | Request rate surge/cliff evaluation (optional) |
| `explanation` | string | Human-readable explanation of SLO evaluation |

### SLO Status Values

| Status | Description |
|--------|-------------|
| `ok` | All metrics well within acceptable thresholds |
| `elevated` | Metrics above acceptable but below warning threshold |
| `warning` | Approaching SLO breach, investigate soon |
| `breached` | SLO threshold exceeded, immediate action needed |

### Operational Impact Values

| Impact | Description |
|--------|-------------|
| `none` | No operational concern |
| `informational` | Anomaly detected but operationally acceptable - log but don't alert |
| `actionable` | Approaching limits or elevated state - should investigate |
| `critical` | SLO breached - requires immediate attention |

### Severity Adjustment Logic

The SLO layer can adjust severity in these ways:

| Scenario | ML Severity | SLO Status | Adjusted Severity |
|----------|-------------|------------|-------------------|
| Anomaly within acceptable limits | critical/high/medium | ok | **low** |
| Anomaly approaching SLO | medium | warning | high |
| SLO breached (regardless of ML) | any | breached | critical |
| No anomaly but SLO elevated | none | elevated | low |

**Key principle**: When `slo_status` is `ok` (all metrics within acceptable thresholds), severity is always adjusted to `low` regardless of the original ML-assigned severity. This ensures alerts reflect operational impact, not just statistical deviation.

**Key insight**: ML answers "is this unusual?" while SLO evaluation answers "does it matter operationally?"

### Database Latency Evaluation

The `database_latency_evaluation` object evaluates database latency using ratio-based thresholds against the training baseline.

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Status: `ok`, `info`, `warning`, `high`, `critical` |
| `value_ms` | float | Current database latency in milliseconds |
| `baseline_mean_ms` | float | Mean database latency from training data |
| `ratio` | float | Current / baseline ratio |
| `floor_ms` | float | Minimum latency to consider (noise filter, default: 5ms) |
| `thresholds` | object | Ratio thresholds for each status level |
| `explanation` | string | Human-readable explanation |

**Ratio Thresholds (defaults):**
| Status | Ratio | Meaning |
|--------|-------|---------|
| `ok` | < 1.5× | Normal database performance |
| `info` | ≥ 1.5× | Slightly elevated, monitor |
| `warning` | ≥ 2.0× | Elevated, investigate |
| `high` | ≥ 3.0× | Significantly elevated |
| `critical` | ≥ 5.0× | Critical database performance issue |

**Note:** Values below the floor (default 5ms) are always considered `ok` to filter out noise from low-latency databases.

### Request Rate Evaluation

The `request_rate_evaluation` object detects sudden traffic surges or cliffs.

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Status: `ok`, `surge`, `cliff` |
| `value` | float | Current request rate (req/s) |
| `baseline_mean` | float | Mean request rate from training |
| `ratio` | float | Current / baseline ratio |
| `surge_threshold` | float | Ratio above which is considered a surge (default: 3.0×) |
| `cliff_threshold` | float | Ratio below which is considered a cliff (default: 0.1×) |
| `explanation` | string | Human-readable explanation |

**Detection Logic:**
| Condition | Status | Meaning |
|-----------|--------|---------|
| ratio > surge_threshold | `surge` | Traffic surge detected (e.g., 3× normal) |
| ratio < cliff_threshold | `cliff` | Traffic cliff detected (e.g., 90% drop) |
| otherwise | `ok` | Normal traffic variation |

---

## Complete Example: Consolidated Anomaly

```json
{
  "alert_type": "anomaly_detected",
  "service_name": "titan",
  "timestamp": "2025-12-17T13:56:06.028585",
  "time_period": "business_hours",
  "model_name": "business_hours",
  "model_type": "time_aware_5period",

  "anomalies": {
    "latency_spike_recent": {
      "type": "consolidated",
      "root_metric": "application_latency",
      "severity": "high",
      "confidence": 0.80,
      "score": -0.5,
      "signal_count": 2,

      "description": "Recent performance degradation: latency increased to 636ms (confirmed by 2 detection methods)",
      "interpretation": "Latency recently increased without traffic change - something changed. Not a capacity issue; likely recent deployment, config change, or dependency degradation.",
      "pattern_name": "latency_spike_recent",

      "value": 636.1,

      "detection_signals": [
        {
          "method": "isolation_forest",
          "type": "ml_isolation",
          "severity": "low",
          "score": -0.01,
          "direction": "high",
          "percentile": 91.7
        },
        {
          "method": "named_pattern_matching",
          "type": "multivariate_pattern",
          "severity": "high",
          "score": -0.5,
          "pattern": "latency_spike_recent"
        }
      ],

      "possible_causes": [
        "Resource exhaustion (CPU, memory, threads)",
        "Garbage collection pressure",
        "Lock contention or thread pool saturation",
        "Downstream service slowness",
        "Database query performance degradation"
      ],

      "recommended_actions": [
        "IMMEDIATE: Check deployments in last 2 hours",
        "CHECK: Configuration changes (feature flags, settings)",
        "CHECK: External dependency response times",
        "CHECK: Database query performance",
        "CHECK: GC behavior and memory pressure"
      ],

      "checks": [
        "Check CPU and memory utilization",
        "Review GC logs for long pauses",
        "Monitor database query times"
      ],

      "comparison_data": {
        "application_latency": {
          "current": 636.1,
          "training_mean": 421.9,
          "training_std": 368.7,
          "training_p95": 668.9,
          "deviation_sigma": 0.58,
          "percentile_estimate": 91.7
        },
        "error_rate": {
          "current": 0.0,
          "training_mean": 0.005,
          "training_std": 0.037,
          "training_p95": 0.0,
          "deviation_sigma": -0.15,
          "percentile_estimate": 0.0
        },
        "request_rate": {
          "current": 0.039,
          "training_mean": 0.19,
          "training_std": 0.4,
          "training_p95": 1.26,
          "deviation_sigma": -0.38,
          "percentile_estimate": 36.9
        }
      },

      "business_impact": "Anomalous behavior detected - monitor for escalation",

      "fingerprint_id": "anomaly_b827fc318ca5",
      "fingerprint_action": "CREATE",
      "incident_id": "incident_7a723c1a2a9e",
      "incident_action": "CREATE",
      "incident_duration_minutes": 0,
      "first_seen": "2025-12-17T13:56:06.028585",
      "last_updated": "2025-12-17T13:56:06.028585",
      "occurrence_count": 1,
      "time_confidence": 0.9,
      "detected_by_model": "business_hours"
    }
  },

  "anomaly_count": 1,
  "overall_severity": "high",

  "current_metrics": {
    "application_latency": 636.1,
    "dependency_latency": 183.3,
    "database_latency": 0.0,
    "error_rate": 0.0,
    "request_rate": 0.039
  },

  "exception_context": null,

  "fingerprinting": {
    "service_name": "titan",
    "model_name": "business_hours",
    "timestamp": "2025-12-17T13:56:06.028585",
    "overall_action": "CREATE",
    "total_open_incidents": 1,
    "action_summary": {
      "incident_creates": 1,
      "incident_continues": 0,
      "incident_closes": 0
    },
    "detection_context": {
      "inference_timestamp": "2025-12-17T13:56:06.028585",
      "model_used": "business_hours"
    },
    "resolved_incidents": []
  },

  "performance_info": {
    "lazy_loaded": true,
    "models_loaded": ["business_hours"],
    "period_used": "business_hours",
    "total_available": 5
  },

  "metadata": {
    "service_name": "titan",
    "detection_timestamp": "2025-12-17T13:56:06.028585",
    "models_used": ["application_latency_isolation", "multivariate_detector"],
    "enhanced_messaging": true,
    "features": {
      "contextual_severity": true,
      "named_patterns": true,
      "recommendations": true,
      "interpretations": true,
      "anomaly_correlation": true
    }
  }
}
```

---

## Complete Example: Error Anomaly with Exception Context

When an error-related anomaly is detected with high severity, the `exception_context` field is populated:

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
      "root_metric": "error_rate",
      "severity": "high",
      "confidence": 0.85,
      "score": -0.45,
      "signal_count": 2,

      "description": "Error rate elevated above normal threshold (5% vs typical 0.5%)",
      "interpretation": "Error rate is significantly above the normal range. Exception analysis shows R2D2Exception (62% of errors) is the dominant exception type.",
      "pattern_name": "elevated_errors",

      "value": 0.05,

      "possible_causes": [
        "Downstream service failures",
        "Database connectivity issues",
        "Application bug in recent deployment"
      ],

      "recommended_actions": [
        "INVESTIGATE: Top exception is R2D2Exception (62% of errors)",
        "CHECK: Application logs for error details",
        "VERIFY: Recent deployments or configuration changes"
      ],

      "comparison_data": {
        "error_rate": {
          "current": 0.05,
          "training_mean": 0.005,
          "training_std": 0.01,
          "training_p95": 0.02,
          "deviation_sigma": 4.5,
          "percentile_estimate": 98.5
        }
      }
    }
  },

  "anomaly_count": 1,
  "overall_severity": "high",

  "current_metrics": {
    "application_latency": 150.5,
    "dependency_latency": 45.2,
    "database_latency": 12.3,
    "error_rate": 0.05,
    "request_rate": 150.5
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

  "fingerprinting": {
    "service_name": "search",
    "model_name": "business_hours",
    "timestamp": "2024-01-15T10:30:00",
    "overall_action": "CONFIRMED",
    "total_active_incidents": 1,
    "total_alerting_incidents": 1
  },

  "metadata": {
    "service_name": "search",
    "detection_timestamp": "2024-01-15T10:30:00",
    "enhanced_messaging": true
  }
}
```

---

## Complete Example: Latency Anomaly with Service Graph Context

When a client latency anomaly is detected and SLO evaluation confirms latency breach, the `service_graph_context` field shows downstream service calls:

```json
{
  "alert_type": "anomaly_detected",
  "service_name": "cmhub",
  "timestamp": "2024-01-15T10:30:00",
  "time_period": "business_hours",
  "model_name": "business_hours",
  "model_type": "time_aware_5period",

  "anomalies": {
    "dependency_latency_elevated": {
      "type": "consolidated",
      "root_metric": "dependency_latency",
      "severity": "high",
      "confidence": 0.82,
      "score": -0.42,
      "signal_count": 2,

      "description": "Client latency elevated above normal threshold (850ms vs typical 200ms)",
      "interpretation": "Downstream service calls are taking longer than expected. Service graph analysis shows database route has highest latency (150ms avg).",
      "pattern_name": "external_dependency_slow",

      "value": 850.0,

      "possible_causes": [
        "Downstream service slowdown",
        "Network latency to dependencies",
        "Connection pool exhaustion"
      ],

      "recommended_actions": [
        "INVESTIGATE: Slowest downstream route is database/query (150ms)",
        "CHECK: r2d2 service performance (highest traffic: 0.117/s)",
        "VERIFY: Network connectivity to downstream services"
      ]
    }
  },

  "anomaly_count": 1,
  "overall_severity": "high",

  "current_metrics": {
    "application_latency": 120.5,
    "dependency_latency": 850.0,
    "database_latency": 45.2,
    "error_rate": 0.001,
    "request_rate": 25.4
  },

  "exception_context": null,

  "service_graph_context": {
    "service_name": "cmhub",
    "timestamp": "2024-01-15T10:30:00",
    "total_request_rate": 2.1,
    "route_count": 3,
    "unique_servers": ["r2d2", "eai.production.smartbox.com", "database"],
    "routes": [
      {
        "server": "r2d2",
        "route": "app_broadcastlistener_roomavailabilitylistener",
        "request_rate": 0.117,
        "avg_latency_ms": 29.0,
        "percentage": 5.6
      },
      {
        "server": "eai.production.smartbox.com",
        "route": null,
        "request_rate": 0.087,
        "avg_latency_ms": null,
        "percentage": 4.1
      },
      {
        "server": "database",
        "route": "query",
        "request_rate": 0.050,
        "avg_latency_ms": 150.0,
        "percentage": 2.4
      }
    ],
    "top_route": {
      "server": "r2d2",
      "route": "app_broadcastlistener_roomavailabilitylistener",
      "request_rate": 0.117
    },
    "slowest_route": {
      "server": "database",
      "route": "query",
      "avg_latency_ms": 150.0
    },
    "summary": "Service graph for cmhub (2.10 req/s total):\n  Downstream services: r2d2, eai.production.smartbox.com, database\n  Top routes by traffic:\n    - r2d2/roomavailabilitylistener: 0.117/s (5.6%), 29ms\n    - eai.production.smartbox.com: 0.087/s (4.1%)\n  Slowest route: database/query (150ms)",
    "query_successful": true,
    "error_message": null
  },

  "slo_evaluation": {
    "original_severity": "high",
    "adjusted_severity": "high",
    "severity_changed": false,
    "slo_status": "warning",
    "latency_evaluation": {
      "status": "warning",
      "value": 850.0,
      "threshold_acceptable": 300,
      "threshold_warning": 500,
      "threshold_critical": 1000
    }
  },

  "fingerprinting": {
    "service_name": "cmhub",
    "model_name": "business_hours",
    "timestamp": "2024-01-15T10:30:00",
    "overall_action": "CONFIRMED",
    "total_active_incidents": 1,
    "total_alerting_incidents": 1
  },

  "metadata": {
    "service_name": "cmhub",
    "detection_timestamp": "2024-01-15T10:30:00",
    "enhanced_messaging": true
  }
}
```

---

## Complete Example: No Anomaly

```json
{
  "alert_type": "no_anomaly",
  "service_name": "booking",
  "timestamp": "2025-12-17T14:00:00.000000",
  "time_period": "business_hours",
  "model_name": "business_hours",
  "model_type": "time_aware_5period",

  "anomalies": {},
  "anomaly_count": 0,
  "overall_severity": "none",

  "current_metrics": {
    "application_latency": 115.2,
    "dependency_latency": 12.5,
    "database_latency": 8.3,
    "error_rate": 0.001,
    "request_rate": 25.4
  },

  "fingerprinting": {
    "service_name": "booking",
    "model_name": "business_hours",
    "timestamp": "2025-12-17T14:00:00.000000",
    "overall_action": "NONE",
    "total_open_incidents": 0,
    "action_summary": {
      "incident_creates": 0,
      "incident_continues": 0,
      "incident_closes": 0
    },
    "resolved_incidents": []
  },

  "metadata": {
    "service_name": "booking",
    "detection_timestamp": "2025-12-17T14:00:00.000000",
    "models_used": [],
    "enhanced_messaging": true,
    "features": {
      "contextual_severity": true,
      "named_patterns": true,
      "recommendations": true,
      "interpretations": true,
      "anomaly_correlation": true
    }
  }
}
```

---

## Complete Example: Metrics Unavailable

When the metrics server (VictoriaMetrics) is unreachable or critical metrics fail to collect, detection is skipped to avoid false alerts:

```json
{
  "alert_type": "metrics_unavailable",
  "service_name": "booking",
  "timestamp": "2025-12-25T16:51:46.000000",

  "error": "Metrics collection failed: 5/5 metrics failed: request_rate, application_latency, dependency_latency, database_latency, error_rate",
  "skipped_reason": "critical_metrics_unavailable",

  "failed_metrics": [
    "request_rate",
    "application_latency",
    "dependency_latency",
    "database_latency",
    "error_rate"
  ],

  "collection_errors": {
    "request_rate": "HTTPSConnectionPool: Max retries exceeded (Connection refused)",
    "application_latency": "HTTPSConnectionPool: Max retries exceeded (Connection refused)",
    "dependency_latency": "HTTPSConnectionPool: Max retries exceeded (Connection refused)",
    "database_latency": "HTTPSConnectionPool: Max retries exceeded (Connection refused)",
    "error_rate": "HTTPSConnectionPool: Max retries exceeded (Connection refused)"
  },

  "anomalies": {},
  "anomaly_count": 0,
  "overall_severity": "none"
}
```

### Metrics Unavailable Fields

| Field | Type | Description |
|-------|------|-------------|
| `alert_type` | string | Always `"metrics_unavailable"` for this response type |
| `error` | string | Human-readable summary of the collection failure |
| `skipped_reason` | string | Why detection was skipped: `"critical_metrics_unavailable"` |
| `failed_metrics` | array | List of metric names that failed to collect |
| `collection_errors` | object | Detailed error message per failed metric |

### When Metrics Unavailable is Returned

Detection is skipped and `metrics_unavailable` is returned when:

1. **Critical metric failed**: `request_rate` failed to collect
   - This prevents false "traffic cliff" alerts (0.0 looks like traffic dropped to zero)

2. **Too many failures**: 3+ metrics failed to collect
   - Too much missing data for reliable detection

### Partial Metrics Failure

If only 1-2 non-critical metrics fail (e.g., `database_latency`), detection proceeds with a warning:

```json
{
  "alert_type": "anomaly_detected",
  "service_name": "booking",

  "partial_metrics_failure": {
    "failed_metrics": ["database_latency"],
    "failure_summary": "1/5 metrics failed: database_latency"
  },

  "anomalies": { ... }
}
```

---

## Key Changes from Previous Format

| Aspect | Previous | Current |
|--------|----------|---------|
| Anomaly grouping | Separate anomalies per detection method | Consolidated by root metric |
| Anomaly naming | `ml_isolation_isolation_forest` | `latency_high` or pattern name |
| Anomaly count | Count of detection signals | Count of distinct issues |
| Severity | Per-method, conflicting | Aggregated maximum |
| Confidence | Not present | Added (0.0-1.0) |
| Detection transparency | Lost after detection | Preserved in `detection_signals` |

---

## API Server Implementation Notes

1. **Anomaly identification**: Use the anomaly key (e.g., `"latency_spike_recent"`) as the primary identifier
2. **Severity handling**: Use the top-level `severity` field for alerting decisions
3. **Confidence threshold**: Consider using `confidence >= 0.7` for high-confidence alerts
4. **Signal inspection**: The `detection_signals` array provides audit trail of what triggered the anomaly
5. **Incident correlation**: Use `fingerprint_id` and `incident_id` for tracking across detections
