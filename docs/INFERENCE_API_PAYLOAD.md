# Inference Engine API Payload Specification

This document describes the JSON payload format sent by the inference engine to the API server after anomaly detection.

---

## Overview

The inference engine performs anomaly detection using multiple methods (Isolation Forest, Pattern Matching, Statistical Correlation) and returns a consolidated view where related anomalies are grouped into single actionable alerts.

---

## Top-Level Response Structure

```json
{
  "alert_type": "anomaly_detected | no_anomaly",
  "service_name": "string",
  "timestamp": "ISO8601 datetime",
  "time_period": "business_hours | evening_hours | night_hours | weekend_day | weekend_night",
  "model_name": "string (e.g., 'business_hours')",
  "model_type": "time_aware_5period | single",

  "anomalies": { ... },
  "anomaly_count": "integer",
  "overall_severity": "critical | high | medium | low | none",

  "current_metrics": { ... },
  "fingerprinting": { ... },
  "performance_info": { ... },
  "metadata": { ... },

  "drift_warning": { ... },
  "validation_warnings": [ ... ],
  "drift_analysis": { ... }
}
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `alert_type` | string | `"anomaly_detected"` if anomalies found, `"no_anomaly"` otherwise |
| `service_name` | string | Name of the service being monitored |
| `timestamp` | string | ISO8601 timestamp of the detection |
| `time_period` | string | Current time period used for detection |
| `model_name` | string | Name of the model used (matches time_period for time-aware) |
| `model_type` | string | Type of model: `"time_aware_5period"` or `"single"` |
| `anomaly_count` | integer | Number of distinct anomalies (after consolidation) |
| `overall_severity` | string | Highest severity among all anomalies |
| `drift_warning` | object | Present when model drift is detected (see Drift Warning section) |
| `validation_warnings` | array | List of input validation issues (see Validation Warnings section) |
| `drift_analysis` | object | Detailed drift analysis results (when `check_drift` is enabled) |

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
| Named pattern | `{pattern_name}` | `recent_degradation`, `fast_rejection`, `traffic_cliff` |
| Latency anomaly | `latency_anomaly` | When consolidated from latency-related detections |
| Error anomaly | `error_rate_anomaly` | When consolidated from error-related detections |
| Traffic anomaly | `traffic_anomaly` | When consolidated from request_rate detections |
| Single detection | `{metric}_{direction}` | `latency_high`, `error_rate_high` |

---

## Consolidated Anomaly Structure

When multiple detection methods identify the same underlying issue, they are consolidated into a single anomaly:

```json
{
  "recent_degradation": {
    "type": "consolidated",
    "root_metric": "application_latency",
    "severity": "high",
    "confidence": 0.80,
    "score": -0.5,
    "signal_count": 2,

    "description": "Latency degradation: 636ms (92nd percentile). (confirmed by 2 detection methods)",
    "interpretation": "Latency recently increased without traffic change - something changed...",
    "pattern_name": "recent_degradation",

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
      "pattern": "recent_degradation"
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
    "client_latency": { ... }
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
    "client_latency": 183.3,
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
| `client_latency` | ms | External client/dependency latency |
| `database_latency` | ms | Database query latency |
| `error_rate` | ratio (0-1) | Error rate as decimal (0.05 = 5%) |
| `request_rate` | req/s | Requests per second |

---

## Fingerprinting Object

Incident tracking and lifecycle information:

```json
{
  "fingerprinting": {
    "service_name": "titan",
    "model_name": "business_hours",
    "timestamp": "2025-12-17T13:56:06.028585",
    "overall_action": "CREATE | UPDATE | MIXED | RESOLVE",
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

    "resolved_incidents": [ ... ]
  }
}
```

### Fingerprinting Fields

| Field | Type | Description |
|-------|------|-------------|
| `overall_action` | string | Summary action: `CREATE`, `UPDATE`, `MIXED`, `RESOLVE` |
| `total_open_incidents` | integer | Current number of open incidents |
| `action_summary` | object | Counts of each action type |
| `resolved_incidents` | array | Details of incidents closed in this detection |

---

## Anomaly-Level Fingerprinting Fields

Each anomaly includes fingerprinting metadata:

```json
{
  "fingerprint_id": "anomaly_d18f6ae2bf62",
  "fingerprint_action": "CREATE | UPDATE | RESOLVE",
  "incident_id": "incident_31e9e23d4b2b",
  "incident_action": "CREATE | CONTINUE | CLOSE",
  "incident_duration_minutes": 0,
  "first_seen": "2025-12-17T13:56:06.028585",
  "last_updated": "2025-12-17T13:56:06.028585",
  "occurrence_count": 1
}
```

| Field | Type | Description |
|-------|------|-------------|
| `fingerprint_id` | string | Unique identifier for this anomaly fingerprint |
| `fingerprint_action` | string | Action taken: `CREATE`, `UPDATE`, `RESOLVE` |
| `incident_id` | string | Incident this anomaly belongs to |
| `incident_action` | string | Incident lifecycle action |
| `incident_duration_minutes` | integer | How long incident has been open |
| `first_seen` | string | When anomaly was first detected |
| `last_updated` | string | Last update timestamp |
| `occurrence_count` | integer | Number of consecutive detections |

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
| `recent_degradation` | high | Latency increased without traffic change |
| `fast_rejection` | high | Requests rejected rapidly (circuit breaker, rate limit, auth) |
| `fast_failure` | high | Fast failure mode without full processing |
| `traffic_cliff` | critical | Sudden traffic drop (upstream issue) |
| `database_bottleneck` | high | Database latency dominating response time |
| `external_dependency_slow` | high | External service causing slowdown |
| `traffic_surge_failing` | critical | High traffic + high latency + high errors |
| `internal_bottleneck` | medium | Internal processing constraint |
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
    "recent_degradation": {
      "type": "consolidated",
      "root_metric": "application_latency",
      "severity": "high",
      "confidence": 0.80,
      "score": -0.5,
      "signal_count": 2,

      "description": "Recent performance degradation: latency increased to 636ms (confirmed by 2 detection methods)",
      "interpretation": "Latency recently increased without traffic change - something changed. Not a capacity issue; likely recent deployment, config change, or dependency degradation.",
      "pattern_name": "recent_degradation",

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
          "pattern": "recent_degradation"
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
    "client_latency": 183.3,
    "database_latency": 0.0,
    "error_rate": 0.0,
    "request_rate": 0.039
  },

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
    "client_latency": 12.5,
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

1. **Anomaly identification**: Use the anomaly key (e.g., `"recent_degradation"`) as the primary identifier
2. **Severity handling**: Use the top-level `severity` field for alerting decisions
3. **Confidence threshold**: Consider using `confidence >= 0.7` for high-confidence alerts
4. **Signal inspection**: The `detection_signals` array provides audit trail of what triggered the anomaly
5. **Incident correlation**: Use `fingerprint_id` and `incident_id` for tracking across detections
