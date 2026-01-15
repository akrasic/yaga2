# API Payload Reference

Reference for the JSON payload structure sent by the inference engine.

## Alert Types

| Type | Description |
|------|-------------|
| `anomaly_detected` | Anomalies detected for service |
| `no_anomaly` | Service is healthy |
| `metrics_unavailable` | Metrics collection failed |

## Top-Level Structure

```json
{
  "alert_type": "anomaly_detected",
  "service_name": "booking",
  "timestamp": "2024-01-15T10:30:00",
  "time_period": "business_hours",
  "model_name": "business_hours",
  "model_type": "time_aware_5period",

  "anomaly_count": 1,
  "overall_severity": "high",

  "anomalies": { ... },
  "current_metrics": { ... },
  "slo_evaluation": { ... },
  "exception_context": { ... },
  "service_graph_context": { ... },
  "fingerprinting": { ... }
}
```

## Current Metrics

```json
{
  "current_metrics": {
    "request_rate": 52.7,
    "application_latency": 110.5,
    "dependency_latency": 1.4,
    "database_latency": 0.8,
    "error_rate": 0.0001
  }
}
```

| Metric | Unit | Description |
|--------|------|-------------|
| request_rate | req/s | Requests per second |
| application_latency | ms | Server processing time |
| dependency_latency | ms | External dependency latency |
| database_latency | ms | Database query time |
| error_rate | ratio | Error rate (0.05 = 5%) |

## Anomaly Object

```json
{
  "latency_spike_recent": {
    "type": "consolidated",
    "root_metric": "application_latency",
    "severity": "high",
    "confidence": 0.80,
    "score": -0.5,
    "signal_count": 2,

    "description": "Latency degradation: 636ms (92nd percentile)",
    "interpretation": "Latency recently increased...",
    "pattern_name": "latency_spike_recent",

    "detection_signals": [ ... ],
    "recommended_actions": [ ... ],
    "comparison_data": { ... },

    "fingerprint_id": "anomaly_d18f6ae2bf62",
    "incident_id": "incident_31e9e23d4b2b",
    "status": "OPEN",
    "occurrence_count": 5,
    "is_confirmed": true
  }
}
```

## Detection Signals

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

## SLO Evaluation

```json
{
  "slo_evaluation": {
    "original_severity": "critical",
    "adjusted_severity": "low",
    "severity_changed": true,
    "slo_status": "ok",
    "slo_proximity": 0.56,
    "operational_impact": "informational",

    "latency_evaluation": {
      "status": "ok",
      "value": 280.0,
      "threshold_acceptable": 300,
      "proximity": 0.93
    },
    "error_rate_evaluation": {
      "status": "ok",
      "value": 0.001,
      "value_percent": "0.10%",
      "within_acceptable": true
    },
    "database_latency_evaluation": {
      "status": "warning",
      "value_ms": 25.0,
      "baseline_mean_ms": 10.0,
      "ratio": 2.5
    },
    "request_rate_evaluation": {
      "status": "ok",
      "type": "normal",
      "value_rps": 52.7,
      "baseline_mean_rps": 50.0,
      "ratio": 1.05
    },

    "explanation": "Severity adjusted from critical to low..."
  }
}
```

## Cascade Analysis

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

## Fingerprinting

```json
{
  "fingerprinting": {
    "service_name": "booking",
    "model_name": "business_hours",
    "timestamp": "2025-12-17T13:56:06",
    "overall_action": "UPDATE",
    "total_active_incidents": 1,
    "total_alerting_incidents": 1,

    "action_summary": {
      "incident_creates": 0,
      "incident_continues": 1,
      "incident_closes": 0,
      "newly_confirmed": 0
    },
    "status_summary": {
      "suspected": 0,
      "confirmed": 1,
      "recovering": 0
    },
    "resolved_incidents": [],
    "newly_confirmed_incidents": []
  }
}
```

## Exception Context

Present when error SLO breached:

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
      }
    ],
    "query_successful": true
  }
}
```

## Service Graph Context

Present when client latency SLO breached:

```json
{
  "service_graph_context": {
    "service_name": "cmhub",
    "total_request_rate": 2.1,
    "routes": [
      {
        "server": "r2d2",
        "route": "roomavailabilitylistener",
        "request_rate": 0.117,
        "avg_latency_ms": 29.0,
        "percentage": 5.6
      }
    ],
    "top_route": { ... },
    "slowest_route": { ... },
    "summary": "Service graph for cmhub..."
  }
}
```

## Resolution Payload

```json
{
  "resolved_incidents": [
    {
      "fingerprint_id": "anomaly_061598e9ca91",
      "incident_id": "incident_abc123def456",
      "anomaly_name": "database_degradation",
      "fingerprint_action": "RESOLVE",
      "incident_action": "CLOSE",
      "final_severity": "medium",
      "resolved_at": "2025-12-17T14:30:00",
      "total_occurrences": 5,
      "incident_duration_minutes": 45,
      "first_seen": "2025-12-17T13:45:00",
      "resolution_reason": "resolved"
    }
  ]
}
```

## Severity Values

| Severity | Priority | Description |
|----------|----------|-------------|
| critical | 1 | Immediate action required |
| high | 2 | Investigate promptly |
| medium | 3 | Monitor closely |
| low | 4 | Informational |
| none | 5 | No anomaly |

## Named Patterns

| Pattern | Severity | Trigger Condition |
|---------|----------|-------------------|
| `traffic_surge_healthy` | low | High traffic, normal latency/errors |
| `traffic_surge_degrading` | high | High traffic, high latency |
| `traffic_surge_failing` | critical | High traffic, high latency, high errors |
| `traffic_cliff` | critical | Very low traffic |
| `latency_spike_recent` | high | Normal traffic, high latency |
| `internal_latency_issue` | high | High latency, healthy deps |
| `error_rate_elevated` | high | Elevated error rate |
| `error_rate_critical` | critical | Very high error rate |
| `fast_failure` | critical | Low latency, high errors |
| `fast_rejection` | critical | Very low latency, very high errors |
| `database_bottleneck` | high | High DB latency, DB dominant |
| `database_degradation` | medium | High DB latency, compensating |
| `upstream_cascade` | high | High latency + upstream anomaly |
