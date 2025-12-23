# Anomaly Detection API Specification

**Version:** 1.0.0
**Last Updated:** 2024-01-15

This document defines the stable JSON contracts for the Smartbox ML Anomaly Detection API. These specifications enable building a consistent ingest API and web UI for anomaly monitoring.

## Table of Contents

1. [Overview](#overview)
2. [API Endpoints](#api-endpoints)
3. [Alert Types](#alert-types)
4. [Data Models](#data-models)
5. [Lifecycle Flow](#lifecycle-flow)
6. [Examples](#examples)
7. [Error Handling](#error-handling)
8. [Schema Versioning](#schema-versioning)

---

## Overview

The anomaly detection pipeline produces two types of alerts:

| Alert Type | Endpoint | Purpose |
|------------|----------|---------|
| `anomaly_detected` | `/api/anomalies/batch` | Active anomalies detected |
| `incident_resolved` | `/api/incidents/resolve` | Previously detected anomalies that cleared |

### Key Concepts

- **Fingerprint ID**: Deterministic hash identifying an anomaly *pattern* (service + anomaly type). Same pattern = same fingerprint across time.
- **Incident ID**: Unique identifier for a specific *occurrence* of an anomaly pattern. Each time a pattern appears after being resolved, it gets a new incident ID.
- **Time Period**: One of 5 behavioral periods that determine which ML model is used (business_hours, evening_hours, night_hours, weekend_day, weekend_night).

---

## API Endpoints

### POST /api/anomalies/batch

Ingests active anomaly alerts.

**Request:**
```json
{
  "alerts": [
    { /* AnomalyDetectedPayload */ },
    { /* AnomalyDetectedPayload */ }
  ],
  "schema_version": "1.0.0"
}
```

**Response:**
```json
{
  "success": true,
  "processed_count": 2,
  "failed_count": 0,
  "errors": [],
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### POST /api/incidents/resolve

Notifies that previously detected anomalies have cleared.

**Request:**
```json
{
  "resolutions": [
    { /* IncidentResolvedPayload */ }
  ],
  "schema_version": "1.0.0"
}
```

**Response:**
```json
{
  "success": true,
  "processed_count": 1,
  "failed_count": 0,
  "errors": [],
  "timestamp": "2024-01-15T11:00:00Z"
}
```

---

## Alert Types

### anomaly_detected

Sent when anomalies are actively detected for a service.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `alert_type` | string | Yes | Always `"anomaly_detected"` |
| `service` | string | Yes | Service name (e.g., `"booking"`) |
| `timestamp` | string | Yes | ISO 8601 timestamp |
| `overall_severity` | string | Yes | Highest severity: `low`, `medium`, `high`, `critical` |
| `anomaly_count` | integer | Yes | Number of anomalies (≥1) |
| `current_metrics` | object | Yes | Metrics at detection time |
| `anomalies` | array | Yes | Array of Anomaly objects |
| `time_period` | string | No | Time period model used |
| `model_type` | string | No | Model type identifier |
| `fingerprinting_metadata` | object | No | Incident tracking data |
| `explanation` | object | No | Human-readable explanations |
| `recommended_actions` | array | No | Suggested remediation steps |

### incident_resolved

Sent when an anomaly clears and the incident is closed.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `alert_type` | string | Yes | Always `"incident_resolved"` |
| `service` | string | Yes | Service name |
| `timestamp` | string | Yes | ISO 8601 timestamp of resolution |
| `incident_id` | string | Yes | Unique incident identifier |
| `fingerprint_id` | string | Yes | Pattern fingerprint |
| `anomaly_name` | string | Yes | Name of resolved anomaly |
| `resolution_details` | object | Yes | Resolution details |
| `model_type` | string | Yes | Always `"incident_resolution"` |

---

## Data Models

### CurrentMetrics

Metric values at the time of detection.

```json
{
  "request_rate": 150.5,
  "application_latency": 85.2,
  "client_latency": 45.0,
  "database_latency": 12.3,
  "error_rate": 0.05
}
```

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `request_rate` | float | ≥0 | Requests per second |
| `application_latency` | float | ≥0 | Server processing time (ms) |
| `client_latency` | float | ≥0 | External call latency (ms) |
| `database_latency` | float | ≥0 | Database query time (ms) |
| `error_rate` | float | 0-1 | Error rate (0.05 = 5%) |

### Anomaly

Single detected anomaly within an alert.

```json
{
  "type": "multivariate_enhanced_isolation_forest",
  "severity": "high",
  "confidence_score": 0.85,
  "description": "Application latency significantly elevated",
  "detection_method": "enhanced_isolation_forest",
  "threshold_value": 100.0,
  "actual_value": 250.0,
  "metadata": { /* AnomalyMetadata */ }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | Yes | Anomaly type identifier |
| `severity` | string | Yes | `low`, `medium`, `high`, `critical` |
| `confidence_score` | float | Yes | 0.0-1.0 confidence level |
| `description` | string | Yes | Human-readable description |
| `detection_method` | string | Yes | Detection method used |
| `threshold_value` | float | No | Expected threshold |
| `actual_value` | float | No | Observed value |
| `metadata` | object | No | Extended metadata |

### CascadeInfo

Information about dependency cascade when an anomaly is caused by an upstream service failure.

```json
{
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
```

| Field | Type | Description |
|-------|------|-------------|
| `is_cascade` | bool | Whether this anomaly is part of a cascade |
| `root_cause_service` | string | Service identified as the root cause |
| `affected_chain` | array | Chain of affected services from root to current |
| `cascade_type` | string | Type: `upstream_cascade`, `chain_degraded`, `dependencies_healthy`, `none` |
| `confidence` | float | Confidence in cascade analysis (0.0-1.0) |
| `propagation_path` | array | Detailed path showing each service's anomaly status |

### AnomalyMetadata

Extended metadata for anomaly tracking and explainability.

```json
{
  "fingerprint_id": "anomaly_8d4a011b83ca",
  "incident_id": "incident_1dcbafc91480",
  "anomaly_name": "multivariate_enhanced_isolation_forest",
  "fingerprint_action": "UPDATE",
  "incident_action": "CONTINUE",
  "occurrence_count": 5,
  "first_seen": "2024-01-15T09:00:00",
  "last_updated": "2024-01-15T10:30:00",
  "incident_duration_minutes": 90,
  "detected_by_model": "evening_hours",
  "time_period": "evening_hours",
  "time_confidence": 0.8,
  "feature_contributions": [
    {"feature": "application_latency", "contribution": 0.65, "direction": "elevated"}
  ],
  "business_impact": "User-facing latency may impact checkout conversion",
  "cascade_analysis": { /* CascadeInfo - present when anomaly is part of a cascade */ }
}
```

#### Fingerprinting Fields

| Field | Type | Description |
|-------|------|-------------|
| `fingerprint_id` | string | Pattern identifier (deterministic hash) |
| `incident_id` | string | Occurrence identifier (UUID-based) |
| `fingerprint_action` | string | `CREATE`, `UPDATE`, `RESOLVE` |
| `incident_action` | string | `CREATE`, `CONTINUE`, `CLOSE` |
| `occurrence_count` | integer | Times detected in this incident |
| `first_seen` | string | First detection timestamp |
| `last_updated` | string | Most recent detection |
| `incident_duration_minutes` | integer | Duration since first_seen |

#### Explainability Fields

| Field | Type | Description |
|-------|------|-------------|
| `feature_contributions` | array | Which metrics drove the detection |
| `comparison_data` | object | Historical statistics comparison |
| `business_impact` | string | Business impact assessment |
| `percentile_position` | float | Where this value ranks historically |

### FingerprintingMetadata

Top-level fingerprinting summary for the alert.

```json
{
  "service_name": "booking",
  "model_name": "evening_hours",
  "timestamp": "2024-01-15T10:30:00",
  "action_summary": {
    "incident_creates": 0,
    "incident_continues": 1,
    "incident_closes": 0
  },
  "overall_action": "UPDATE",
  "resolved_incidents": [],
  "total_open_incidents": 1,
  "detection_context": {
    "model_used": "evening_hours",
    "inference_timestamp": "2024-01-15T10:30:00"
  }
}
```

### ResolutionDetails

Details about incident resolution.

```json
{
  "final_severity": "high",
  "total_occurrences": 12,
  "incident_duration_minutes": 120,
  "first_seen": "2024-01-15T09:00:00",
  "last_detected_by_model": "evening_hours"
}
```

---

## Lifecycle Flow

### Anomaly Detection → Resolution Flow

```
┌─────────────────┐
│  No Anomaly     │
│  (Normal)       │
└────────┬────────┘
         │ Anomaly detected
         ▼
┌─────────────────┐
│  CREATE         │  ← New incident_id generated
│  incident_action│    fingerprint_id assigned
│  = CREATE       │
└────────┬────────┘
         │ Same anomaly detected again
         ▼
┌─────────────────┐
│  CONTINUE       │  ← Same incident_id
│  incident_action│    occurrence_count++
│  = CONTINUE     │
└────────┬────────┘
         │ Anomaly clears
         ▼
┌─────────────────┐
│  RESOLVE        │  ← incident_resolved sent
│  incident_action│    incident closed
│  = CLOSE        │
└────────┬────────┘
         │ Same pattern reappears later
         ▼
┌─────────────────┐
│  CREATE         │  ← NEW incident_id
│  (new incident) │    SAME fingerprint_id
└─────────────────┘
```

### JSON Payload at Each State

#### State: CREATE (New Incident)

```json
{
  "alert_type": "anomaly_detected",
  "service": "booking",
  "anomalies": [{
    "type": "high_latency",
    "severity": "high",
    "metadata": {
      "fingerprint_id": "anomaly_abc123",
      "incident_id": "incident_xyz789",
      "incident_action": "CREATE",
      "occurrence_count": 1,
      "first_seen": "2024-01-15T10:00:00"
    }
  }]
}
```

#### State: CONTINUE (Ongoing Incident)

```json
{
  "alert_type": "anomaly_detected",
  "service": "booking",
  "anomalies": [{
    "type": "high_latency",
    "severity": "high",
    "metadata": {
      "fingerprint_id": "anomaly_abc123",
      "incident_id": "incident_xyz789",
      "incident_action": "CONTINUE",
      "occurrence_count": 5,
      "first_seen": "2024-01-15T10:00:00",
      "incident_duration_minutes": 30
    }
  }]
}
```

#### State: RESOLVE (Incident Cleared)

```json
{
  "alert_type": "incident_resolved",
  "service": "booking",
  "incident_id": "incident_xyz789",
  "fingerprint_id": "anomaly_abc123",
  "anomaly_name": "high_latency",
  "resolution_details": {
    "final_severity": "high",
    "total_occurrences": 8,
    "incident_duration_minutes": 45,
    "first_seen": "2024-01-15T10:00:00"
  }
}
```

---

## Examples

### Complete anomaly_detected Payload

```json
{
  "alert_type": "anomaly_detected",
  "service": "booking",
  "timestamp": "2024-01-15T10:30:00",
  "overall_severity": "high",
  "anomaly_count": 1,
  "time_period": "evening_hours",
  "model_type": "time_aware_explainable",
  "current_metrics": {
    "request_rate": 150.5,
    "application_latency": 250.0,
    "client_latency": 45.0,
    "database_latency": 12.3,
    "error_rate": 0.02
  },
  "anomalies": [
    {
      "type": "multivariate_enhanced_isolation_forest",
      "severity": "high",
      "confidence_score": 0.85,
      "description": "Application latency significantly elevated at 250ms (expected <100ms)",
      "detection_method": "enhanced_isolation_forest",
      "threshold_value": 100.0,
      "actual_value": 250.0,
      "metadata": {
        "fingerprint_id": "anomaly_8d4a011b83ca",
        "incident_id": "incident_1dcbafc91480",
        "anomaly_name": "multivariate_enhanced_isolation_forest",
        "fingerprint_action": "UPDATE",
        "incident_action": "CONTINUE",
        "occurrence_count": 5,
        "first_seen": "2024-01-15T09:45:00",
        "last_updated": "2024-01-15T10:30:00",
        "incident_duration_minutes": 45,
        "detected_by_model": "evening_hours",
        "time_period": "evening_hours",
        "feature_contributions": [
          {"feature": "application_latency", "contribution": 0.65, "direction": "elevated", "percentile": 99.2},
          {"feature": "request_rate", "contribution": 0.20, "direction": "elevated", "percentile": 85.0}
        ],
        "business_impact": "High latency affecting checkout flow - potential revenue impact"
      }
    }
  ],
  "fingerprinting_metadata": {
    "service_name": "booking",
    "model_name": "evening_hours",
    "timestamp": "2024-01-15T10:30:00",
    "action_summary": {
      "incident_creates": 0,
      "incident_continues": 1,
      "incident_closes": 0
    },
    "overall_action": "UPDATE",
    "total_open_incidents": 1
  },
  "recommended_actions": [
    "Check database connection pool - latency spike correlates with DB metrics",
    "Review recent deployments to booking service",
    "Scale horizontally if traffic has increased"
  ]
}
```

### Complete incident_resolved Payload

```json
{
  "alert_type": "incident_resolved",
  "service": "booking",
  "timestamp": "2024-01-15T11:00:00",
  "incident_id": "incident_1dcbafc91480",
  "fingerprint_id": "anomaly_8d4a011b83ca",
  "anomaly_name": "multivariate_enhanced_isolation_forest",
  "model_type": "incident_resolution",
  "resolution_details": {
    "final_severity": "high",
    "total_occurrences": 8,
    "incident_duration_minutes": 75,
    "first_seen": "2024-01-15T09:45:00",
    "last_detected_by_model": "evening_hours"
  }
}
```

---

## Error Handling

### Error Payload

```json
{
  "alert_type": "error",
  "service": "booking",
  "timestamp": "2024-01-15T10:30:00",
  "error_message": "Failed to collect metrics: connection timeout",
  "error_code": "METRICS_TIMEOUT"
}
```

### Common Error Codes

| Code | Description |
|------|-------------|
| `METRICS_TIMEOUT` | VictoriaMetrics query timed out |
| `MODEL_NOT_FOUND` | No trained model for service/period |
| `CIRCUIT_BREAKER_OPEN` | Too many failures, circuit breaker tripped |
| `VALIDATION_ERROR` | Invalid metric values |
| `INFERENCE_ERROR` | ML model inference failed |

---

## Schema Versioning

The API uses semantic versioning for the schema:

- **Major version** (1.x.x): Breaking changes to required fields
- **Minor version** (x.1.x): New optional fields added
- **Patch version** (x.x.1): Documentation/description updates

Clients should send `schema_version` in batch requests. The server should accept any compatible version and include its version in responses.

```json
{
  "alerts": [...],
  "schema_version": "1.0.0"
}
```

---

## Web UI Considerations

When building a web UI for this data:

### Dashboard Views

1. **Active Incidents Table**
   - Group by service
   - Show: severity, duration, occurrence_count
   - Filter by severity, time_period
   - Sort by first_seen or last_updated

2. **Incident Timeline**
   - Plot incidents on timeline
   - Show CREATE → CONTINUE → RESOLVE lifecycle
   - Color-code by severity

3. **Service Health Grid**
   - Card per service
   - Current status (healthy/anomaly)
   - Active incident count

4. **Incident Detail View**
   - Full anomaly metadata
   - Feature contributions chart
   - Metrics at detection time
   - Recommended actions

### Real-time Updates

- Use WebSockets or SSE for live updates
- Push both `anomaly_detected` and `incident_resolved` events
- Update dashboard counters in real-time

### Historical Analysis

- Store all payloads for trend analysis
- Track MTTR (mean time to resolution)
- Pattern frequency analysis using fingerprint_id
- Severity distribution over time

---

## Implementation Notes

### Pydantic Models

See `models.py` for validated Pydantic models:

```python
from models import (
    AnomalyDetectedPayload,
    IncidentResolvedPayload,
    create_anomaly_payload,
    create_resolution_payload,
)

# Create validated payload
payload = create_anomaly_payload(
    service="booking",
    anomalies=[...],
    metrics={"request_rate": 150.0},
)

# Serialize to JSON
json_data = payload.model_dump_json()
```

### Validation

All payloads are validated at creation:
- Severity values are constrained to valid enum values
- Confidence scores are clamped to 0.0-1.0
- Required fields are enforced
- anomaly_count is auto-corrected to match array length
