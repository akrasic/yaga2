# Anomaly Observability API Documentation

This document describes the API endpoints for the Anomaly Observability service, which receives and displays ML anomaly detection results from the yaga2 pipeline.

## Overview

The API receives two types of payloads from `yaga2/inference.py`:

1. **Anomaly Alerts** - Active anomalies detected by the ML pipeline
2. **Incident Resolutions** - Notifications when anomalies clear and return to normal

## Architecture

```
yaga2/inference.py
        │
        ├── POST /api/anomalies/batch ──────► alerts table
        │                                          │
        │                                          ▼
        │                                    incidents table
        │                                    (auto-created from
        │                                     anomaly metadata)
        │
        └── POST /api/incidents/resolve ───► incidents table
                                            (status → RESOLVED)
```

---

## Endpoints

### POST /api/anomalies/batch

Ingest a batch of anomaly alerts from the ML pipeline.

**Request Body:** Array of `IngestAlert` objects

```json
[
  {
    "alert_type": "anomaly_detected",
    "service": "booking",
    "timestamp": "2024-01-15T10:30:00",
    "overall_severity": "high",
    "anomaly_count": 2,
    "current_metrics": {
      "request_rate": 150.5,
      "application_latency": 1250.0,
      "client_latency": 85.0,
      "database_latency": 45.0,
      "error_rate": 0.05
    },
    "anomalies": [
      {
        "type": "multivariate_isolation",
        "severity": "high",
        "confidence_score": 0.85,
        "description": "Unusual combination of metrics detected",
        "detection_method": "enhanced_isolation_forest",
        "threshold_value": null,
        "actual_value": null,
        "metadata": {
          "fingerprint_id": "anomaly_abc123def456",
          "incident_id": "incident_xyz789ghi012",
          "anomaly_name": "multivariate_isolation_forest",
          "incident_action": "CREATE",
          "occurrence_count": 1,
          "first_seen": "2024-01-15T10:30:00",
          "time_period": "business_hours",
          "time_confidence": 0.95,
          "business_impact": "Multiple metrics showing unusual patterns"
        }
      }
    ],
    "time_period": "business_hours",
    "model_type": "time_aware_explainable",
    "fingerprinting_metadata": {
      "service_name": "booking",
      "model_name": "booking_business_hours",
      "timestamp": "2024-01-15T10:30:00",
      "action_summary": {
        "incident_creates": 1,
        "incident_continues": 0,
        "incident_closes": 0
      },
      "overall_action": "CREATE",
      "resolved_incidents": [],
      "total_open_incidents": 1
    },
    "historical_context": {...},
    "metric_analysis": {...},
    "explanation": {...},
    "recommended_actions": ["Check system resources", "Review recent deployments"]
  }
]
```

**Response:**

```json
{
  "success": true,
  "processed_count": 1,
  "failed_count": 0,
  "errors": [],
  "ids": [42]
}
```

---

### POST /api/incidents/resolve

Process incident resolutions when anomalies clear.

**Request Body:** Array of `IncidentResolution` objects

```json
[
  {
    "alert_type": "incident_resolved",
    "service": "booking",
    "timestamp": "2024-01-15T11:15:00",
    "incident_id": "incident_xyz789ghi012",
    "fingerprint_id": "anomaly_abc123def456",
    "anomaly_name": "multivariate_isolation_forest",
    "resolution_details": {
      "final_severity": "high",
      "total_occurrences": 5,
      "incident_duration_minutes": 45,
      "first_seen": "2024-01-15T10:30:00",
      "last_detected_by_model": "booking_business_hours"
    },
    "model_type": "incident_resolution"
  }
]
```

**Response:**

```json
{
  "success": true,
  "processed_count": 1,
  "failed_count": 0,
  "errors": [],
  "ids": [15]
}
```

---

### GET /api/anomalies

List anomaly alerts with filtering and pagination.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| page | int | 1 | Page number |
| per_page | int | 20 | Items per page (max 200) |
| service | string | null | Filter by service name |
| severity | string | null | Filter by severity (critical/high/medium/low) |
| time_period | string | null | Filter by time period |
| has_fingerprinting | bool | null | Filter by fingerprinting presence |
| sort | string | -timestamp | Sort order (-timestamp, timestamp, severity, -severity) |

**Response:**

```json
{
  "page": 1,
  "per_page": 20,
  "count": 15,
  "total": 150,
  "data": [
    {
      "alert_type": "anomaly_detected",
      "service": "booking",
      "timestamp": "2024-01-15T10:30:00",
      ...
    }
  ]
}
```

---

### GET /api/anomalies/{alert_id}

Get a single anomaly alert by database ID.

**Response:** Full alert payload (JSON)

---

### GET /api/incidents

List incidents with filtering and pagination.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| page | int | 1 | Page number |
| per_page | int | 20 | Items per page (max 200) |
| service | string | null | Filter by service name |
| status | string | null | Filter by status (OPEN/RESOLVED) |
| severity | string | null | Filter by severity |
| sort | string | -first_seen | Sort order |

**Response:**

```json
{
  "page": 1,
  "per_page": 20,
  "count": 10,
  "total": 50,
  "open_count": 5,
  "resolved_count": 45,
  "data": [
    {
      "id": 15,
      "incident_id": "incident_xyz789ghi012",
      "fingerprint_id": "anomaly_abc123def456",
      "service": "booking",
      "anomaly_name": "multivariate_isolation_forest",
      "status": "RESOLVED",
      "severity": "high",
      "first_seen": "2024-01-15T10:30:00",
      "last_updated": "2024-01-15T11:15:00",
      "resolved_at": "2024-01-15T11:15:00",
      "occurrence_count": 5,
      "duration_minutes": 45,
      "time_period": "business_hours"
    }
  ]
}
```

---

### GET /api/incidents/{incident_id}

Get a single incident by incident_id string.

**Response:**

```json
{
  "id": 15,
  "incident_id": "incident_xyz789ghi012",
  "fingerprint_id": "anomaly_abc123def456",
  "service": "booking",
  "anomaly_name": "multivariate_isolation_forest",
  "status": "RESOLVED",
  "severity": "high",
  "first_seen": "2024-01-15T10:30:00",
  "last_updated": "2024-01-15T11:15:00",
  "resolved_at": "2024-01-15T11:15:00",
  "occurrence_count": 5,
  "duration_minutes": 45,
  "time_period": "business_hours",
  "resolution_payload": {...}
}
```

---

### GET /api/stats

Get dashboard statistics.

**Response:**

```json
{
  "total_alerts": 1500,
  "total_incidents": 200,
  "open_incidents": 5,
  "resolved_incidents": 195,
  "severity_breakdown": {
    "critical": 10,
    "high": 50,
    "medium": 100,
    "low": 40
  },
  "top_services": {
    "booking": 50,
    "search": 30,
    "mobile-api": 25
  }
}
```

---

## Data Models

### IngestAlert

Primary payload for anomaly detection alerts.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| alert_type | string | Yes | Always "anomaly_detected" |
| service | string | Yes | Service name |
| timestamp | datetime | Yes | Detection timestamp |
| overall_severity | string | Yes | Highest severity (critical/high/medium/low) |
| anomaly_count | int | Yes | Number of anomalies |
| current_metrics | dict | No | Metrics at detection time |
| anomalies | array | Yes | List of detected anomalies |
| time_period | string | No | Time period (business_hours, etc.) |
| model_type | string | No | Model type used |
| fingerprinting_metadata | dict | No | Fingerprinting/incident tracking data |
| historical_context | dict | No | Historical analysis |
| metric_analysis | dict | No | Detailed metric analysis |
| explanation | dict | No | Human-readable explanations |
| recommended_actions | array | No | Suggested actions |

### Anomaly

Single detected anomaly within an alert.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| type | string | Yes | Anomaly type identifier |
| severity | string | Yes | Severity level |
| confidence_score | float | No | Detection confidence (0-1) |
| description | string | Yes | Human-readable description |
| detection_method | string | No | Detection method used |
| threshold_value | float | No | Threshold that was exceeded |
| actual_value | float | No | Actual observed value |
| metadata | AnomalyMetadata | No | Extended metadata |

### AnomalyMetadata

Extended metadata for fingerprinting and explainability.

| Field | Type | Description |
|-------|------|-------------|
| fingerprint_id | string | Deterministic pattern identifier |
| incident_id | string | Unique incident occurrence ID |
| anomaly_name | string | Generated anomaly name |
| incident_action | string | Action taken (CREATE/CONTINUE/RESOLVE) |
| occurrence_count | int | Times this incident detected |
| first_seen | string | ISO timestamp of first detection |
| time_period | string | Time period (business_hours, etc.) |
| time_confidence | float | Period confidence score |
| business_impact | string | Business impact assessment |

### IncidentResolution

Payload for incident resolution notifications.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| alert_type | string | Yes | Always "incident_resolved" |
| service | string | Yes | Service name |
| timestamp | datetime | Yes | Resolution timestamp |
| incident_id | string | Yes | Unique incident identifier |
| fingerprint_id | string | Yes | Pattern fingerprint identifier |
| anomaly_name | string | Yes | Name of the resolved anomaly |
| resolution_details | dict | No | Resolution details |

### ResolutionDetails

Details about an incident resolution.

| Field | Type | Description |
|-------|------|-------------|
| final_severity | string | Severity at resolution |
| total_occurrences | int | Total times detected |
| incident_duration_minutes | int | Total duration in minutes |
| first_seen | string | ISO timestamp of first detection |
| last_detected_by_model | string | Last model to detect |

---

## Database Schema

### alerts table

Stores all anomaly alerts received from the ML pipeline.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| alert_type | VARCHAR(64) | Alert type |
| service | VARCHAR(256) | Service name (indexed) |
| timestamp | DATETIME | Detection timestamp (indexed) |
| overall_severity | VARCHAR(32) | Severity level (indexed) |
| anomaly_count | INTEGER | Number of anomalies |
| model_type | VARCHAR(64) | Model type used |
| time_period | VARCHAR(32) | Time period |
| error_rate_current | FLOAT | Error rate (denormalized) |
| request_rate_current | FLOAT | Request rate (denormalized) |
| application_latency_current | FLOAT | App latency (denormalized) |
| client_latency_current | FLOAT | Client latency (denormalized) |
| database_latency_current | FLOAT | DB latency (denormalized) |
| has_fingerprinting | BOOLEAN | Has fingerprinting data |
| incident_creates | INTEGER | Incidents created |
| incident_continues | INTEGER | Incidents continued |
| incident_closes | INTEGER | Incidents closed |
| payload | JSON | Full alert payload |
| created_at | DATETIME | Record creation time |

### incidents table

Tracks incident lifecycle from creation to resolution.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| incident_id | VARCHAR(128) | Unique incident ID (unique, indexed) |
| fingerprint_id | VARCHAR(128) | Pattern fingerprint (indexed) |
| service | VARCHAR(256) | Service name (indexed) |
| anomaly_name | VARCHAR(256) | Anomaly type name |
| status | VARCHAR(32) | OPEN or RESOLVED (indexed) |
| severity | VARCHAR(32) | Current severity |
| first_seen | DATETIME | First detection (indexed) |
| last_updated | DATETIME | Last update time |
| resolved_at | DATETIME | Resolution time (nullable) |
| occurrence_count | INTEGER | Total occurrences |
| duration_minutes | INTEGER | Total duration (nullable) |
| last_detected_by_model | VARCHAR(128) | Last detecting model |
| time_period | VARCHAR(32) | Time period |
| resolution_payload | JSON | Full resolution payload |
| created_at | DATETIME | Record creation time |

---

## Usage with yaga2

### Sending Anomalies

```python
# In yaga2/inference.py
import requests

def _send_active_anomalies(anomalies: List[Dict], verbose: bool) -> None:
    r = requests.post(
        "http://localhost:8000/api/anomalies/batch",
        json=anomalies,
        timeout=5
    )
    r.raise_for_status()
```

### Sending Resolutions

```python
# In yaga2/inference.py
def _send_resolved_incidents(resolutions: List[Dict], verbose: bool) -> None:
    r = requests.post(
        "http://localhost:8000/api/incidents/resolve",
        json=resolutions,
        timeout=5
    )
    r.raise_for_status()
```

---

## Web UI Routes

| Route | Description |
|-------|-------------|
| / | Main alerts dashboard |
| /anomalies/{id} | Alert detail page |
| /incidents | Incidents dashboard |
| /live-metrics | Real-time metrics (Prometheus) |

---

## Running the Service

```bash
cd ~/test/yaga-web
source .venv/bin/activate
python app.py
# or
uvicorn app:app --reload --port 8000
```

Access at: http://localhost:8000
API docs: http://localhost:8000/docs (FastAPI auto-generated)
