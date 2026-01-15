# VictoriaMetrics Queries Reference

This document provides a complete reference of all PromQL queries used by Yaga2 to pull metrics from VictoriaMetrics.

---

## Quick Reference

| Metric | Purpose | Underlying Metric |
|--------|---------|-------------------|
| `request_rate` | Traffic volume (req/s) | `http_requests:count:rate_5m` |
| `application_latency` | Server processing time (ms) | `duration_milliseconds` (server spans) |
| `dependency_latency` | External dependency latency (ms) | `duration_milliseconds` (non-DB client spans) |
| `database_latency` | Database query time (ms) | `duration_milliseconds` (DB client spans) |
| `error_rate` | Error percentage (0-1) | `duration_milliseconds` (5xx status codes) |

---

## Core Metrics Queries

**Location:** `smartbox_anomaly/metrics/client.py:246-273`

### 1. Request Rate

```promql
http_requests:count:rate_5m
```

- **Metric Name:** `request_rate`
- **Description:** Pre-aggregated HTTP requests per second (5-minute window)
- **Unit:** requests/second
- **Usage:** Detects traffic surges and cliffs

### 2. Application Latency

```promql
sum(rate(duration_milliseconds_sum{
    span_kind="SPAN_KIND_SERVER",
    deployment_environment_name=~"production"
}[5m])) by (service_name)
/
sum(rate(duration_milliseconds_count{
    span_kind="SPAN_KIND_SERVER",
    deployment_environment_name=~"production"
}[5m])) by (service_name)
```

- **Metric Name:** `application_latency`
- **Description:** Average server-side processing latency
- **Unit:** milliseconds
- **Key Labels:**
  - `span_kind="SPAN_KIND_SERVER"` - OpenTelemetry server spans only
  - `deployment_environment_name=~"production"` - Production environment
- **Calculation:** sum(latency) / count(spans) = average latency

### 3. Client Latency (External Dependencies)

```promql
sum(rate(duration_milliseconds_sum{
    span_kind="SPAN_KIND_CLIENT",
    deployment_environment_name=~"production",
    db_system="",
    db_system_name=""
}[5m])) by (service_name)
/
sum(rate(duration_milliseconds_count{
    span_kind="SPAN_KIND_CLIENT",
    deployment_environment_name=~"production",
    db_system="",
    db_system_name=""
}[5m])) by (service_name)
```

- **Metric Name:** `dependency_latency`
- **Description:** Average latency of external service calls (non-database)
- **Unit:** milliseconds
- **Key Labels:**
  - `span_kind="SPAN_KIND_CLIENT"` - Outbound calls only
  - `db_system=""` and `db_system_name=""` - Excludes database calls
- **Usage:** Detects downstream service slowness

### 4. Database Latency

```promql
sum(rate(duration_milliseconds_sum{
    span_kind="SPAN_KIND_CLIENT",
    deployment_environment_name=~"production",
    db_system_name!=""
}[5m])) by (service_name)
/
sum(rate(duration_milliseconds_count{
    span_kind="SPAN_KIND_CLIENT",
    deployment_environment_name=~"production",
    db_system_name!=""
}[5m])) by (service_name)
```

- **Metric Name:** `database_latency`
- **Description:** Average database query latency
- **Unit:** milliseconds
- **Key Labels:**
  - `db_system_name!=""` - Only includes database calls
- **Usage:** Detects database performance bottlenecks

### 5. Error Rate

```promql
sum(rate(duration_milliseconds_count{
    span_kind="SPAN_KIND_SERVER",
    deployment_environment_name=~"production",
    http_response_status_code=~"5.*|"
}[5m])) by (service_name)
/
sum(rate(duration_milliseconds_count{
    span_kind="SPAN_KIND_SERVER",
    deployment_environment_name=~"production"
}[5m])) by (service_name)
```

- **Metric Name:** `error_rate`
- **Description:** Fraction of requests resulting in 5xx errors
- **Unit:** decimal (0.0-1.0), e.g., 0.05 = 5%
- **Key Labels:**
  - `http_response_status_code=~"5.*|"` - 5xx HTTP status codes
- **Calculation:** (5xx errors) / (total requests)

---

## Enrichment Queries

### Exception Breakdown (Error Context)

**Location:** `smartbox_anomaly/enrichment/exceptions.py:140-152`

**When Used:** On HIGH or CRITICAL error-related anomalies when error_rate > 1%

```promql
sum(rate(events_total{
    service_name="{service}",
    deployment_environment_name=~"production"
}[{window}])) by (exception_type)
```

- **Metric:** `events_total` (OpenTelemetry exceptions)
- **Parameters:**
  - `{service}` - Service name (e.g., "search")
  - `{window}` - Time window (default: 5m)
- **Output:** Exception rate by type
- **Purpose:** Identifies which exception types are causing errors

### Service Graph (Downstream Dependencies)

**Location:** `smartbox_anomaly/enrichment/service_graph.py:179-191`

**When Used:** On latency anomalies when dependency_latency SLO is breached

#### Request Rate to Downstream Services

```promql
sum(rate(traces_service_graph_request_total{
    client="{service}"
}[{window}])) by (client, server, server_http_route)
```

- **Metric:** `traces_service_graph_request_total`
- **Parameters:**
  - `{service}` - Client service name
  - `{window}` - Time window (default: 5m)
- **Output:** Request rates per downstream server/route

#### Latency to Downstream Services

```promql
sum(rate(traces_service_graph_request_server_seconds_sum{
    client="{service}"
}[{window}])) by (client, server, server_http_route)
/
sum(rate(traces_service_graph_request_server_seconds_count{
    client="{service}"
}[{window}])) by (client, server, server_http_route)
```

- **Metrics:**
  - `traces_service_graph_request_server_seconds_sum`
  - `traces_service_graph_request_server_seconds_count`
- **Output:** Average latency per downstream server/route (seconds, converted to ms)
- **Purpose:** Identifies slowest downstream routes

---

## Query Execution Contexts

### Inference (Real-time Detection)

| Setting | Value |
|---------|-------|
| Frequency | Every 10 minutes (configurable) |
| Query Type | Instant queries |
| Window | 5 minutes |
| Timeout | 10 seconds |
| Retries | Up to 3 attempts |

### Training (Model Building)

| Setting | Value |
|---------|-------|
| Frequency | Daily at 2 AM (configurable) |
| Query Type | Range queries |
| Window | 30 days (configurable) |
| Timeout | 120 seconds |
| Step | 5-minute resolution |

### Enrichment (Anomaly Context)

| Setting | Value |
|---------|-------|
| Trigger | HIGH/CRITICAL anomalies only |
| Query Type | Instant queries |
| Window | 5 minutes |
| Time Alignment | Matches anomaly timestamp |

---

## Service Name Injection

Queries are modified at runtime to filter by service:

```python
# Original query
"http_requests:count:rate_5m"

# After injection for service "booking"
'http_requests:count:rate_5m{service_name="booking"}'
```

For queries with `by (service_name)`, the filter is added inside the selector.

---

## OpenTelemetry Label Conventions

| Component | Label |
|-----------|-------|
| Service Name | `service_name` |
| Environment | `deployment_environment_name` |
| Span Type | `span_kind` |
| HTTP Status | `http_response_status_code` |
| Database System | `db_system`, `db_system_name` |
| Exception Type | `exception_type` |
| Service Graph Client | `client` |
| Service Graph Server | `server` |
| HTTP Route | `server_http_route` |

---

## Metric Validation Boundaries

**Location:** `smartbox_anomaly/core/constants.py:215-218`

| Metric | Maximum Value | Notes |
|--------|---------------|-------|
| `request_rate` | 1,000,000 req/s | Values above are capped |
| Latencies | 300,000 ms (5 min) | Values above are capped |
| `error_rate` | 1.0 (100%) | Values above are capped |

Invalid values (NaN, Inf, negative) are replaced with 0.0 and generate validation warnings.

---

## VictoriaMetrics Configuration

**Location:** `config.json` under `victoria_metrics`

```json
{
  "victoria_metrics": {
    "endpoint": "https://otel-metrics.production.smartbox.com",
    "timeout_seconds": 10,
    "max_retries": 3,
    "circuit_breaker_threshold": 5,
    "circuit_breaker_timeout_seconds": 300
  }
}
```

| Setting | Default | Description |
|---------|---------|-------------|
| `endpoint` | See config | VictoriaMetrics server URL |
| `timeout_seconds` | 10 | Request timeout |
| `max_retries` | 3 | Retry attempts |
| `circuit_breaker_threshold` | 5 | Failures before circuit opens |
| `circuit_breaker_timeout_seconds` | 300 | Circuit breaker reset time |

---

## Summary

Yaga2 queries **9 distinct PromQL queries** from VictoriaMetrics:

| # | Query | Metric Type | Context |
|---|-------|-------------|---------|
| 1 | Request Rate | Core | Inference/Training |
| 2 | Application Latency | Core | Inference/Training |
| 3 | Client Latency | Core | Inference/Training |
| 4 | Database Latency | Core | Inference/Training |
| 5 | Error Rate | Core | Inference/Training |
| 6 | Exception Rate | Enrichment | Error anomalies |
| 7 | Exception Count | Enrichment | Error anomalies |
| 8 | Service Graph Request Rate | Enrichment | Latency anomalies |
| 9 | Service Graph Latency | Enrichment | Latency anomalies |
