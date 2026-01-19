# Envoy Metrics Enrichment Design

**Date:** 2026-01-15
**Status:** Discovery Complete - Ready for Implementation
**Author:** ML Platform Team

---

## Overview

This document describes the integration of Envoy edge/ingress metrics with the existing OpenTelemetry-based anomaly detection system. The goal is to enrich anomaly alerts with edge-level context showing how the service appears from the ingress perspective.

---

## Data Sources Discovery

### 1. OpenTelemetry Metrics (Primary)

**Endpoint:** `https://otel-metrics.production.smartbox.com`

**Metrics Available:**
- `duration_milliseconds_count` - Request counts with HTTP status codes
- `duration_milliseconds_sum` - Total latency
- `duration_milliseconds_bucket` - Latency histogram

**Key Labels:**
- `service_name` - Application service name
- `span_kind` - `SPAN_KIND_SERVER` for incoming requests
- `http_response_status_code` - Individual HTTP status codes (200, 302, 401, 500, etc.)

**Sample Data (booking service):**
| HTTP Status | Rate (req/s) |
|-------------|--------------|
| 200 | 48.53 |
| 302 | 2.91 |
| 401 | 0.33 |
| 500 | 0.01 |

### 2. Envoy/Mimir Metrics (Enrichment)

**Endpoint:** `https://mimir.sbxtest.net/prometheus`

**Metrics Available:**
| Metric | Series Count | Description |
|--------|--------------|-------------|
| `envoy_cluster_upstream_rq_total` | 76 | Total upstream requests |
| `envoy_cluster_upstream_rq_xx` | 226 | Requests by response class (2xx/3xx/4xx/5xx) |
| `envoy_cluster_upstream_rq_time_bucket` | 1240 | Latency histogram buckets |
| `envoy_cluster_upstream_cx_active` | 76 | Active connections |
| `envoy_cluster_upstream_cx_total` | 76 | Total connections |

**Key Labels:**
- `envoy_cluster_name` - Envoy cluster identifier
- `envoy_response_code_class` - Response class ('2', '3', '4', '5')
- `le` - Histogram bucket boundaries (0.5, 1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500ms)

**Histogram Bucket Boundaries:**
```
0.5ms, 1ms, 5ms, 10ms, 25ms, 50ms, 100ms, 250ms, 500ms, 1000ms, 2500ms, +Inf
```

---

## Service Name Mapping

Envoy cluster names don't always match OTel service names. A mapping is required:

| OTel Service | Envoy Cluster | Match Confidence |
|--------------|---------------|------------------|
| `booking` | `booking` | Exact |
| `search` | `search_k8s` | High |
| `mobile-api` | `mobile-api` | Exact |
| `shire-api` | `shireapi_cluster` | High |
| `fa5` | `fa5-public` | High |
| `titan` | `titan` | Exact |
| `friday` | `friday` | Exact (0 traffic) |
| `vms` | - | No match found |
| `gambit` | - | No match found |
| `r2d2` | - | No match found |
| `catalog` | - | No match found |

**Additional Envoy Clusters (no OTel equivalent):**
- `booking-api`, `cmhub-prod`, `eai`, `exchange`
- `m1_smartbox`, `m1_varnish`, `m2_admin`, `m2_checkout`
- `shire_production_smartbox`, `shireapp_cluster`, `gandalf`, `svag`

---

## Current Envoy Metrics Snapshot

### Request Rate by Cluster

| Cluster | 2xx/s | 3xx/s | 4xx/s | 5xx/s | Total |
|---------|-------|-------|-------|-------|-------|
| search_k8s | 215.07 | 0 | 0.10 | 0.26 | 215.43 |
| eai | 125.37 | 0 | 0.04 | 1.45 | 126.86 |
| shireapp_cluster | 98.36 | 9.72 | 0.03 | 0 | 108.11 |
| booking | 68.88 | 5.80 | 0.51 | 0.003 | 75.19 |
| shireapi_cluster | 69.71 | 0 | 2.79 | 0.01 | 72.51 |
| m2_checkout | 54.21 | 0.37 | 0.41 | 0 | 54.99 |
| exchange | 27.67 | 0.90 | 0.15 | 0.05 | 28.77 |
| m1_smartbox | 21.02 | 6.62 | 0.43 | 0.19 | 28.26 |
| mobile-api | 14.07 | 0 | 2.58 | 0.05 | 16.70 |
| cmhub-prod | 13.29 | 0 | 0 | 0 | 13.29 |

### P99 Latency by Cluster

| Cluster | P99 Latency (ms) | Status |
|---------|------------------|--------|
| m1_smartbox | 4490.7 | Critical |
| m1_varnish | 4347.1 | Critical |
| mobile-api | 3059.0 | High |
| booking | 1920.1 | Elevated |
| exchange | 948.2 | Elevated |
| m2_checkout | 924.7 | Elevated |
| eai | 912.4 | Elevated |
| search_k8s | 827.8 | Moderate |
| booking-api | 808.3 | Moderate |
| m2_admin | 705.0 | Moderate |

### Active Connections

| Cluster | Active Connections |
|---------|-------------------|
| eai | 104 |
| booking | 48 |
| shireapi_cluster | 48 |
| m1_smartbox | 47 |
| m2_checkout | 41 |
| mobile-api | 36 |
| cmhub-prod | 36 |

---

## Implementation Design

### EnvoyEnrichmentService

Following the existing enrichment pattern (ExceptionEnrichmentService, ServiceGraphEnrichmentService):

```python
@dataclass
class EnvoyMetricsContext:
    """Envoy edge metrics context for a service."""
    service_name: str
    envoy_cluster_name: str | None
    timestamp: datetime

    # Request rates by response class
    request_rate_2xx: float
    request_rate_3xx: float
    request_rate_4xx: float
    request_rate_5xx: float
    total_request_rate: float

    # Error rate from edge perspective
    edge_error_rate: float  # (4xx + 5xx) / total
    edge_server_error_rate: float  # 5xx / total

    # Latency percentiles
    latency_p50_ms: float | None
    latency_p90_ms: float | None
    latency_p99_ms: float | None

    # Connection info
    active_connections: int

    # Query status
    query_successful: bool
    error_message: str | None

    # Summary for UI display
    summary: str


class EnvoyEnrichmentService:
    """Enriches anomaly detection with Envoy edge metrics."""

    def __init__(
        self,
        mimir_endpoint: str = "https://mimir.sbxtest.net/prometheus",
        lookback_minutes: int = 5,
        enabled: bool = True,
    ):
        self.mimir_endpoint = mimir_endpoint
        self.lookback_minutes = lookback_minutes
        self.enabled = enabled
        self._cluster_mapping = self._load_cluster_mapping()

    def _load_cluster_mapping(self) -> dict[str, str]:
        """Load OTel service name to Envoy cluster mapping."""
        return {
            "booking": "booking",
            "search": "search_k8s",
            "mobile-api": "mobile-api",
            "shire-api": "shireapi_cluster",
            "fa5": "fa5-public",
            "titan": "titan",
            "friday": "friday",
        }

    def get_envoy_context(
        self,
        service_name: str,
        timestamp: datetime | None = None,
    ) -> EnvoyMetricsContext | None:
        """Get Envoy metrics context for a service."""
        if not self.enabled:
            return None

        cluster_name = self._cluster_mapping.get(service_name)
        if not cluster_name:
            return None

        # Query Mimir for Envoy metrics
        # ... implementation details ...
```

### Configuration

Add to `config.json`:

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
      "shire-api": "shireapi_cluster",
      "fa5": "fa5-public",
      "titan": "titan"
    }
  }
}
```

### API Payload Enhancement

Add `envoy_context` field to anomaly payloads:

```json
{
  "alert_type": "anomaly_detected",
  "service_name": "booking",
  "anomalies": { ... },
  "current_metrics": { ... },
  "exception_context": { ... },
  "service_graph_context": { ... },

  "envoy_context": {
    "service_name": "booking",
    "envoy_cluster_name": "booking",
    "timestamp": "2026-01-15T10:30:00",

    "request_rates": {
      "2xx": 68.88,
      "3xx": 5.80,
      "4xx": 0.51,
      "5xx": 0.003,
      "total": 75.19
    },

    "edge_error_rate": 0.0068,
    "edge_server_error_rate": 0.00004,

    "latency_percentiles": {
      "p50_ms": 245.0,
      "p90_ms": 892.0,
      "p99_ms": 1920.1
    },

    "active_connections": 48,

    "comparison_to_otel": {
      "otel_request_rate": 52.7,
      "envoy_request_rate": 75.19,
      "rate_delta_percent": 42.7,
      "note": "Envoy sees more traffic (includes health checks, internal calls)"
    },

    "summary": "Edge: 75.19 req/s (68.88 2xx, 0.51 4xx, 0.003 5xx). P99: 1920ms. 48 active connections.",

    "query_successful": true,
    "error_message": null
  }
}
```

---

## When Envoy Enrichment Triggers

Envoy context is populated when:

1. **Service has Envoy mapping** - The service name maps to an Envoy cluster
2. **Any anomaly detected** - Unlike exception enrichment (errors only), edge context is valuable for all anomaly types
3. **Mimir is reachable** - Query succeeds within timeout

This differs from exception enrichment (only on error anomalies) and service graph enrichment (only on latency anomalies) because edge metrics provide valuable context for any type of anomaly.

---

## Use Cases

### 1. Traffic Discrepancy Detection

Compare OTel `request_rate` with Envoy `total_request_rate`:
- Large delta may indicate dropped requests, health checks, or internal traffic
- Useful for debugging traffic cliff anomalies

### 2. Edge Error Rate Correlation

Compare OTel `error_rate` with Envoy `edge_error_rate`:
- If Envoy 5xx > OTel error rate: Issues at edge/proxy level
- If OTel error rate > Envoy 5xx: Application-level errors being retried

### 3. Latency Attribution

Compare OTel `application_latency` with Envoy latency percentiles:
- Envoy P99 >> OTel P99: Network/serialization overhead
- Envoy P99 ≈ OTel P99: Latency is application-side

### 4. Connection Pool Analysis

Monitor `active_connections` for:
- Connection pool exhaustion during traffic spikes
- Unusual connection patterns during errors

---

## Implementation Phases

### Phase 1: Basic Integration
1. Create `EnvoyEnrichmentService` class
2. Implement Mimir query methods
3. Add cluster name mapping configuration
4. Integrate into enrichment pipeline

### Phase 2: Enhanced Analysis
1. Add OTel vs Envoy comparison logic
2. Implement latency attribution analysis
3. Add connection pool monitoring

### Phase 3: Anomaly Detection
1. Consider adding Envoy-specific anomaly patterns
2. Edge traffic cliff detection
3. Edge error rate spikes

---

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `smartbox_anomaly/enrichment/envoy.py` | Create | EnvoyEnrichmentService implementation |
| `smartbox_anomaly/core/config.py` | Modify | Add EnvoyConfig dataclass |
| `config.json` | Modify | Add envoy_enrichment section |
| `smartbox_anomaly/inference/pipeline.py` | Modify | Integrate EnvoyEnrichmentService |
| `docs/INFERENCE_API_PAYLOAD.md` | Modify | Document envoy_context field |
| `docs/API_CHANGELOG.md` | Modify | Add changelog entry |

---

## PromQL Queries for Mimir

### Request Rate by Response Class
```promql
sum(rate(envoy_cluster_upstream_rq_xx{envoy_cluster_name="booking"}[5m]))
  by (envoy_response_code_class)
```

### Latency Percentiles
```promql
# P50
histogram_quantile(0.50,
  sum(rate(envoy_cluster_upstream_rq_time_bucket{envoy_cluster_name="booking"}[5m]))
  by (le))

# P90
histogram_quantile(0.90,
  sum(rate(envoy_cluster_upstream_rq_time_bucket{envoy_cluster_name="booking"}[5m]))
  by (le))

# P99
histogram_quantile(0.99,
  sum(rate(envoy_cluster_upstream_rq_time_bucket{envoy_cluster_name="booking"}[5m]))
  by (le))
```

### Active Connections
```promql
sum(envoy_cluster_upstream_cx_active{envoy_cluster_name="booking"})
```

### Total Request Rate
```promql
sum(rate(envoy_cluster_upstream_rq_total{envoy_cluster_name="booking"}[5m]))
```

---

## Open Questions

1. **Should Envoy anomalies trigger independently?** Or only enrich existing OTel-detected anomalies?

2. **How to handle services without Envoy mapping?** Currently returns `null` - is this acceptable?

3. **Should we auto-discover cluster mappings?** By querying both sources and matching by traffic patterns?

4. **Mimir authentication?** Is the endpoint protected or open?

5. **Rate limiting?** Does Mimir have query rate limits we need to respect?

---

## References

- `/Users/antun.krasic/test/yaga2/ENVOY_METRICS.md` - Original Grafana dashboard queries
- `/Users/antun.krasic/test/yaga2/smartbox_anomaly/enrichment/` - Existing enrichment service patterns
- Mimir endpoint: `https://mimir.sbxtest.net/prometheus`
- OTel endpoint: `https://otel-metrics.production.smartbox.com`
