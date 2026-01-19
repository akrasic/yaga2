# Envoy Edge Metrics Enrichment

**Version**: 1.5.0
**Last Updated**: 2026-01-15
**Status**: Implemented, partial coverage

---

## Overview

The Envoy enrichment feature adds edge-level metrics from Envoy proxy to anomaly detection payloads. This provides correlation between OpenTelemetry application metrics and ingress-level observations.

### Why It Matters

| Perspective | Metrics Source | What It Shows |
|-------------|----------------|---------------|
| Application | OTel (VictoriaMetrics) | Internal service health, processing time, error rates |
| Edge/Ingress | Envoy (Mimir) | How service appears to external clients, edge latency, connection issues |

Correlating both views helps identify:
- Edge-level issues not visible to application metrics (connection limits, TLS errors)
- Discrepancies between internal and external error rates
- Latency added by infrastructure vs application

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Anomaly Detection Pipeline                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌────────────────────────┐ │
│  │   ML         │    │   SLO        │    │   Enrichment           │ │
│  │   Detection  │───▶│   Evaluation │───▶│   Runner               │ │
│  └──────────────┘    └──────────────┘    └────────────────────────┘ │
│                                                   │                  │
│                           ┌───────────────────────┼──────────────┐  │
│                           │                       │              │  │
│                           ▼                       ▼              ▼  │
│                    ┌─────────────┐    ┌─────────────┐   ┌──────────┐│
│                    │ Exception   │    │ Service     │   │ Envoy    ││
│                    │ Enrichment  │    │ Graph       │   │ Enrichmt ││
│                    └──────┬──────┘    └──────┬──────┘   └────┬─────┘│
│                           │                  │               │      │
└───────────────────────────┼──────────────────┼───────────────┼──────┘
                            │                  │               │
                            ▼                  ▼               ▼
                    ┌─────────────┐    ┌─────────────┐   ┌──────────┐
                    │ Victoria    │    │ Victoria    │   │  Mimir   │
                    │ Metrics     │    │ Metrics     │   │          │
                    │ (OTel)      │    │ (traces)    │   │ (Envoy)  │
                    └─────────────┘    └─────────────┘   └──────────┘
```

---

## Current Coverage Status

### Mapped Services (7 of 37 = 19%)

| OTel Service | Envoy Cluster | Category | Status |
|--------------|---------------|----------|--------|
| booking | booking | critical | ✅ Mapped |
| search | search_k8s | critical | ✅ Mapped |
| mobile-api | mobile-api | critical | ✅ Mapped |
| shire-api | shireapi_cluster | critical | ✅ Mapped |
| titan | titan | standard | ✅ Mapped |
| friday | friday | standard | ✅ Mapped |
| fa5 | fa5-public | micro | ✅ Mapped |

### Missing Services (30 of 37 = 81%)

| Category | Services | Notes |
|----------|----------|-------|
| Critical | vms | Likely has Envoy exposure |
| Standard | gambit, r2d2, catalog, tc14 | May have Envoy exposure |
| Admin | 12× m2-*-adm services | Likely internal-only, no Envoy |
| Core | 13× m2-* services | Likely internal-only, no Envoy |

---

## Expanding Coverage

### Step 1: Discover Available Clusters

Run the discovery script from a machine with Mimir access:

```bash
# Generate a full report
uv run python scripts/discover_envoy_clusters.py

# Output JSON for direct config update
uv run python scripts/discover_envoy_clusters.py --output-json
```

The script:
1. Queries Mimir for all `envoy_cluster_name` label values
2. Compares with configured OTel services
3. Uses string similarity to suggest mappings
4. Outputs a report and config snippet

### Step 2: Verify Suggested Mappings

For each suggested mapping, verify it returns data:

```bash
# Check if cluster has traffic
curl -s "https://mimir.sbxtest.net/prometheus/api/v1/query" \
  --data-urlencode 'query=sum(rate(envoy_cluster_upstream_rq_xx{envoy_cluster_name="CLUSTER_NAME"}[5m]))' \
  | jq '.data.result'
```

### Step 3: Update Configuration

Add verified mappings to `config.json`:

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
      "titan": "titan",
      "friday": "friday",

      "vms": "vms",
      "gambit": "gambit"
    }
  }
}
```

### Step 4: Restart Service

The configuration is read at startup. Restart the inference service to pick up new mappings.

---

## Configuration Reference

### config.json Structure

```json
{
  "envoy_enrichment": {
    "enabled": true,
    "mimir_endpoint": "https://mimir.sbxtest.net/prometheus",
    "lookback_minutes": 5,
    "timeout_seconds": 10,
    "cluster_mapping": {
      "<otel_service_name>": "<envoy_cluster_name>"
    }
  }
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `false` | Enable/disable Envoy enrichment |
| `mimir_endpoint` | string | `https://mimir.sbxtest.net/prometheus` | Mimir Prometheus API URL |
| `lookback_minutes` | int | `5` | Time window for rate queries |
| `timeout_seconds` | int | `10` | HTTP request timeout |
| `cluster_mapping` | object | `{}` | OTel service → Envoy cluster mapping |

### Cluster Mapping Patterns

Common naming patterns observed:

| Pattern | Example |
|---------|---------|
| Exact match | `booking` → `booking` |
| `_k8s` suffix | `search` → `search_k8s` |
| `_cluster` suffix | `shire-api` → `shireapi_cluster` |
| `-public` suffix | `fa5` → `fa5-public` |
| Concatenated | `shire-api` → `shireapi_cluster` |

---

## Mimir Queries Used

The EnvoyEnrichmentService queries these metrics:

### Request Rates by Response Class

```promql
sum(rate(envoy_cluster_upstream_rq_xx{envoy_cluster_name="<cluster>"}[5m]))
  by (envoy_response_code_class)
```

Returns: `2xx`, `3xx`, `4xx`, `5xx` rates per second

### Latency Percentiles

```promql
histogram_quantile(0.99,
  sum(rate(envoy_cluster_upstream_rq_time_bucket{envoy_cluster_name="<cluster>"}[5m]))
  by (le))
```

Returns: p50, p90, p99 latency in seconds

### Active Connections

```promql
sum(envoy_cluster_upstream_cx_active{envoy_cluster_name="<cluster>"})
```

Returns: Current active upstream connections

---

## Output Schema

When `envoy_context` is populated in anomaly payloads:

```json
{
  "envoy_context": {
    "service_name": "booking",
    "envoy_cluster_name": "booking",
    "timestamp": "2025-01-15T10:30:00",
    "request_rates": {
      "2xx": 147.2,
      "3xx": 0.0,
      "4xx": 2.1,
      "5xx": 1.2,
      "total": 150.5
    },
    "edge_error_rate": 0.022,
    "edge_server_error_rate": 0.008,
    "latency_percentiles": {
      "p50_ms": 45.2,
      "p90_ms": 120.5,
      "p99_ms": 250.0
    },
    "active_connections": 42,
    "summary": "Edge: 150.50 req/s (147.20 2xx, 2.10 4xx, 1.200 5xx). P99: 250ms. 42 active connections.",
    "query_successful": true,
    "error_message": null
  }
}
```

---

## Files Reference

| File | Purpose |
|------|---------|
| `smartbox_anomaly/enrichment/envoy.py` | EnvoyEnrichmentService implementation |
| `smartbox_anomaly/enrichment/__init__.py` | Module exports |
| `smartbox_anomaly/inference/enrichment_runner.py` | `apply_envoy_enrichment()` method |
| `smartbox_anomaly/inference/pipeline.py` | Service initialization |
| `smartbox_anomaly/core/config.py` | `EnvoyEnrichmentConfig` dataclass |
| `config.json` | Runtime configuration |
| `scripts/discover_envoy_clusters.py` | Cluster discovery script |
| `docs/ENVOY_ENRICHMENT.md` | This documentation |

---

## Troubleshooting

### No Envoy Context in Output

1. Check `envoy_enrichment.enabled` is `true` in config
2. Verify service has a cluster mapping
3. Check Mimir connectivity from the inference host
4. Check logs for query errors

### Query Timeouts

Increase `timeout_seconds` in config or check Mimir load.

### Missing Metrics

Some services may not have Envoy exposure:
- Internal-only services (admin panels, background workers)
- Services accessed via different ingress path
- New services not yet in Envoy config

---

## Future Improvements

### Planned

- [ ] Auto-discovery of new clusters on startup
- [ ] Correlation scoring between OTel and Envoy error rates
- [ ] Alert when edge and application metrics diverge significantly

### Potential

- [ ] Support for multiple Envoy clusters per service (canary/stable)
- [ ] Historical Envoy metrics in resolution context
- [ ] Edge latency vs application latency comparison in patterns

---

## Changelog

### v1.5.0 (2026-01-15)

- Initial implementation of Envoy enrichment
- Added `envoy_context` field to anomaly payloads
- Created cluster discovery script
- Mapped 7 critical/standard services

---

## Contact

For questions about Envoy enrichment or to request additional cluster mappings, contact the ML Platform team.
