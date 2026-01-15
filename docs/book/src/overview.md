# System Overview

The Yaga2 anomaly detection system processes metrics through multiple layers to produce actionable alerts.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Inference Pipeline                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐                                                        │
│  │ Victoria     │                                                        │
│  │ Metrics      │──┐                                                     │
│  └──────────────┘  │                                                     │
│                    ▼                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    Detection Layer                               │    │
│  │  ┌─────────────────┐    ┌─────────────────┐                     │    │
│  │  │ Isolation Forest│───▶│ Pattern Matching │                     │    │
│  │  │ (ML Anomaly)    │    │ (Interpretation) │                     │    │
│  │  └─────────────────┘    └─────────────────┘                     │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                    │                                                     │
│                    ▼                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    SLO Evaluation Layer                          │    │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────┐            │    │
│  │  │ Latency │ │ Errors  │ │ DB Lat  │ │ Request Rate│            │    │
│  │  └────┬────┘ └────┬────┘ └────┬────┘ └──────┬──────┘            │    │
│  │       └───────────┴───────────┴─────────────┘                    │    │
│  │                         │                                         │    │
│  │                         ▼                                         │    │
│  │              ┌─────────────────────┐                             │    │
│  │              │ Severity Adjustment │                             │    │
│  │              └─────────────────────┘                             │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                    │                                                     │
│                    ▼                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    Incident Lifecycle                            │    │
│  │                                                                   │    │
│  │  SUSPECTED ──▶ OPEN ──▶ RECOVERING ──▶ CLOSED                   │    │
│  │      │          │           │             │                      │    │
│  │   (wait)    (alert)     (grace)      (resolve)                  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Metrics Collected

| Metric | Description | Unit |
|--------|-------------|------|
| `request_rate` | Incoming requests per second | req/s |
| `application_latency` | Server-side processing time | ms |
| `dependency_latency` | External dependency call time | ms |
| `database_latency` | Database query time | ms |
| `error_rate` | Percentage of failed requests | ratio (0-1) |

## Time-Aware Detection

The system uses **time-aware models** trained separately for each behavioral period:

| Period | Hours | Days |
|--------|-------|------|
| `business_hours` | 08:00 - 18:00 | Mon-Fri |
| `evening_hours` | 18:00 - 22:00 | Mon-Fri |
| `night_hours` | 22:00 - 06:00 | Mon-Fri |
| `weekend_day` | 08:00 - 22:00 | Sat-Sun |
| `weekend_night` | 22:00 - 08:00 | Sat-Sun |

This prevents false positives from expected behavioral differences (e.g., low traffic at 3 AM is normal).

## Two-Pass Detection

For accurate cascade analysis, detection runs in two passes:

1. **Pass 1**: Detect anomalies for all services (no dependency context)
2. **Pass 2**: Re-analyze latency anomalies with dependency context

This enables identifying root cause services in dependency chains.

## Output

Each inference run produces:

- **Anomaly alerts** for services with detected issues
- **Resolution notifications** for incidents that cleared
- **Enrichment data** (exceptions, service graph) when available

See [API Payload](./reference/api-payload.md) for the complete output format.
