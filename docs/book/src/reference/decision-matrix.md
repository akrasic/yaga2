# Decision Matrix

Quick reference for how different conditions map to alert decisions.

## End-to-End Decision Flow

```
Metrics
   │
   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ 1. Detection: Is this behavior unusual?                                  │
│    Isolation Forest + Pattern Matching                                   │
│    Output: anomaly detected (yes/no), severity, pattern name             │
└────────────────────────────────────────────────────────────────────────┬─┘
                                                                         │
   ┌─────────────────────────────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ 2. SLO Evaluation: Does it matter operationally?                         │
│    Latency, Errors, DB Latency, Request Rate vs thresholds              │
│    Output: slo_status (ok/warning/breached), adjusted severity           │
└────────────────────────────────────────────────────────────────────────┬─┘
                                                                         │
   ┌─────────────────────────────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ 3. Incident Lifecycle: Should we alert now?                              │
│    Confirmation cycles, grace periods                                    │
│    Output: alert (yes/no), status (SUSPECTED/OPEN/RECOVERING/CLOSED)     │
└──────────────────────────────────────────────────────────────────────────┘
```

## Detection Decision Matrix

| Traffic | Latency | Errors | Pattern | Severity |
|---------|---------|--------|---------|----------|
| High | Normal | Normal | `traffic_surge_healthy` | low |
| High | High | Normal | `traffic_surge_degrading` | high |
| High | High | High | `traffic_surge_failing` | critical |
| Very Low | Any | Any | `traffic_cliff` | critical |
| Normal | High | Normal | `latency_spike_recent` | high |
| Normal | High | Normal (deps healthy) | `internal_latency_issue` | high |
| Normal | Normal | High | `error_rate_elevated` | high |
| Normal | Normal | Very High | `error_rate_critical` | critical |
| Normal | Low | High | `fast_failure` | critical |
| Normal | Very Low | Very High | `fast_rejection` | critical |
| Normal | High (DB dominant) | Normal | `database_bottleneck` | high |
| Normal | Normal (DB high) | Normal | `database_degradation` | medium |

## SLO Severity Adjustment Matrix

| ML Severity | Latency SLO | Error SLO | DB SLO | Final Severity |
|-------------|-------------|-----------|--------|----------------|
| critical | ok | ok | ok | **low** |
| critical | ok | ok | warning | **low** |
| critical | warning | ok | ok | high |
| critical | ok | warning | ok | high |
| critical | breached | ok | ok | **critical** |
| critical | ok | breached | ok | **critical** |
| high | ok | ok | ok | **low** |
| high | warning | ok | ok | high |
| medium | ok | ok | ok | **low** |
| any | breached | breached | any | **critical** |

**Key Rule**: SLO status `ok` → Final severity `low` (regardless of ML severity)

## Incident State Decision Matrix

| Current State | Anomaly Detected? | Consecutive | Action |
|---------------|-------------------|-------------|--------|
| None | Yes | 1 | CREATE (SUSPECTED) |
| SUSPECTED | Yes | < threshold | Stay SUSPECTED |
| SUSPECTED | Yes | ≥ threshold | → OPEN (alert) |
| SUSPECTED | No | < grace | Stay SUSPECTED |
| SUSPECTED | No | ≥ grace | → CLOSED (silent) |
| OPEN | Yes | any | Continue OPEN |
| OPEN | No | 1 | → RECOVERING |
| RECOVERING | Yes | any | → OPEN (resume) |
| RECOVERING | No | < grace | Stay RECOVERING |
| RECOVERING | No | ≥ grace | → CLOSED (resolve) |

## Alert Decision Summary

| Condition | Alert Sent? | Reason |
|-----------|-------------|--------|
| First detection | No | SUSPECTED, waiting for confirmation |
| Second consecutive | Yes | Confirmed (OPEN) |
| Continuing OPEN | No | Already alerting |
| First non-detection | No | Grace period (RECOVERING) |
| Resolved | Resolution | CLOSED after grace period |
| Stale gap (>30 min) | New alert | Old closed, new SUSPECTED |

## Complete Example Scenarios

### Scenario 1: Transient Spike (No Alert)

```
10:00 - Latency 450ms detected
        → ML: latency_spike_recent (high)
        → SLO: 450ms < 500ms acceptable → status: ok
        → Adjusted severity: low
        → Status: SUSPECTED (first detection)
        → Alert: ❌ NO

10:03 - Latency 120ms normal
        → No anomaly detected
        → Status: SUSPECTED (missed: 1)

10:06 - Latency 115ms normal
        → No anomaly detected
        → Status: SUSPECTED (missed: 2)

10:09 - Latency 118ms normal
        → Status: CLOSED (suspected_expired)
        → Alert: ❌ NO (never confirmed)
```

### Scenario 2: Real Incident (Alert)

```
10:00 - Latency 850ms detected
        → ML: latency_spike_recent (critical)
        → SLO: 850ms > 800ms warning → status: warning
        → Adjusted severity: high
        → Status: SUSPECTED
        → Alert: ❌ NO (not confirmed)

10:03 - Latency 900ms detected
        → Status: OPEN (confirmed!)
        → Alert: ✅ YES

10:06 - Latency 920ms detected
        → Status: OPEN (continue)
        → Alert: Already sent

...

10:30 - Latency 200ms normal
        → Status: RECOVERING (grace: 1)

10:33 - Latency 180ms normal
        → Status: RECOVERING (grace: 2)

10:36 - Latency 175ms normal
        → Status: CLOSED (resolved)
        → Resolution: ✅ YES
```

### Scenario 3: SLO Suppression (Low Priority)

```
10:00 - Latency 280ms detected
        → ML: latency_spike_recent (high)
        → SLO: 280ms < 300ms acceptable → status: ok
        → Adjusted severity: low (!!!)
        → Status: SUSPECTED
        → Alert: ❌ NO

10:03 - Latency 285ms detected
        → Status: OPEN (confirmed)
        → Severity: low
        → Alert: ✅ YES (low priority)
```

## Quick Reference: When to Alert

| Must be True | Description |
|-------------|-------------|
| Anomaly detected | ML flagged unusual behavior |
| Pattern matched | Known operational scenario |
| 2+ consecutive | Confirmed, not transient |
| SLO evaluated | Operational impact assessed |

| Final Severity | Action |
|----------------|--------|
| critical | PagerDuty, immediate action |
| high | Alert, investigate soon |
| low | Log only, informational |
| none | No action |
