# Confirmation Logic

Confirmation prevents alert noise by requiring multiple consecutive detections before alerting. This chapter explains why confirmation is essential, how it works, and how to tune it for your environment.

## What is Confirmation?

**Confirmation** is the process of validating that an anomaly is real and persistent before triggering an alert. Instead of alerting immediately when something looks wrong, the system waits to see if the issue persists across multiple detection cycles.

```
Without Confirmation:           With Confirmation:

Detection 1 → ALERT!           Detection 1 → Wait...
(might be noise)               (could be transient)

Detection 2 → ALERT!           Detection 2 → ALERT!
(might be same issue)          (confirmed: real issue)
```

Think of it like a smoke detector: you don't want it to alarm for every wisp of steam from a shower. You want it to confirm there's actual smoke before waking everyone up.

## Why Confirmation Matters

### The Problem: Alert Fatigue

In production environments, metrics naturally fluctuate. A latency spike might last 30 seconds then disappear. An error rate might briefly increase during a garbage collection pause. Without confirmation, every transient blip becomes an alert.

**Without confirmation, operators experience:**
- Multiple alerts per hour for transient issues
- "Resolved" notifications seconds after alerts
- Loss of trust in the alerting system
- Critical alerts buried in noise

### Real-World Example

Consider a service during normal operation:

```
TIME      LATENCY   WITHOUT CONFIRMATION    WITH CONFIRMATION
────────────────────────────────────────────────────────────────
10:00     120ms     Normal                  Normal
10:03     450ms     🔴 ALERT: Latency!      ⏳ SUSPECTED (1/2)
10:06     125ms     🟢 RESOLVED             ⏳ Expires (no alert)
10:09     448ms     🔴 ALERT: Latency!      ⏳ SUSPECTED (1/2)
10:12     455ms     (still alerting)        🔴 CONFIRMED (2/2)
10:15     460ms     (still alerting)        📊 Continue tracking
10:18     130ms     🟢 RESOLVED             ⏳ Recovering (1/3)
10:21     128ms     Normal                  ⏳ Recovering (2/3)
10:24     125ms     Normal                  🟢 RESOLVED
```

**Results:**
- Without confirmation: 2 alerts, 2 resolutions (noisy)
- With confirmation: 1 alert, 1 resolution (accurate)

The confirmed alert represents the real issue (10:09-10:15), while the transient spike at 10:03 was correctly filtered out.

## How Confirmation Works

### The Confirmation Counter

Each incident tracks how many consecutive cycles it has been detected:

```
┌────────────────────────────────────────────────────────────────────┐
│                    consecutive_detections Counter                   │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│   Cycle 1: Anomaly detected                                        │
│   ┌────────────────────────────────────────────────────────────┐   │
│   │  consecutive_detections: 1                                 │   │
│   │  confirmation_cycles: 2  (configured)                      │   │
│   │  cycles_to_confirm: 1   (remaining)                        │   │
│   │                                                            │   │
│   │  Is 1 >= 2? No → Stay SUSPECTED                           │   │
│   └────────────────────────────────────────────────────────────┘   │
│                                                                    │
│   Cycle 2: Anomaly detected again                                  │
│   ┌────────────────────────────────────────────────────────────┐   │
│   │  consecutive_detections: 2                                 │   │
│   │  confirmation_cycles: 2  (configured)                      │   │
│   │  cycles_to_confirm: 0   (remaining)                        │   │
│   │                                                            │   │
│   │  Is 2 >= 2? Yes → Transition to OPEN                      │   │
│   └────────────────────────────────────────────────────────────┘   │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### Confirmation Flow Diagram

```
                    ┌─────────────────────┐
                    │  Anomaly Detected   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
              ┌─────│ Active Incident?    │─────┐
              │     └─────────────────────┘     │
              │NO                               │YES
              ▼                                 ▼
    ┌─────────────────────┐           ┌─────────────────────┐
    │  Create SUSPECTED   │           │  What's the status? │
    │  consecutive = 1    │           └──────────┬──────────┘
    └──────────┬──────────┘                      │
               │                    ┌────────────┼────────────┐
               ▼                    │            │            │
    ┌─────────────────────┐    SUSPECTED       OPEN      RECOVERING
    │  Wait for next      │         │            │            │
    │  detection cycle    │         ▼            ▼            ▼
    └─────────────────────┘    Increment     Continue     Resume to
                               counter       tracking       OPEN
                                   │
                                   ▼
                          ┌─────────────────────┐
                          │ consecutive >=      │
                          │ confirmation_cycles?│
                          └──────────┬──────────┘
                               │           │
                              YES          NO
                               │           │
                               ▼           ▼
                    ┌─────────────────┐  ┌─────────────────┐
                    │  CONFIRMED!     │  │  Still waiting  │
                    │  Status → OPEN  │  │  Stay SUSPECTED │
                    │  Send Alert     │  │  No alert       │
                    └─────────────────┘  └─────────────────┘
```

## Detection Cycles Explained

### Cycle 1: First Detection (SUSPECTED)

When an anomaly is first detected, it enters the SUSPECTED state:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Detection Cycle 1                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Time: 10:00:00                                                    │
│   Event: Latency spike detected (450ms, normally 120ms)             │
│                                                                     │
│   Actions taken:                                                    │
│   ─────────────                                                     │
│   1. Generate fingerprint_id from pattern content                   │
│   2. Check database for active incident with this fingerprint       │
│   3. No active incident found → Create new incident                 │
│   4. Set status = SUSPECTED                                         │
│                                                                     │
│   Result:                                                           │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  fingerprint_id: anomaly_8d4a011b83ca                       │   │
│   │  incident_id: incident_1dcbafc91480                         │   │
│   │  status: SUSPECTED                                          │   │
│   │  consecutive_detections: 1                                  │   │
│   │  confirmation_pending: true                                 │   │
│   │  cycles_to_confirm: 1                                       │   │
│   │  is_confirmed: false                                        │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│   Web API: ❌ NOT notified                                          │
│   Dashboard: ❌ No alert displayed                                  │
│   Reason: Waiting for confirmation                                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Cycle 2: Confirmation (SUSPECTED → OPEN)

If the same anomaly is detected again in the next cycle:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Detection Cycle 2                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Time: 10:03:00 (3 minutes later)                                  │
│   Event: Same latency pattern detected again                        │
│                                                                     │
│   Actions taken:                                                    │
│   ─────────────                                                     │
│   1. Generate fingerprint_id from pattern content                   │
│   2. Check database → Found existing SUSPECTED incident             │
│   3. Increment consecutive_detections: 1 → 2                        │
│   4. Check: 2 >= confirmation_cycles (2)? YES!                      │
│   5. Transition status: SUSPECTED → OPEN                            │
│   6. Set newly_confirmed = true                                     │
│   7. Send alert to Web API                                          │
│                                                                     │
│   Result:                                                           │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  fingerprint_id: anomaly_8d4a011b83ca                       │   │
│   │  incident_id: incident_1dcbafc91480                         │   │
│   │  status: OPEN                                               │   │
│   │  previous_status: SUSPECTED  ← For tracking the transition  │   │
│   │  consecutive_detections: 2                                  │   │
│   │  confirmation_pending: false                                │   │
│   │  is_confirmed: true                                         │   │
│   │  newly_confirmed: true  ← Signals this is fresh confirmation│   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│   Web API: ✅ Alert sent!                                           │
│   Dashboard: ✅ Alert displayed to operators                        │
│   Reason: Confirmed after 2 consecutive detections                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### The `newly_confirmed` Flag

The `newly_confirmed` flag is crucial for downstream consumers:

```
When newly_confirmed = true:
─────────────────────────────
• This is the FIRST time this incident is being sent as confirmed
• Web UI should create a new alert entry
• Notification systems should send alerts
• Only set on the exact cycle of confirmation

When newly_confirmed = false:
──────────────────────────────
• Incident was already confirmed in a previous cycle
• Web UI should update existing alert entry
• Notification systems should NOT re-alert
• Set for all subsequent cycles
```

## Key Fields Reference

| Field | Type | Description |
|-------|------|-------------|
| `consecutive_detections` | integer | How many cycles in a row this anomaly was detected |
| `confirmation_pending` | boolean | `true` while still in SUSPECTED state |
| `cycles_to_confirm` | integer | Remaining cycles needed for confirmation (0 when confirmed) |
| `is_confirmed` | boolean | `true` once status becomes OPEN |
| `newly_confirmed` | boolean | `true` only on the exact cycle where SUSPECTED → OPEN |
| `previous_status` | string | What the status was before this cycle (for tracking transitions) |

## Fingerprinting Summary Object

The top-level `fingerprinting` object provides a summary of all confirmation activity:

```json
{
  "fingerprinting": {
    "service_name": "booking",
    "model_name": "business_hours",
    "timestamp": "2025-12-17T10:03:00",
    "overall_action": "CONFIRMED",

    "status_summary": {
      "suspected": 0,
      "confirmed": 1,
      "recovering": 0
    },

    "action_summary": {
      "incident_creates": 0,
      "incident_continues": 0,
      "incident_closes": 0,
      "newly_confirmed": 1
    },

    "newly_confirmed_incidents": [
      {
        "fingerprint_id": "anomaly_8d4a011b83ca",
        "incident_id": "incident_1dcbafc91480",
        "anomaly_name": "latency_spike_recent",
        "severity": "high"
      }
    ]
  }
}
```

### Overall Action Values

| Action | Meaning | When It Happens |
|--------|---------|-----------------|
| `CREATE` | New incident(s) created in SUSPECTED state | First detection of new anomaly patterns |
| `CONFIRMED` | Incident(s) transitioned SUSPECTED → OPEN | Anomaly detected for confirmation_cycles times |
| `UPDATE` | Existing OPEN incident(s) continued | Ongoing anomaly still being detected |
| `RESOLVE` | Incident(s) closed | Grace period exceeded without detection |
| `MIXED` | Multiple different actions in one cycle | E.g., one confirmed while another closes |
| `NO_CHANGE` | No significant state changes | Only RECOVERING incidents still waiting |

## Web API Integration

### Confirmed-Only Alerts (v1.3.2)

Starting with version 1.3.2, only **confirmed** anomalies are sent to the web API. This is a critical feature for preventing orphaned incidents.

```python
# How the inference engine filters before sending to web API

def process_results(anomalies):
    # Filter to only confirmed anomalies
    confirmed_anomalies = {
        name: anomaly for name, anomaly in anomalies.items()
        if anomaly.get('is_confirmed', False) or
           anomaly.get('status') in ('OPEN', 'RECOVERING')
    }

    if confirmed_anomalies:
        # Send only confirmed anomalies to web API
        send_alert(confirmed_anomalies)
    else:
        # SUSPECTED anomalies are NOT sent
        # This prevents orphaned incidents
        pass
```

### Why This Matters

Before v1.3.2, all anomalies (including SUSPECTED) were sent to the web API. This caused problems:

```
OLD BEHAVIOR (problematic):
───────────────────────────
10:00  Detection → SUSPECTED → Sent to Web API → Web API creates OPEN incident
10:03  Not detected → SUSPECTED expires → No resolution sent (suspected_expired)
10:06  ...
Result: Orphaned OPEN incident in Web API that never gets resolved!

NEW BEHAVIOR (correct):
───────────────────────────
10:00  Detection → SUSPECTED → NOT sent to Web API (waiting for confirmation)
10:03  Not detected → SUSPECTED expires → Nothing to resolve (never sent)
10:06  ...
Result: No orphaned incident - Web API never knew about it!

OR if it gets confirmed:

10:00  Detection → SUSPECTED → NOT sent to Web API
10:03  Detection → OPEN (confirmed) → Sent to Web API → Web API creates incident
10:06  Detection → OPEN continues → Update sent
...
10:15  Not detected × 3 cycles → CLOSED → Resolution sent
Result: Complete lifecycle - incident created when confirmed, resolved when cleared
```

## SUSPECTED Expiration

When an anomaly is detected but then disappears before confirmation:

```
Timeline of SUSPECTED Expiration:
─────────────────────────────────

10:00:00  Anomaly detected
          ├─ Status: SUSPECTED
          ├─ consecutive_detections: 1
          └─ Web API: NOT notified

10:03:00  Anomaly NOT detected
          ├─ Status: still SUSPECTED
          ├─ missed_cycles: 1
          └─ Web API: still not notified

10:06:00  Anomaly NOT detected
          ├─ Status: still SUSPECTED
          ├─ missed_cycles: 2
          └─ Web API: still not notified

10:09:00  Anomaly NOT detected
          ├─ missed_cycles: 3 >= resolution_grace_cycles
          ├─ Status: SUSPECTED → CLOSED
          ├─ resolution_reason: "suspected_expired"
          ├─ Web API: NO resolution sent (never was an incident there)
          └─ Incident removed from tracking

Result:
─────────
• No alert was ever sent
• No resolution is needed
• Transient issue correctly filtered
• Zero noise in the alerting system
```

### Resolution Reasons for SUSPECTED

| Reason | Description | Web API Impact |
|--------|-------------|----------------|
| `suspected_expired` | Never confirmed, disappeared before confirmation | Nothing sent (no orphan) |
| `resolved` | Normal resolution after grace period | N/A (only applies to OPEN) |
| `auto_stale` | Time gap exceeded threshold | N/A (only applies to OPEN) |

## Tuning Confirmation Cycles

The `confirmation_cycles` configuration determines how many consecutive detections are required:

```json
{
  "fingerprinting": {
    "confirmation_cycles": 2
  }
}
```

### Trade-off Analysis

| Value | Confirmation Time* | Pros | Cons |
|-------|-------------------|------|------|
| 1 | Immediate | Fastest response | No filtering, all noise |
| **2** | ~4-6 min | **Good balance (default)** | 1 cycle delay |
| 3 | ~6-9 min | Fewer false positives | May miss short incidents |
| 4 | ~8-12 min | Very strict filtering | Risk of missing real issues |
| 5+ | 10+ min | Maximum noise reduction | Likely too slow for production |

*Assuming 2-3 minute detection cycles

### Choosing the Right Value

**Use `confirmation_cycles: 1`** when:
- Testing or debugging the system
- You need immediate alerts regardless of noise
- You have other mechanisms to filter alerts downstream

**Use `confirmation_cycles: 2`** (default) when:
- Running in production
- You want balanced noise reduction
- Detection cycle is 2-5 minutes

**Use `confirmation_cycles: 3`** when:
- You have a noisy environment with frequent transient issues
- False positives are more costly than delayed detection
- You can tolerate 6-9 minute confirmation delay

**Use `confirmation_cycles: 4+`** when:
- You have very long detection cycles (10+ minutes)
- You're monitoring non-critical services
- Alert fatigue is a severe problem

## Edge Cases

### Pattern Changes During Confirmation

If the anomaly pattern changes during confirmation, it's treated as a new anomaly:

```
10:00  latency_spike_recent detected → SUSPECTED
10:03  traffic_surge_failing detected → NEW SUSPECTED (different pattern)
       (latency_spike_recent starts expiration countdown)
```

### Multiple Anomalies Confirming Simultaneously

Multiple anomalies can confirm in the same cycle:

```json
{
  "fingerprinting": {
    "overall_action": "MIXED",
    "newly_confirmed_incidents": [
      {"anomaly_name": "latency_spike_recent", ...},
      {"anomaly_name": "error_rate_elevated", ...}
    ],
    "action_summary": {
      "newly_confirmed": 2
    }
  }
}
```

### Confirmation After Recovery

If an incident is in RECOVERING state and the anomaly reappears, it doesn't need re-confirmation:

```
10:00  SUSPECTED (1/2)
10:03  OPEN (confirmed)
10:06  OPEN (continuing)
10:09  RECOVERING (not detected, 1/3)
10:12  OPEN (detected again - immediately returns to OPEN, no confirmation needed)
```

This is because the incident was already confirmed before entering RECOVERING.

## Summary

Confirmation is a critical noise-reduction mechanism that:

1. **Prevents alert fatigue** by filtering transient spikes
2. **Ensures reliability** by only alerting on persistent issues
3. **Protects the web API** from orphaned incidents (confirmed-only alerts)
4. **Provides clear lifecycle** with trackable state transitions

The default configuration (`confirmation_cycles: 2`) provides a good balance between responsiveness and noise reduction for most production environments.
