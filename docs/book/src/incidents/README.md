# Incident Lifecycle

The incident lifecycle manages anomaly state over time, reducing alert noise through confirmation requirements and grace periods. This chapter explains how Yaga2 tracks anomalies as they are detected, confirmed, and eventually resolved.

## What Problem Does Incident Lifecycle Solve?

Raw anomaly detection produces a stream of "yes/no" decisions every inference cycle. Without state management, this creates significant operational problems:

### Problem 1: Transient Spikes Cause Alert Noise

```
Without lifecycle management:
─────────────────────────────
10:00  Latency 450ms  → ALERT!  (ops team paged)
10:03  Latency 120ms  → Resolved
10:06  Latency 455ms  → ALERT!  (ops team paged again)
10:09  Latency 118ms  → Resolved
10:12  Latency 460ms  → ALERT!  (ops team paged again)
10:15  Latency 115ms  → Resolved

Result: 3 alerts in 15 minutes for what might be normal variance
        Ops team frustrated, starts ignoring alerts
```

### Problem 2: Brief Recovery Causes Flapping

```
Without lifecycle management:
─────────────────────────────
10:00  Error rate 5%   → ALERT!
10:03  Error rate 5%   → Continue
10:06  Error rate 4.8% → Continue
10:09  Error rate 0.5% → Resolved  (brief dip)
10:12  Error rate 5.2% → ALERT!   (new alert!)
10:15  Error rate 5%   → Continue

Result: Same ongoing incident treated as two separate incidents
        Resolution was premature - the issue wasn't actually fixed
```

### Problem 3: No Incident Correlation

```
Without lifecycle management:
─────────────────────────────
Each detection is independent. No way to answer:
- "How long has this been happening?"
- "Is this the same issue from yesterday?"
- "How many times has this pattern occurred?"
```

## The Solution: Stateful Incident Tracking

Yaga2 solves these problems with a **state machine** that tracks anomalies through a defined lifecycle:

```
┌───────────────────────────────────────────────────────────────────────────┐
│                    THE INCIDENT LIFECYCLE                                  │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│    First Detection                                                        │
│         │                                                                 │
│         ▼                                                                 │
│    ┌─────────────┐                                                        │
│    │  SUSPECTED  │   "We saw something - let's wait to be sure"           │
│    │             │   • No alert sent yet                                  │
│    │  (waiting)  │   • Needs 2 consecutive detections                     │
│    └──────┬──────┘                                                        │
│           │                                                               │
│           │ Detected again (confirmed!)                                   │
│           ▼                                                               │
│    ┌─────────────┐                                                        │
│    │    OPEN     │   "This is real - notify the team"                     │
│    │             │◄─────────────┐                                         │
│    │ (alerting)  │   detected   │ • Alert sent to web API                 │
│    └──────┬──────┘   again      │ • Tracking duration and occurrences     │
│           │          │          │                                         │
│           │ Not detected                                                  │
│           ▼          │          │                                         │
│    ┌─────────────┐   │          │                                         │
│    │ RECOVERING  │───┘          │                                         │
│    │             │   "Anomaly cleared - but let's wait"                   │
│    │  (waiting)  │   • No resolution yet                                  │
│    └──────┬──────┘   • Grace period in progress                           │
│           │                                                               │
│           │ Still not detected (confirmed resolved!)                      │
│           ▼                                                               │
│    ┌─────────────┐                                                        │
│    │   CLOSED    │   "Incident resolved - notify the team"                │
│    │             │   • Resolution sent to web API                         │
│    │ (resolved)  │   • Incident history preserved                         │
│    └─────────────┘                                                        │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

## Core Concepts

### Fingerprint vs Incident

These two concepts are central to incident tracking:

| Concept | What It Is | Analogy |
|---------|------------|---------|
| **Fingerprint ID** | Pattern identifier (same pattern = same ID) | The "type" of crime |
| **Incident ID** | Unique occurrence identifier | A specific "case file" |

**Fingerprint ID** is **deterministic** - it's computed from the anomaly content:
```
hash("booking_business_hours_latency_spike_recent")
  → anomaly_8d4a011b83ca
```

The same anomaly pattern on the same service always gets the same fingerprint ID.

**Incident ID** is **unique** - each new occurrence gets a fresh identifier:
```
Random UUID generation
  → incident_1dcbafc91480
```

### Why Both IDs?

The same anomaly pattern can occur multiple times:

```
                    Fingerprint: anomaly_abc123
               (latency_spike_recent on booking)
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
    Incident #1         Incident #2         Incident #3
    Jan 10, 14:00       Jan 15, 09:30       Jan 22, 16:45
    Duration: 45min     Duration: 2hrs      Duration: 20min

    Each is a separate occurrence of the same pattern type
```

This enables powerful analysis:
- "This pattern has occurred 3 times this month"
- "Average duration is 58 minutes"
- "Longest incident was 2 hours on Jan 15"

## States Explained

### SUSPECTED - First Detection

When an anomaly is detected for the first time, it enters the SUSPECTED state:

```
┌─────────────────────────────────────────────────────────────────────┐
│  STATE: SUSPECTED                                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  What happened:                                                     │
│    ML detected something unusual                                    │
│                                                                     │
│  What we do:                                                        │
│    Wait for confirmation (not alert yet)                            │
│                                                                     │
│  Why wait?                                                          │
│    Single-point anomalies are often noise:                          │
│    • Brief network blip                                             │
│    • One slow database query                                        │
│    • Measurement variance                                           │
│                                                                     │
│  Payload fields:                                                    │
│    status: "SUSPECTED"                                              │
│    is_confirmed: false                                              │
│    confirmation_pending: true                                       │
│    cycles_to_confirm: 1                                             │
│                                                                     │
│  Web API:  NOT notified (to prevent orphaned incidents)             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### OPEN - Confirmed Incident

After 2+ consecutive detections, the incident is **confirmed**:

```
┌─────────────────────────────────────────────────────────────────────┐
│  STATE: OPEN                                                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  What happened:                                                     │
│    Anomaly persisted for 2+ detection cycles                        │
│                                                                     │
│  What we do:                                                        │
│    Send alert to web API                                            │
│    Track duration and occurrence count                              │
│                                                                     │
│  Why confirm?                                                       │
│    Persistent anomalies are likely real issues:                     │
│    • Not a brief spike                                              │
│    • Sustained degradation                                          │
│    • Worth investigating                                            │
│                                                                     │
│  Payload fields:                                                    │
│    status: "OPEN"                                                   │
│    is_confirmed: true                                               │
│    newly_confirmed: true (on first cycle only)                      │
│    occurrence_count: incrementing                                   │
│                                                                     │
│  Web API:  Alert sent!                                              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### RECOVERING - Grace Period

When the anomaly stops being detected, the incident enters a grace period:

```
┌─────────────────────────────────────────────────────────────────────┐
│  STATE: RECOVERING                                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  What happened:                                                     │
│    Anomaly not detected in last cycle                               │
│                                                                     │
│  What we do:                                                        │
│    Wait before sending resolution                                   │
│    Watch for anomaly return                                         │
│                                                                     │
│  Why wait?                                                          │
│    Brief recovery is often not real resolution:                     │
│    • Issue may return in next cycle                                 │
│    • Prevents "flapping" alerts                                     │
│    • More reliable resolution signal                                │
│                                                                     │
│  Payload fields:                                                    │
│    status: "RECOVERING"                                             │
│    missed_cycles: 1, 2, ...                                         │
│                                                                     │
│  Web API:  No update yet (waiting for confirmation)                 │
│                                                                     │
│  If anomaly returns:  → Back to OPEN                                │
│  If still clear:      → Continue grace period                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### CLOSED - Resolved

After 3+ cycles without detection, the incident is confirmed resolved:

```
┌─────────────────────────────────────────────────────────────────────┐
│  STATE: CLOSED                                                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  What happened:                                                     │
│    Anomaly not detected for 3+ consecutive cycles                   │
│                                                                     │
│  What we do:                                                        │
│    Send resolution to web API                                       │
│    Record incident history                                          │
│                                                                     │
│  Resolution contains:                                               │
│    • Total duration                                                 │
│    • Occurrence count                                               │
│    • Final severity                                                 │
│    • Resolution reason                                              │
│                                                                     │
│  Payload fields:                                                    │
│    incident_action: "CLOSE"                                         │
│    resolution_reason: "resolved"                                    │
│    incident_duration_minutes: final value                           │
│    total_occurrences: final count                                   │
│                                                                     │
│  Web API:  Resolution sent!                                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Cycle-Based Timing

The lifecycle is based on **detection cycles**, not wall-clock time:

| Parameter | Default | With 3-min inference | Purpose |
|-----------|---------|---------------------|---------|
| `confirmation_cycles` | 2 | ~6 min to confirm | Prevent false alerts |
| `resolution_grace_cycles` | 3 | ~9 min grace period | Prevent flapping |
| `incident_separation_minutes` | 30 | 30 min gap = new incident | Prevent zombie incidents |

### Example Timeline

```
Time     Detection   Status      Action
────────────────────────────────────────────────────
10:00    Spike!      SUSPECTED   Wait for confirmation
10:03    Spike!      OPEN        Alert sent (confirmed after 2 cycles)
10:06    Spike!      OPEN        Continue tracking
10:09    Spike!      OPEN        Continue tracking
10:12    Normal      RECOVERING  Grace period starts
10:15    Normal      RECOVERING  Grace period continues
10:18    Normal      CLOSED      Resolution sent (3 cycles without detection)
```

## Why Confirmation?

**The problem**: Single-detection alerts create noise.

```
WITHOUT CONFIRMATION:
─────────────────────
Every detection immediately fires an alert.

┌──────────────────────────────────────────────────────────────────┐
│  10:00  450ms latency detected  →  ALERT!  (page on-call)        │
│  10:03  120ms latency (normal)  →  Resolved                      │
│  10:06  455ms latency detected  →  ALERT!  (page on-call again)  │
│  10:09  118ms latency (normal)  →  Resolved                      │
│  10:12  460ms latency detected  →  ALERT!  (page on-call again)  │
│  10:15  115ms latency (normal)  →  Resolved                      │
└──────────────────────────────────────────────────────────────────┘

Result: 3 alerts for transient spikes
        On-call engineer interrupted 3 times
        Nothing was actually wrong
```

**The solution**: Require consecutive detections.

```
WITH CONFIRMATION (2 cycles):
─────────────────────────────
Only sustained anomalies fire alerts.

┌──────────────────────────────────────────────────────────────────┐
│  10:00  450ms latency detected  →  SUSPECTED (wait)              │
│  10:03  120ms latency (normal)  →  SUSPECTED expires quietly     │
│  10:06  455ms latency detected  →  SUSPECTED (wait)              │
│  10:09  460ms latency detected  →  OPEN - ALERT! (confirmed)     │
│  10:12  455ms latency detected  →  OPEN (continuing)             │
│  10:15  450ms latency detected  →  OPEN (continuing)             │
│  10:18  120ms latency (normal)  →  RECOVERING (grace period)     │
│  10:21  115ms latency (normal)  →  RECOVERING (grace period)     │
│  10:24  118ms latency (normal)  →  CLOSED - Resolved             │
└──────────────────────────────────────────────────────────────────┘

Result: 1 alert for a real sustained issue
        Resolution only after confirmed recovery
```

## Why Grace Period?

**The problem**: Brief dips in anomaly cause flapping.

```
WITHOUT GRACE PERIOD:
─────────────────────
Every non-detection immediately resolves.

┌──────────────────────────────────────────────────────────────────┐
│  10:00  Error 5%  →  ALERT!                                      │
│  10:03  Error 5%  →  Continue                                    │
│  10:06  Error 0.5% →  Resolved!  (brief dip)                     │
│  10:09  Error 5%  →  ALERT!  (new incident)                      │
│  10:12  Error 5%  →  Continue                                    │
│  10:15  Error 0.5% →  Resolved!  (another brief dip)             │
│  10:18  Error 5%  →  ALERT!  (third incident)                    │
└──────────────────────────────────────────────────────────────────┘

Result: 3 separate incidents for what's really one ongoing issue
        "Flapping" alerts every time there's a brief improvement
```

**The solution**: Wait before confirming resolution.

```
WITH GRACE PERIOD (3 cycles):
─────────────────────────────
Resolution requires sustained recovery.

┌──────────────────────────────────────────────────────────────────┐
│  10:00  Error 5%  →  ALERT!                                      │
│  10:03  Error 5%  →  Continue                                    │
│  10:06  Error 0.5% →  RECOVERING (grace period, not resolved)    │
│  10:09  Error 5%  →  OPEN (back to alerting - wasn't resolved)   │
│  10:12  Error 5%  →  Continue                                    │
│  10:15  Error 0.5% →  RECOVERING (grace period)                  │
│  10:18  Error 0.5% →  RECOVERING (grace continues)               │
│  10:21  Error 0.5% →  CLOSED - Resolved (confirmed recovery)     │
└──────────────────────────────────────────────────────────────────┘

Result: 1 incident throughout entire episode
        Resolution only after confirmed sustained recovery
```

## Staleness Check

**The problem**: Old incidents continuing forever.

If an anomaly is detected, disappears for hours, then reappears, is it the same incident?

```
10:00  - OPEN incident for latency_spike_recent
10:30  - Last detection
...
15:00  - Same pattern detected again (4.5 hours later!)

Is this the same incident? Probably not - it's a new occurrence.
```

**The solution**: `incident_separation_minutes` threshold.

```
If gap > incident_separation_minutes (default: 30 min):
  → Old incident auto-closed (reason: "auto_stale")
  → New SUSPECTED incident created

10:00  - OPEN incident created
10:30  - Last detection
...
11:15  - Same pattern detected (45 min gap > 30 min threshold)
         → Old incident CLOSED (auto_stale)
         → New SUSPECTED incident created
```

This prevents "zombie incidents" that span unrelated events.

## Confirmed-Only Alerts (v1.3.2)

**Important**: Only confirmed anomalies are sent to the web API.

| State | Sent to Web API? | Why? |
|-------|-----------------|------|
| SUSPECTED | **No** | Not yet confirmed - might be noise |
| OPEN | **Yes** | Confirmed - real incident |
| RECOVERING | **Yes** | Still tracking - might return |
| CLOSED | **Resolution only** | Final status update |

### Why Filter SUSPECTED?

Before v1.3.2, all detections were sent to the web API. This caused **orphaned incidents**:

```
OLD BEHAVIOR (problematic):
───────────────────────────
10:00  SUSPECTED detected  →  Sent to API  →  API creates OPEN incident
10:03  Not detected        →  SUSPECTED expires quietly
10:06  Not detected        →  (nothing sent)

Result: API has an OPEN incident that will never resolve
        "Orphaned incident" - no resolution was ever sent
```

```
NEW BEHAVIOR (v1.3.2):
──────────────────────
10:00  SUSPECTED detected  →  NOT sent to API (wait for confirmation)
10:03  Not detected        →  SUSPECTED expires quietly

Result: API never knew about this transient detection
        No orphaned incidents
```

## Summary Table

| State | Alert Sent? | Resolution Sent? | Purpose |
|-------|-------------|------------------|---------|
| SUSPECTED | No | No | Wait for confirmation |
| OPEN | Yes | No | Active alerting |
| RECOVERING | No | No | Grace period |
| CLOSED | No | Yes | Final notification |

## Sections

- [State Machine](./state-machine.md) - Detailed state transitions and rules
- [Confirmation Logic](./confirmation.md) - How confirmation works
- [Fingerprinting](./fingerprinting.md) - How incidents are identified and tracked
