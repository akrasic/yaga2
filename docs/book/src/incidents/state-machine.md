# State Machine

The incident state machine defines exactly how anomalies transition through the lifecycle. This page provides detailed rules, diagrams, and examples for each transition.

## What is a State Machine?

A **state machine** is a model where:
- The system is always in exactly one **state**
- **Events** trigger transitions between states
- **Conditions** determine which transition occurs
- **Actions** execute when transitions happen

For incident tracking:
- **States**: SUSPECTED, OPEN, RECOVERING, CLOSED
- **Events**: "anomaly detected" or "anomaly not detected"
- **Conditions**: cycle counts, time gaps
- **Actions**: send alert, send resolution, update counters

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        STATE MACHINE BASICS                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│    ┌─────────┐   Event + Condition    ┌─────────┐                       │
│    │ State A │ ──────────────────────►│ State B │                       │
│    └─────────┘         Action         └─────────┘                       │
│                                                                         │
│    Example:                                                             │
│    ┌──────────────┐  Detected again   ┌──────────────┐                  │
│    │  SUSPECTED   │  consecutive ≥ 2  │     OPEN     │                  │
│    │              │ ─────────────────►│              │                  │
│    └──────────────┘  Send alert       └──────────────┘                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Complete Transition Rules

This table defines all possible state transitions:

| From State | Event | Condition | To State | Action |
|------------|-------|-----------|----------|--------|
| *(none)* | Detected | - | SUSPECTED | Create incident |
| SUSPECTED | Detected | `consecutive < N` | SUSPECTED | Increment `consecutive_detections` |
| SUSPECTED | Detected | `consecutive ≥ N` | OPEN | **Send alert** |
| SUSPECTED | Not detected | `missed < M` | SUSPECTED | Increment `missed_cycles` |
| SUSPECTED | Not detected | `missed ≥ M` | CLOSED | Silent close (`suspected_expired`) |
| OPEN | Detected | - | OPEN | Continue, reset `missed_cycles` |
| OPEN | Not detected | - | RECOVERING | Start grace period |
| RECOVERING | Detected | - | OPEN | Resume incident |
| RECOVERING | Not detected | `missed < M` | RECOVERING | Increment `missed_cycles` |
| RECOVERING | Not detected | `missed ≥ M` | CLOSED | **Send resolution** |
| *(any)* | Detected | `gap > threshold` | SUSPECTED | Close stale, create new |

Where:
- `N` = `confirmation_cycles` (default: 2)
- `M` = `resolution_grace_cycles` (default: 3)
- `gap` = time since `last_updated`
- `threshold` = `incident_separation_minutes` (default: 30)

## Visual State Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        INCIDENT STATE MACHINE                                 │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                                                                              │
│         ┌───────────────────────────────────────────────────────────┐        │
│         │                     NO INCIDENT                           │        │
│         │             (no active tracking for this pattern)         │        │
│         └─────────────────────────┬─────────────────────────────────┘        │
│                                   │                                          │
│                                   │ Anomaly detected                         │
│                                   │ (first detection of pattern)             │
│                                   ▼                                          │
│         ┌───────────────────────────────────────────────────────────┐        │
│         │                     SUSPECTED                             │        │
│         │                                                           │        │
│         │  • incident_action: CREATE                                │        │
│         │  • consecutive_detections: 1                              │        │
│         │  • missed_cycles: 0                                       │        │
│         │  • is_confirmed: false                                    │        │
│         │  • confirmation_pending: true                             │        │
│         │                                                           │        │
│         │  Web API: NOT notified                                    │        │
│         └─────────────────────────┬─────────────────────────────────┘        │
│                                   │                                          │
│              ┌────────────────────┼────────────────────┐                     │
│              │                    │                    │                     │
│              │                    │                    │                     │
│              ▼                    ▼                    ▼                     │
│    ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────────────┐       │
│    │  Stay SUSPECTED │ │      OPEN       │ │        CLOSED           │       │
│    │                 │ │   (CONFIRMED)   │ │   (suspected_expired)   │       │
│    │ Event: Detected │ │                 │ │                         │       │
│    │ Cond: consec<N  │ │ Event: Detected │ │ Event: Not detected     │       │
│    │                 │ │ Cond: consec≥N  │ │ Cond: missed≥M          │       │
│    │ Action:         │ │                 │ │                         │       │
│    │ consec++        │ │ Action:         │ │ Action:                 │       │
│    │                 │ │ SEND ALERT      │ │ Silent close            │       │
│    └─────────────────┘ │ newly_confirmed │ │ No alert ever sent      │       │
│                        │ = true          │ │ No resolution sent      │       │
│                        └────────┬────────┘ └─────────────────────────┘       │
│                                 │                                            │
│                                 │                                            │
│                 ┌───────────────┴───────────────┐                            │
│                 │                               │                            │
│                 │ Detected again                │ Not detected               │
│                 ▼                               ▼                            │
│    ┌──────────────────────┐       ┌──────────────────────┐                   │
│    │        OPEN          │       │     RECOVERING       │                   │
│    │     (continue)       │◄──────│                      │                   │
│    │                      │       │ • missed_cycles: 1+  │                   │
│    │ • Reset missed to 0  │Detect │ • Grace period       │                   │
│    │ • occurrence_count++ │ again │ • Watching for       │                   │
│    │ • Update severity    │       │   anomaly return     │                   │
│    │                      │       │                      │                   │
│    └──────────────────────┘       └──────────┬───────────┘                   │
│                                              │                               │
│                                              │ Not detected                  │
│                                              │ (missed ≥ grace_cycles)       │
│                                              ▼                               │
│                                   ┌──────────────────────┐                   │
│                                   │       CLOSED         │                   │
│                                   │     (resolved)       │                   │
│                                   │                      │                   │
│                                   │ Action:              │                   │
│                                   │ SEND RESOLUTION      │                   │
│                                   │                      │                   │
│                                   │ • resolution_reason: │                   │
│                                   │   "resolved"         │                   │
│                                   │ • Final metrics      │                   │
│                                   │   recorded           │                   │
│                                   └──────────────────────┘                   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## State-by-State Details

### NO INCIDENT → SUSPECTED

**Trigger**: First detection of an anomaly pattern

```
Before:   No active incident for this fingerprint
Event:    ML detects anomaly_latency_spike_recent
After:    SUSPECTED incident created

┌─────────────────────────────────────────────────────────────────────┐
│  Transition: NO INCIDENT → SUSPECTED                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  What happens:                                                      │
│    1. Generate fingerprint_id from pattern content                  │
│    2. Check database - no active incident exists                    │
│    3. Create new incident record                                    │
│    4. Set status = SUSPECTED                                        │
│    5. Initialize counters                                           │
│                                                                     │
│  Payload fields set:                                                │
│    fingerprint_id: "anomaly_abc123"  (deterministic hash)           │
│    incident_id: "incident_xyz789"    (new UUID)                     │
│    fingerprint_action: "CREATE"                                     │
│    incident_action: "CREATE"                                        │
│    status: "SUSPECTED"                                              │
│    consecutive_detections: 1                                        │
│    missed_cycles: 0                                                 │
│    occurrence_count: 1                                              │
│    first_seen: <current timestamp>                                  │
│    is_confirmed: false                                              │
│    confirmation_pending: true                                       │
│    cycles_to_confirm: 1                                             │
│                                                                     │
│  Web API: NOT notified (filtered out before sending)                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### SUSPECTED → SUSPECTED (Still Waiting)

**Trigger**: Anomaly detected again, but not enough cycles yet

```
Before:   SUSPECTED with consecutive_detections = 1
Event:    Same anomaly detected
Cond:     consecutive_detections < confirmation_cycles
After:    SUSPECTED with consecutive_detections = 2

(With default confirmation_cycles=2, this transition goes to OPEN instead)
```

### SUSPECTED → OPEN (Confirmed!)

**Trigger**: Anomaly confirmed after N consecutive detections

```
Before:   SUSPECTED with consecutive_detections = 1
Event:    Same anomaly detected again
Cond:     consecutive_detections ≥ confirmation_cycles (default: 2)
After:    OPEN - alert sent!

┌─────────────────────────────────────────────────────────────────────┐
│  Transition: SUSPECTED → OPEN                                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  This is the KEY transition - the moment an alert fires             │
│                                                                     │
│  What happens:                                                      │
│    1. Anomaly detected for 2nd consecutive cycle                    │
│    2. Threshold met: consecutive ≥ confirmation_cycles              │
│    3. Status changes: SUSPECTED → OPEN                              │
│    4. newly_confirmed flag set to true (this cycle only)            │
│    5. Alert sent to web API                                         │
│                                                                     │
│  Payload fields change:                                             │
│    status: "SUSPECTED" → "OPEN"                                     │
│    previous_status: "SUSPECTED"                                     │
│    is_confirmed: false → true                                       │
│    confirmation_pending: true → false                               │
│    newly_confirmed: true  ← Important for tracking!                 │
│    consecutive_detections: 2                                        │
│    occurrence_count: 2                                              │
│                                                                     │
│  Web API: Alert SENT (now included in payload)                      │
│                                                                     │
│  Fingerprinting summary:                                            │
│    overall_action: "CONFIRMED"                                      │
│    newly_confirmed_incidents: [this incident]                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### SUSPECTED → CLOSED (Expired Quietly)

**Trigger**: SUSPECTED anomaly not confirmed before grace period ends

```
Before:   SUSPECTED with missed_cycles = 2
Event:    Anomaly NOT detected
Cond:     missed_cycles ≥ resolution_grace_cycles (default: 3)
After:    CLOSED with reason "suspected_expired"

┌─────────────────────────────────────────────────────────────────────┐
│  Transition: SUSPECTED → CLOSED (suspected_expired)                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  This is a SILENT close - no alert was ever sent                    │
│                                                                     │
│  What happens:                                                      │
│    1. Anomaly detected once (SUSPECTED)                             │
│    2. Not detected in next 3 cycles                                 │
│    3. Incident closes without ever alerting                         │
│    4. No resolution sent to web API                                 │
│                                                                     │
│  Why no resolution?                                                 │
│    Web API was never notified of this incident.                     │
│    Sending a resolution would be confusing.                         │
│                                                                     │
│  Timeline example:                                                  │
│    10:00  Detected → SUSPECTED (not sent to API)                    │
│    10:03  Not detected → missed: 1                                  │
│    10:06  Not detected → missed: 2                                  │
│    10:09  Not detected → CLOSED (suspected_expired)                 │
│                                                                     │
│  Result: A transient spike that no one ever knew about              │
│          This is the desired behavior for noise reduction           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### OPEN → OPEN (Continue)

**Trigger**: Anomaly continues to be detected

```
Before:   OPEN with occurrence_count = 5
Event:    Same anomaly detected
After:    OPEN with occurrence_count = 6

┌─────────────────────────────────────────────────────────────────────┐
│  Transition: OPEN → OPEN (continue)                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  What happens:                                                      │
│    1. Anomaly still detected                                        │
│    2. Counters updated                                              │
│    3. missed_cycles reset to 0 (not in grace period)                │
│    4. Duration updated                                              │
│                                                                     │
│  Payload fields update:                                             │
│    incident_action: "CONTINUE"                                      │
│    occurrence_count: ++                                             │
│    consecutive_detections: ++                                       │
│    missed_cycles: 0 (reset)                                         │
│    incident_duration_minutes: updated                               │
│    last_updated: <current timestamp>                                │
│                                                                     │
│  Severity can change:                                               │
│    If anomaly severity changes (e.g., high → critical):             │
│    severity: "high" → "critical"                                    │
│    severity_changed: true                                           │
│    previous_severity: "high"                                        │
│                                                                     │
│  Web API: Continued alerting                                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### OPEN → RECOVERING

**Trigger**: Anomaly not detected, entering grace period

```
Before:   OPEN
Event:    Anomaly NOT detected
After:    RECOVERING with missed_cycles = 1

┌─────────────────────────────────────────────────────────────────────┐
│  Transition: OPEN → RECOVERING                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  What happens:                                                      │
│    1. Anomaly not detected this cycle                               │
│    2. Enter grace period (don't close yet!)                         │
│    3. Watch for anomaly return                                      │
│                                                                     │
│  Why grace period?                                                  │
│    Anomalies often briefly clear before returning:                  │
│    • Brief improvement in latency                                   │
│    • One good measurement among bad                                 │
│    • Flapping behavior                                              │
│                                                                     │
│  Payload fields:                                                    │
│    status: "RECOVERING"                                             │
│    missed_cycles: 1                                                 │
│                                                                     │
│  The fingerprinting summary shows:                                  │
│    status_summary.recovering: 1                                     │
│                                                                     │
│  Web API: No resolution yet                                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### RECOVERING → OPEN (Anomaly Returns)

**Trigger**: Anomaly detected again during grace period

```
Before:   RECOVERING with missed_cycles = 2
Event:    Anomaly detected again
After:    OPEN - back to active alerting

┌─────────────────────────────────────────────────────────────────────┐
│  Transition: RECOVERING → OPEN                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  This is why grace periods exist!                                   │
│                                                                     │
│  What happens:                                                      │
│    1. Anomaly returns before grace period ends                      │
│    2. Resume tracking same incident                                 │
│    3. Reset missed_cycles to 0                                      │
│    4. Increment occurrence_count                                    │
│                                                                     │
│  Without grace period:                                              │
│    - Would have resolved after first miss                           │
│    - Would create NEW incident when anomaly returned                │
│    - Same issue tracked as multiple incidents                       │
│                                                                     │
│  With grace period:                                                 │
│    - Brief clearance doesn't trigger resolution                     │
│    - Same incident continues when anomaly returns                   │
│    - Accurate duration tracking                                     │
│                                                                     │
│  Payload fields:                                                    │
│    status: "RECOVERING" → "OPEN"                                    │
│    missed_cycles: 2 → 0                                             │
│    occurrence_count: ++                                             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### RECOVERING → CLOSED (Resolved)

**Trigger**: Grace period completed without anomaly return

```
Before:   RECOVERING with missed_cycles = 2
Event:    Anomaly NOT detected
Cond:     missed_cycles ≥ resolution_grace_cycles (default: 3)
After:    CLOSED with reason "resolved"

┌─────────────────────────────────────────────────────────────────────┐
│  Transition: RECOVERING → CLOSED                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  What happens:                                                      │
│    1. Anomaly not detected for 3+ consecutive cycles                │
│    2. Grace period completed                                        │
│    3. Incident closed                                               │
│    4. Resolution sent to web API                                    │
│                                                                     │
│  Resolution payload:                                                │
│    fingerprint_id: "anomaly_abc123"                                 │
│    incident_id: "incident_xyz789"                                   │
│    fingerprint_action: "RESOLVE"                                    │
│    incident_action: "CLOSE"                                         │
│    resolution_reason: "resolved"                                    │
│    final_severity: "high"                                           │
│    resolved_at: <current timestamp>                                 │
│    total_occurrences: 8                                             │
│    incident_duration_minutes: 45                                    │
│    first_seen: <original timestamp>                                 │
│                                                                     │
│  Web API: Resolution SENT                                           │
│                                                                     │
│  Fingerprinting summary:                                            │
│    overall_action: "RESOLVE"                                        │
│    resolved_incidents: [this incident]                              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Staleness: Any → CLOSED + SUSPECTED

**Trigger**: Same pattern detected after long gap

```
Before:   OPEN (or RECOVERING) last updated 45 minutes ago
Event:    Anomaly detected
Cond:     gap > incident_separation_minutes (default: 30)
After:    Old incident CLOSED (auto_stale) + New SUSPECTED created

┌─────────────────────────────────────────────────────────────────────┐
│  Staleness Transition                                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  What happens:                                                      │
│    1. Same anomaly pattern detected                                 │
│    2. But time gap exceeds threshold (30 min default)               │
│    3. Old incident auto-closed                                      │
│    4. New incident created from scratch                             │
│                                                                     │
│  Why?                                                               │
│    Long gaps usually indicate separate occurrences:                 │
│    • Incident at 10:00, resolved by 10:30                           │
│    • Same pattern at 15:00 - probably new issue                     │
│    • Should be tracked separately                                   │
│                                                                     │
│  Two things happen in same cycle:                                   │
│                                                                     │
│  1. Old incident closed:                                            │
│     resolved_incidents: [{                                          │
│       incident_id: "incident_old",                                  │
│       resolution_reason: "auto_stale",                              │
│       incident_duration_minutes: 30                                 │
│     }]                                                              │
│                                                                     │
│  2. New incident created:                                           │
│     anomalies: {                                                    │
│       "latency_spike_recent": {                                     │
│         fingerprint_id: "anomaly_abc123",  (same pattern)           │
│         incident_id: "incident_new",       (new UUID)               │
│         status: "SUSPECTED"                                         │
│       }                                                             │
│     }                                                               │
│                                                                     │
│  Web API receives:                                                  │
│    - Resolution for old incident (auto_stale)                       │
│    - NOT the new SUSPECTED (filtered)                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Payload Fields by State

### SUSPECTED State

```json
{
  "fingerprint_id": "anomaly_d18f6ae2bf62",
  "incident_id": "incident_31e9e23d4b2b",
  "fingerprint_action": "CREATE",
  "incident_action": "CREATE",
  "status": "SUSPECTED",
  "consecutive_detections": 1,
  "missed_cycles": 0,
  "occurrence_count": 1,
  "first_seen": "2025-12-17T13:56:06.028585",
  "last_updated": "2025-12-17T13:56:06.028585",
  "incident_duration_minutes": 0,
  "confirmation_pending": true,
  "cycles_to_confirm": 1,
  "is_confirmed": false,
  "newly_confirmed": false
}
```

### OPEN State (Newly Confirmed)

```json
{
  "fingerprint_id": "anomaly_d18f6ae2bf62",
  "incident_id": "incident_31e9e23d4b2b",
  "fingerprint_action": "UPDATE",
  "incident_action": "CONTINUE",
  "status": "OPEN",
  "previous_status": "SUSPECTED",
  "consecutive_detections": 2,
  "missed_cycles": 0,
  "occurrence_count": 2,
  "first_seen": "2025-12-17T13:56:06.028585",
  "last_updated": "2025-12-17T13:59:06.028585",
  "incident_duration_minutes": 3,
  "confirmation_pending": false,
  "is_confirmed": true,
  "newly_confirmed": true
}
```

### OPEN State (Continuing)

```json
{
  "fingerprint_id": "anomaly_d18f6ae2bf62",
  "incident_id": "incident_31e9e23d4b2b",
  "fingerprint_action": "UPDATE",
  "incident_action": "CONTINUE",
  "status": "OPEN",
  "consecutive_detections": 5,
  "missed_cycles": 0,
  "occurrence_count": 5,
  "first_seen": "2025-12-17T13:56:06.028585",
  "last_updated": "2025-12-17T14:08:06.028585",
  "incident_duration_minutes": 12,
  "is_confirmed": true,
  "newly_confirmed": false
}
```

### RECOVERING State

Note: RECOVERING incidents appear in `status_summary.recovering` count but individual anomalies are not in the payload (not detected this cycle).

Fingerprinting summary shows:
```json
{
  "fingerprinting": {
    "overall_action": "NO_CHANGE",
    "status_summary": {
      "suspected": 0,
      "confirmed": 0,
      "recovering": 1
    }
  }
}
```

### CLOSED State (Resolution)

```json
{
  "resolved_incidents": [
    {
      "fingerprint_id": "anomaly_d18f6ae2bf62",
      "incident_id": "incident_31e9e23d4b2b",
      "anomaly_name": "latency_spike_recent",
      "fingerprint_action": "RESOLVE",
      "incident_action": "CLOSE",
      "final_severity": "high",
      "resolved_at": "2025-12-17T14:30:00.000000",
      "total_occurrences": 8,
      "incident_duration_minutes": 34,
      "first_seen": "2025-12-17T13:56:06.028585",
      "service_name": "booking",
      "last_detected_by_model": "business_hours",
      "resolution_reason": "resolved"
    }
  ]
}
```

## Resolution Reasons

| Reason | When | Alert Sent? | Resolution Sent? |
|--------|------|-------------|------------------|
| `resolved` | Grace period completed | Yes (earlier) | Yes |
| `suspected_expired` | SUSPECTED never confirmed | No | No |
| `auto_stale` | Time gap exceeded threshold | Yes (earlier) | Yes |

## Configuration

```json
{
  "fingerprinting": {
    "confirmation_cycles": 2,
    "resolution_grace_cycles": 3,
    "incident_separation_minutes": 30,
    "cleanup_max_age_hours": 72
  }
}
```

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| `confirmation_cycles` | 2 | 1-10 | Higher = fewer false positives, more detection delay |
| `resolution_grace_cycles` | 3 | 1-10 | Higher = fewer flapping alerts, longer resolution time |
| `incident_separation_minutes` | 30 | 5-1440 | Higher = more likely to continue existing incident |
| `cleanup_max_age_hours` | 72 | 1-720 | How long closed incidents stay in database |

## Complete Example: Full Lifecycle

```
Time     Event           State          Actions
─────────────────────────────────────────────────────────────────────
10:00    Detected        SUSPECTED      Create incident_abc, no alert
10:03    Detected        OPEN           ALERT SENT! (confirmed)
10:06    Detected        OPEN           Continue tracking
10:09    Detected        OPEN           Continue tracking (count: 4)
10:12    Not detected    RECOVERING     Grace period starts (missed: 1)
10:15    Detected        OPEN           Resume incident (missed: 0)
10:18    Detected        OPEN           Continue tracking
10:21    Not detected    RECOVERING     Grace period (missed: 1)
10:24    Not detected    RECOVERING     Grace period (missed: 2)
10:27    Not detected    CLOSED         RESOLUTION SENT! (resolved)

Result:
  - 1 alert sent (at 10:03)
  - 1 resolution sent (at 10:27)
  - Duration: 27 minutes
  - Occurrences: 6
```
