# Fingerprinting

Fingerprinting assigns stable identifiers to anomaly patterns, enabling tracking across detection cycles. This chapter explains the identification system, how it works, and why two different types of IDs are needed.

## What is Fingerprinting?

**Fingerprinting** is the process of creating stable, reproducible identifiers for anomaly patterns. When the system detects an anomaly, it generates a "fingerprint" - a unique identifier based on what the anomaly is, not when it happened.

Think of it like identifying a person:
- **Fingerprint** = A person's actual fingerprint (uniquely identifies the person)
- **Incident** = A specific encounter with that person (when and where you met them)

```
                   Same Person (Fingerprint)
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
   Meeting #1          Meeting #2          Meeting #3
   (Jan 10, Coffee)    (Jan 15, Office)    (Jan 22, Park)

Each meeting is a unique incident, but it's the same person.
```

## Why Two ID Types?

The system uses two different identifiers that serve complementary purposes:

| ID Type | What It Identifies | Characteristics |
|---------|-------------------|-----------------|
| **Fingerprint ID** | The anomaly pattern itself | Stable, deterministic, content-based |
| **Incident ID** | A specific occurrence of the pattern | Unique, random, time-bound |

### Why Not Just Use One ID?

Using only one ID would create problems:

**If only fingerprint_id:**
```
Problem: Can't distinguish between separate occurrences

Jan 10  latency_spike (fingerprint_abc) - Incident A
Jan 15  latency_spike (fingerprint_abc) - Is this Incident A continuing?
                                          Or a new incident?
                                          We can't tell!
```

**If only incident_id:**
```
Problem: Can't tell if issues are related

Jan 10  incident_123 (latency spike)
Jan 15  incident_456 (latency spike) - Are these the same type of issue?
                                        Different issue types?
                                        We can't tell!
```

**With both IDs:**
```
Jan 10  fingerprint_abc / incident_123 (latency spike)
Jan 15  fingerprint_abc / incident_456 (latency spike)

Now we know:
- Same TYPE of anomaly (same fingerprint_abc)
- Different OCCURRENCES (different incident IDs)
- We can track patterns AND individual events
```

## Fingerprint ID: Pattern Identity

### What Is It?

A **deterministic hash** based on the anomaly's content - what it is, not when it happened.

### How It's Generated

```python
# The fingerprint is generated from these components:
content = f"{service_name}_{model_name}_{anomaly_name}"
fingerprint_id = f"anomaly_{sha256(content)[:12]}"
```

**Example:**
```
Components:
  service_name: booking
  model_name: business_hours
  anomaly_name: latency_spike_recent

Concatenated: "booking_business_hours_latency_spike_recent"
SHA256 hash:  d18f6ae2bf62a7c9...
Fingerprint:  anomaly_d18f6ae2bf62

Same inputs ALWAYS produce the same fingerprint!
```

### Properties

| Property | Description | Example |
|----------|-------------|---------|
| **Deterministic** | Same pattern always gets same ID | `booking_business_hours_latency_spike_recent` → always `anomaly_d18f6ae2bf62` |
| **Content-based** | Based on what, not when | Doesn't include timestamp |
| **Stable** | Doesn't change across time | Same ID today, tomorrow, next year |
| **Reproducible** | Can be regenerated | Given the same inputs, always same output |

### What Changes the Fingerprint?

The fingerprint changes when the pattern type changes:

```
SAME fingerprint (anomaly_abc):
─────────────────────────────
• booking + business_hours + latency_spike_recent (Monday)
• booking + business_hours + latency_spike_recent (Tuesday)
• booking + business_hours + latency_spike_recent (Different severity)

DIFFERENT fingerprint (anomaly_xyz):
───────────────────────────────────
• booking + business_hours + traffic_surge_failing (different anomaly type)
• booking + evening_hours + latency_spike_recent (different time period)
• search + business_hours + latency_spike_recent (different service)
```

## Incident ID: Occurrence Identity

### What Is It?

A **unique identifier** for each specific occurrence of an anomaly pattern.

### How It's Generated

```python
# The incident ID is generated randomly when a new incident is created:
incident_id = f"incident_{uuid4().hex[:16]}"
```

**Example:**
```
Each new incident gets a new random ID:
  incident_1dcbafc91480
  incident_7a23b9f4c8e1
  incident_95d2e61a8b43

Even for the same pattern type, each occurrence has a unique ID!
```

### Properties

| Property | Description | Example |
|----------|-------------|---------|
| **Unique** | Each occurrence gets its own ID | No two incidents share an ID |
| **UUID-based** | Random generation | Not predictable from inputs |
| **Transient** | New ID when pattern reappears | After resolution, next occurrence = new ID |
| **Time-bound** | Associated with specific time period | Tracks one continuous occurrence |

### When Is a New Incident ID Created?

```
New incident_id created when:
────────────────────────────
• Anomaly detected for first time (no active incident)
• Anomaly reappears after being resolved
• Anomaly reappears after staleness threshold (>30 min gap)

Same incident_id continues when:
─────────────────────────────────
• Anomaly detected while incident is OPEN
• Anomaly detected while incident is RECOVERING
• Anomaly detected within staleness threshold
```

## The Relationship Between IDs

```
┌─────────────────────────────────────────────────────────────────────┐
│                 Fingerprint: anomaly_d18f6ae2bf62                   │
│                 (Pattern: booking latency spike)                    │
├─────────────────────────────────────────────────────────────────────┤
│                              │                                      │
│                              │                                      │
│    ┌─────────────────────────┼─────────────────────────┐           │
│    │                         │                         │           │
│    ▼                         ▼                         ▼           │
│ ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐    │
│ │   Incident A    │   │   Incident B    │   │   Incident C    │    │
│ │ incident_abc123 │   │ incident_def456 │   │ incident_ghi789 │    │
│ ├─────────────────┤   ├─────────────────┤   ├─────────────────┤    │
│ │ Jan 10, 10:00   │   │ Jan 15, 14:00   │   │ Jan 22, 09:00   │    │
│ │ Duration: 45min │   │ Duration: 2hr   │   │ Duration: 30min │    │
│ │ Status: CLOSED  │   │ Status: CLOSED  │   │ Status: OPEN    │    │
│ │ Occurrences: 8  │   │ Occurrences: 24 │   │ Occurrences: 5  │    │
│ └─────────────────┘   └─────────────────┘   └─────────────────┘    │
│                                                                     │
│ All three incidents represent the SAME type of anomaly              │
│ (same fingerprint) but are SEPARATE occurrences (different IDs)     │
└─────────────────────────────────────────────────────────────────────┘
```

### What Each ID Enables

| Capability | Fingerprint ID | Incident ID |
|------------|----------------|-------------|
| Track same issue across time | ✅ | ❌ |
| Identify individual occurrences | ❌ | ✅ |
| Correlate related events | ✅ | ❌ |
| Calculate MTTR per incident | ❌ | ✅ |
| Find recurring patterns | ✅ | ❌ |
| Link alert to specific event | ❌ | ✅ |

## How Fingerprinting Works

### The Complete Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Fingerprinting Process                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Step 1: Detection Result                                          │
│   ─────────────────────────                                         │
│   anomaly: latency_spike_recent                                     │
│   service: booking                                                  │
│   model: business_hours                                             │
│                                                                     │
│                          │                                          │
│                          ▼                                          │
│                                                                     │
│   Step 2: Generate Fingerprint ID                                   │
│   ──────────────────────────────                                    │
│   hash("booking_business_hours_latency_spike_recent")               │
│   → anomaly_d18f6ae2bf62                                            │
│                                                                     │
│                          │                                          │
│                          ▼                                          │
│                                                                     │
│   Step 3: Database Lookup                                           │
│   ───────────────────────                                           │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  SELECT * FROM anomaly_incidents                            │   │
│   │  WHERE fingerprint_id = 'anomaly_d18f6ae2bf62'              │   │
│   │    AND status IN ('SUSPECTED', 'OPEN', 'RECOVERING')        │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│                          │                                          │
│                    ┌─────┴─────┐                                    │
│                    │           │                                    │
│               NOT FOUND      FOUND                                  │
│                    │           │                                    │
│                    ▼           ▼                                    │
│                                                                     │
│   Step 4a: Create New           Step 4b: Update Existing            │
│   ───────────────────           ───────────────────────             │
│   • Generate new incident_id    • Keep same incident_id             │
│   • Set status = SUSPECTED      • Check staleness                   │
│   • Initialize counters         • Increment occurrence_count        │
│   • Set first_seen = now        • Update last_updated               │
│                                 • Handle state transitions          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Decision Tree

```
                    ┌───────────────────────┐
                    │   Anomaly Detected    │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │ Generate fingerprint  │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │ Active incident with  │
              ┌─────│ this fingerprint?     │─────┐
              │     └───────────────────────┘     │
              │NO                                 │YES
              │                                   │
              ▼                                   ▼
    ┌─────────────────────┐           ┌─────────────────────┐
    │   CREATE incident   │           │  Is it stale?       │
    │   fingerprint_action│           │  (gap > 30 min)     │
    │   = CREATE          │           └──────────┬──────────┘
    │   incident_action   │                      │
    │   = CREATE          │             ┌────────┴────────┐
    └─────────────────────┘             │YES              │NO
                                        │                 │
                                        ▼                 ▼
                              ┌─────────────────┐  ┌─────────────────┐
                              │ Close stale     │  │ UPDATE incident │
                              │ Create new      │  │ fingerprint_    │
                              │                 │  │ action = UPDATE │
                              │ Two actions:    │  │ incident_action │
                              │ RESOLVE + CREATE│  │ = CONTINUE      │
                              └─────────────────┘  └─────────────────┘
```

## Database Schema

The fingerprinting system uses SQLite for persistence:

```sql
CREATE TABLE anomaly_incidents (
    -- Identity
    fingerprint_id TEXT NOT NULL,          -- Pattern identifier
    incident_id TEXT PRIMARY KEY,          -- Occurrence identifier

    -- Context
    service_name TEXT NOT NULL,            -- Service (e.g., "booking")
    anomaly_name TEXT NOT NULL,            -- Pattern name
    detected_by_model TEXT,                -- Time period model

    -- State
    status TEXT NOT NULL,                  -- SUSPECTED, OPEN, RECOVERING, CLOSED
    severity TEXT NOT NULL,                -- low, medium, high, critical

    -- Timestamps
    first_seen TIMESTAMP NOT NULL,         -- When incident started
    last_updated TIMESTAMP NOT NULL,       -- Most recent detection
    resolved_at TIMESTAMP NULL,            -- When closed (null if active)

    -- Counters
    occurrence_count INTEGER NOT NULL,     -- Total detections
    consecutive_detections INTEGER NOT NULL, -- In a row (for confirmation)
    missed_cycles INTEGER NOT NULL,        -- Not detected (for grace period)

    -- Optional data
    current_value REAL,                    -- Current metric value
    threshold_value REAL,                  -- Threshold if applicable
    confidence_score REAL,                 -- ML confidence
    detection_method TEXT,                 -- How it was detected
    description TEXT,                      -- Human-readable description
    metadata TEXT                          -- JSON for extra data
);

-- Indexes for fast lookups
CREATE INDEX idx_fingerprint_status ON anomaly_incidents(fingerprint_id, status);
CREATE INDEX idx_service_timeline ON anomaly_incidents(service_name, first_seen DESC);
CREATE INDEX idx_active_incidents ON anomaly_incidents(status, last_updated DESC)
    WHERE status IN ('SUSPECTED', 'OPEN', 'RECOVERING');
```

### Why SQLite?

| Advantage | Description |
|-----------|-------------|
| **Simple** | No external dependencies |
| **Reliable** | ACID transactions built-in |
| **Fast** | Excellent for read-heavy workloads |
| **Portable** | Single file, easy to backup |
| **Lightweight** | Minimal resource usage |

For most deployments, SQLite handles the expected volume easily (< 1000 active incidents).

## Staleness Check

### What Is Staleness?

When the time gap between detections exceeds a threshold, the system considers the incident "stale" - meaning it's probably not the same ongoing issue, even if it's the same pattern type.

### Why Staleness Matters

Without staleness check:
```
10:00  Latency spike detected → incident_123 created
10:03  Latency spike detected → incident_123 continues
10:06  Latency spike resolved

(long gap - different issue)

15:00  Latency spike detected → incident_123 continues??? NO!
       This is likely a different issue, not the same one from 5 hours ago!
```

With staleness check:
```
10:00  Latency spike detected → incident_123 created
10:03  Latency spike detected → incident_123 continues
10:06  Latency spike resolved

(long gap - different issue)

15:00  Latency spike detected
       Gap check: 15:00 - 10:06 = 4h 54min > 30min threshold
       Action: Close incident_123 as "auto_stale", create incident_456
       Now we have two separate incidents (correct!)
```

### Staleness Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Staleness Check                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Inputs:                                                           │
│   • last_updated: 10:00:00 (from database)                          │
│   • current_time: 11:15:00                                          │
│   • incident_separation_minutes: 30 (config)                        │
│                                                                     │
│   Calculation:                                                      │
│   ─────────────                                                     │
│   gap = current_time - last_updated                                 │
│   gap = 11:15 - 10:00 = 75 minutes                                  │
│                                                                     │
│   is_stale = gap > incident_separation_minutes                      │
│   is_stale = 75 > 30 = TRUE                                         │
│                                                                     │
│   Result:                                                           │
│   ────────                                                          │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  Old incident closed:                                       │   │
│   │    incident_action: CLOSE                                   │   │
│   │    resolution_reason: "auto_stale"                          │   │
│   │                                                             │   │
│   │  New incident created:                                      │   │
│   │    incident_id: incident_new456                             │   │
│   │    incident_action: CREATE                                  │   │
│   │    status: SUSPECTED                                        │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Resolution Reasons

| Reason | Description | When It Happens |
|--------|-------------|-----------------|
| `resolved` | Normal resolution | Anomaly cleared after grace period |
| `auto_stale` | Stale incident closed | Time gap exceeded threshold, pattern reappeared |
| `suspected_expired` | Never confirmed | SUSPECTED state expired without confirmation |

## Per-Anomaly Fingerprinting Fields

Each anomaly in the output includes comprehensive fingerprinting metadata:

```json
{
  "anomalies": {
    "latency_spike_recent": {
      "type": "consolidated",
      "severity": "high",
      "description": "Latency spike: 450ms (normally 120ms)",

      "fingerprint_id": "anomaly_d18f6ae2bf62",
      "fingerprint_action": "UPDATE",

      "incident_id": "incident_31e9e23d4b2b",
      "incident_action": "CONTINUE",

      "status": "OPEN",
      "previous_status": "OPEN",

      "incident_duration_minutes": 15,
      "first_seen": "2025-12-17T13:45:00",
      "last_updated": "2025-12-17T14:00:00",

      "occurrence_count": 5,
      "consecutive_detections": 5,

      "is_confirmed": true,
      "newly_confirmed": false
    }
  }
}
```

### Field Reference

| Field | Type | Description |
|-------|------|-------------|
| `fingerprint_id` | string | Pattern identifier (deterministic hash) |
| `fingerprint_action` | string | Pattern action: CREATE, UPDATE, RESOLVE |
| `incident_id` | string | Occurrence identifier (random UUID) |
| `incident_action` | string | Incident action: CREATE, CONTINUE, CLOSE |
| `status` | string | Current status: SUSPECTED, OPEN, RECOVERING, CLOSED |
| `previous_status` | string | Status before this cycle |
| `incident_duration_minutes` | integer | Time since first_seen |
| `first_seen` | datetime | When this incident started |
| `last_updated` | datetime | Most recent detection time |
| `occurrence_count` | integer | Total times detected |
| `consecutive_detections` | integer | Detections in a row |
| `is_confirmed` | boolean | True if status is OPEN |
| `newly_confirmed` | boolean | True if just confirmed this cycle |

## Action Types

### Fingerprint Actions

The fingerprint action describes what happened to the pattern tracking:

| Action | Description | When |
|--------|-------------|------|
| `CREATE` | New pattern encountered | First time this exact pattern is seen |
| `UPDATE` | Existing pattern updated | Pattern detected, already being tracked |
| `RESOLVE` | Pattern no longer active | Grace period exceeded, pattern cleared |

### Incident Actions

The incident action describes what happened to the specific occurrence:

| Action | Description | When |
|--------|-------------|------|
| `CREATE` | New incident created | New occurrence of a pattern (including after stale) |
| `CONTINUE` | Incident continues | Same occurrence still active |
| `CLOSE` | Incident closed | Occurrence resolved (grace period exceeded) |

### Combined Actions

The combination tells the full story:

| fingerprint_action | incident_action | Meaning |
|--------------------|-----------------|---------|
| CREATE | CREATE | Brand new pattern, new incident |
| UPDATE | CONTINUE | Ongoing issue, same incident |
| UPDATE | CREATE | Same pattern reappeared (was stale or resolved) |
| RESOLVE | CLOSE | Pattern cleared, incident resolved |

## Resolution Payload

When an incident is resolved, the system sends detailed resolution information:

```json
{
  "fingerprinting": {
    "resolved_incidents": [
      {
        "fingerprint_id": "anomaly_d18f6ae2bf62",
        "incident_id": "incident_31e9e23d4b2b",
        "anomaly_name": "latency_spike_recent",

        "fingerprint_action": "RESOLVE",
        "incident_action": "CLOSE",

        "final_severity": "high",
        "resolved_at": "2025-12-17T14:30:00",
        "total_occurrences": 8,
        "incident_duration_minutes": 45,
        "first_seen": "2025-12-17T13:45:00",
        "service_name": "booking",
        "last_detected_by_model": "business_hours",
        "resolution_reason": "resolved"
      }
    ]
  }
}
```

### Resolution Fields

| Field | Type | Description |
|-------|------|-------------|
| `fingerprint_id` | string | Pattern that was resolved |
| `incident_id` | string | Specific occurrence that closed |
| `anomaly_name` | string | Human-readable pattern name |
| `final_severity` | string | Severity at time of resolution |
| `resolved_at` | datetime | When the incident was closed |
| `total_occurrences` | integer | How many times detected during incident |
| `incident_duration_minutes` | integer | Total duration |
| `first_seen` | datetime | When it started |
| `service_name` | string | Affected service |
| `resolution_reason` | string | Why it was closed |

## Database Cleanup

To prevent unbounded database growth, closed incidents are automatically cleaned up:

```json
{
  "fingerprinting": {
    "cleanup_max_age_hours": 72
  }
}
```

### Cleanup Process

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Cleanup Process                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Every cleanup run:                                                │
│   ──────────────────                                                │
│                                                                     │
│   1. Find CLOSED incidents older than cleanup_max_age_hours         │
│                                                                     │
│   DELETE FROM anomaly_incidents                                     │
│   WHERE status = 'CLOSED'                                           │
│     AND resolved_at < (NOW - cleanup_max_age_hours)                 │
│                                                                     │
│   2. Leave active incidents untouched                               │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  SUSPECTED incidents: KEPT (may still confirm)             │   │
│   │  OPEN incidents:      KEPT (active issue)                  │   │
│   │  RECOVERING incidents: KEPT (may resume or close)          │   │
│   │  CLOSED incidents:    DELETED if older than 72h            │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│   Result: Database stays bounded while preserving active state      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Why 72 Hours?

| Shorter (24h) | Longer (168h/1 week) |
|---------------|----------------------|
| Less storage | More history |
| Faster queries | Slower queries |
| Less post-incident analysis | Better trend analysis |
| Risk of losing relevant history | More database bloat |

72 hours (3 days) is a balance that:
- Covers most incident review periods
- Allows weekend incident analysis on Monday
- Keeps database size manageable

## Configuration Reference

All fingerprinting settings in `config.json`:

```json
{
  "fingerprinting": {
    "db_path": "./anomaly_state.db",
    "confirmation_cycles": 2,
    "resolution_grace_cycles": 3,
    "incident_separation_minutes": 30,
    "cleanup_max_age_hours": 72
  }
}
```

| Setting | Default | Description |
|---------|---------|-------------|
| `db_path` | `./anomaly_state.db` | Path to SQLite database |
| `confirmation_cycles` | 2 | Detections required before alerting |
| `resolution_grace_cycles` | 3 | Non-detections before resolving |
| `incident_separation_minutes` | 30 | Gap threshold for staleness |
| `cleanup_max_age_hours` | 72 | Age threshold for cleanup |

## Summary

Fingerprinting provides the foundation for intelligent incident tracking:

1. **Fingerprint ID** enables pattern recognition across time
2. **Incident ID** enables precise occurrence tracking
3. **Staleness check** prevents confusion between separate issues
4. **Database persistence** maintains state across restarts
5. **Automatic cleanup** keeps the system healthy

Together, these mechanisms enable Yaga2 to:
- Track recurring issues
- Calculate accurate metrics (MTTR, frequency)
- Prevent duplicate alerts
- Provide complete incident lifecycle visibility
