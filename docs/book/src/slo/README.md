# SLO Evaluation Layer

The SLO (Service Level Objective) evaluation layer adjusts anomaly severity based on operational thresholds. This chapter explains what SLOs are, why they matter for anomaly detection, and how Yaga2 uses them.

## What is an SLO?

**SLO (Service Level Objective)** is a target level of service reliability that you promise to maintain. It's the answer to: *"What does 'good enough' look like for this service?"*

### The SLI → SLO → SLA Hierarchy

Understanding SLOs requires understanding the full hierarchy:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Service Level Hierarchy                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌───────────────────────────────────────────────────────────┐     │
│   │                        SLA                                │     │
│   │              Service Level Agreement                      │     │
│   │                                                           │     │
│   │   "If we breach this, there are consequences"             │     │
│   │   Example: 99.9% uptime or customer gets credit           │     │
│   │                                                           │     │
│   │   ┌───────────────────────────────────────────────────┐   │     │
│   │   │                      SLO                          │   │     │
│   │   │            Service Level Objective                │   │     │
│   │   │                                                   │   │     │
│   │   │   "This is our internal target"                   │   │     │
│   │   │   Example: 99.95% uptime (stricter than SLA)      │   │     │
│   │   │                                                   │   │     │
│   │   │   ┌───────────────────────────────────────────┐   │   │     │
│   │   │   │                  SLI                      │   │   │     │
│   │   │   │        Service Level Indicator            │   │   │     │
│   │   │   │                                           │   │   │     │
│   │   │   │   "This is what we measure"               │   │   │     │
│   │   │   │   Example: Request success rate           │   │   │     │
│   │   │   │                                           │   │   │     │
│   │   │   └───────────────────────────────────────────┘   │   │     │
│   │   └───────────────────────────────────────────────────┘   │     │
│   └───────────────────────────────────────────────────────────┘     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

| Term | Definition | Example |
|------|------------|---------|
| **SLI** (Indicator) | The metric you measure | Request latency in milliseconds |
| **SLO** (Objective) | The target value for that metric | 99% of requests < 500ms |
| **SLA** (Agreement) | Legal/business commitment with consequences | 99.9% uptime or refund |

### Common SLO Types

| SLO Type | What It Measures | Example Target |
|----------|------------------|----------------|
| **Availability** | Is the service responding? | 99.9% of requests succeed |
| **Latency** | How fast does it respond? | 95% of requests < 300ms |
| **Error Rate** | How many requests fail? | < 0.1% error rate |
| **Throughput** | How much can it handle? | 1000 req/s capacity |

### Why SLOs Matter

Without SLOs, you have two problems:

**Problem 1: Over-alerting**
```
❌ Static threshold: Alert if latency > 200ms
   ↓
   You get 50 alerts per day for "normal" spikes
   ↓
   Alert fatigue → Team ignores alerts
   ↓
   Real incidents get missed
```

**Problem 2: Under-alerting**
```
❌ ML-only detection: Alert when "unusual"
   ↓
   Latency at 400ms is unusual (normally 100ms)
   ↓
   But users don't notice 400ms latency
   ↓
   Alert noise for non-issues
```

**The SLO Solution:**
```
✅ SLO-aware detection:
   ↓
   ML detects 400ms latency as unusual
   ↓
   SLO evaluation: 400ms < 500ms acceptable threshold
   ↓
   Severity adjusted to "low" (no page, just log)
   ↓
   Team focuses on real problems
```

## Why SLOs Matter for Anomaly Detection

ML-based anomaly detection answers: **"Is this unusual?"**

But unusual doesn't mean bad:
- A traffic spike during a sale is unusual but good
- Latency at 300ms when it's usually 100ms is unusual but might be acceptable
- Error rate dropping to 0% is unusual but definitely not a problem

SLO evaluation answers a different question: **"Does this impact users?"**

### The Two-Question Model

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Anomaly Evaluation                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Question 1: Is this unusual?                                      │
│   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━                                      │
│   Answered by: Machine Learning (Isolation Forest)                  │
│   Method: Compare against historical baseline                       │
│   Output: "Yes, this is 2σ above normal"                           │
│                                                                     │
│                         │                                           │
│                         ▼                                           │
│                                                                     │
│   Question 2: Does it matter operationally?                         │
│   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                          │
│   Answered by: SLO Evaluation                                       │
│   Method: Compare against operational thresholds                    │
│   Output: "No, still within acceptable SLO"                        │
│                                                                     │
│                         │                                           │
│                         ▼                                           │
│                                                                     │
│   Final Decision                                                    │
│   ━━━━━━━━━━━━━━━                                                   │
│   Unusual: ✓ Yes                                                    │
│   Impactful: ✗ No                                                   │
│   Action: Log for awareness, don't page                            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Example: The 280ms Latency Case

Consider a booking service:

| Metric | Value |
|--------|-------|
| Current latency | 280ms |
| Historical baseline | 150ms |
| Statistical deviation | +2.3σ (p91 percentile) |

**ML-only approach:**
```
ML says: "280ms is unusual! That's 2.3σ above normal!"
Result: HIGH severity alert
Problem: Engineer pages at 3 AM for non-issue
```

**SLO-aware approach:**
```
ML says: "280ms is unusual"
SLO says: "280ms < 300ms acceptable threshold"
SLO says: "280ms is 93% toward the limit (proximity = 0.93)"
Result: LOW severity (logged but no page)
Benefit: Engineer sleeps, system still tracks the deviation
```

## How Yaga2's SLO Layer Works

### Evaluation Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SLO Evaluation Pipeline                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Input: ML Detection Result                                        │
│   ┌────────────────────────────────────┐                           │
│   │ pattern: latency_spike_recent      │                           │
│   │ severity: high                     │                           │
│   │ score: -0.35                       │                           │
│   │ metrics:                           │                           │
│   │   latency: 280ms                   │                           │
│   │   error_rate: 0.1%                 │                           │
│   │   request_rate: 150/s              │                           │
│   └─────────────────┬──────────────────┘                           │
│                     │                                               │
│                     ▼                                               │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │              Evaluate Each SLO Component                    │   │
│   ├──────────────┬──────────────┬─────────────┬────────────────┤   │
│   │   Latency    │  Error Rate  │  DB Latency │   Traffic      │   │
│   │   280ms      │   0.1%       │  2.5ms      │   150/s        │   │
│   │   vs 300ms   │   vs 0.5%    │  vs 5ms     │   vs 50/s      │   │
│   │   ↓          │   ↓          │   ↓         │   ↓            │   │
│   │   ok (93%)   │   ok (20%)   │  ok (below  │   ok (3x but   │   │
│   │              │              │     floor)  │   no impact)   │   │
│   └──────┬───────┴──────┬───────┴──────┬──────┴───────┬────────┘   │
│          │              │              │              │             │
│          └──────────────┴──────────────┴──────────────┘             │
│                         │                                           │
│                         ▼                                           │
│              ┌─────────────────────┐                               │
│              │  Combined Status    │                               │
│              │  ─────────────────  │                               │
│              │  Status: ok         │                               │
│              │  Proximity: 0.93    │  (closest to breach)          │
│              │  Impact: none       │                               │
│              └──────────┬──────────┘                               │
│                         │                                           │
│                         ▼                                           │
│              ┌─────────────────────┐                               │
│              │  Severity Adjust    │                               │
│              │  ─────────────────  │                               │
│              │  Original: high     │                               │
│              │  Adjusted: low      │  (SLO ok → low)               │
│              │  Changed: true      │                               │
│              └─────────────────────┘                               │
│                                                                     │
│   Output: Adjusted Detection Result                                 │
│   ┌────────────────────────────────────┐                           │
│   │ pattern: latency_spike_recent      │                           │
│   │ severity: low                      │  ← adjusted               │
│   │ original_severity: high            │  ← preserved              │
│   │ slo_status: ok                     │                           │
│   │ slo_proximity: 0.93                │                           │
│   │ explanation: "Within acceptable    │                           │
│   │   SLO thresholds"                  │                           │
│   └────────────────────────────────────┘                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### SLO Components in Yaga2

| Component | Threshold Type | Evaluation Method |
|-----------|---------------|-------------------|
| [Latency](./latency.md) | Absolute (ms) | Compare against acceptable/warning/critical thresholds |
| [Error Rate](./error-rate.md) | Absolute (%) | Compare against acceptable/warning/critical with floor suppression |
| [Database Latency](./database-latency.md) | Ratio-based | Compare ratio against training baseline (1.5x, 2x, 3x, 5x) |
| [Request Rate](./request-rate.md) | Ratio-based | Detect surge (≥200%) or cliff (≤50%) with correlation |

### SLO Status

| Status | What It Means | Severity Impact |
|--------|---------------|-----------------|
| `ok` | All metrics within acceptable limits | Severity → `low` |
| `elevated` | Above acceptable, below warning | Severity stays as-is |
| `warning` | Approaching SLO breach | Severity → `high` |
| `breached` | SLO threshold exceeded | Severity → `critical` |

### Severity Adjustment Rules

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Severity Adjustment Matrix                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Original Severity    SLO Status      Final Severity               │
│   ─────────────────    ──────────      ──────────────               │
│                                                                     │
│   critical             ok              → low                        │
│   critical             elevated        → high                       │
│   critical             warning         → critical                   │
│   critical             breached        → critical                   │
│                                                                     │
│   high                 ok              → low                        │
│   high                 elevated        → high                       │
│   high                 warning         → high                       │
│   high                 breached        → critical                   │
│                                                                     │
│   medium               ok              → low                        │
│   medium               warning         → high                       │
│                                                                     │
│   low                  any             → low                        │
│                                                                     │
│   (any)                breached        → critical (always)          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Key Insight:** When SLO status is `ok`, severity is **always** adjusted to `low`, regardless of the original ML-assigned severity. This ensures alerts reflect operational impact, not just statistical deviation.

## Practical Examples

### Example 1: Acceptable Anomaly

```
Service: booking
─────────────────────────────────────────────────────────────

Current Metrics:
  latency: 420ms (usually 200ms)
  error_rate: 0.2%

ML Detection:
  Pattern: latency_spike_recent
  Severity: high
  Reason: Latency is 3σ above baseline

SLO Evaluation:
  Latency: 420ms < 500ms acceptable ✓
  Errors: 0.2% < 0.5% acceptable ✓
  Status: ok

Final Decision:
  Severity: low
  Action: Log for awareness, no page
  Message: "Anomaly detected but within SLO"
```

### Example 2: Warning Zone

```
Service: search
─────────────────────────────────────────────────────────────

Current Metrics:
  latency: 750ms (acceptable: 500ms, warning: 800ms)
  error_rate: 0.8% (acceptable: 0.5%, warning: 1%)

ML Detection:
  Pattern: latency_spike_recent
  Severity: high

SLO Evaluation:
  Latency: 750ms > 500ms acceptable, < 800ms warning
  Errors: 0.8% > 0.5% acceptable, < 1% warning
  Status: warning
  Proximity: 0.94 (close to warning threshold)

Final Decision:
  Severity: high (maintained)
  Action: Alert on-call for investigation
  Message: "Approaching SLO limits"
```

### Example 3: SLO Breach

```
Service: checkout
─────────────────────────────────────────────────────────────

Current Metrics:
  latency: 2500ms (critical: 1000ms)
  error_rate: 5%

ML Detection:
  Pattern: traffic_surge_failing
  Severity: critical

SLO Evaluation:
  Latency: 2500ms > 1000ms critical ✗
  Errors: 5% > 2% critical ✗
  Status: breached

Final Decision:
  Severity: critical (confirmed)
  Action: Immediate page, potential incident
  Message: "SLO breached - users impacted"
```

## Configuration

### Default SLO Thresholds

```json
{
  "slos": {
    "enabled": true,
    "allow_downgrade_to_informational": true,
    "require_slo_breach_for_critical": true,
    "defaults": {
      "latency_acceptable_ms": 500,
      "latency_warning_ms": 800,
      "latency_critical_ms": 1000,
      "error_rate_acceptable": 0.005,
      "error_rate_warning": 0.01,
      "error_rate_critical": 0.02,
      "error_rate_floor": 0.001,
      "database_latency_floor_ms": 5.0,
      "database_latency_ratios": {
        "info": 1.5,
        "warning": 2.0,
        "high": 3.0,
        "critical": 5.0
      }
    }
  }
}
```

### Per-Service Overrides

Different services have different requirements:

```json
{
  "slos": {
    "services": {
      "checkout": {
        "latency_acceptable_ms": 300,
        "latency_critical_ms": 500,
        "error_rate_acceptable": 0.001
      },
      "search": {
        "latency_acceptable_ms": 200,
        "latency_critical_ms": 400
      },
      "admin-panel": {
        "latency_acceptable_ms": 2000,
        "error_rate_acceptable": 0.01
      }
    }
  }
}
```

### Busy Period Configuration

During high-traffic periods (holidays, sales), thresholds can be automatically relaxed:

```json
{
  "slos": {
    "defaults": {
      "busy_period_factor": 1.5
    },
    "busy_periods": [
      {
        "start": "2024-12-20T00:00:00",
        "end": "2025-01-05T23:59:59"
      }
    ]
  }
}
```

During busy periods:
- 500ms acceptable → 750ms acceptable
- 0.5% error acceptable → 0.75% error acceptable

## Best Practices

### 1. Set SLOs Based on User Experience

```
❌ Bad: "Let's set latency SLO at 100ms because it sounds good"
   Problem: Your P50 latency is already 150ms

✓ Good: "Users report issues above 500ms, let's set acceptable at 400ms"
   Benefit: SLO reflects actual user impact
```

### 2. Use the Error Budget Concept

If your SLO is 99.9% availability:
- Error budget = 0.1% of requests can fail
- Monthly budget ≈ 43 minutes of downtime
- Use this to decide when to deploy vs stabilize

### 3. Have Different SLOs for Different Services

| Service | Latency SLO | Why |
|---------|-------------|-----|
| Checkout | 300ms | Revenue-critical, users waiting |
| Search | 200ms | User experience, interactive |
| Admin Panel | 2000ms | Internal tool, less critical |
| Batch Jobs | 30000ms | Background, no users waiting |

### 4. Review and Adjust SLOs Quarterly

- Analyze false positive rate
- Check if SLOs match user complaints
- Tighten SLOs as systems improve

## Further Reading

- [Latency Evaluation](./latency.md) - Response time SLO checking
- [Error Rate Evaluation](./error-rate.md) - Error percentage SLO checking
- [Database Latency](./database-latency.md) - Ratio-based DB latency evaluation
- [Request Rate](./request-rate.md) - Traffic surge and cliff detection
- [Severity Adjustment](./severity-adjustment.md) - How severity is adjusted based on SLO status
