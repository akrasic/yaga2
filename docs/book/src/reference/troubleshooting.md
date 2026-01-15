# Troubleshooting

This guide helps you diagnose and resolve common issues with the anomaly detection system. Each section follows a systematic approach: identify symptoms, understand root causes, and apply targeted solutions.

## How to Use This Guide

When troubleshooting:

1. **Identify the symptom** - What behavior are you observing?
2. **Gather evidence** - Collect relevant logs and configuration
3. **Understand the cause** - Why is this happening?
4. **Apply the fix** - Make targeted changes
5. **Verify the solution** - Confirm the issue is resolved

```
┌─────────────────────────────────────────────────────────────────┐
│                   TROUBLESHOOTING WORKFLOW                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│    Symptom                                                      │
│       │                                                         │
│       ▼                                                         │
│    Which category?                                              │
│       │                                                         │
│       ├──▶ Detection Issues (no alerts, false positives)        │
│       ├──▶ Incident Lifecycle (confirmation, resolution)        │
│       ├──▶ Metrics Issues (unavailable, validation)             │
│       ├──▶ SLO Issues (evaluation, thresholds)                  │
│       ├──▶ Training Issues (failures, drift)                    │
│       └──▶ Container Issues (restarts, resources)               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Detection Issues

### No Anomalies Detected (When Expected)

**Symptoms**:
- Real incidents are occurring but the system generates no alerts
- Dashboards show problems but Yaga2 reports `alert_type: "no_anomaly"`
- Users report issues but your alerting is silent

**Why This Happens**:

The detection pipeline has multiple stages where anomalies can be "filtered out":

```
Metrics Collection → ML Detection → Pattern Matching → SLO Evaluation → Alerting
        │                 │                │                 │
        │                 │                │                 │
     No data?         No model?      No pattern?      SLO suppressed?
     (fail here)      (fail here)    (no match)       (low severity)
```

**Diagnostic Decision Tree**:

```
No alerts generated?
        │
        ├─▶ Are metrics being collected?
        │       │
        │       ├─ NO ──▶ Check VictoriaMetrics connectivity
        │       │         (See: metrics_unavailable section)
        │       │
        │       └─ YES ─▶ Do models exist for this service?
        │                       │
        │                       ├─ NO ──▶ Train models
        │                       │
        │                       └─ YES ─▶ Is the service in config?
        │                                       │
        │                                       ├─ NO ──▶ Add to config
        │                                       │
        │                                       └─ YES ─▶ Check contamination
        │                                                 and SLO settings
```

**Step 1: Verify models exist**:

```bash
# Check if models directory exists for your service
ls -la smartbox_models/<service-name>/

# Expected output shows time period subdirectories:
# drwxr-xr-x  business_hours/
# drwxr-xr-x  evening_hours/
# drwxr-xr-x  night_hours/
# drwxr-xr-x  weekend_day/
# drwxr-xr-x  weekend_night/
```

If no models exist, train them:

```bash
docker compose run --rm yaga train
```

**Step 2: Check inference logs**:

```bash
# View recent inference activity
docker compose exec yaga tail -50 /app/logs/inference.log

# Look for your service
docker compose exec yaga grep "<service-name>" /app/logs/inference.log | tail -20
```

**Step 3: Run verbose inference**:

```bash
# This shows detailed detection output
docker compose run --rm yaga inference --verbose
```

Look for output like:

```
Service: booking
  Time period: business_hours
  Metrics collected: ✓
  ML detection: No anomalies (all scores normal)
  Pattern matching: No patterns matched
  Result: alert_type = no_anomaly
```

**Common Causes and Solutions**:

| Cause | How to Identify | Solution |
|-------|-----------------|----------|
| No trained model | `ls smartbox_models/<service>/` is empty | Run `docker compose run --rm yaga train` |
| Service not in config | Service not listed in `config.json` | Add to appropriate `services` category |
| Contamination too high | Model trained with high contamination (e.g., 0.10) | Lower to 0.03-0.05 and retrain |
| Model stale | Model trained on old data; current behavior differs | Retrain with recent data |
| Wrong time period | Checking business_hours during night | Verify timezone configuration |
| SLO suppression | Anomaly detected but SLO says "ok" | Check SLO thresholds (may be too lenient) |

**Example Investigation**:

```
Scenario: booking service had a 5-minute outage but no alert was generated

Step 1: Check models
$ ls smartbox_models/booking/
business_hours/  evening_hours/  night_hours/  weekend_day/  weekend_night/
✓ Models exist

Step 2: Check if service is in config
$ grep -A5 '"services"' config.json
"services": {
  "critical": ["booking", "search"],
  ...
}
✓ Service is configured

Step 3: Run verbose inference during the issue
$ docker compose run --rm yaga inference --verbose
...
booking: ML detected anomaly (score: -0.45, severity: high)
booking: SLO evaluation: latency 250ms < 500ms acceptable → status: ok
booking: Severity adjusted: high → low (SLO status ok)
...

Diagnosis: SLO threshold too lenient (500ms acceptable)
Fix: Lower latency_acceptable_ms to 200ms
```

---

### Too Many False Positives

**Symptoms**:
- Alerts fire during normal operation
- Team ignores alerts due to noise (alert fatigue)
- Anomalies detected don't correlate with real problems

**Understanding False Positives**:

False positives occur when the ML model flags behavior as "anomalous" when it's actually normal. This happens because:

```
┌─────────────────────────────────────────────────────────────────┐
│                    WHY FALSE POSITIVES HAPPEN                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Contamination Too Low                                       │
│     ┌──────────────────────────────────┐                       │
│     │ Training data distribution       │                       │
│     │                                  │                       │
│     │         ●●●●●●                   │                       │
│     │        ●●●●●●●●                  │                       │
│     │       ●●●●●●●●●●   ← Model       │                       │
│     │        ●●●●●●●●      learns      │                       │
│     │         ●●●●●●       "normal"    │                       │
│     │           ▲                      │                       │
│     │     If contamination=0.02,      │                       │
│     │     only top 2% flagged         │                       │
│     └──────────────────────────────────┘                       │
│     → Normal variance at edges gets flagged                    │
│                                                                 │
│  2. Seasonality Not Captured                                    │
│     Model trained during quiet period, now in busy season       │
│                                                                 │
│  3. Recent Behavior Change                                      │
│     Deployment changed baseline, model is stale                 │
│                                                                 │
│  4. Time Period Mismatch                                        │
│     Night traffic patterns flagged by business_hours model      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Solution 1: Increase contamination rate**

The contamination rate tells the model what percentage of training data to consider "anomalous". Higher values = less sensitive detection.

```json
{
  "model": {
    "contamination_by_service": {
      "noisy-service": 0.08
    }
  }
}
```

| Contamination | Sensitivity | Best For |
|---------------|-------------|----------|
| 0.02-0.03 | Very high | Critical services where every alert matters |
| 0.05 | Balanced | Most production services |
| 0.08-0.10 | Lower | Noisy services, variable traffic |

**Solution 2: Adjust SLO thresholds**

SLO evaluation can suppress anomalies that don't have operational impact:

```json
{
  "slos": {
    "services": {
      "noisy-service": {
        "latency_acceptable_ms": 600,
        "error_rate_floor": 0.005
      }
    }
  }
}
```

This tells the system: "Even if ML detects an anomaly, don't alert unless latency exceeds 600ms or error rate exceeds 0.5%."

**Solution 3: Retrain models**

After changing configuration, always retrain:

```bash
docker compose run --rm yaga train
```

**Measuring Improvement**:

Track your false positive rate before and after changes:

```
Before tuning:
  Total alerts (7 days): 145
  Actionable: 12 (8%)
  False positives: 133 (92%)

After increasing contamination 0.03 → 0.06:
  Total alerts (7 days): 34
  Actionable: 11 (32%)
  False positives: 23 (68%)

After adjusting SLO thresholds:
  Total alerts (7 days): 18
  Actionable: 10 (56%)
  False positives: 8 (44%)
```

---

### Alerts Always Low Severity

**Symptom**:
- All alerts show `severity: low`
- Never see `high` or `critical` alerts
- Real incidents don't escalate properly

**Understanding Severity Flow**:

```
ML Detection (original severity)
        │
        ▼
SLO Evaluation
        │
        ├─ SLO breached ──▶ Keep/Escalate to CRITICAL
        │
        ├─ SLO warning ───▶ Keep HIGH
        │
        └─ SLO ok ────────▶ Downgrade to LOW
```

When SLO status is `ok` (all metrics within acceptable thresholds), severity is **always** downgraded to `low`. This is intentional—if metrics are operationally acceptable, the alert shouldn't be high priority.

**The Problem**:

If your SLO thresholds are too lenient, anomalies will never breach them:

```
Example: Latency threshold too high

Current latency: 450ms (genuinely slow for this service)
SLO acceptable: 500ms (too lenient)
SLO status: "ok" (450 < 500)
Result: severity = low (even though users are impacted)
```

**Solution: Tighten SLO thresholds**

Review and lower your acceptable thresholds:

```json
{
  "slos": {
    "services": {
      "booking": {
        "latency_acceptable_ms": 200,
        "latency_warning_ms": 300,
        "latency_critical_ms": 400,
        "error_rate_acceptable": 0.002,
        "error_rate_warning": 0.005,
        "error_rate_critical": 0.01
      }
    }
  }
}
```

**Before vs After**:

```
BEFORE (lenient thresholds):
  Latency acceptable: 500ms
  Current latency: 350ms
  SLO status: ok (350 < 500)
  ML severity: high → Adjusted: low

AFTER (tightened thresholds):
  Latency acceptable: 200ms
  Current latency: 350ms
  SLO status: warning (350 > 200, < 400)
  ML severity: high → Adjusted: high (kept)
```

**Finding the Right Thresholds**:

1. Look at your historical latency distribution:
   ```bash
   # In VictoriaMetrics, query p50, p90, p99 for your service
   ```

2. Set thresholds based on percentiles:
   - `acceptable`: Around p75 (allows normal variance)
   - `warning`: Around p90 (getting concerning)
   - `critical`: Around p99 (definitely a problem)

---

## Incident Lifecycle Issues

### Alerts Not Being Sent

**Symptom**:
- Verbose output shows anomaly detected
- Web API never receives the alert
- Dashboard shows no incidents

**Understanding Alert Flow**:

```
Anomaly Detected
        │
        ▼
Is incident confirmed?
        │
        ├─ NO (SUSPECTED) ──▶ NOT sent to web API
        │                      └─ Wait for next cycle
        │
        └─ YES (OPEN) ───────▶ Sent to web API
                                └─ Dashboard shows alert
```

**Diagnostic Steps**:

**Step 1: Check if anomaly is in SUSPECTED state**

```bash
docker compose run --rm yaga inference --verbose
```

Look for:
```
fingerprint_action: CREATE
status: SUSPECTED
is_confirmed: false
cycles_to_confirm: 1
```

If `status: SUSPECTED`, the anomaly hasn't been confirmed yet. Wait for the next detection cycle.

**Step 2: Check web API connectivity**

```bash
# Test if web API is reachable
docker compose exec yaga curl http://observability-api:8000/health

# Expected: {"status": "healthy"}
```

If this fails, check:
- Is the observability API running?
- Is the network configured correctly?
- Are there firewall rules blocking traffic?

**Step 3: Check if API is enabled**

```json
{
  "observability_api": {
    "enabled": true,
    "base_url": "http://observability-api:8000"
  }
}
```

If `enabled: false`, alerts won't be sent.

**Step 4: Check for API errors in logs**

```bash
docker compose exec yaga grep -i "api" /app/logs/inference.log | tail -20

# Look for:
# - "API call failed"
# - "Connection refused"
# - "Timeout"
```

---

### Orphaned Incidents in Web API

**Symptom**:
- Web API shows OPEN incidents that never resolve
- Incidents stuck in dashboard for days/weeks
- `consecutive_detections = 1` on stuck incidents

**Why This Happens**:

Before v1.3.2, SUSPECTED incidents were sent to the web API. If they expired without confirmation, no resolution was sent:

```
OLD BEHAVIOR (pre-v1.3.2):
─────────────────────────
10:00  Anomaly detected → SUSPECTED incident created
10:00  Alert sent to web API → Web API creates OPEN incident
10:03  Anomaly not detected → SUSPECTED expires
10:03  No resolution sent → Web API incident stays OPEN forever!

NEW BEHAVIOR (v1.3.2+):
───────────────────────
10:00  Anomaly detected → SUSPECTED incident created
       (NOT sent to web API - waiting for confirmation)
10:03  Anomaly not detected → SUSPECTED expires silently
       Web API never knew about it → No orphan created
```

**Identifying Orphaned Incidents**:

Query your web API database:

```sql
-- Find orphaned incidents (never confirmed but sent)
SELECT incident_id, service_name, created_at, consecutive_detections
FROM incidents
WHERE status = 'OPEN'
  AND consecutive_detections = 1
  AND created_at < NOW() - INTERVAL '24 hours';
```

**Solutions**:

1. **Manual cleanup**: Close orphaned incidents in the web API
2. **Wait for upgrade**: v1.3.2+ won't create new orphans
3. **Automated cleanup**: Add a job to close incidents older than X days with `consecutive_detections = 1`

---

### Incidents Restarting Unexpectedly

**Symptom**:
- Same anomaly pattern creates multiple incident IDs
- Incident history shows many short incidents instead of one long one
- Resolution reason shows `auto_stale`

**Understanding Staleness**:

The system considers an incident "stale" if the time gap since last detection exceeds `incident_separation_minutes` (default: 30):

```
Timeline with 45-minute gap:
──────────────────────────────────────────────────────────────
10:00  Anomaly detected → incident_abc created (SUSPECTED)
10:03  Detected again → incident_abc confirmed (OPEN)
10:06  Detected again → incident_abc continues
...
10:30  Last detection
       (service recovers but then has a second issue)
11:15  Anomaly detected → 45 min gap > 30 min threshold
                        → incident_abc auto-closed (stale)
                        → incident_xyz created (new incident)
```

**When This Is Correct vs. Problematic**:

```
CORRECT (two separate issues):
──────────────────────────────
10:00-10:30: Database slow due to query
11:15-12:00: Database slow due to disk I/O

These ARE two different incidents - staleness helps track them separately.


PROBLEMATIC (one issue, intermittent symptoms):
───────────────────────────────────────────────
10:00-10:30: Network flaky (detected)
10:30-11:15: Network stable (not detected)
11:15-11:45: Network flaky again (same issue)

This is ONE issue but creates multiple incidents.
```

**Solution: Increase separation threshold**

```json
{
  "fingerprinting": {
    "incident_separation_minutes": 60
  }
}
```

| Threshold | Use Case |
|-----------|----------|
| 15 min | Fast-resolving issues, want precise tracking |
| 30 min | Default, good balance |
| 60 min | Intermittent issues, prefer fewer incidents |
| 120 min | Very intermittent, consolidate aggressively |

---

## Metrics Issues

### metrics_unavailable Alerts

**Symptom**:
- `alert_type: "metrics_unavailable"` instead of detection results
- Detection skipped for some/all services
- Circuit breaker messages in logs

**Understanding This Alert**:

The system returns `metrics_unavailable` when it cannot collect metrics reliably. This prevents false alerts—if we can't read metrics, we shouldn't guess.

```
┌─────────────────────────────────────────────────────────────────┐
│                  METRICS UNAVAILABLE SCENARIOS                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Scenario 1: VictoriaMetrics Down                              │
│  ─────────────────────────────────                             │
│  All metrics fail → All services return metrics_unavailable    │
│                                                                 │
│  Scenario 2: Critical Metric Failed                            │
│  ─────────────────────────────────                             │
│  request_rate failed → Detection skipped (prevents false       │
│  "traffic cliff" from 0.0 value)                               │
│                                                                 │
│  Scenario 3: Partial Failure (Non-Critical)                    │
│  ──────────────────────────────────────────                    │
│  database_latency failed, others OK → Detection runs with      │
│  warning, partial_metrics_failure in output                    │
│                                                                 │
│  Scenario 4: Circuit Breaker Open                              │
│  ────────────────────────────────                              │
│  Too many failures → Circuit breaker prevents further calls    │
│  Wait for timeout (5 min) or fix underlying issue              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Diagnostic Steps**:

**Step 1: Test VictoriaMetrics connectivity**

```bash
# From inside the container
docker compose exec yaga curl -s "http://vm:8428/api/v1/status/buildinfo"

# Expected: JSON with version info
# If this fails, VM is unreachable
```

**Step 2: Check for circuit breaker**

```bash
docker compose exec yaga tail -50 /app/logs/inference.log | grep -i "circuit"

# Look for:
# "Circuit breaker OPEN - too many failures"
# "Circuit breaker will reset in X seconds"
```

**Step 3: Check which metrics failed**

Look at the output:
```json
{
  "alert_type": "metrics_unavailable",
  "failed_metrics": ["request_rate", "application_latency"],
  "collection_errors": {
    "request_rate": "Connection timeout after 10s",
    "application_latency": "Connection timeout after 10s"
  }
}
```

**Solutions**:

| Problem | Solution |
|---------|----------|
| VM down | Restart VictoriaMetrics, check VM health |
| Network issue | Check firewall, DNS, routing |
| Circuit breaker open | Wait 5 minutes or fix underlying issue |
| Timeout | Increase `timeout_seconds` in config |

**Circuit Breaker Behavior**:

```
Normal Operation:
  Requests succeed → Circuit CLOSED

5 consecutive failures:
  Circuit OPENS → All requests fail fast (no retry)

After 5 minutes (configurable):
  Circuit HALF-OPEN → One request allowed

  If succeeds → Circuit CLOSES
  If fails → Circuit stays OPEN for another timeout
```

---

### Validation Warnings

**Symptom**:
- `validation_warnings` array in output
- Messages like "capping at 1.0" or "using 0.0"

**Understanding Validation**:

Before metrics are processed, they're validated at the inference boundary:

```
Raw Metric Value
        │
        ▼
   Validation
        │
        ├─ NaN/Inf? ──────────▶ Replace with 0.0
        ├─ Negative rate? ────▶ Cap at 0.0
        ├─ Error rate > 1.0? ─▶ Cap at 1.0
        ├─ Latency > 5 min? ──▶ Cap at 300,000ms
        └─ Request rate > 1M? ▶ Cap at 1,000,000
        │
        ▼
   Sanitized Value (used for detection)
   + Warning (logged for investigation)
```

**Example Warnings**:

```json
{
  "validation_warnings": [
    "error_rate: value 1.5 > 1.0, capping at 1.0",
    "application_latency: negative value -50, using 0.0",
    "request_rate: NaN detected, using 0.0"
  ]
}
```

**Why This Happens**:

| Warning | Likely Cause | Investigation |
|---------|--------------|---------------|
| Error rate > 1.0 | Metric miscalculation upstream | Check error rate query definition |
| Negative latency | Clock skew or calculation bug | Check latency metric source |
| NaN/Inf | Division by zero, no data | Check if service was down |
| Extreme values | Metric spike or bug | Verify in VictoriaMetrics UI |

**Action Required**:

Validation warnings are **informational**. Detection still runs with sanitized values. However, frequent warnings indicate upstream data quality issues that should be investigated.

```bash
# Track warning frequency
docker compose exec yaga grep "validation_warnings" /app/logs/inference.log | wc -l
```

---

## SLO Issues

### SLO Evaluation Failed

**Symptom**:
- Warning in logs about SLO evaluation
- Anomalies not being severity-adjusted
- Missing `slo_evaluation` in output

**Common Causes**:

**1. SLO not enabled**

```json
{
  "slos": {
    "enabled": false
  }
}
```

Set to `true` to enable SLO evaluation.

**2. Service not configured**

If a service isn't in the SLO config, it uses defaults:

```json
{
  "slos": {
    "services": {
      "booking": {
        "latency_acceptable_ms": 300
      }
    }
  }
}
```

Services not listed use values from `slos.defaults`.

**3. Missing training statistics**

SLO evaluation for database latency requires training baseline. If the model doesn't have statistics, evaluation is skipped.

```bash
# Check if training statistics exist
docker compose exec yaga cat smartbox_models/<service>/business_hours/metadata.json | python -c "import sys,json; d=json.load(sys.stdin); print(d.get('statistics', {}))"
```

---

### Database Latency Always OK

**Symptom**:
- ML detects database latency anomaly
- But `database_latency_evaluation.status: "ok"`
- Even when database is clearly slow

**Understanding the Noise Floor**:

Database latency uses a hybrid approach:

```
Current DB Latency
        │
        ▼
   Below noise floor?
        │
        ├─ YES (e.g., 2ms < 5ms floor)
        │       │
        │       └─▶ status: "ok" (noise filtered)
        │
        └─ NO (above floor)
                │
                └─▶ Calculate ratio to baseline
                        │
                        └─▶ Ratio-based status
```

**Why This Design**:

For services with very fast databases (sub-millisecond), small absolute changes are operationally meaningless:

```
Without noise floor:
  Baseline: 0.3ms
  Current: 0.6ms
  Ratio: 2.0x → status: "warning"

  But 0.6ms is still FAST! This alert is noise.

With noise floor (5ms):
  Current: 0.6ms < 5ms floor
  status: "ok" (filtered)

  Operationally correct - 0.6ms is fine.
```

**Adjusting the Floor**:

For services with fast databases where you want to detect small changes:

```json
{
  "slos": {
    "services": {
      "fast-db-service": {
        "database_latency_floor_ms": 1.0
      }
    }
  }
}
```

For services with slow databases where 5ms is normal:

```json
{
  "slos": {
    "services": {
      "slow-db-service": {
        "database_latency_floor_ms": 10.0
      }
    }
  }
}
```

**Checking Current Configuration**:

```json
{
  "database_latency_evaluation": {
    "value_ms": 2.0,
    "baseline_mean_ms": 1.5,
    "floor_ms": 5.0,
    "below_floor": true,
    "ratio": 0.0,
    "status": "ok"
  }
}
```

If `below_floor: true`, the value is being filtered. Lower the floor if needed.

---

## Training Issues

### Training Fails

**Symptoms**:
- Models not updating (old timestamps)
- Training command exits with error
- No model files created

**Diagnostic Checklist**:

```
Training Failed?
        │
        ├─▶ 1. Can we reach VictoriaMetrics?
        │       │
        │       └─ No → Fix network/VM connectivity
        │
        ├─▶ 2. Is there enough historical data?
        │       │
        │       └─ < 30 days → Wait for data accumulation
        │
        ├─▶ 3. Is there enough disk space?
        │       │
        │       └─ Low → Clean old models/logs
        │
        ├─▶ 4. Is there enough memory?
        │       │
        │       └─ OOM → Increase container memory
        │
        └─▶ 5. Is the config valid?
                │
                └─ Invalid JSON → Fix syntax
```

**Step 1: Check training logs**

```bash
docker compose exec yaga cat /app/logs/train.log

# Look for:
# - "Training failed for service X"
# - "Not enough data points"
# - "Connection refused"
# - "MemoryError"
```

**Step 2: Check VictoriaMetrics**

```bash
docker compose exec yaga curl -s "http://vm:8428/api/v1/status/buildinfo"

# If this fails, VM is unreachable
```

**Step 3: Check disk space**

```bash
df -h

# Models typically need 10-50MB per service
# Ensure at least 1GB free
```

**Step 4: Run training manually**

```bash
docker compose run --rm yaga train 2>&1 | tee train_output.log

# This shows real-time output for diagnosis
```

**Common Causes and Solutions**:

| Cause | Symptoms | Solution |
|-------|----------|----------|
| VM unreachable | "Connection refused" | Check network, VM status |
| Not enough data | "Insufficient samples" | Wait for 30 days of data |
| Disk full | "No space left" | Clean old models: `rm -rf smartbox_models/old-service/` |
| Memory exhausted | "MemoryError" or OOM killed | Increase memory limit in docker-compose |
| Invalid config | "JSON decode error" | Validate config.json syntax |

**Minimum Data Requirements**:

| Model Type | Minimum Samples | Typical Duration |
|------------|-----------------|------------------|
| Univariate (per-metric) | 500 | ~2 days at 5-min intervals |
| Multivariate (combined) | 1000 | ~4 days at 5-min intervals |
| Full time-period coverage | 8640 | 30 days (captures weekly patterns) |

---

### Model Drift Detected

**Symptom**:
- `drift_warning` in inference output
- Confidence scores reduced
- Recommendation to retrain

**Understanding Drift**:

Model drift occurs when production data differs significantly from training data:

```
Training Data (30 days ago)          Current Data
─────────────────────────            ────────────
Mean latency: 100ms                  Mean latency: 200ms

Distribution:                        Distribution:
    ●●●●●                               ●●●●●
   ●●●●●●●                            ●●●●●●●●●
  ●●●●●●●●●                            ●●●●●●●●●●●●
 ●●●●●●●●●●●                              ●●●●●●●●●●●
    100ms                                    200ms

Model expects ~100ms                 Seeing ~200ms → DRIFT
```

**Drift Score Interpretation**:

| Score | Status | Action |
|-------|--------|--------|
| < 3 | Normal | No action needed |
| 3-5 | Moderate drift | Monitor, consider retraining |
| > 5 | Severe drift | Retrain soon |

**Output Example**:

```json
{
  "drift_warning": {
    "type": "model_drift",
    "overall_drift_score": 4.2,
    "affected_metrics": ["application_latency", "request_rate"],
    "recommendation": "WARNING: Moderate drift detected. Consider retraining.",
    "confidence_penalty_applied": 0.15
  }
}
```

**Why Drift Happens**:

1. **Seasonal changes**: Holiday traffic patterns vs. normal
2. **Deployments**: New code changed latency characteristics
3. **Infrastructure**: Migrated to faster/slower hardware
4. **Business changes**: New features changed usage patterns

**Solution: Retrain**

```bash
docker compose run --rm yaga train
```

This replaces old models with ones trained on recent data.

**Preventing Drift Issues**:

- Schedule regular retraining (daily or weekly)
- Retrain after significant deployments
- Monitor drift scores in dashboards

---

## Container Issues

### Container Keeps Restarting

**Symptoms**:
- `docker compose ps` shows restart count increasing
- Container never stays "Up" for long
- Application logs show repeated startup/shutdown

**Diagnostic Steps**:

**Step 1: Check exit code**

```bash
docker compose ps -a

# Look for:
# smartbox-anomaly  Exit 1   (error exit)
# smartbox-anomaly  Exit 137 (OOM killed)
# smartbox-anomaly  Exit 0   (clean exit, shouldn't restart)
```

**Step 2: Check recent logs**

```bash
docker compose logs --tail 100

# Look for error messages near the end
```

**Step 3: Check container events**

```bash
docker events --filter container=smartbox-anomaly --since 1h
```

**Common Exit Codes**:

| Exit Code | Meaning | Solution |
|-----------|---------|----------|
| 0 | Clean exit | Check restart policy, should be `unless-stopped` |
| 1 | Application error | Check logs for error message |
| 137 | OOM killed | Increase memory limit |
| 139 | Segmentation fault | Report bug, check for corruption |

**Solution by Cause**:

**Config file missing**:
```bash
# Check if config exists
ls -la config.json

# If missing, create from template
cp config.example.json config.json
```

**Invalid JSON in config**:
```bash
# Validate JSON syntax
python -m json.tool config.json

# If error, fix the syntax issue
```

**OOM killed**:
```yaml
# In docker-compose.yml, increase memory:
deploy:
  resources:
    limits:
      memory: 4G
```

---

### Database Locked

**Symptom**:
- `database is locked` error in logs
- SQLite error messages
- Fingerprinting failures

**Understanding the Issue**:

SQLite doesn't handle concurrent writes well. This happens when:

```
Process A: Running inference
                │
                ├─▶ UPDATE anomaly_incidents SET ...
                │
Process B: Also running inference (concurrent)
                │
                └─▶ UPDATE anomaly_incidents SET ...
                         │
                         └─▶ ERROR: database is locked
```

**How This Can Happen**:

1. **Manual inference + scheduled inference**: Running `docker compose run --rm yaga inference` while the scheduled inference is also running
2. **Multiple containers**: Two containers sharing the same volume
3. **Long-running queries**: One process holds lock while another waits

**Solutions**:

**1. Only run one inference at a time**:

```bash
# Check if inference is already running
docker compose ps | grep yaga

# If running, wait for it to finish
```

**2. Use proper container orchestration**:

Ensure only one inference container runs at a time via scheduling:

```yaml
# In docker-compose.yml
command: ["scheduler"]  # Uses cron, prevents overlap
```

**3. Check for stuck processes**:

```bash
# Look for zombie inference processes
docker compose exec yaga ps aux | grep python

# If stuck, restart the container
docker compose restart
```

**Corrupt Database Recovery**:

If the database becomes corrupted:

```bash
# Backup current state
cp data/anomaly_state.db data/anomaly_state.db.backup

# Remove corrupted database
rm data/anomaly_state.db

# Restart - new database will be created
docker compose restart
```

Note: This loses incident history. All incidents will start fresh (new incident IDs).

---

## Debugging

### Enable Verbose Logging

For detailed diagnostic output:

**Option 1: Command-line flag**

```bash
docker compose run --rm yaga inference --verbose
```

This shows:
- Metrics being collected
- ML detection results
- Pattern matching output
- SLO evaluation details
- Fingerprinting actions

**Option 2: Config file**

```json
{
  "logging": {
    "level": "DEBUG"
  }
}
```

Then rebuild and restart:
```bash
docker compose build
docker compose up -d
```

**Log Level Guide**:

| Level | Use Case |
|-------|----------|
| INFO | Normal operation (default) |
| DEBUG | Troubleshooting, detailed output |
| WARNING | Only problems and warnings |
| ERROR | Only errors |

---

### Inspect Model Details

**View model metadata**:

```bash
# Check training info
docker compose exec yaga cat /app/smartbox_models/<service>/business_hours/metadata.json | python -m json.tool
```

Expected output:

```json
{
  "service_name": "booking",
  "period": "business_hours",
  "trained_at": "2024-01-15T02:00:00Z",
  "training_samples": 8640,
  "contamination": 0.02,
  "n_estimators": 250,
  "statistics": {
    "application_latency": {
      "mean": 110.3,
      "std": 45.2,
      "p50": 105.0,
      "p95": 180.0,
      "p99": 250.0
    }
  },
  "calibrated_thresholds": {
    "critical": -0.58,
    "high": -0.32,
    "medium": -0.15,
    "low": -0.08
  }
}
```

**Check model age**:

```bash
# Find oldest and newest models
find smartbox_models -name "metadata.json" -exec sh -c 'echo {} && grep trained_at {}' \; | sort
```

---

### Check Fingerprint Database

**List active incidents**:

```bash
docker compose exec yaga sqlite3 /app/data/anomaly_state.db \
  "SELECT fingerprint_id, status, severity, occurrence_count,
          datetime(first_seen), datetime(last_updated)
   FROM anomaly_incidents
   WHERE status != 'CLOSED'
   ORDER BY last_updated DESC;"
```

**Count incidents by status**:

```bash
docker compose exec yaga sqlite3 /app/data/anomaly_state.db \
  "SELECT status, COUNT(*) FROM anomaly_incidents GROUP BY status;"
```

**Find long-running incidents**:

```bash
docker compose exec yaga sqlite3 /app/data/anomaly_state.db \
  "SELECT fingerprint_id, service_name,
          ROUND((julianday('now') - julianday(first_seen)) * 24 * 60) as duration_minutes
   FROM anomaly_incidents
   WHERE status = 'OPEN'
   ORDER BY duration_minutes DESC
   LIMIT 10;"
```

**Database schema reference**:

```sql
-- Main table: anomaly_incidents
fingerprint_id TEXT       -- Pattern identifier (hash)
incident_id TEXT PRIMARY  -- Unique occurrence ID
service_name TEXT         -- Service name
anomaly_name TEXT         -- Anomaly type
status TEXT               -- SUSPECTED, OPEN, RECOVERING, CLOSED
severity TEXT             -- low, medium, high, critical
first_seen TIMESTAMP      -- When first detected
last_updated TIMESTAMP    -- Last detection time
resolved_at TIMESTAMP     -- When closed (NULL if open)
occurrence_count INTEGER  -- Times detected
consecutive_detections INTEGER -- For confirmation
missed_cycles INTEGER     -- For grace period
```

---

## Quick Reference: Common Commands

```bash
# Health check
docker compose ps
docker compose exec yaga python -c "import smartbox_anomaly; print('OK')"

# View logs
docker compose logs --tail 100
docker compose exec yaga tail -f /app/logs/inference.log

# Manual operations
docker compose run --rm yaga inference --verbose
docker compose run --rm yaga train

# Check models
ls -la smartbox_models/
docker compose exec yaga cat smartbox_models/<service>/business_hours/metadata.json

# Check database
docker compose exec yaga sqlite3 /app/data/anomaly_state.db ".tables"
docker compose exec yaga sqlite3 /app/data/anomaly_state.db "SELECT * FROM anomaly_incidents WHERE status='OPEN';"

# Network connectivity
docker compose exec yaga curl -s http://vm:8428/api/v1/status/buildinfo
docker compose exec yaga curl http://observability-api:8000/health

# Container management
docker compose restart
docker compose build
docker compose up -d
```

---

## Getting Help

If you can't resolve an issue:

1. **Collect diagnostic information**:
   ```bash
   docker compose logs --tail 200 > diagnostic.log
   docker compose run --rm yaga inference --verbose >> diagnostic.log 2>&1
   docker compose exec yaga cat config.json >> diagnostic.log
   ```

2. **Check documentation**:
   - [Configuration Guide](../slo/README.md) - SLO and threshold settings
   - [Detection Pipeline](../detection/pipeline.md) - How detection works
   - [Incident Lifecycle](../incidents/README.md) - State machine details

3. **File an issue** with:
   - Symptom description
   - Expected vs. actual behavior
   - Diagnostic logs
   - Configuration (sanitized)
   - Steps to reproduce
