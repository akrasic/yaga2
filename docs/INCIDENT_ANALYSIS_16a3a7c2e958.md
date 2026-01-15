# Incident Analysis: incident_16a3a7c2e958

**Date:** 2026-01-13
**Service:** booking
**Anomaly Type:** error_rate_high
**Status:** RECOVERING

> **Resolution Status:** FIXED in v1.3.3
> - Added `error_rate_floor` config option
> - Implemented error rate suppression in SLO evaluator
> - Configured booking service with `error_rate_floor: 0.002` (0.2%)

---

## Executive Summary

This incident is part of a **recurring pattern of false positive alerts** for the booking service. The ML model is detecting tiny statistical deviations in error rate that are **well below operational SLO thresholds**.

**Key Finding:** 84% of detections (215/256 incidents) have error rates below the SLO acceptable threshold of 0.2%. These alerts provide no operational value and create noise.

**Recommendation:** Implement error rate suppression similar to the existing database latency floor filtering.

---

## Incident Details

### Current Incident

| Field | Value |
|-------|-------|
| Incident ID | `incident_16a3a7c2e958` |
| Fingerprint ID | `anomaly_0b5456c076e1` |
| Service | booking |
| Status | RECOVERING |
| Severity | low |
| First Seen | 2026-01-13 06:48:50 |
| Last Updated | 2026-01-13 16:42:51 |
| Duration | ~10 hours |
| Occurrences | 139 |

### Metric Values at Last Detection

| Metric | Current | Training Mean | Deviation (σ) | Percentile |
|--------|---------|---------------|---------------|------------|
| request_rate | 52.7 req/s | 42.5 req/s | +0.40σ | 69th |
| application_latency | 110.5 ms | 110.3 ms | +0.03σ | 56th |
| dependency_latency | 1.43 ms | 1.47 ms | -0.17σ | 48th |
| **error_rate** | **0.013%** | **0.012%** | **+0.02σ** | **78th** |

**Observation:** All metrics are within normal ranges. The error rate deviation is only 0.02 standard deviations - essentially noise.

---

## Pattern Analysis (Fingerprint History)

This fingerprint (`anomaly_0b5456c076e1`) has been extremely active:

| Metric | Value |
|--------|-------|
| Total Incidents | 256 |
| Total Occurrences | 1,369 |
| Time Span | 7 days (Jan 6-13, 2026) |
| Avg Duration | 19 minutes |
| Max Duration | 594 minutes (~10 hours) |

### Severity Distribution

| Severity | Count | Percentage |
|----------|-------|------------|
| low | 250 | 97.7% |
| medium | 6 | 2.3% |
| high | 0 | 0% |
| critical | 0 | 0% |

### Time Period Distribution

| Model | Count |
|-------|-------|
| business_hours | 84 |
| weekend_day | 63 |
| evening_hours | 41 |
| night_hours | 38 |
| weekend_night | 30 |

**Finding:** Alerts fire across all time periods, suggesting this is a systemic issue with detection sensitivity, not a time-specific problem.

---

## SLO Comparison

### Booking Service SLO Thresholds (from config.json)

| Threshold | Value |
|-----------|-------|
| error_rate_acceptable | 0.2% (0.002) |
| error_rate_warning | 0.5% (0.005) |
| error_rate_critical | 1.0% (0.01) |

### Detected Error Rates vs SLO

| Category | Count | % of Incidents |
|----------|-------|----------------|
| **Below acceptable (< 0.2%)** | **215** | **84%** |
| Between acceptable and warning | 35 | 14% |
| Between warning and critical | 6 | 2% |
| Above critical | 0 | 0% |

**Critical Finding:** 84% of incidents have error rates below the "acceptable" SLO threshold. These detections are statistically valid but operationally meaningless.

### Sample Detected Values

| Error Rate | Description |
|------------|-------------|
| 0.01% | "Elevated: 0.01% (normally 0.01%, threshold 0.10%)" |
| 0.02% | "Elevated: 0.02% (normally 0.01%, threshold 0.10%)" |
| 0.03% | "Elevated: 0.03% (normally 0.01%, threshold 0.10%)" |
| 0.12% | "Elevated: 0.12% (normally 0.08%, threshold 0.67%)" |

These values are **1-2 orders of magnitude below** the SLO acceptable threshold.

---

## Root Cause Analysis

### Why Alerts Are Firing

1. **ML Model Sensitivity:** Isolation Forest detects any deviation from the training distribution, even tiny ones
2. **Low Baseline Error Rate:** booking service has a very low baseline (~0.01%), so even tiny increases are "unusual"
3. **Confidence Score:** IF scores around -0.03 to -0.21 (barely anomalous)
4. **No Error Rate Floor:** Unlike database latency, there's no suppression for error rates below SLO thresholds

### Gap in Current Implementation

The SLO evaluator has suppression logic for database latency:

```python
# From smartbox_anomaly/slo/evaluator.py:925-932
if db_latency < floor_ms:
    logger.debug(f"Suppressing {anomaly_name}: database_latency below floor")
    continue  # Skip this anomaly entirely
```

**But no equivalent suppression exists for error rate:**

```python
# Error anomalies only get SLO context added, but are NOT suppressed
elif "error" in anomaly_lower:
    anomaly_copy["slo_context"] = {
        "within_acceptable": error_rate <= slo.error_rate_acceptable,
        # ... but no suppression logic
    }
```

The `within_acceptable: true` flag is set but not acted upon.

---

## Impact Assessment

### Alert Fatigue

- **256 incidents in 7 days** = ~37 incidents/day
- **1,369 occurrences** = ~196 detections/day
- All low severity, providing no actionable signal

### Resource Waste

- Database storage for tracking non-issues
- API calls for non-actionable alerts
- Dashboard clutter with false positives

### Masking Real Issues

High volume of false positives can lead to:
- Operators ignoring "booking error rate" alerts
- Delayed response when a real error spike occurs

---

## Recommendations

### 1. ~~Implement Error Rate Suppression (High Priority)~~ DONE

Added suppression logic similar to database latency floor in `smartbox_anomaly/slo/evaluator.py`:

```python
# Implemented in _evaluate_anomalies()
elif "error" in anomaly_lower:
    error_rate = metrics.get("error_rate", 0.0)

    # Determine suppression threshold
    suppression_threshold = (
        slo.error_rate_floor if slo.error_rate_floor > 0
        else slo.error_rate_acceptable
    )

    # Suppress if error rate is below threshold
    if error_rate < suppression_threshold:
        logger.debug(...)
        continue  # Skip this anomaly entirely
```

### 2. ~~Add Minimum Error Rate Floor (Medium Priority)~~ DONE

Added `error_rate_floor` config option to `ServiceSLOConfig`:

```json
{
  "slos": {
    "defaults": {
      "error_rate_floor": 0  // 0 = use error_rate_acceptable
    },
    "services": {
      "booking": {
        "error_rate_floor": 0.002  // 0.2% - matches acceptable threshold
      }
    }
  }
}
```

### 3. Improve Anomaly Naming (Low Priority) - Future Work

The name `error_rate_high` is misleading for 0.01% error rates. Consider:
- `error_rate_elevated` - for rates between training mean and SLO acceptable
- `error_rate_high` - only for rates above SLO warning
- `error_rate_critical` - only for rates above SLO critical

### 4. Adjust Contamination Rate (Alternative) - Not Needed

With error rate suppression implemented, adjusting contamination is no longer necessary.

---

## Lessons Learned

1. **ML ≠ Operational Significance:** Statistical anomalies are not always operational problems
2. **SLO Integration Must Be Complete:** Partial SLO integration (severity adjustment without suppression) creates noise
3. **Low-Error-Rate Services Need Special Handling:** Services with very low baseline error rates will have many false positives
4. **Monitor Alert Quality:** Track false positive rates to identify detection tuning needs

---

## Files Modified

| File | Change | Status |
|------|--------|--------|
| `smartbox_anomaly/slo/evaluator.py` | Add error rate suppression in `_evaluate_anomalies()` | DONE |
| `smartbox_anomaly/core/config.py` | Add `error_rate_floor` to `ServiceSLOConfig` and parsing | DONE |
| `config.json` | Add `error_rate_floor` to defaults and booking service | DONE |
| `docs/CONFIGURATION.md` | Document `error_rate_floor` option | DONE |
| `docs/API_CHANGELOG.md` | Add v1.3.3 changelog entry | DONE |

---

## Related Documentation

- [KNOWN_ISSUES.md](./KNOWN_ISSUES.md) - Add this as a new known issue
- [CONFIGURATION.md](./CONFIGURATION.md) - Document error_rate_floor option
- [FINGERPRINTING.md](./FINGERPRINTING.md) - Consider stale fingerprint cleanup for noisy patterns
