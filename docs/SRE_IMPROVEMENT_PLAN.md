# SRE Improvement Plan: Alert Quality Enhancement

**Created**: 2026-01-15
**Status**: In Progress
**Version**: 1.0

---

## Overview

This document tracks the implementation of SRE-recommended improvements to the Smartbox anomaly detection system, focusing on alert quality, noise reduction, and operational clarity.

**Goals**:
- Reduce alert volume by 60-80% through correlation
- Eliminate false-positive categories (healthy patterns)
- Standardize naming conventions
- Improve MTTR through clearer alert identification

---

## Implementation Status

### P0 - Critical (Target: 2 weeks)

| # | Task | Status | Notes |
|---|------|--------|-------|
| 1.1 | Add severity threshold configuration | ✅ Done | `AlertingConfig` in `config.py` with `severity_threshold` |
| 1.2 | Implement alert suppression by severity | ✅ Done | `should_alert()` method filters in `inference.py` |
| 2.1 | Design alert correlation system | ✅ Done | `CorrelationConfig` with primary selection strategies |
| 2.2 | Implement correlation in inference | ✅ Done | `_correlate_service_anomalies()` in `inference.py` |

### P1 - High (Target: 1 month)

| # | Task | Status | Notes |
|---|------|--------|-------|
| 3.1 | Remove `traffic_surge_healthy` from alerting | ✅ Done | Default in `non_alerting_patterns` |
| 3.2 | Audit other "healthy" patterns | ⏳ Pending | Check for similar patterns to exclude |
| 4.1 | Define naming convention standard | ⏳ Pending | Document `{metric}_{state}_{modifier}` formally |
| 4.2 | Rename inconsistent patterns | ⏳ Pending | Update remaining old-style names |
| 5.1 | Add root cause service to cascade names | ⏳ Pending | Modify `_build_cascade_name()` |

### P2 - Medium (Target: Quarter)

| # | Task | Status | Notes |
|---|------|--------|-------|
| 6.1 | Clean orphaned RECOMMENDATION_RULES | ⏳ Pending | Remove rules for non-existent patterns |
| 6.2 | Clean orphaned pattern references | ⏳ Pending | Audit all pattern name usage |
| 7.1 | Add service name to fallback naming | ⏳ Pending | Format: `{service}:{metric}_anomaly` |
| 8.1 | Document naming convention in CLAUDE.md | ⏳ Pending | Add pattern reference table |

### P3 - Low (Backlog)

| # | Task | Status | Notes |
|---|------|--------|-------|
| 9.1 | Optional time context prefix | ⏳ Backlog | e.g., `night:latency_spike` |
| 10.1 | Cross-model deduplication | ⏳ Backlog | If night_hours and business_hours both detect |

---

## Detailed Implementation Plans

### 1. Severity Threshold for Alerting

**Goal**: Only send alerts to Web API for anomalies at or above a configurable severity threshold. Lower severity anomalies are logged for analytics but don't create incidents.

**Configuration Addition** (`config.json`):
```json
{
  "alerting": {
    "severity_threshold": "medium",
    "log_below_threshold": true,
    "below_threshold_log_level": "INFO"
  }
}
```

**Files to Modify**:
- `smartbox_anomaly/core/config.py` - Add AlertingConfig dataclass
- `inference.py` - Filter anomalies by severity before Web API call
- `docs/CONFIGURATION.md` - Document new options

**Behavior**:
- `severity_threshold: "medium"` → Only medium, high, critical sent to API
- `severity_threshold: "low"` → All anomalies sent (current behavior)
- `severity_threshold: "high"` → Only high, critical sent
- Below-threshold anomalies logged with configurable level

---

### 2. Alert Correlation System

**Goal**: Group multiple anomalies for the same service within a time window into a single correlated incident.

**Design**:
```
Before Correlation:
  booking: [dependency_latency_high, application_latency_high, downstream_cascade]
  → 3 separate alerts

After Correlation:
  booking: {
    primary: "downstream_cascade",  // Highest confidence pattern
    contributing: ["dependency_latency_high", "application_latency_high"],
    correlation_id: "corr_abc123",
    anomaly_count: 3
  }
  → 1 alert with context
```

**Configuration Addition**:
```json
{
  "alerting": {
    "correlation": {
      "enabled": true,
      "window_seconds": 300,
      "primary_selection": "highest_confidence"
    }
  }
}
```

**Files to Modify**:
- `smartbox_anomaly/core/config.py` - Add CorrelationConfig
- `smartbox_anomaly/fingerprinting/fingerprinter.py` - Add correlation logic
- `inference.py` - Apply correlation before API call

**Primary Selection Strategy**:
1. Named pattern > ML-only detection
2. Higher confidence > lower confidence
3. More specific name > generic name
4. Earlier detection > later detection (for ties)

---

### 3. Remove "Healthy" Patterns from Alerting

**Goal**: Patterns indicating healthy behavior should not generate alerts.

**Patterns to Convert to Log-Only**:
- `traffic_surge_healthy` - System handling load well
- `recovery_in_progress` - Returning to normal (debatable)

**Implementation Options**:

**Option A**: Add `alertable: false` to pattern definition
```python
"traffic_surge_healthy": PatternDefinition(
    severity="low",
    alertable=False,  # New field
    ...
)
```

**Option B**: Filter by pattern name in inference
```python
NON_ALERTING_PATTERNS = {"traffic_surge_healthy", "recovery_in_progress"}
```

**Recommendation**: Option A is more maintainable.

---

### 4. Standardize Naming Convention

**Convention**: `{metric}_{state}_{modifier}`

**Rename Map**:
| Current | New | Reason |
|---------|-----|--------|
| `traffic_surge_failing` | `request_rate_surge_failing` | Consistent metric prefix |
| `traffic_surge_degrading` | `request_rate_surge_degrading` | Consistent metric prefix |
| `traffic_surge_healthy` | `request_rate_surge_healthy` | Consistent metric prefix |
| `traffic_cliff` | `request_rate_cliff` | Consistent metric prefix |
| `elevated_errors` | `error_rate_elevated` | Matches existing pattern |
| `fast_rejection` | `error_rate_fast_rejection` | Add metric prefix |
| `fast_failure` | `error_rate_fast_failure` | Add metric prefix |
| `partial_rejection` | `error_rate_partial_rejection` | Add metric prefix |
| `downstream_cascade` | `dependency_latency_cascade` | Specific metric |
| `internal_bottleneck` | `application_latency_bottleneck` | Specific metric |
| `database_bottleneck` | `database_latency_bottleneck` | Consistent format |
| `database_degradation` | `database_latency_degraded` | Consistent format |

**Migration Strategy**:
1. Add new pattern names alongside old (aliasing)
2. Update fingerprinter to normalize old → new
3. Deprecation period (2 weeks)
4. Remove old names

---

### 5. Root Cause in Cascade Names

**Goal**: When cascade analysis identifies root cause with high confidence, include it in the alert.

**Implementation**:
```python
def _build_cascade_name(base_name: str, cascade_analysis: dict) -> str:
    """Enhance pattern name with root cause when confident."""
    if not cascade_analysis:
        return base_name

    confidence = cascade_analysis.get("confidence", 0)
    root_service = cascade_analysis.get("root_cause_service")

    if confidence >= 0.8 and root_service:
        return f"{base_name}:{root_service}"

    return base_name

# Result: "dependency_latency_cascade:titan"
```

---

## Success Metrics

| Metric | Baseline | Target | Measurement |
|--------|----------|--------|-------------|
| Alerts per incident | ~7 | 1-2 | Correlation effectiveness |
| Low severity % of alerts | 85% | <20% | After threshold filter |
| Generic names % | ~15% | <1% | Pattern name audit |
| On-call pages per week | TBD | -50% | Alert volume tracking |

---

## Rollout Plan

### Phase 1: Severity Threshold (Week 1)
1. Add configuration
2. Implement filtering
3. Deploy with `severity_threshold: "low"` (no change)
4. Monitor baseline metrics
5. Switch to `severity_threshold: "medium"`

### Phase 2: Healthy Pattern Removal (Week 1-2)
1. Add `alertable` field to PatternDefinition
2. Mark healthy patterns as non-alertable
3. Deploy and monitor

### Phase 3: Naming Standardization (Week 2-3)
1. Create alias mapping
2. Update fingerprinter normalization
3. Deploy with both old and new names active
4. Monitor for issues
5. Remove old names after 2 weeks

### Phase 4: Alert Correlation (Week 3-4)
1. Implement correlation logic
2. Test with production data replay
3. Deploy with `correlation.enabled: false`
4. Enable for single service (booking)
5. Monitor and tune
6. Enable for all services

---

## Changelog

| Date | Change | Author |
|------|--------|--------|
| 2026-01-15 | Initial plan created | SRE Assessment |

