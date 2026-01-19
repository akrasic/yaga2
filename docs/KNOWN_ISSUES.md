# Known Issues and Limitations

This document tracks known issues, limitations, and planned improvements identified through expert reviews and production experience.

---

## Summary

| Category | Issues | Critical | High | Medium | Fixed |
|----------|--------|----------|------|--------|-------|
| Training Pipeline | 10 | 1 | 3 | 2 | 10 |
| Message Semantics | 12 | 1 | 2 | 8 | 1 |
| Web API Integration | 1 | 0 | 1 | 0 | 1 |
| **Total** | **23** | **2** | **6** | **10** | **12** |

**Recently Fixed:**
- Issue #1: Statistics/Training Data Mismatch → Unified data cleaning pipeline, no outlier removal
- Issue #2: No Model Validation → Implemented temporal train/validation split
- Issue #3: Pattern Overlap → Replaced with distinct patterns
- Issue #4: Hardcoded Contamination Rates → Implemented knee-based contamination estimation
- Issue #5: No Temporal Validation → Implemented temporal split with config option
- Issue #6: Insufficient Sample Requirements → Increased to 500/1000
- Issue #7: Low Latency Misinterpreted → Added lower_is_better_metrics()
- Issue #10: No Feature Correlation Analysis → Added correlation matrix analysis
- Issue #11: Missing Training Data Quality Report → Added DataQualityReport with gap detection
- Issue #12: No Model Drift Detection → Implemented z-score and Mahalanobis drift detection
- Issue #20: Silent Query Failures → Added QueryResult with explicit error tracking
- Issue #21: SUSPECTED Alerts Sent to Web API → Filter confirmed-only before sending (v1.3.2)

---

## Critical Issues

### 1. Statistics/Training Data Mismatch

**Status**: Fixed
**Severity**: Critical
**Location**: `smartbox_anomaly/detection/detector.py:1232-1419`

**Problem**: Statistics (p90, p95, etc.) were calculated on raw data, but models were trained on filtered data with outliers removed. This created inconsistent thresholds.

**Resolution**: Unified the data cleaning pipeline so both statistics and model training use identical data:

1. **Outliers are NO LONGER removed** from IF training - they are what IF should learn to detect
2. **Robust statistics** (trimmed mean, IQR-based std) are used instead of standard mean/std
3. Both `_calculate_training_statistics()` and `_train_univariate_model()` now use the same `_clean_metric_data_for_training()` method
4. Only NaN and infinity values are removed from both paths

Key code comment (line 1271):
```python
# NOTE: We do NOT remove outliers for IF training - outliers are what IF should detect!
```

This ensures:
- Statistics accurately represent the data distribution the IF model sees
- Severity thresholds are correctly calibrated against the trained model
- Robust statistics provide outlier-resistant summary without removing outliers

---

### 2. No Model Validation

**Status**: Fixed
**Severity**: Critical
**Location**: `smartbox_anomaly/detection/time_aware.py:80-168`, `main.py:574-576`

**Problem**: Models were trained on 100% of data with no validation split, cross-validation, or performance metrics.

**Resolution**: Implemented temporal train/validation split in `train_time_aware_models()`:
- Default validation fraction: 20% (configurable via `validation_fraction` in config.json)
- Temporal split: First 80% for training, last 20% for validation
- Validation data used for threshold calibration
- `ValidationMetrics` dataclass tracks false positive rate and detection rate
- `_calibrate_on_validation()` method calibrates severity thresholds on held-out data

The split respects temporal ordering to prevent future data leakage.

---

### 3. Pattern Overlap: silent_degradation vs resource_contention

**Status**: Fixed
**Severity**: Critical
**Location**: `smartbox_anomaly/detection/interpretations.py`

**Problem**: These patterns had overlapping conditions but different interpretations.

**Resolution**: Both patterns were removed and replaced with better-designed patterns:
- `latency_spike_recent` - Uses `latency_change=recent_increase` condition to detect recent latency increases
- `internal_bottleneck` - Requires `dependency_latency=normal` and `database_latency=normal` to confirm issue is internal

The new patterns have distinct conditions and don't overlap. The orphan recommendation entry for `silent_degradation` was also removed.

---

## High Priority Issues

### 4. Hardcoded Contamination Rates

**Status**: Fixed
**Severity**: High
**Location**: `smartbox_anomaly/detection/detector.py:2030-2104`

**Problem**: Contamination rates were hardcoded per service type without data-driven validation.

**Resolution**: Implemented automatic contamination estimation using knee detection:
- `_estimate_contamination()` method with "knee" and "gap" detection methods
- `_find_knee_contamination()` analyzes score distribution to find natural anomaly threshold
- Estimated contamination is bounded by service category limits (50%-300% of base)
- Hardcoded values now serve as defaults/fallbacks, not fixed values
- Configuration option `contamination_estimation.method` in config.json

The system now automatically adapts contamination to each service's actual anomaly distribution while maintaining reasonable bounds.

---

### 5. No Temporal Validation for Time-Series

**Status**: Fixed
**Severity**: High
**Location**: `smartbox_anomaly/detection/time_aware.py:123-137`

**Problem**: Time-series data was split by period but not validated temporally. Random sampling during training could leak future information.

**Resolution**: Implemented temporal validation split in `train_time_aware_models()`:
- Data is sorted by datetime index before splitting (`period_data_sorted = period_data.sort_index()`)
- First (1 - validation_fraction) samples used for training
- Last validation_fraction samples used for validation
- No random sampling - strict temporal ordering maintained
- Configuration option `validation_fraction` in config.json (default: 0.2)

This prevents future data leakage and ensures models are validated on temporally held-out data.

---

### 6. Insufficient Minimum Sample Requirements

**Status**: Fixed
**Severity**: High
**Location**: `smartbox_anomaly/core/config.py`, `smartbox_anomaly/core/constants.py`

**Problem**: Previous minimums (50 univariate, 100 multivariate) were too low for stable Isolation Forest training.

**Resolution**: Increased minimum sample requirements:
- `MIN_TRAINING_SAMPLES`: 50 → 500 (univariate)
- `MIN_MULTIVARIATE_SAMPLES`: 100 → 1000 (multivariate)

These values align with Isolation Forest recommendations for stable tree construction with n_estimators=200.

---

### 7. Low Latency Misinterpreted as Positive

**Status**: Fixed
**Severity**: High
**Location**: `smartbox_anomaly/detection/detector.py:632-685`, `smartbox_anomaly/core/constants.py:188-199`

**Problem**: "Unusually fast responses" message sounds positive, but low latency often indicates:
- Error responses (no processing done)
- Circuit breaker activations
- Rate limiting rejections

**Solution Implemented**:
1. Added `lower_is_better_metrics()` in `constants.py` defining metrics where low values are improvements (`database_latency`, `dependency_latency`, `error_rate`)
2. Updated `_signals_to_levels()` to treat low values for these metrics as `"normal"` instead of `"low"`
3. `application_latency` is intentionally NOT in this list - low latency with high errors should still match fast-fail patterns
4. Pattern matching now uses fail-closed validation - unknown conditions don't silently pass

**Result**:
- Low `database_latency`/`dependency_latency`/`error_rate` → treated as normal (no alert)
- Low `application_latency` + high errors → still matches `fast_failure` pattern (correct alert)
- Low `application_latency` + normal errors → no pattern match (no false positive)

---

## Medium Priority Issues

### 8. No Stationarity Check

**Status**: Open
**Severity**: Medium
**Location**: Not implemented

**Problem**: Anomaly detection assumes stationary data (stable mean/variance), but this isn't verified.

**Impact**:
- Training on trending data produces invalid statistics
- Models become stale faster than expected

**Workaround**: Retrain frequently (daily).

---

### 9. RobustScaler May Compress Signal

**Status**: Open
**Severity**: Medium
**Location**: `smartbox_anomaly/detection/detector.py:362-363`

**Problem**: RobustScaler (median/IQR) works well for latency but may over-normalize traffic metrics with legitimate high variance.

**Impact**: Large traffic spikes become indistinguishable from smaller ones.

**Workaround**: Set lower contamination for traffic-heavy services.

**Fix Required**: Use metric-specific transformations (log for rates, logit for proportions).

---

### 10. No Feature Correlation Analysis

**Status**: Fixed
**Severity**: Medium
**Location**: `smartbox_anomaly/detection/detector.py:1983-2028`

**Problem**: Multivariate model treated all features equally, even highly correlated ones (app_latency & dependency_latency).

**Resolution**: Implemented feature correlation analysis during training:
- `_analyze_feature_correlation()` method computes correlation matrix
- Identifies highly correlated pairs (r > 0.8) and logs warnings
- Results stored in `correlation_analysis` and persisted with model
- Warnings help identify potential multicollinearity issues
- Correlation data available in model metadata for inspection

---

### 11. Missing Training Data Quality Report

**Status**: Fixed
**Severity**: Medium
**Location**: `smartbox_anomaly/metrics/quality.py`, `main.py:305-327`

**Problem**: No visibility into training data quality - missing values, outliers, gaps are silently handled.

**Resolution**: Implemented comprehensive data quality analysis module:
- `DataQualityReport` class with quality scoring (A-F grades)
- `analyze_combined_data_quality()` for multi-metric analysis
- `detect_time_gaps()` for identifying gaps in time series data
- Automatic quality logging during training with per-metric breakdown
- Coverage percentage tracking (expected vs actual data points)
- Gap count and maximum gap duration reporting
- Integration into training pipeline (`_combine_metrics()` method)

---

### 12. No Model Drift Detection

**Status**: Fixed
**Severity**: Medium
**Location**: `smartbox_anomaly/detection/detector.py:127-147, 2145-2220`

**Problem**: No infrastructure to detect when models become stale.

**Resolution**: Implemented comprehensive drift detection with two methods:
- **Univariate drift (z-score)**: Compares each metric to training mean/std
- **Multivariate drift (Mahalanobis distance)**: Detects covariate shift in feature relationships
- `DriftAnalysis` dataclass captures drift results
- `check_drift()` method performs analysis at inference time
- Training computes and stores inverse covariance matrix for Mahalanobis distance
- Drift warnings included in output with severity levels (moderate: z>3, severe: z>5)
- Confidence penalties applied when drift detected (15% moderate, 30% severe)
- Configuration option `inference.check_drift` enables/disables drift checking

---

### 13. traffic_cliff Pattern Too Simple

**Status**: Open
**Severity**: Medium
**Location**: `smartbox_anomaly/detection/interpretations.py`

**Problem**: Only checks request_rate, fires during normal low-traffic periods (3 AM).

**Impact**: False positives during expected quiet periods.

**Workaround**: Time-aware models should reduce this, but not eliminate it.

---

### 14. error_rate_critical Severity May Be Too High

**Status**: Open
**Severity**: Medium
**Location**: `smartbox_anomaly/detection/interpretations.py`

**Problem**: Any "very_high" error rate triggers `error_rate_critical` with critical severity, but 6% errors (just above threshold) isn't as severe as 50% errors.

**Impact**: Alert fatigue from minor error elevations.

**Note**: Pattern was renamed from `partial_outage` to `error_rate_critical` for clarity (v1.3.1).

---

### 15. Missing error_rate Low Interpretation

**Status**: Open
**Severity**: Medium
**Location**: `smartbox_anomaly/detection/interpretations.py`

**Problem**: No interpretation for error rate dropping to near-zero, which could indicate:
- Recovery from incident (good)
- Error logging broken (bad)

---

### 16. "Spike" Terminology Misleading

**Status**: Open
**Severity**: Low
**Location**: `smartbox_anomaly/detection/interpretations.py`

**Problem**: "Traffic spike" implies sudden change, but detection doesn't verify rate of change. A gradual increase is also labeled "spike".

---

### 17. Statistics Don't Account for Autocorrelation

**Status**: Open
**Severity**: Low
**Location**: `smartbox_anomaly/detection/detector.py:270-320`

**Problem**: Standard deviation assumes independent samples, but time series data is autocorrelated.

**Impact**: Confidence intervals are overconfident.

---

### 18. No Concept Drift Handling in Time Periods

**Status**: Open
**Severity**: Low
**Location**: `smartbox_anomaly/detection/time_aware.py`

**Problem**: All historical data for a period weighted equally (Monday 3 months ago = yesterday).

**Fix Required**: Use exponential weighting or sliding window to favor recent data.

---

### 19. Missing HTTP Status Code Context in Fast-Fail Patterns

**Status**: Backlog (Future Enhancement)
**Severity**: Low
**Location**: `smartbox_anomaly/detection/interpretations.py`

**Problem**: Fast-fail patterns (`fast_rejection`, `fast_failure`, `partial_rejection`) detect rapid failures but cannot distinguish between different failure modes because HTTP status codes are not currently tracked.

**Current Behavior**:
```
fast_rejection: very low latency + very high errors
  → "Requests rejected before processing"
```

**Desired Behavior**:
```
fast_rejection_401: very low latency + very high 401s
  → "Authentication failures - likely auth service issue or token expiry"

fast_rejection_429: very low latency + very high 429s
  → "Rate limiting active - traffic exceeds configured limits"

fast_rejection_503: very low latency + very high 503s
  → "Circuit breaker open - downstream dependency unavailable"
```

**Why Deferred**:
- HTTP status code metrics are not currently collected in the inference engine
- Requires new metrics pipeline to track `status_code` labels from VictoriaMetrics
- Model training would need to incorporate status code distributions

**Future Implementation**:
1. Add status code metrics collection to VictoriaMetrics client
2. Extend pattern conditions to support status code checks
3. Create status-code-specific fast-fail patterns
4. Update SLO evaluation to consider status code distributions

**Added**: 2026-01-15 (SRE Expert Review)

---

### 20. Silent Query Failures in VictoriaMetrics Client

**Status**: Fixed
**Severity**: Medium
**Location**: `smartbox_anomaly/metrics/client.py:509-634`

**Problem**: VictoriaMetrics query failures (timeouts, connection errors, VM errors) were silently returning empty data, making it impossible to distinguish between "no data exists" and "query failed".

**Resolution**: Implemented structured query result pattern:
- `QueryResult` dataclass with explicit success/failure status
- `query_range()` now returns `QueryResult` with error tracking
- `query_instant()` added as new method with same pattern
- Backward-compatible `query()` and `query_range_raw()` methods preserved
- Configurable timeout for training queries (default 120s for 30-day ranges)
- Per-query duration tracking for performance monitoring
- Detailed error logging for timeout, connection, and VM errors
- Training pipeline updated to handle query failures and report them

---

### 21. SUSPECTED Alerts Sent to Web API

**Status**: Fixed (v1.3.2)
**Severity**: High
**Location**: `inference.py:2550-2591`

**Problem**: The inference engine was sending ALL anomalies to the web API, including those in SUSPECTED state (not yet confirmed). When these anomalies expired without confirmation (`suspected_expired`), no resolution was sent, leaving orphaned OPEN incidents in the web API.

**Evidence**:
- 72 orphaned OPEN incidents found in production web API database
- All had `consecutive_detections = 1` (never confirmed)
- All had `resolution_reason = 'suspected_expired'` in inference DB
- No resolution was ever sent (correct for unconfirmed, but shouldn't have been created)

**Resolution**: Modified `_update_results_processor()` to filter anomalies by confirmation status before sending to web API:

```python
# Filter to only confirmed anomalies (OPEN or RECOVERING status)
confirmed_anomalies = {
    name: anomaly for name, anomaly in anomalies.items()
    if anomaly.get('is_confirmed', False) or
       anomaly.get('status') in ('OPEN', 'RECOVERING')
}

if confirmed_anomalies:
    # Only send confirmed anomalies to web API
    filtered_result = {**result, 'anomalies': confirmed_anomalies}
    formatted_payload = pipeline.results_processor._format_time_aware_alert_json(filtered_result)
```

**Also Fixed**: KeyError in verbose logging at line 2752 - `resolution['service']` changed to `resolution.get('service_name', ...)`.

**Related**: See `BUG_REPORT_SUSPECTED_ALERTS.md` for full investigation and fix details.

---

### 22. Low Application Latency False Positives

**Status**: Fixed (v1.3.3)
**Severity**: Medium
**Location**: `smartbox_anomaly/detection/detector.py:715-725`

**Problem**: The anomaly detector was generating `application_latency_low` alerts for services with fast response times, even when errors were 0%. This is a false positive - fast responses with no errors are good, not anomalous.

**Root Cause**: `application_latency` was intentionally NOT in `lower_is_better_metrics()` because low latency + high errors = fast-fail (bad). However, the detection logic didn't differentiate between:
- Low latency + high errors = fast-fail (should alert)
- Low latency + low errors = fast responses (should NOT alert)

**Evidence from Production**:
- Multiple services showing `application_latency_low` with 0% error rate
- Examples: m2-it (28ms vs 52ms mean), m2-bn (57ms vs 133ms mean), all with error_rate=0.0

**Resolution**: Added conditional filtering in `_interpret_anomaly_signals()`:

```python
# Low application_latency is only concerning when combined with errors
error_rate = metrics.get(MetricName.ERROR_RATE, 0.0)
error_threshold = 0.01  # 1% - below this, low latency is not concerning
if error_rate < error_threshold:
    actionable_signals = [
        s for s in actionable_signals
        if not (s.metric_name == MetricName.APPLICATION_LATENCY and s.direction == "low")
    ]
```

**Result**: Fast responses with low errors are now correctly filtered out. Fast-fail patterns still match when errors are elevated.

---

### 23. Database Latency Sub-millisecond Noise

**Status**: Fixed (v1.3.3)
**Severity**: Medium
**Location**: `smartbox_anomaly/detection/detector.py:727-742`

**Problem**: The `database_latency_degraded` pattern was firing for sub-millisecond database latency changes, such as 0.29ms → 0.37ms. While statistically significant, these changes are operationally meaningless.

**Root Cause**: Pattern matching used percentile-based levels (e.g., >90th percentile = "high"). Sub-millisecond values could still be at high percentiles statistically, but the absolute values are too small to matter.

**Evidence from Production**:
- m2-se: 0.37ms flagged as "high" (1.93σ above 0.29ms mean)
- m2-df-adm: 0.83ms flagged (7σ above 0.53ms mean)
- All values were sub-millisecond and operationally insignificant

**Resolution**: Added absolute floor check in `_interpret_anomaly_signals()`:

```python
# Sub-millisecond database_latency is operationally insignificant
db_latency_floor_ms = 1.0
db_latency_value = metrics.get(MetricName.DATABASE_LATENCY, 0.0)
if db_latency_value < db_latency_floor_ms:
    actionable_signals = [
        s for s in actionable_signals
        if s.metric_name != MetricName.DATABASE_LATENCY
    ]
```

**Result**: Database latency signals below 1ms are now filtered out entirely, preventing noise alerts from statistically valid but operationally meaningless changes.

---

## Recommended Improvements by Priority

### Immediate (Before Next Release)

1. ~~Fix statistics/training data mismatch~~ (Fixed: unified data cleaning pipeline, no outlier removal)
2. ~~Increase minimum sample requirements to 256~~ (Fixed: now 500/1000)
3. ~~Add training data quality report (logging)~~ (Fixed: DataQualityReport implemented)

### Short-Term (Next Sprint)

4. ~~Implement temporal validation split~~ (Fixed: validation_fraction config option)
5. Add model performance metrics
6. ~~Correlate low latency with error rate~~ (Fixed: lower_is_better_metrics)

### Medium-Term (Next Quarter)

7. ~~Data-driven contamination tuning~~ (Fixed: knee-based estimation implemented)
8. ~~Model drift detection~~ (Fixed: z-score + Mahalanobis distance)
9. Stationarity checks

### Long-Term (Roadmap)

10. Walk-forward cross-validation
11. Metric-specific transformations
12. Automated retraining triggers

---

## How to Report New Issues

When discovering new issues:

1. **Document the issue** with:
   - Location (file and line number)
   - Problem description
   - Impact assessment
   - Reproduction steps (if applicable)

2. **Assign severity**:
   - **Critical**: Causes incorrect detection or missed incidents
   - **High**: Significant impact on accuracy or operations
   - **Medium**: Moderate impact, workarounds available
   - **Low**: Minor impact or cosmetic

3. **Add to this document** with status "Open"

4. **Update status** when fixed:
   - Open → In Progress → Fixed (version X.Y.Z)

---

## References

- [SRE_IMPROVEMENT_PLAN.md](./SRE_IMPROVEMENT_PLAN.md) - SRE-recommended improvements for alert quality
- [SEMANTIC_ANALYSIS_EXPERT_REVIEW.md](./SEMANTIC_ANALYSIS_EXPERT_REVIEW.md) - Full semantic analysis
- [TRAINING_CODE_EXPERT_REVIEW.md](./TRAINING_CODE_EXPERT_REVIEW.md) - Full training pipeline review
- [MACHINE_LEARNING.md](./MACHINE_LEARNING.md) - ML concepts and limitations
