# Known Issues and Limitations

This document tracks known issues, limitations, and planned improvements identified through expert reviews and production experience.

---

## Summary

| Category | Issues | Critical | High | Medium | Fixed |
|----------|--------|----------|------|--------|-------|
| Training Pipeline | 10 | 2 | 4 | 4 | 0 |
| Message Semantics | 12 | 1 | 2 | 8 | 1 |
| **Total** | **22** | **3** | **6** | **12** | **1** |

---

## Critical Issues

### 1. Statistics/Training Data Mismatch

**Status**: Open
**Severity**: Critical
**Location**: `smartbox_anomaly/detection/detector.py:270-389`

**Problem**: Statistics (p90, p95, etc.) are calculated on raw data, but models are trained on filtered data with outliers removed. This creates inconsistent thresholds.

```python
# Statistics calculated on RAW data (includes outliers)
stats = TrainingStatistics(p90=float(data.quantile(0.90)))

# Model trained on FILTERED data (outliers removed)
clean_data = data[(data <= p999) & (data >= p001)]
model.fit(clean_data)
```

**Impact**:
- Detection thresholds may be too high or too low
- p90 might include outlier values the model never saw
- Inconsistent severity classifications

**Workaround**: None currently. Accept some threshold inconsistency.

**Fix Required**: Calculate statistics on the same cleaned data used for training.

---

### 2. No Model Validation

**Status**: Open
**Severity**: Critical
**Location**: `smartbox_anomaly/detection/detector.py:146-199`

**Problem**: Models are trained on 100% of data with no validation split, cross-validation, or performance metrics.

**Impact**:
- No way to detect overfitting
- No quality metrics to compare model versions
- No early warning if model quality degrades

**Workaround**: Monitor false positive rates manually in production.

**Fix Required**: Implement temporal train/validation split (last 20% for validation).

---

### 3. Pattern Overlap: silent_degradation vs resource_contention

**Status**: Open
**Severity**: Critical
**Location**: `smartbox_anomaly/detection/interpretations.py`

**Problem**: These patterns have overlapping conditions but different interpretations:

| Condition | silent_degradation | resource_contention |
|-----------|-------------------|---------------------|
| request_rate | normal | normal |
| application_latency | high | high |
| error_rate | normal | low |

When both match, the wrong diagnosis may be presented.

**Impact**: Misleading root cause analysis in alerts.

**Workaround**: Manually verify root cause when seeing either pattern.

**Fix Required**: Add temporal analysis - recent latency increase = degradation, sustained high = contention.

---

## High Priority Issues

### 4. Hardcoded Contamination Rates

**Status**: Open
**Severity**: High
**Location**: `smartbox_anomaly/detection/service_config.py`

**Problem**: Contamination rates are hardcoded per service type without data-driven validation.

```python
KNOWN_SERVICE_PARAMS = {
    "booking": ServiceParameters(base_contamination=0.02),
    "search": ServiceParameters(base_contamination=0.04),
}
```

**Impact**:
- Rates may not reflect actual anomaly frequency
- Become stale as services evolve

**Workaround**: Manually adjust contamination in `config.json` based on observed false positive rates.

**Fix Required**: Implement contamination tuning via grid search with validation set.

---

### 5. No Temporal Validation for Time-Series

**Status**: Open
**Severity**: High
**Location**: `smartbox_anomaly/detection/time_aware.py:79-127`

**Problem**: Time-series data is split by period but not validated temporally. Random sampling during training can leak future information.

**Impact**:
- Model may not generalize to future patterns
- Overly optimistic performance expectations

**Workaround**: None. Accept potential generalization issues.

**Fix Required**: Implement walk-forward validation for time series.

---

### 6. Insufficient Minimum Sample Requirements

**Status**: Open
**Severity**: High
**Location**: `smartbox_anomaly/detection/detector.py:349,395`

**Problem**: Current minimums (50 univariate, 100 multivariate) are too low.

| Current | Recommended |
|---------|-------------|
| Univariate: 50 | 256 (IF recommendation) |
| Multivariate: 100 | 256 × n_features |

**Impact**:
- Unstable anomaly scores
- High variance in percentile estimation

**Workaround**: Ensure services have >256 samples per time period before relying on detection.

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
1. Added `lower_is_better_metrics()` in `constants.py` defining metrics where low values are improvements (`database_latency`, `client_latency`, `error_rate`)
2. Updated `_signals_to_levels()` to treat low values for these metrics as `"normal"` instead of `"low"`
3. `application_latency` is intentionally NOT in this list - low latency with high errors should still match fast-fail patterns
4. Pattern matching now uses fail-closed validation - unknown conditions don't silently pass

**Result**:
- Low `database_latency`/`client_latency`/`error_rate` → treated as normal (no alert)
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

**Status**: Open
**Severity**: Medium
**Location**: `smartbox_anomaly/detection/detector.py:391-452`

**Problem**: Multivariate model treats all features equally, even highly correlated ones (app_latency & client_latency).

**Impact**: Correlated features may dominate detection, harder to interpret results.

---

### 11. Missing Training Data Quality Report

**Status**: Open
**Severity**: Medium
**Location**: Not implemented

**Problem**: No visibility into training data quality - missing values, outliers, gaps are silently handled.

**Impact**: Can't detect data pipeline issues affecting model quality.

---

### 12. No Model Drift Detection

**Status**: Open
**Severity**: Medium
**Location**: Not implemented

**Problem**: No infrastructure to detect when models become stale.

**Impact**: Models silently degrade over time with no alerting.

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

## Recommended Improvements by Priority

### Immediate (Before Next Release)

1. Fix statistics/training data mismatch
2. Increase minimum sample requirements to 256
3. Add training data quality report (logging)

### Short-Term (Next Sprint)

4. Implement temporal validation split
5. Add model performance metrics
6. Correlate low latency with error rate

### Medium-Term (Next Quarter)

7. Data-driven contamination tuning
8. Model drift detection
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

- [SEMANTIC_ANALYSIS_EXPERT_REVIEW.md](./SEMANTIC_ANALYSIS_EXPERT_REVIEW.md) - Full semantic analysis
- [TRAINING_CODE_EXPERT_REVIEW.md](./TRAINING_CODE_EXPERT_REVIEW.md) - Full training pipeline review
- [MACHINE_LEARNING.md](./MACHINE_LEARNING.md) - ML concepts and limitations
