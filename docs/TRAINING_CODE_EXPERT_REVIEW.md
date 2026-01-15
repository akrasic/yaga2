# Training Code Expert Review

> **⚠️ HISTORICAL DOCUMENT - December 2025**
>
> This expert review was conducted on the original training pipeline. **All critical and high-priority issues identified in this document have been addressed** in subsequent updates. See the changelog below for implementation status.
>
> | Issue | Status | Implementation |
> |-------|--------|----------------|
> | Statistics/training data mismatch | ✅ Fixed | Robust statistics (trimmed mean, IQR-based std) |
> | No validation split | ✅ Fixed | Temporal 80/20 train/validation split |
> | Hardcoded thresholds | ✅ Fixed | Per-model calibrated thresholds from percentiles |
> | Contamination estimation | ✅ Fixed | Knee/gap detection for empirical estimation |
> | Minimum sample size | ✅ Fixed | 500 univariate, 1000 multivariate |
> | Rolling feature leakage | ✅ Fixed | Separate rolling computation per split |
> | No drift detection | ✅ Fixed | Z-score + Mahalanobis distance |
>
> For current implementation details, see [ML_TRAINING.md](./ML_TRAINING.md) and [MACHINE_LEARNING.md](./MACHINE_LEARNING.md).

---

## Executive Summary

This document provides an expert review of the anomaly detection training pipeline from both **observability** and **machine learning** best practices perspectives. The analysis covers data preparation, model training, statistical computation, and validation approaches.

---

## Critical Issues

### 1. Statistics Calculated on Unfiltered Data, Model Trained on Filtered Data

**Location:** `detector.py:270-320` (statistics) vs `detector.py:322-389` (training)

**Problem:**
```python
# _calculate_training_statistics - uses RAW data
data = features_df[metric].dropna()
stats = TrainingStatistics(
    p90=float(data.quantile(0.90)),  # Calculated on ALL data including outliers
    ...
)

# _train_univariate_model - uses FILTERED data
clean_data = data.dropna().replace([np.inf, -np.inf], np.nan).dropna()
if len(clean_data) > 100:
    p999 = clean_data.quantile(0.999)
    p001 = clean_data.quantile(0.001)
    clean_data = clean_data[(clean_data <= p999) & (clean_data >= p001)]  # Outliers removed!
```

**Impact:** The statistics used for level classification (`_calculate_metric_levels`) include outliers, but the model was trained without them. This creates a mismatch:
- p90 might be artificially high due to outliers in statistics
- Model doesn't know about those outliers
- Detection thresholds become inconsistent

**Fix:** Calculate statistics AFTER the same cleaning applied during model training, or use the same data reference.

---

### 2. No Cross-Validation or Model Validation

**Location:** `detector.py:146-199`

**Problem:** Models are trained on 100% of data with no validation:
```python
def train(self, features_df: pd.DataFrame) -> None:
    # No train/validation split
    # No cross-validation
    # No hold-out test set
    model.fit(scaled_data)  # Just fit and done
```

**Impact:**
- No way to detect overfitting
- No performance metrics to compare model versions
- No early warning if model quality degrades

**Best Practice:**
```python
# Should implement:
# 1. Time-based train/validation split (last 20% for validation)
# 2. Cross-validation for contamination parameter tuning
# 3. Validation metrics: reconstruction error, false positive rate on known-good data
```

---

### 3. Hardcoded Contamination Without Data-Driven Validation

**Location:** `service_config.py` and `detector.py:355-359`

**Problem:** Contamination rate is either hardcoded or heuristically adjusted:
```python
# service_config.py - hardcoded per service type
KNOWN_SERVICE_PARAMS = {
    "booking": ServiceParameters(base_contamination=0.02, ...),
    "search": ServiceParameters(base_contamination=0.04, ...),
}

# Auto-adjustment based on CV, not actual anomaly rate
if cv > 3.0:
    contamination = min(0.20, base * 2.0)
```

**Impact:**
- Contamination should reflect actual expected anomaly rate
- Different services have different baseline failure rates
- Hardcoded values become stale as services evolve

**Best Practice:**
- Use historical incident data to estimate true anomaly rate
- Implement contamination tuning via grid search with validation set
- Log and monitor false positive rates in production

---

### 4. No Temporal Validation for Time-Series Data

**Location:** `time_aware.py:79-127`

**Problem:** Time-series data is split by period but not validated temporally:
```python
for period in self.TIME_PERIODS:
    period_data = features_df[features_df["time_period"] == period]
    model.train(period_data)  # No temporal ordering considered
```

**Impact:**
- Random sampling during training can leak future information
- Model may not generalize to future patterns
- Seasonal patterns within periods not captured

**Best Practice:**
```python
# Should implement:
# 1. Temporal train/test split (train on earlier data, test on later)
# 2. Walk-forward validation for time series
# 3. Seasonal decomposition awareness
```

---

### 5. Inconsistent Minimum Sample Requirements

**Location:** `detector.py:349` (univariate: 50), `detector.py:395` (multivariate: 100)

**Problem:** Sample requirements are arbitrary and don't scale with feature count:
```python
if len(clean_df) < 50:  # Univariate - 50 samples
    return False

if len(clean_data) < 100:  # Multivariate - 100 samples for 5 features
    return False
```

**Impact:**
- 100 samples for 5 features = 20 samples per feature (too low for reliable estimation)
- Isolation Forest recommends `n_samples >= 256` for stable anomaly scores
- Percentile estimation with 50 samples has high variance

**Best Practice:**
- Minimum 256 samples for Isolation Forest
- For multivariate: `min_samples = 50 * n_features` (at least)
- For percentile estimation: `min_samples = 100 / (1 - percentile)` (e.g., 1000 for p99)

---

### 6. No Stationarity Check

**Location:** Not implemented

**Problem:** Time-series anomaly detection assumes stationarity, but this isn't verified:
```python
# Current code doesn't check for:
# - Trends (gradual increase/decrease)
# - Seasonality beyond time-of-day
# - Regime changes (step changes in mean/variance)
```

**Impact:**
- Training on non-stationary data produces invalid statistics
- p90 from 3 months ago may not apply to current data
- Model becomes stale faster than expected

**Best Practice:**
```python
# Should implement:
from scipy.stats import adfuller

def check_stationarity(data):
    result = adfuller(data.dropna())
    p_value = result[1]
    return p_value < 0.05  # Stationary if p < 0.05
```

---

### 7. RobustScaler May Remove Important Signal

**Location:** `detector.py:362-363`

**Problem:** RobustScaler uses median/IQR, which can over-normalize legitimate patterns:
```python
scaler = RobustScaler()  # Uses median and IQR
scaled_data = scaler.fit_transform(clean_df)
```

**Impact:**
- For metrics with legitimate high variance (request_rate), scaling compresses signal
- Spikes that are 5x median become indistinguishable from 3x median
- Better for latency metrics than traffic metrics

**Best Practice:**
```python
# Consider metric-specific scaling:
if metric_name == MetricName.REQUEST_RATE:
    # Log-transform for multiplicative patterns
    scaled_data = np.log1p(clean_df)
elif metric_name == MetricName.ERROR_RATE:
    # Logit transform for bounded [0,1] data
    scaled_data = np.log(clean_df / (1 - clean_df + 1e-8))
else:
    # RobustScaler for latency metrics
    scaler = RobustScaler()
```

---

### 8. No Feature Correlation Analysis

**Location:** `detector.py:391-452`

**Problem:** Multivariate model treats all features equally:
```python
def _train_multivariate_model(self, core_metrics_df: pd.DataFrame) -> bool:
    # No correlation analysis
    # No feature selection
    # No collinearity check
    model.fit(scaled_data)  # All features weighted equally
```

**Impact:**
- Highly correlated features (app_latency & dependency_latency) may dominate
- Redundant features reduce model effectiveness
- Harder to interpret which features drive anomaly

**Best Practice:**
```python
# Should implement:
correlation_matrix = core_metrics_df.corr()
# Remove features with correlation > 0.9
# Or use PCA for dimensionality reduction
# Log correlation structure for explainability
```

---

### 9. Missing Training Data Quality Report

**Location:** Not implemented

**Problem:** No visibility into training data quality:
```python
# Current code silently handles:
# - Missing values (just dropped)
# - Outliers (removed without logging counts)
# - Zero values (handled but not reported)
```

**Impact:**
- Operators don't know if training data is representative
- Can't detect data pipeline issues (e.g., metrics stopped reporting)
- No audit trail for model decisions

**Best Practice:**
```python
class TrainingDataQualityReport:
    total_samples: int
    missing_values: dict[str, int]  # per metric
    outliers_removed: dict[str, int]
    zero_values: dict[str, int]
    time_coverage: dict[str, float]  # % of each period covered
    data_gaps: list[tuple[datetime, datetime]]  # gaps > 1 hour
```

---

### 10. No Model Drift Detection Setup

**Location:** Not implemented

**Problem:** No infrastructure to detect when model becomes stale:
```python
# Current model stores:
model_metadata = {
    "trained_at": datetime.now().isoformat(),  # Just timestamp
    # No baseline metrics for drift detection
}
```

**Impact:**
- Models silently degrade over time
- No alerting when retraining is needed
- No comparison between model versions

**Best Practice:**
```python
model_metadata = {
    "trained_at": ...,
    "training_data_hash": hash_training_data(features_df),
    "baseline_metrics": {
        "mean_anomaly_score": np.mean(model.decision_function(validation_data)),
        "score_std": np.std(model.decision_function(validation_data)),
        "expected_anomaly_rate": contamination,
    },
    "drift_thresholds": {
        "score_mean_shift": 0.1,  # Alert if mean shifts by 0.1
        "anomaly_rate_shift": 0.5,  # Alert if rate doubles
    },
}
```

---

## Medium Priority Issues

### 11. Isolation Forest Parameters Not Optimized

**Location:** `detector.py:369`

**Problem:** Uses mostly default parameters:
```python
model = IsolationForest(**params.to_isolation_forest_params())
# max_features defaults to 1.0 (all features)
# max_samples defaults to "auto" (256 or n_samples)
```

**Best Practice:**
- `max_features`: Use sqrt(n_features) for multivariate
- `max_samples`: 256 is usually sufficient, larger just adds computation
- `n_estimators`: 100-200 is usually optimal; 500 is overkill

---

### 12. No Handling of Concept Drift in Time Periods

**Location:** `time_aware.py:79-127`

**Problem:** All historical data for a period is treated equally:
```python
period_data = features_df[features_df["time_period"] == period]
model.train(period_data)  # Monday 3 months ago weighted same as yesterday
```

**Best Practice:**
- Use exponential weighting to favor recent data
- Or sliding window training (last N weeks only)
- Detect regime changes within periods

---

### 13. Statistics Don't Account for Autocorrelation

**Location:** `detector.py:270-320`

**Problem:** Standard deviation assumes independent samples:
```python
std=float(data.std()),  # Assumes i.i.d. samples
```

**Impact:**
- Time series data is autocorrelated (consecutive samples are similar)
- Effective sample size is smaller than actual sample count
- Confidence intervals are overconfident

**Best Practice:**
```python
# Calculate effective sample size
from statsmodels.tsa.stattools import acf
autocorr = acf(data, nlags=50)
effective_n = len(data) / (1 + 2 * sum(autocorr[1:]))
```

---

## Recommendations by Priority

### Immediate (Before Next Training Run)

1. **Fix statistics/training data mismatch** - Calculate statistics on the same cleaned data used for training
2. **Increase minimum sample requirements** - At least 256 for univariate, 256 * n_features for multivariate
3. **Add training data quality report** - Log missing values, outliers removed, coverage

### Short-Term (Next Sprint)

4. **Implement validation split** - Use last 20% of time-ordered data for validation
5. **Add model performance metrics** - Track validation set anomaly scores, false positive rate
6. **Implement stationarity check** - Warn if training data shows strong trends

### Medium-Term (Next Quarter)

7. **Data-driven contamination tuning** - Grid search with validation set
8. **Model drift detection** - Store baseline metrics, alert on drift
9. **Feature correlation analysis** - Log correlation structure, consider PCA

### Long-Term (Roadmap)

10. **Walk-forward validation** - Proper time series cross-validation
11. **Metric-specific transformations** - Log for rates, logit for proportions
12. **Automated retraining triggers** - Based on drift detection

---

## Proposed Code Structure

```python
class EnhancedAnomalyTrainer:
    """Production-grade anomaly detection trainer."""

    def train(self, features_df: pd.DataFrame) -> TrainingResult:
        # 1. Data quality assessment
        quality_report = self._assess_data_quality(features_df)
        if not quality_report.is_sufficient:
            raise InsufficientDataError(quality_report)

        # 2. Stationarity check
        stationarity = self._check_stationarity(features_df)
        if not stationarity.is_stationary:
            logger.warning(f"Non-stationary data: {stationarity.diagnostics}")

        # 3. Clean data (single pipeline for stats AND model)
        clean_df = self._clean_data(features_df)

        # 4. Train/validation split (temporal)
        train_df, val_df = self._temporal_split(clean_df, val_ratio=0.2)

        # 5. Calculate statistics on TRAINING data only
        self._calculate_training_statistics(train_df)

        # 6. Train model with hyperparameter tuning
        best_params = self._tune_contamination(train_df, val_df)
        self._train_models(train_df, best_params)

        # 7. Validate on held-out data
        val_metrics = self._validate_model(val_df)

        # 8. Store drift detection baseline
        self._store_drift_baseline(val_df, val_metrics)

        return TrainingResult(
            quality_report=quality_report,
            stationarity=stationarity,
            validation_metrics=val_metrics,
            model_metadata=self.model_metadata,
        )
```

---

## Summary

The current training pipeline is functional but lacks several production-grade safeguards:

| Category | Current | Recommended |
|----------|---------|-------------|
| Validation | None | Temporal train/val split |
| Data Quality | Silent handling | Explicit quality report |
| Statistics | On raw data | On cleaned training data |
| Sample Size | 50-100 | 256+ (IF recommendation) |
| Drift Detection | None | Baseline metrics + alerting |
| Contamination | Hardcoded | Data-driven tuning |

Implementing these improvements will significantly increase model reliability and operational confidence.
