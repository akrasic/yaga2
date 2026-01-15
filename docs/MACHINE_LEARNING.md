# Machine Learning Guide

This document explains the machine learning concepts, algorithms, and design decisions behind the Smartbox Anomaly Detection system. It's designed for operators, developers, and anyone who wants to understand how the system works.

---

## Overview

The system uses **Isolation Forest**, an unsupervised machine learning algorithm, to detect anomalies in service metrics. It learns what "normal" behavior looks like from historical data and flags deviations.

```
Historical Metrics (30 days)
         │
         ▼
   ┌─────────────┐
   │  Training   │  → Learn normal patterns
   └─────────────┘
         │
         ▼
   ┌─────────────┐
   │   Models    │  → Stored per service + time period
   └─────────────┘
         │
         ▼
   ┌─────────────┐
   │  Inference  │  → Score new metrics, detect anomalies
   └─────────────┘
```

---

## Why Isolation Forest?

### The Core Idea

Isolation Forest is based on a simple observation: **anomalies are easier to isolate than normal points**.

Imagine plotting your service metrics as points in space:
- Normal behavior forms dense clusters (many similar data points)
- Anomalies are isolated points far from the clusters

The algorithm builds random decision trees that try to isolate each point. Anomalies get isolated quickly (short path), while normal points require many splits (long path).

```
Normal Point (hard to isolate):     Anomaly (easy to isolate):
        ┌───┬───┬───┐                      ┌───────────┐
        │ · │ · │ · │                      │     ·     │ ← isolated in 1 split
        ├───┼───┼───┤                      └───────────┘
        │ · │ X │ · │ ← many splits
        ├───┼───┼───┤
        │ · │ · │ · │
        └───┴───┴───┘
```

### Why It's Ideal for Service Metrics

| Property | Why It Matters |
|----------|----------------|
| **Unsupervised** | No labeled "anomaly" data required - learns from normal operation |
| **Fast** | Linear time complexity - handles high-volume metrics efficiently |
| **Robust to outliers** | Doesn't assume Gaussian distribution - handles latency spikes |
| **Works with skewed data** | Latency and error metrics are naturally skewed (long tails) |
| **Handles multiple metrics** | Can detect anomalies across correlated metrics |

### Alternatives Considered

| Algorithm | Why Not Used |
|-----------|--------------|
| Statistical (Z-score) | Assumes normal distribution - fails with latency data |
| LSTM/RNN | Requires labeled training data, computationally expensive |
| Local Outlier Factor | Slow at inference time, memory intensive |
| One-Class SVM | Sensitive to parameter tuning, doesn't scale well |

---

## Metrics Collected

The system monitors five core metrics per service:

| Metric | Description | Unit |
|--------|-------------|------|
| `request_rate` | Incoming requests per second | req/s |
| `application_latency` | Server-side processing time | ms |
| `dependency_latency` | Total round-trip time seen by clients | ms |
| `database_latency` | Time spent on database operations | ms |
| `error_rate` | Percentage of failed requests | % (0-1) |

Data is collected at **5-minute granularity** from VictoriaMetrics.

---

## Feature Engineering

Raw metrics are transformed into features that help the model understand patterns:

### Rolling Window Statistics

For each metric, we compute aggregates over multiple time windows:

| Window | Features Computed |
|--------|-------------------|
| 1 hour | mean, std, min, max |
| 6 hours | mean, std, min, max |
| 24 hours | mean, std, min, max |

This captures both short-term fluctuations and longer-term baselines.

### Additional Features

- **Lagged deltas**: How much did the metric change from the previous period?
- **Percentage change**: Rate of change relative to baseline
- **Time encodings**: Hour of day, day of week (cyclical encoding)

### Normalization

All features are normalized using **RobustScaler**, which uses median and interquartile range (IQR) instead of mean and standard deviation. This is resistant to outliers in the training data.

```
scaled_value = (value - median) / IQR
```

---

## Time-Aware Detection

Service behavior varies significantly by time period. A traffic pattern that's normal during business hours might be anomalous at 3 AM.

### Time Periods

| Period | Hours | Days |
|--------|-------|------|
| `business_hours` | 08:00 - 18:00 | Mon-Fri |
| `evening_hours` | 18:00 - 22:00 | Mon-Fri |
| `night_hours` | 22:00 - 06:00 | Mon-Fri |
| `weekend_day` | 08:00 - 22:00 | Sat-Sun |
| `weekend_night` | 22:00 - 08:00 | Sat-Sun |

### Separate Models Per Period

The system trains **separate models for each time period**. This prevents false positives from expected behavioral differences.

```
smartbox_models/
└── booking/
    ├── business_hours/
    │   ├── model.joblib
    │   └── metadata.json
    ├── evening_hours/
    │   └── ...
    ├── night_hours/
    │   └── ...
    ├── weekend_day/
    │   └── ...
    └── weekend_night/
        └── ...
```

---

## Anomaly Scoring

### How Scores Work

Isolation Forest produces a **decision function score** in the range `[-1, 1]`:

| Score Range | Meaning |
|-------------|---------|
| Close to 0 | Normal (near the cluster center) |
| Negative | Potentially anomalous |
| More negative | More confidently anomalous |

### Calibrated Severity Thresholds

**Important**: Severity thresholds are now **calibrated per model** based on the actual score distribution from training data, rather than using fixed hardcoded values.

During training, the system calculates severity thresholds from score percentiles:

| Severity | Percentile | Description |
|----------|------------|-------------|
| **Critical** | Bottom 0.1% | Extreme outliers |
| **High** | Bottom 1% | Significant anomalies |
| **Medium** | Bottom 5% | Moderate deviations |
| **Low** | Bottom 10% | Minor anomalies |

This approach ensures thresholds are appropriate for each service's unique characteristics.

### Fallback Thresholds

If calibrated thresholds are unavailable, the system falls back to these defaults:

| Severity | Score Threshold | Interpretation |
|----------|-----------------|----------------|
| **Critical** | < -0.6 | Severe anomaly, likely user-visible incident |
| **High** | -0.6 to -0.3 | Significant deviation, investigate promptly |
| **Medium** | -0.3 to -0.1 | Moderate anomaly, monitor closely |
| **Low** | >= -0.1 | Minor deviation or normal variance |

### Why Calibration Matters

Fixed thresholds can be problematic because:
- Different services have different score distributions
- A score of -0.3 might be rare for one service but common for another
- Calibration adapts to each service's natural variability

---

## Contamination Rate

### What Is Contamination?

**Contamination** is the expected proportion of anomalies in your training data. It tells the model "approximately X% of the training data should be considered anomalous."

```python
IsolationForest(contamination=0.05)  # Expect 5% anomalies
```

### Empirical Contamination Estimation

The system now **automatically estimates** the optimal contamination rate from your data using two methods:

#### Knee Detection Method (Default)
1. Train a preliminary model with `contamination="auto"`
2. Sort all anomaly scores from the training data
3. Find the "knee" point where the score curve bends sharply
4. The proportion of points beyond the knee becomes the estimated contamination

#### Gap Detection Method (Fallback)
1. Compute anomaly scores for all training data
2. Find the largest gap in the sorted score distribution
3. Points beyond the gap are considered anomalies

The estimated contamination is then bounded by service category limits to prevent extreme values:
- Minimum: 50% of category base contamination
- Maximum: 300% of category base (capped at 15%)

### How It Affects Detection

| Contamination | Effect |
|---------------|--------|
| **Lower (0.01-0.03)** | Stricter detection, fewer false positives, might miss real anomalies |
| **Higher (0.08-0.10)** | More sensitive, catches more anomalies, more false positives |

### Default Values by Category

| Category | Contamination | Rationale |
|----------|---------------|-----------|
| `critical` | 0.03 (3%) | Revenue-critical services - minimize false positives |
| `core` | 0.04 (4%) | Platform infrastructure - balanced detection |
| `standard` | 0.05 (5%) | Normal services - default sensitivity |
| `admin` | 0.06 (6%) | Back-office tools - slightly more tolerant |
| `micro` | 0.08 (8%) | Low-traffic services - higher variance expected |
| `background` | 0.08 (8%) | Workers/jobs - naturally variable |

### Tuning Contamination

**If you're getting too many false positives:**
```json
"contamination_by_service": {
  "my-noisy-service": 0.08
}
```

**If you're missing real anomalies:**
```json
"contamination_by_service": {
  "my-critical-service": 0.02
}
```

Note: Manual overrides take precedence over empirical estimation.

---

## Univariate vs Multivariate Detection

### Univariate Detection

Trains a **separate model for each metric**. Detects anomalies in individual metrics.

```
request_rate     → IF model → anomaly score
application_latency → IF model → anomaly score
error_rate       → IF model → anomaly score
```

**Best for**: Detecting when a single metric goes out of bounds.

### Multivariate Detection

Trains a **single model on all metrics together**. Detects anomalies in the relationships between metrics.

```
[request_rate, latency, errors] → IF model → anomaly score
```

**Best for**: Detecting unusual combinations, like:
- High latency with low traffic (unexpected)
- High errors with normal latency (fast failures)
- Traffic spike with no latency increase (capacity issue)

### How They Work Together

The system runs both types and combines results:

1. Check univariate anomalies for each metric
2. Check multivariate anomalies for pattern detection
3. Apply rule-based overrides for known critical conditions
4. Return the highest severity detected

---

## Rule-Based Overrides

Some conditions are always critical, regardless of ML score:

| Rule | Threshold | Severity |
|------|-----------|----------|
| Error rate spike | > 5% | Critical |
| Extreme latency | > 2000ms | Critical |
| High latency | > 1000ms | High |
| Traffic cliff | < 30% of normal | High |

These domain-knowledge rules ensure critical issues are never missed, even if the model hasn't seen similar patterns.

---

## Training Pipeline

### Data Requirements

| Requirement | Value | Rationale |
|-------------|-------|-----------|
| Minimum training period | 30 days | Capture weekly patterns |
| Minimum univariate samples | 500 | Statistical significance for single-metric models |
| Minimum multivariate samples | 1000 | Higher requirement for cross-metric correlations |
| Metric granularity | 5 minutes | Balance detail vs noise |

### Training Process

1. **Fetch historical data** from VictoriaMetrics (30 days)
2. **Segment by time period** (business hours, night, weekend, etc.)
3. **Engineer features** (rolling windows, deltas, scaling)
4. **Temporal train/validation split** (80% train, 20% validation - chronologically)
5. **Estimate contamination** empirically from score distribution (knee/gap detection)
6. **Train Isolation Forest** for each metric (univariate) on training split
7. **Train multivariate model** on combined features
8. **Calibrate severity thresholds** from validation data (percentile-based)
9. **Compute robust statistics** (trimmed mean, IQR-based std)
10. **Analyze feature correlation** for multivariate model health
11. **Store drift detection baseline** (mean, covariance for Mahalanobis distance)
12. **Validate** anomaly rate on held-out validation data
13. **Save models** with metadata (statistics, calibrated thresholds, feature names)

### Retraining Schedule

| Trigger | Recommendation |
|---------|----------------|
| Scheduled | Daily at low-traffic time (default: 2 AM) |
| Service changes | After deployments that change behavior |
| High false positive rate | Retrain with adjusted contamination |
| New service added | Train immediately after adding to config |

---

## Model Storage

Each trained model produces the following artifacts:

```
smartbox_models/<service>/<period>/
├── model.joblib           # Serialized Isolation Forest
├── scaler.joblib          # RobustScaler for normalization
└── metadata.json          # Training statistics and parameters
```

### Metadata Contents

```json
{
  "service_name": "booking",
  "period": "business_hours",
  "trained_at": "2024-01-15T02:00:00Z",
  "training_samples": 8640,
  "feature_columns": ["request_rate", "application_latency", ...],
  "contamination": 0.02,
  "n_estimators": 250,
  "statistics": {
    "request_rate": {
      "mean": 150.5,
      "std": 45.2,
      "p50": 145.0,
      "p95": 230.0,
      "p99": 280.0
    }
  }
}
```

---

## Explainability

Every anomaly includes explanatory context:

### Feature Contributions

Which metrics contributed most to the anomaly score:

```json
{
  "primary_contributor": "application_latency",
  "contribution_score": 0.72,
  "secondary_contributors": ["error_rate", "database_latency"]
}
```

### Statistical Context

How unusual the current values are:

```json
{
  "application_latency": {
    "current_value": 850,
    "baseline_mean": 200,
    "percentile": 99.5,
    "sigma_deviation": 3.2
  }
}
```

### Pattern Detection

Known patterns identified:

```json
{
  "pattern": "database_bottleneck",
  "confidence": 0.85,
  "description": "High database latency driving overall response time"
}
```

---

## Drift Detection

### What Is Drift?

**Model drift** occurs when the production data distribution differs significantly from the training data. This can happen due to:
- Seasonal changes in traffic patterns
- Infrastructure changes
- New features or user behavior changes
- Gradual degradation of model accuracy

### How Drift Detection Works

The system uses two complementary methods:

#### Univariate Drift (Z-Score)
For each metric, compare the current value to training statistics:

```
z_score = |current_value - training_mean| / training_std
```

| Z-Score | Interpretation |
|---------|----------------|
| < 3 | Normal variance |
| 3-5 | Moderate drift |
| > 5 | Severe drift |

#### Multivariate Drift (Mahalanobis Distance)
Detects drift in the relationships between metrics:

```
mahalanobis = sqrt((x - μ)ᵀ Σ⁻¹ (x - μ))
```

Where:
- `x` = current metric vector
- `μ` = training mean vector
- `Σ⁻¹` = inverse covariance matrix from training

A threshold based on feature count (`p + 3√(2p) + 3`) determines significant drift.

### Drift Impact on Confidence

When drift is detected, confidence scores are reduced:

| Drift Severity | Confidence Penalty |
|----------------|-------------------|
| Moderate (z > 3) | 15% reduction |
| Severe (z > 5) | 30% reduction |

This helps operators understand when detection results may be less reliable.

### Enabling Drift Detection

```json
{
  "inference": {
    "check_drift": true
  }
}
```

When enabled, each detection result includes a `drift_warning` field if drift is detected.

---

## Robust Statistics

### Why Robust Statistics?

Traditional mean and standard deviation are sensitive to outliers. Since outliers are exactly what we're trying to detect, using them in statistics can skew results.

### Robust Alternatives

| Traditional | Robust Alternative | Benefit |
|-------------|-------------------|---------|
| Mean | Trimmed Mean (1% trim) | Ignores extreme values |
| Standard Deviation | IQR / 1.349 | Based on quartiles, not affected by outliers |

### Implementation

```python
from scipy.stats import trim_mean, iqr

robust_mean = trim_mean(data, proportiontocut=0.01)
robust_std = iqr(data) / 1.349  # Normalized IQR
```

The factor 1.349 converts IQR to an estimate of standard deviation for normal distributions.

---

## Input Validation

### Inference Boundary Validation

Before metrics are passed to models, they are validated:

| Check | Action |
|-------|--------|
| NaN/Inf values | Replaced with 0.0 |
| Negative rates | Capped at 0.0 |
| Negative latencies | Capped at 0.0 |
| Extreme latencies (>5 min) | Capped at 300,000ms |
| Extreme request rates (>1M/s) | Capped at 1,000,000 |
| Error rates > 100% | Capped at 1.0 |

Invalid values generate warnings in the output.

---

## Limitations and Edge Cases

### Known Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| Cold start | New services have no baseline | Use conservative thresholds initially |
| Concept drift | Behavior changes over time | Daily retraining |
| Rare events | Once-a-year events may trigger false positives | Rule-based overrides for known events |
| Multimodal distributions | IF assumes single cluster | Time-period separation helps |

### When Detection May Fail

1. **Gradual degradation**: Slow drift may not trigger until severe
2. **Novel failure modes**: Patterns never seen in training
3. **Correlated failures**: Multiple services failing together may mask individual issues
4. **Seasonal patterns**: Annual events (Black Friday) may need special handling

### Recommendations

- Monitor false positive/negative rates
- Adjust contamination based on observed accuracy
- Add rule-based overrides for known failure patterns
- Consider manual review of alerts for the first 2 weeks after training

---

## Glossary

| Term | Definition |
|------|------------|
| **Calibrated Thresholds** | Severity thresholds derived from actual score percentiles, not hardcoded |
| **Contamination** | Expected proportion of anomalies in training data |
| **Decision function** | IF scoring function, returns anomaly score |
| **Drift Detection** | Monitoring for distribution changes between training and production |
| **Feature engineering** | Transforming raw metrics into model inputs |
| **Isolation Forest (IF)** | Unsupervised anomaly detection algorithm |
| **Knee Detection** | Method to find natural breakpoints in score distributions |
| **Mahalanobis Distance** | Multivariate distance measure accounting for correlations |
| **Multivariate** | Analysis considering multiple metrics together |
| **RobustScaler** | Normalization resistant to outliers |
| **Temporal Split** | Train/validation split respecting time order (no future leakage) |
| **Trimmed Mean** | Mean calculated after removing extreme values |
| **Univariate** | Analysis of a single metric |
| **Time period** | Behavioral segment (business hours, night, etc.) |
| **Validation Split** | Hold-out data for threshold calibration (default 20%) |

---

## Further Reading

- [Isolation Forest Paper](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf) - Original algorithm description
- [ML_TRAINING.md](./ML_TRAINING.md) - Technical deep-dive for ML engineers
- [CONFIGURATION.md](./CONFIGURATION.md) - All configuration options
- [FINGERPRINTING.md](./FINGERPRINTING.md) - Incident tracking system
