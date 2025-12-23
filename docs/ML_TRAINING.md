# Smartbox Anomaly Detection – ML Training & Inference Deep Dive

*Audience: ML engineers, data scientists, advanced practitioners*

This document provides a **detailed technical deep dive** into the training and inference pipeline for Smartbox’s anomaly detection system. It builds on the high-level documentation and focuses on ML methodology, design rationale, and system tradeoffs.

---

## 1. End-to-End Workflow Overview

The anomaly detection pipeline consists of three main phases:

1. **Data ingestion** – pull raw metrics from VictoriaMetrics.
2. **Training** – feature engineering, model selection, parameter tuning, validation, persistence.
3. **Inference** – load trained models, score new metrics, aggregate anomaly signals, add contextual explainability.

Training is performed periodically (e.g., daily/weekly), while inference runs continuously on fresh metric snapshots.

---

## 2. Data Representation

### 2.1 Raw metrics

Pulled at **5-minute granularity** per service, including:

* Request rate
* Application latency
* Client latency
* Database latency
* Error rate

### 2.2 Engineered features

Raw series are transformed into a **feature matrix** `X ∈ R^(T×d)` with T timesteps and d features. Transformations include:

* **Rolling aggregates** (mean, std, min, max) over windows W ∈ {1h, 6h, 24h}.
* **Lagged deltas & pct\_change** for short-term dynamics.
* **Time-of-day and day-of-week encodings**.
* **Robust scaling** (median/IQR normalization).

This representation provides both **short-term fluctuations** and **longer contextual baselines**.

---

## 3. Training Pipeline

### 3.1 Period-based partitioning

Data is segmented into **behavioral periods**:

* Business hours, Evening, Night, Weekend-Day, Weekend-Night.

Each segment yields a **separate submodel**, improving specificity and reducing false positives.

### 3.2 Temporal train/validation split

**Critical for proper evaluation**: Data is split chronologically, not randomly.

* **Training set**: First 80% of data (chronologically)
* **Validation set**: Last 20% of data (chronologically)

This prevents **data leakage** where future information contaminates training.

#### Rolling Feature Leakage Prevention

Rolling window features (mean, std over past N hours) are computed **separately** for each split:

1. Compute base features on full dataset
2. Split into train/validation chronologically
3. Add rolling features to train set using only train data
4. Add rolling features to validation set using only validation data
5. Drop warm-up rows from validation (first 12 rows = 1 hour at 5-min granularity)

### 3.3 Model classes

#### Univariate detectors

For each metric `m`:

* Minimum samples: **500** (increased from 50 for statistical significance)
* Train `IsolationForest(IF_m)` on feature vector `[m_t]` across time.
* Scaling: `RobustScaler_m`.
* Training distribution statistics captured using **robust estimates**:
  - Trimmed mean (1% trim) instead of standard mean
  - IQR-based std (IQR/1.349) instead of standard deviation
  - Quantiles (p25–p99)

#### Multivariate detector

On subset of available metrics `M ⊆ {request_rate, latencies, error_rate}`:

* Minimum samples: **1000** (increased from 100 for reliable cross-metric patterns)
* Train `IsolationForest(IF_multi)` on `[m1_t, m2_t, …]`.
* Scaling: `RobustScaler_multi`.
* **Correlation analysis** performed to identify highly correlated features (r > 0.8).

Multivariate IF provides **contextual anomaly scoring** across correlated metrics.

### 3.4 Contamination estimation

Instead of using fixed contamination rates, the system **estimates optimal contamination** from the data:

#### Knee Detection Method (Default)
```python
# 1. Fit preliminary model
preliminary_model = IsolationForest(contamination="auto")
preliminary_model.fit(scaled_data)

# 2. Get sorted scores
scores = preliminary_model.decision_function(scaled_data)
sorted_scores = np.sort(scores)

# 3. Find knee point (second derivative maximum)
knee_idx = find_knee_point(sorted_scores)

# 4. Contamination = proportion of points beyond knee
estimated_contamination = (len(scores) - knee_idx) / len(scores)
```

#### Gap Detection Method (Fallback)
Finds the largest gap in the sorted score distribution.

Estimated contamination is **bounded** by service category limits:
* Minimum: 50% of category base
* Maximum: 300% of category base (capped at 15%)

### 3.5 Threshold calibration

Severity thresholds are **calibrated per model** from validation data percentiles:

| Severity | Percentile | Interpretation |
|----------|------------|----------------|
| Critical | 0.1% | Extreme outliers (1 in 1000) |
| High | 1% | Significant anomalies |
| Medium | 5% | Moderate deviations |
| Low | 10% | Minor anomalies |

```python
scores = model.decision_function(validation_data)
calibrated_thresholds = {
    "critical": float(np.percentile(scores, 0.1)),
    "high": float(np.percentile(scores, 1)),
    "medium": float(np.percentile(scores, 5)),
    "low": float(np.percentile(scores, 10)),
}
```

### 3.6 Auto-tuning & service-aware priors

* **Contamination rate (ρ)** estimated from data, bounded by service category.
* **n\_estimators** tuned upward for complex services (e.g., booking) vs leaner microservices.
* Bootstrap sampling enabled for robustness on long histories.

### 3.7 Drift detection baseline

For each trained model, drift detection baselines are stored:

* **Univariate**: Mean and std for each metric (for z-score calculation)
* **Multivariate**: Mean vector and inverse covariance matrix (for Mahalanobis distance)

```python
# Multivariate drift baseline
self.multivariate_mean = np.mean(scaled_data, axis=0)
self.multivariate_cov_inv = np.linalg.inv(np.cov(scaled_data.T) + 1e-6 * np.eye(p))
```

### 3.8 Validation

Each trained model is validated using:

* **Anomaly rate sanity check:** proportion of flagged points on validation set must fall below threshold.
* **Sample sufficiency:** minimum 500 (univariate) or 1000 (multivariate) points per period.
* **Explainability completeness:** must record per-feature statistics.
* **Correlation analysis:** warnings logged for highly correlated feature pairs (r > 0.8).

Models failing checks are skipped or flagged for retraining.

### 3.9 Persistence

Artifacts stored per `(service, period)`:

* Scaler(s) + Isolation Forest(s) serialized with `joblib`.
* Metadata JSON containing:
  - Feature columns
  - Robust statistics (trimmed mean, IQR-based std, quantiles)
  - **Calibrated severity thresholds**
  - **Estimated contamination**
  - **Drift detection baselines** (mean, covariance inverse)
  - **Correlation analysis results**
  - Tuned parameters

---

## 4. Isolation Forest Explained

Isolation Forest (IF) is the core model used for anomaly detection. It is based on the principle that **anomalies are easier to isolate than normal points**.

### 4.1 How it works

1. **Random partitioning**: IF builds many decision trees by recursively splitting feature space with randomly chosen split values.
2. **Isolation depth**: For each sample, the algorithm measures the number of splits (tree depth) required to isolate it from the rest.
3. **Scoring**:

   * Normal points → require more splits to separate (they are embedded in dense clusters).
   * Anomalies → isolated very quickly (few splits, shallow depth).
4. **Aggregation**: Average path lengths over all trees are normalized to produce an anomaly score in `[-1, 1]`.

### 4.2 Advantages

* **No labels required** (unsupervised).
* **Linear time complexity** → efficient for high-volume metrics.
* **Handles high-dimensional, skewed distributions**.
* **Robust to irrelevant features** since splits are random.

### 4.3 Why IF is suitable here

* Latency and error metrics often have **heavy-tailed distributions**. IF handles this naturally.
* Services differ drastically in volume; IF adapts since it only relies on relative path length.
* Works well for both **univariate (per metric)** and **multivariate (cross-metric)** training.

### 4.4 Mathematical intuition

Expected path length for a point `x`:

```
E[h(x)] ≈ c(n)
```

where `c(n)` is the average path length of unsuccessful search in a Binary Search Tree of size `n`.

Anomaly score:

```
s(x, n) = 2^(-E[h(x)] / c(n))
```

* If `s(x)` is close to 1 → highly anomalous.
* If `s(x)` is around 0 → normal.
* Smartbox rescales this score to \[-1, 1] for consistency.

---

## 5. Inference Pipeline

### 5.1 Input validation

Before processing, metrics are validated at the inference boundary:

| Check | Action |
|-------|--------|
| NaN/Inf values | Replaced with 0.0 |
| Negative rates/latencies | Capped at 0.0 |
| Extreme latencies (>5 min) | Capped at 300,000ms |
| Extreme request rates (>1M/s) | Capped at 1,000,000 |
| Error rates > 100% | Capped at 1.0 |

Validation issues are captured in `validation_warnings` array.

### 5.2 Input

* Current metrics snapshot `x_t` (dict or dataclass).
* Timestamp `t` (per-service to avoid time period mismatch).

### 5.3 Period routing

* Map `t` → behavioral period P via deterministic function.
* Load `{service, P}` model set lazily (cached in memory).

### 5.4 Scoring

For each detector:

* Scale input `x_t` via trained `RobustScaler`.
* Compute IF anomaly score `s ∈ [-1, 1]`.

### 5.5 Thresholds and severity mapping

Severity thresholds are now **calibrated per model** based on validation data percentiles:

| Severity | How determined |
|----------|---------------|
| Critical | Score below 0.1th percentile of validation scores |
| High | Score below 1st percentile |
| Medium | Score below 5th percentile |
| Low | Score below 10th percentile |

#### Fallback thresholds

If calibrated thresholds unavailable:

* **Critical:** `s < -0.6`
* **High:** `-0.6 ≤ s < -0.3`
* **Medium:** `-0.3 ≤ s < -0.1`
* **Low:** `s ≥ -0.1`

### 5.6 Drift detection at inference

When `check_drift: true` in config, each inference checks for distribution drift:

#### Univariate drift (z-score)
```python
z_score = abs((current_value - training_mean) / (training_std + 1e-8))
```

| Z-Score | Meaning | Confidence Penalty |
|---------|---------|-------------------|
| < 3 | Normal | 0% |
| 3-5 | Moderate drift | 15% |
| > 5 | Severe drift | 30% |

#### Multivariate drift (Mahalanobis distance)
```python
diff = current_vector - multivariate_mean
mahalanobis = sqrt(diff @ multivariate_cov_inv @ diff)
threshold = p + 3 * sqrt(2 * p) + 3  # p = feature count
```

If Mahalanobis distance exceeds threshold, multivariate drift is flagged.

#### Drift impact

When drift is detected:
1. `drift_warning` added to output
2. Confidence scores on anomalies reduced by penalty
3. Anomalies marked with `drift_warning: true`

### 5.7 Rule-based overrides

Independent thresholds enforce domain knowledge:

* `error_rate > 5%` ⇒ Critical anomaly regardless of IF score.
* Latency spikes beyond 3σ from baseline trigger medium/high.

### 5.8 Explainability

Alongside severity, the system outputs:

* Feature contributions: which metrics shifted most.
* Percentile context: how unusual current values are (p90, p95, p99 reference).
* Business impact hints based on service category.
* **Drift analysis**: when enabled, per-metric drift scores and recommendations.
* **Validation warnings**: any input sanitization that was performed.

Result is an **anomaly report object** containing structured JSON for downstream alerting.

---

## 6. Design Rationale

* **Isolation Forest** chosen for robustness on high-dimensional, unlabeled, skewed data.
* **RobustScaler** protects against heavy tails common in latency distributions.
* **Period-aware training** prevents misclassification of expected traffic patterns.
* **Rule-based checks** add domain priors to catch anomalies outside ML’s sensitivity.
* **Explainability layer** increases trust and actionability of alerts.

---

## 7. Mathematical Summary

Given input features `X = {x_1, …, x_T}` for a service period:

1. Scale: `z_t = RobustScaler(x_t)`
2. Train IF: `IF.fit({z_1, …, z_T})`
3. Inference: `score_t = IF.decision_function(z_t)`
4. Predict anomaly: `label_t = 1[score_t < τ]` where τ ∈ {−0.1, −0.3, −0.6} depending on severity band.
5. Map severity via piecewise thresholds.

