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

### 3.2 Model classes

#### Univariate detectors

For each metric `m`:

* Train `IsolationForest(IF_m)` on feature vector `[m_t]` across time.
* Scaling: `RobustScaler_m`.
* Training distribution statistics captured: mean, std, quantiles (p25–p99).

#### Multivariate detector

On subset of available metrics `M ⊆ {request_rate, latencies, error_rate}`:

* Train `IsolationForest(IF_multi)` on `[m1_t, m2_t, …]`.
* Scaling: `RobustScaler_multi`.

Multivariate IF provides **contextual anomaly scoring** across correlated metrics.

### 3.3 Auto-tuning & service-aware priors

* **Contamination rate (ρ)** adapted by service category (high-variance vs low-variance services).
* **n\_estimators** tuned upward for complex services (e.g., booking) vs leaner microservices.
* Bootstrap sampling enabled for robustness on long histories.

### 3.4 Validation

Each trained model is validated using:

* **Anomaly rate sanity check:** proportion of flagged points must fall below threshold.
* **Sample sufficiency:** enough points per period to generalize.
* **Explainability completeness:** must record per-feature statistics.

Models failing checks are skipped or flagged for retraining.

### 3.5 Persistence

Artifacts stored per `(service, period)`:

* Scaler(s) + Isolation Forest(s) serialized with `joblib`.
* Metadata JSON: feature columns, quantiles, zero-distribution stats, feature importances, tuned parameters.

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

### 5.1 Input

* Current metrics snapshot `x_t` (dict or dataclass).
* Timestamp `t`.

### 5.2 Period routing

* Map `t` → behavioral period P via deterministic function.
* Load `{service, P}` model set lazily (cached in memory).

### 5.3 Scoring

For each detector:

* Scale input `x_t` via trained `RobustScaler`.
* Compute IF anomaly score `s ∈ [-1, 1]`.

### 5.4 Thresholds and severity mapping

The decision function of Isolation Forest returns a **score** where higher means more normal, lower means more abnormal. In Smartbox we interpret scores with the following bands:

* **Critical:** `s < -0.6`
* **High:** `-0.6 ≤ s < -0.3`
* **Medium:** `-0.3 ≤ s < -0.1`
* **Low:** `s ≥ -0.1`

#### Why these thresholds?

* The scale of IF scores is **relative** to the training data distribution. In practice, values closer to 0 indicate the point is near the learned “normal” cluster boundary, while increasingly negative values indicate deeper isolation (i.e., easier to separate, hence more anomalous).
* **-0.1 boundary**: marks the start of mild deviation. Anything above this is considered low‑severity noise.
* **-0.3 boundary**: empirically found to correspond to the point where anomalies occur consistently outside 1–2 standard deviations of baseline. These often correspond to moderate operational issues.
* **-0.6 boundary**: much deeper outliers, often corresponding to events several standard deviations away from normal. Historically, such cases correlated strongly with user-visible incidents (e.g., spikes in latency >3× baseline or error rate surges).

Thus, -0.3 and -0.6 are “bad” because they indicate increasingly **confident anomalies** under the Isolation Forest: the more negative the score, the more the tree ensemble agrees that the point is isolated from the bulk of the distribution.

These boundaries were chosen through **empirical calibration** on historical service incidents, aligning model scores with known severities.

### 5.5 Rule-based overrides

Independent thresholds enforce domain knowledge:

* `error_rate > 5%` ⇒ Critical anomaly regardless of IF score.
* Latency spikes beyond 3σ from baseline trigger medium/high.

### 5.6 Explainability

Alongside severity, the system outputs:

* Feature contributions: which metrics shifted most.
* Percentile context: how unusual current values are (p90, p95, p99 reference).
* Business impact hints based on service category.

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

