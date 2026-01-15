# Isolation Forest (ML Detection)

Isolation Forest is an unsupervised machine learning algorithm that detects anomalies by measuring how easy it is to "isolate" a data point. This chapter explains how it works, why it's effective for service metrics, and how Yaga2 uses it.

## What is Unsupervised Learning?

Machine learning algorithms fall into two main categories:

- **Supervised learning**: Requires labeled training data. For anomaly detection, you'd need examples labeled "normal" and "anomaly."
- **Unsupervised learning**: Learns patterns from unlabeled data. No need to manually identify anomalies in advance.

**Why unsupervised matters for anomaly detection:**

Labeling anomalies is impractical:
- You can't predict all future failure modes
- What's "anomalous" changes as services evolve
- Manual labeling is expensive and error-prone

Isolation Forest learns what's "normal" from your historical data and flags anything that deviates—no labels required.

## The Core Insight

**Anomalies are easier to isolate than normal points.**

Think about it visually:

```
Normal data (clustered):           Anomaly (isolated):

        ●  ●  ●
      ● ● ●● ● ●                              ●
     ● ● ●●●● ● ●                     (far from cluster)
      ● ● ●● ● ●
        ●  ●  ●

   Many similar points               Alone in feature space
   Hard to distinguish               Easy to identify
```

If you were to randomly draw dividing lines to separate data points:
- **Normal points**: Buried in a cluster, need many cuts to isolate
- **Anomalies**: Sitting alone, isolated in just 1-2 cuts

## How Isolation Forest Works

### Step 1: Build Random Trees (Training)

Isolation Forest builds many random decision trees (called "isolation trees"):

```
                    All Data Points
                          │
                ┌─────────┴─────────┐
                │ Split on latency  │
                │ at random value   │
                │ (e.g., 234ms)     │
                └─────────┬─────────┘
                          │
            ┌─────────────┴─────────────┐
            │                           │
     latency < 234ms              latency >= 234ms
            │                           │
            ▼                           ▼
     ┌──────────────┐            ┌──────────────┐
     │ Split on     │            │ Split on     │
     │ request_rate │            │ error_rate   │
     │ at 150 req/s │            │ at 0.05      │
     └──────────────┘            └──────────────┘
            │                           │
           ...                         ...
     (continue until each             (continue)
      point is isolated)
```

Each tree:
1. **Randomly picks a feature** (e.g., latency, error_rate)
2. **Randomly picks a split value** between the min and max of that feature
3. **Recursively partitions** until each point is isolated

### Step 2: Measure Path Length

For each data point, count how many splits were needed to isolate it:

```
Anomaly path:                    Normal point path:

Root ──▶ Left ──▶ ISOLATED       Root ──▶ Right ──▶ Left ──▶ Right ──▶ Left ──▶ ISOLATED

Path length: 2                   Path length: 5
(suspicious!)                    (normal)
```

**Key principle**:
- Short paths → Anomalous (easy to isolate)
- Long paths → Normal (hard to isolate)

### Step 3: Compute Anomaly Score

Average the path lengths across all trees and normalize:

```
                        2^(-average_path_length / c(n))
Anomaly Score = ─────────────────────────────────────────
                              (normalized)
```

Where `c(n)` is the average path length of an unsuccessful search in a binary search tree of n samples.

**Score interpretation:**
- **Score ≈ 1**: Highly anomalous (very short paths)
- **Score ≈ 0.5**: On the boundary
- **Score ≈ 0**: Normal (long paths)

Yaga2 uses the scikit-learn `decision_function()` which returns scores in [-1, 1]:
- **Negative scores**: Anomalous (more negative = more anomalous)
- **Positive scores**: Normal

## Visual Example

Consider 2D data with latency and request_rate:

```
request_rate
     │
 500 │                    ★ Anomaly A
     │                      (high rate,
     │                       isolated quickly)
 300 │    ●●●●●
     │   ●●●●●●●
 200 │  ●●●●●●●●●   Normal cluster
     │   ●●●●●●●    (many splits needed)
 100 │    ●●●●●
     │
  50 │  ★ Anomaly B
     │  (low rate, isolated quickly)
     │
     └────────────────────────────────── latency
          100    200    300    400    500

Random split 1: latency < 350ms
Random split 2: request_rate < 400

Result:
- Anomaly A: Isolated in 2 splits (score ≈ -0.5)
- Anomaly B: Isolated in 2 splits (score ≈ -0.5)
- Normal points: Need 5-8 splits (score ≈ 0.1)
```

## Why Isolation Forest is Ideal for Service Metrics

| Property | Why It Matters for Metrics |
|----------|---------------------------|
| **Unsupervised** | No need to label historical incidents |
| **Fast** | O(n log n) - handles high-volume metrics |
| **Handles skew** | Latency distributions are typically long-tailed |
| **No distribution assumptions** | Unlike z-score, doesn't assume Gaussian |
| **Works with multiple features** | Can detect unusual metric *combinations* |

### Comparison with Other Methods

| Method | Limitation for Metrics |
|--------|----------------------|
| **Z-score / Standard Deviation** | Assumes normal distribution - latency is skewed |
| **Static thresholds** | Brittle, need constant tuning |
| **LSTM / Deep Learning** | Requires labeled data, computationally expensive |
| **Local Outlier Factor** | Slow at inference time |
| **One-Class SVM** | Sensitive to hyperparameters |

## Model Types in Yaga2

### Univariate Models (Per-Metric)

Separate model for each metric, detecting single-metric anomalies:

| Model | What It Detects | Example |
|-------|-----------------|---------|
| `request_rate_isolation` | Traffic anomalies | Sudden drop to 10% of normal |
| `application_latency_isolation` | Response time issues | Latency spike to 500ms (normally 100ms) |
| `error_rate_isolation` | Error spikes | Error rate jumps to 5% (normally 0.1%) |
| `dependency_latency_isolation` | Dependency slowness | External calls taking 2s (normally 200ms) |
| `database_latency_isolation` | Database issues | Query time spikes |

### Multivariate Model (Combined)

One model considering all metrics together. This detects unusual **combinations** that look normal individually:

```
Example: Fast-Fail Pattern

Individual metrics:
  request_rate: 100 req/s     ← Normal
  application_latency: 5ms    ← Very fast (good?)
  error_rate: 40%             ← High

Analysis:
  - Each metric might not trigger univariate detection
  - But the COMBINATION is suspicious:
    "Very fast responses + high errors = requests failing before processing"

  Multivariate model catches this pattern!
```

## Contamination Rate

**Contamination** is a key hyperparameter—the expected proportion of anomalies in training data.

```python
IsolationForest(contamination=0.05)  # Expect 5% anomalies
```

| Contamination | Effect | Use Case |
|---------------|--------|----------|
| **0.01-0.03** | Very strict, few anomalies flagged | Critical services where false positives are costly |
| **0.05** | Balanced (default) | Most services |
| **0.08-0.10** | More sensitive, more anomalies flagged | Noisy services, development environments |

### How Yaga2 Sets Contamination

Yaga2 estimates contamination automatically using **knee detection**:

1. Train a preliminary model with `contamination="auto"`
2. Get anomaly scores for all training data
3. Sort scores and find the "knee" (bend in the curve)
4. Points beyond the knee are considered anomalies
5. Their proportion becomes the estimated contamination

```
Anomaly Scores (sorted)

     │
  0  │────────────────────●●●●●●●●●●●●●●●
     │                   ●
     │                  ●
-0.2 │                 ●  ← Knee point
     │               ●
     │             ●
-0.5 │          ●
     │       ●
     │    ●
-0.8 │ ●
     └─────────────────────────────────────
       1   20   40   60   80  100 (samples)

Knee at sample 95 → contamination ≈ 5%
```

## Severity Thresholds

Anomaly scores are mapped to severity levels:

### Calibrated Thresholds (Preferred)

During training, thresholds are calibrated from validation data:

| Severity | Percentile | Meaning |
|----------|------------|---------|
| **Critical** | Bottom 0.1% | 1 in 1000 - extremely rare |
| **High** | Bottom 1% | 1 in 100 - significant |
| **Medium** | Bottom 5% | 1 in 20 - moderate |
| **Low** | Bottom 10% | 1 in 10 - minor |

### Fallback Thresholds

If calibration data unavailable:

| Severity | Score Range |
|----------|-------------|
| **Critical** | < -0.6 |
| **High** | -0.6 to -0.3 |
| **Medium** | -0.3 to -0.1 |
| **Low** | >= -0.1 |

## Training Process

### Data Requirements

| Requirement | Value | Why |
|-------------|-------|-----|
| Training period | 30 days | Capture weekly patterns |
| Min univariate samples | 500 | Statistical significance |
| Min multivariate samples | 1000 | Need more for cross-metric patterns |
| Data granularity | 5 minutes | Balance detail vs noise |

### Training Steps

1. **Fetch 30 days of metrics** from VictoriaMetrics
2. **Segment by time period** (business hours, night, weekend)
3. **Temporal split**: 80% training, 20% validation (chronological)
4. **Estimate contamination** from data distribution
5. **Train univariate models** for each metric
6. **Train multivariate model** on all metrics
7. **Calibrate severity thresholds** on validation data
8. **Save models and metadata**

### Why Temporal Split?

```
WRONG: Random split
─────────────────────
Training:   ●  ●  ●  ●  ●  ●  ●  ●  (random samples)
Validation:    ●     ●     ●     ●  (random samples)

Problem: Training might include future data → data leakage


CORRECT: Temporal split
───────────────────────
Training:   ●──●──●──●──●──●──●──●──●──│  (first 80%)
Validation:                            │──●──●──●──●  (last 20%)
                                       │
                                    Split point

Model learns only from past, tested on "future" data
```

## Time-Aware Models

Service behavior varies by time of day:

| Time Period | Characteristics |
|-------------|-----------------|
| business_hours (08-18 weekdays) | High traffic, tight latency |
| evening_hours (18-22 weekdays) | Moderate traffic |
| night_hours (22-06 weekdays) | Low traffic, batch jobs |
| weekend_day (08-22 Sat-Sun) | Variable patterns |
| weekend_night (22-08 Sat-Sun) | Very low traffic |

**Why separate models?**

Without time awareness:
```
3 AM: 10 requests/second
  → Model: "This is way below the 500 req/s average - ANOMALY!"
  → Reality: This is normal at 3 AM
  → Result: False positive
```

With time-aware models:
```
3 AM: 10 requests/second
  → Night model average: 15 req/s
  → Model: "This is close to the night baseline - NORMAL"
  → Result: Correct!
```

## Strengths and Limitations

### Strengths

| Strength | Explanation |
|----------|-------------|
| **No labeled data needed** | Learns from normal patterns, no need to tag anomalies |
| **Fast training and inference** | O(n log n) complexity, scales to millions of points |
| **Works with skewed data** | Latency distributions are naturally long-tailed |
| **Detects novel anomalies** | Can find patterns never seen before |
| **Handles multiple dimensions** | Multivariate model catches unusual combinations |

### Limitations

| Limitation | How Yaga2 Mitigates |
|------------|---------------------|
| **No semantic understanding** | Pattern matching interprets what anomalies mean |
| **Context-blind** | Time-aware models reduce false positives |
| **Can produce false positives** | SLO evaluation filters operationally insignificant anomalies |
| **Training data quality matters** | Data quality checks and minimum sample requirements |
| **Gradual drift may be missed** | Daily retraining, drift detection |

## How Detection Works at Inference Time

```
1. Current Metrics Arrive
   {request_rate: 450, latency: 280, errors: 0.02, ...}
                    │
                    ▼
2. Determine Time Period
   Current time: 10:30 AM Wednesday → "business_hours"
                    │
                    ▼
3. Load Appropriate Models
   booking/business_hours/model.joblib
                    │
                    ▼
4. Scale Metrics
   RobustScaler normalizes to training distribution
                    │
                    ▼
5. Score Each Model
   Univariate: latency_score = -0.35 (high!)
               request_score = 0.1 (normal)
               error_score = 0.05 (normal)
   Multivariate: combined_score = -0.28 (elevated)
                    │
                    ▼
6. Map to Severity
   latency_score -0.35 → severity: "high"
                    │
                    ▼
7. Pattern Matching
   High latency + normal traffic + normal errors
   → Pattern: "latency_spike_recent"
                    │
                    ▼
8. Output Anomaly
   {
     "pattern_name": "latency_spike_recent",
     "severity": "high",
     "score": -0.35,
     "description": "Latency spike: 280ms (normally 120ms)"
   }
```

## Further Reading

- [Pattern Matching](./pattern-matching.md) - How anomalies are interpreted
- [Detection Pipeline](./pipeline.md) - End-to-end detection flow
- [SLO Evaluation](../slo/README.md) - How severity is adjusted
