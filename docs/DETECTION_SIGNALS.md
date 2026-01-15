# Detection Signals

This document explains the two primary detection methods used by the Smartbox Anomaly Detection system: **Isolation Forest** (ML-based) and **Named Pattern Matching** (rule-based).

---

## Overview

The anomaly detection system uses a **sequential pipeline** where Isolation Forest (IF) acts as an anomaly trigger and Named Pattern Matching interprets the IF output:

| Method | Role | Strength | Best For |
|--------|------|----------|----------|
| Isolation Forest | Trigger | Detects novel anomalies | Unknown/unexpected deviations |
| Named Pattern Matching | Interpreter | Actionable interpretations | Known operational scenarios |

The pipeline flow is: **IF Detection → Pattern Interpretation → Single Alert**

When IF detects anomalies, patterns interpret what they mean and produce a single, actionable alert.

---

## 1. Isolation Forest

### What Is Isolation Forest?

Isolation Forest is an unsupervised machine learning algorithm designed for anomaly detection. Unlike other methods that try to model "normal" behavior, Isolation Forest works by **isolating anomalies**.

The core insight: **anomalies are easier to isolate than normal points**.

### How It Works

1. **Random Partitioning**: The algorithm builds many random decision trees by:
   - Randomly selecting a feature (metric)
   - Randomly selecting a split value between min and max
   - Recursively partitioning the data

2. **Path Length**: For each data point, count how many splits are needed to isolate it:
   - **Short path** = Easy to isolate = **Anomaly**
   - **Long path** = Hard to isolate = **Normal**

3. **Anomaly Score**: The average path length across all trees produces an anomaly score:
   - Score < 0 = Anomalous (more negative = more anomalous)
   - Score > 0 = Normal

```
Normal data point:         Anomaly:
     ┌─────┐                  ┌─────┐
     │  ●  │ ← buried deep    │     │
     │ ●●● │   in cluster     │  ●  │ ← isolated
     │●●●●●│                  │     │   in 1-2 splits
     │ ●●● │                  └─────┘
     └─────┘
```

### Implementation in Smartbox

The system trains **two types** of Isolation Forest models:

#### Univariate Models (One Per Metric)

Each metric gets its own model:
- `request_rate_isolation`
- `application_latency_isolation`
- `error_rate_isolation`
- `dependency_latency_isolation`
- `database_latency_isolation`

```python
# Training (from detector.py:554-613)
model = IsolationForest(
    contamination=0.05,      # Expected anomaly rate
    n_estimators=200,        # Number of trees
    random_state=42          # Reproducibility
)
model.fit(scaled_data)
```

#### Multivariate Model (Combined Metrics)

One model considers all metrics together to detect unusual **combinations**:

```python
# Example: Normal individually, anomalous together
{
    "request_rate": 500,      # Normal
    "application_latency": 200,  # Normal
    "error_rate": 0.01        # Normal
}
# BUT: High traffic with low latency AND low errors might be
# suspicious (cache bypass? database not being hit?)
```

### Detection Output

When Isolation Forest detects an anomaly:

```json
{
    "application_latency_isolation": {
        "type": "ml_isolation",
        "severity": "high",
        "score": -0.35,
        "description": "Latency degradation: 450ms (92nd percentile, 2.1σ above normal)",
        "detection_method": "isolation_forest",
        "value": 450,
        "threshold": 350,
        "direction": "high",
        "deviation_sigma": 2.1,
        "percentile": 92.0
    }
}
```

### Severity Mapping

Anomaly scores map to severity levels:

| Score Range | Severity |
|-------------|----------|
| < -0.6 | critical |
| -0.6 to -0.3 | high |
| -0.3 to -0.1 | medium |
| > -0.1 | low |

### Strengths of Isolation Forest

1. **No assumptions about data distribution** - Works on any metric shape
2. **Detects novel anomalies** - Catches patterns not seen before
3. **Handles multivariate relationships** - Finds unusual combinations
4. **Scales well** - O(n log n) time complexity

### Limitations

1. **No semantic understanding** - Doesn't know WHY something is anomalous
2. **Context-blind** - Doesn't consider time of day, deployments, etc.
3. **False positives** - Unusual != bad (a traffic spike could be a sale)
4. **Training data quality** - If trained on bad data, learns bad patterns

---

## 2. Named Pattern Matching

### What Is Named Pattern Matching?

Named Pattern Matching is a **rule-based** detection system that identifies known operational scenarios by examining metric combinations and their relationships.

Unlike ML which asks "is this unusual?", pattern matching asks "does this match a known problem type?"

### How It Works

1. **Metric Level Classification**: Each metric is classified based on percentile:

   | Percentile Range | Level |
   |------------------|-------|
   | > 95 | `very_high` |
   | 90 - 95 | `high` |
   | 80 - 90 | `elevated` |
   | 70 - 80 | `moderate` |
   | 10 - 70 | direction fallback (`high`/`low`) |
   | 5 - 10 | `low` |
   | < 5 | `very_low` |
   | no IF signal | `normal` |

2. **Semantic Interpretation**: Some metrics have "lower is better" semantics:
   - `database_latency`: Lower = faster queries (improvement)
   - `dependency_latency`: Lower = faster downstream (improvement)
   - `error_rate`: Lower = fewer errors (ideal is 0%)

   For these metrics, when IF flags them as "low", they are treated as `normal` since low values are desirable, not anomalous.

3. **Pattern Matching**: Current levels are compared against defined pattern conditions

4. **Interpretation**: Matched patterns provide:
   - Human-readable description
   - Root cause interpretation
   - Recommended actions

### Pattern Definitions

Patterns are defined in `smartbox_anomaly/detection/interpretations.py`:

```python
MULTIVARIATE_PATTERNS = {
    "traffic_surge_failing": PatternDefinition(
        name="traffic_surge_failing",
        conditions={
            "request_rate": "high",
            "application_latency": "high",
            "error_rate": "high",
        },
        severity="critical",
        message_template=(
            "Traffic surge overwhelming service: {request_rate:.1f} req/s "
            "with {application_latency:.0f}ms latency and {error_rate:.2%} errors"
        ),
        interpretation=(
            "Service at or beyond capacity - users actively affected. "
            "Both latency and errors elevated indicates system cannot handle current load."
        ),
        recommended_actions=[
            "IMMEDIATE: Enable auto-scaling or manually scale horizontally",
            "IMMEDIATE: Consider activating rate limiting to protect backend",
            "INVESTIGATE: Confirm this is legitimate traffic vs attack",
            "PREPARE: Have rollback ready if recent deployment is contributing",
        ],
    ),
}
```

### Available Patterns

#### Traffic Patterns

| Pattern | Conditions | Severity | Meaning |
|---------|------------|----------|---------|
| `traffic_surge_healthy` | high traffic, normal latency, normal errors | low | System handling load well |
| `traffic_surge_degrading` | high traffic, high latency, normal errors | high | Approaching capacity |
| `traffic_surge_failing` | high traffic, high latency, high errors | critical | Beyond capacity |
| `traffic_cliff` | very low traffic, sudden drop | critical | Service may be unreachable |

#### Error Patterns

| Pattern | Conditions | Severity | Meaning |
|---------|------------|----------|---------|
| `elevated_errors` | normal traffic/latency, high errors | high | Specific endpoint or operation failing |
| `error_rate_critical` | normal traffic/latency, very high errors | critical | Significant portion of requests failing |

#### Latency Patterns

| Pattern | Conditions | Severity | Meaning |
|---------|------------|----------|---------|
| `latency_spike_recent` | normal traffic, high latency, normal errors | high | Recent change caused slowdown |
| `internal_bottleneck` | high app latency, normal external deps | high | Application itself is slow |
| `internal_latency_issue` | high app latency, all deps healthy | high | Issue is internal to service |
| `database_degradation` | high DB latency, normal app latency | medium | DB slow but app compensating |
| `database_bottleneck` | high DB latency, DB >70% of total latency | high | DB is primary constraint |
| `downstream_cascade` | high client latency, >60% of total latency | high | External dependency is bottleneck |

#### Fast-Fail Patterns

| Pattern | Conditions | Severity | Meaning |
|---------|------------|----------|---------|
| `fast_rejection` | very low latency, very high errors | critical | Requests rejected before processing |
| `fast_failure` | low latency, high errors | critical | Quick failures (circuit breaker, auth, etc.) |
| `partial_rejection` | low latency, moderate errors | high | Some operations being rejected |

### Detection Output

When Pattern Matching detects an anomaly:

```json
{
    "traffic_surge_degrading": {
        "type": "multivariate_pattern",
        "pattern_name": "traffic_surge_degrading",
        "severity": "high",
        "score": -0.5,
        "description": "Traffic surge causing slowdown: 850.0 req/s driving latency to 450ms (errors stable at 1.50%)",
        "interpretation": "Service is slowing under load but not failing - approaching capacity. Users experiencing degraded performance but requests are completing.",
        "detection_method": "named_pattern_matching",
        "recommended_actions": [
            "IMMEDIATE: Scale horizontally if possible",
            "CHECK: Resource bottlenecks (CPU, memory, connections)",
            "MONITOR: Error rate for signs of impending failure",
            "CONSIDER: Enable request throttling to protect backend"
        ],
        "contributing_metrics": ["request_rate", "application_latency", "error_rate"],
        "metric_levels": {
            "request_rate": "high",
            "application_latency": "high",
            "error_rate": "normal"
        }
    }
}
```

### Ratio-Based Patterns

Some patterns use metric ratios rather than absolute levels:

```python
"database_bottleneck": PatternDefinition(
    conditions={
        "database_latency": "high",
        "application_latency": "high",
        "db_latency_ratio": "> 0.7",  # DB is >70% of total latency
    },
)
```

This catches cases where absolute values might look normal but the **proportion** is problematic.

### Strengths of Pattern Matching

1. **Semantic understanding** - Knows what patterns mean operationally
2. **Actionable recommendations** - Provides specific next steps
3. **Explainable** - Easy to understand why an alert fired
4. **Deterministic** - Same inputs always produce same outputs
5. **Domain knowledge encoded** - Captures expert knowledge about failure modes

### Limitations

1. **Only detects known patterns** - Novel failure modes are missed
2. **Requires maintenance** - New patterns must be manually added
3. **Threshold sensitivity** - Edge cases near thresholds can flip-flop
4. **No learning** - Doesn't adapt to changing baselines

---

## Sequential Interpretation (Replaces Consolidation)

With the sequential pipeline, IF and Pattern Matching no longer run in parallel and consolidation is not needed. Instead:

1. **IF Detection**: Identifies which metrics are anomalous and produces `AnomalySignal` objects
2. **Pattern Interpretation**: Takes IF signals and matches them against named patterns
3. **Single Output**: Produces one interpreted anomaly (or "unknown" if no pattern matches)

### Example: IF Triggers, Pattern Interprets

```
Phase 1 - IF Detection:
  application_latency: AnomalySignal(score=-0.35, direction="high", percentile=92)
  request_rate: AnomalySignal(score=-0.22, direction="high", percentile=88)

Phase 2 - Pattern Interpretation:
  metric_levels = {application_latency: "high", request_rate: "high", error_rate: "normal"}
  matched_pattern = "traffic_surge_degrading"

Output:
  traffic_surge_degrading:
    - type: "consolidated"
    - severity: "high"
    - confidence: 0.80
    - signal_count: 2
    - detection_signals: [
        {method: "isolation_forest", metric: "application_latency", score: -0.35},
        {method: "isolation_forest", metric: "request_rate", score: -0.22},
        {method: "named_pattern_matching", pattern: "traffic_surge_degrading"}
      ]
```

### Confidence Calculation

Confidence is calculated based on the number of IF signals and pattern match:

```python
# Base confidence from pattern match
confidence = 0.6

# Boost for multiple IF signals
if len(signals) > 1:
    confidence += 0.1 * min(len(signals), 3)

# Boost for strong IF scores
strong_signals = sum(1 for s in signals if s.score < -0.3)
confidence += 0.05 * strong_signals

confidence = min(0.95, confidence)
```

---

## Detection Flow

The system uses a **sequential pipeline** (not parallel):

```
                    Current Metrics
                          │
                          ▼
     ┌────────────────────────────────────┐
     │     Phase 1: IF Detection          │
     │  (Univariate + Multivariate IF)    │
     │                                    │
     │  Output: List of AnomalySignals    │
     │  [metric, score, direction, value] │
     └────────────────┬───────────────────┘
                      │
                      ▼
     ┌────────────────────────────────────┐
     │   Phase 2: Pattern Interpretation  │
     │                                    │
     │  - Convert signals to metric levels│
     │  - Match against named patterns    │
     │  - Build interpreted anomaly       │
     └────────────────┬───────────────────┘
                      │
                      ▼
     ┌────────────────────────────────────┐
     │   Phase 3: Cascade Analysis        │
     │   (Two-Pass for Dependencies)      │
     │                                    │
     │  - Check upstream dependencies     │
     │  - Identify root cause service     │
     │  - Add cascade_analysis to output  │
     └────────────────┬───────────────────┘
                      │
                      ▼
                Single Alert
           (with pattern interpretation
            and cascade analysis)
```

### Key Differences from Previous Architecture

| Aspect | Old (Parallel) | New (Sequential) |
|--------|----------------|------------------|
| IF and Patterns | Run independently | IF triggers, Patterns interpret |
| Output | Multiple anomalies | Single interpreted anomaly |
| Consolidation | Required post-hoc | Not needed |
| Cascade Detection | Not available | Two-pass inference with dependency context |

---

## Dependency-Aware Patterns

The system includes patterns that detect cascade failures across service dependencies.

### Available Dependency Patterns

| Pattern | Conditions | Severity | Description |
|---------|------------|----------|-------------|
| `upstream_cascade` | high latency + `_dependency_context: upstream_anomaly` | high | Root cause in upstream dependency |
| `dependency_chain_degradation` | high latency + `_dependency_context: chain_degraded` | high | Multiple services in chain affected |
| `internal_latency_issue` | high latency + `_dependency_context: dependencies_healthy` | high | All deps healthy, issue is internal |

### How Dependency Detection Works

1. **Two-Pass Inference**: The inference engine runs detection in two passes:
   - Pass 1: Detect anomalies for all services without dependency context
   - Pass 2: Re-run for services with latency anomalies, with dependency context

2. **Dependency Context**: Built from Pass 1 results, contains:
   - Status of each upstream dependency (has_anomaly, anomaly_type, severity)
   - Service dependency graph from config

3. **Pattern Matching**: Dependency patterns use special `_dependency_context` conditions:
   - `upstream_anomaly`: At least one dependency has an anomaly
   - `chain_degraded`: Multiple dependencies in the chain are affected
   - `dependencies_healthy`: All dependencies are healthy

### Example Output

When a cascade is detected:

```json
{
  "upstream_cascade": {
    "type": "consolidated",
    "severity": "high",
    "description": "Cascade from upstream: vms is degraded, affecting booking",
    "cascade_analysis": {
      "is_cascade": true,
      "root_cause_service": "titan",
      "affected_chain": ["titan", "vms"],
      "cascade_type": "upstream_cascade",
      "confidence": 0.85
    },
    "recommended_actions": [
      "FOCUS: Investigate titan first - it is the root cause"
    ]
  }
}
```

### Configuration

Dependencies are configured in `config.json`:

```json
{
  "dependencies": {
    "graph": {
      "booking": ["search", "vms"],
      "vms": ["titan"]
    },
    "cascade_detection": {
      "enabled": true,
      "max_depth": 5
    }
  }
}
```

---

## When to Use Each Method

| Scenario | Best Method | Reason |
|----------|-------------|--------|
| Known failure modes | Pattern Matching | Has actionable recommendations |
| Novel/unknown issues | Isolation Forest | Detects without prior knowledge |
| Production monitoring | Both | Redundancy improves reliability |
| Initial deployment | Isolation Forest first | Learn patterns before encoding rules |
| Mature service | Emphasize Pattern Matching | Faster, more explainable |

---

## Pattern Definition Files

All named patterns are defined in a single file:

```
smartbox_anomaly/detection/interpretations.py
```

### File Structure

The file contains several key sections:

| Line Range | Section | Purpose |
|------------|---------|---------|
| 20-50 | Data Classes | `PatternDefinition`, `MetricInterpretation`, `SeverityContext` |
| 57-258 | `METRIC_INTERPRETATIONS` | Per-metric messages for high/low values |
| 266-558 | `MULTIVARIATE_PATTERNS` | Main pattern definitions (traffic, errors, latency) |
| 566-664 | `FAST_FAIL_PATTERNS` | Fast failure scenario patterns |
| 672-765 | `OPERATIONAL_PATTERNS` | Operational patterns (gradual degradation, recovery, flapping) |
| 773-847 | `RECOMMENDATION_RULES` | Severity-specific action recommendations |
| 855-867 | `BUSINESS_IMPACT_MAP` | Impact descriptions by severity/type |

### Pattern Dictionaries

There are **three pattern dictionaries** that the detector checks:

1. **`MULTIVARIATE_PATTERNS`** - Primary patterns (checked first)
2. **`FAST_FAIL_PATTERNS`** - Fast failure scenarios
3. **`OPERATIONAL_PATTERNS`** - Trending/operational patterns

The `get_pattern_definition()` helper function (line 883-889) searches all three:

```python
def get_pattern_definition(pattern_name: str) -> PatternDefinition | None:
    """Get definition for a multivariate pattern."""
    return (
        MULTIVARIATE_PATTERNS.get(pattern_name)
        or FAST_FAIL_PATTERNS.get(pattern_name)
        or OPERATIONAL_PATTERNS.get(pattern_name)
    )
```

---

## Adding New Patterns

To add a new pattern, edit `smartbox_anomaly/detection/interpretations.py`:

```python
MULTIVARIATE_PATTERNS["my_new_pattern"] = PatternDefinition(
    name="my_new_pattern",
    conditions={
        "request_rate": "normal",
        "application_latency": "high",
        "error_rate": "low",
        # Can also use ratio conditions:
        # "db_latency_ratio": "> 0.5",
    },
    severity="medium",
    message_template="My pattern: latency {application_latency:.0f}ms at {request_rate:.1f} req/s",
    interpretation="What this pattern means operationally...",
    recommended_actions=[
        "IMMEDIATE: First action to take",
        "CHECK: Something to verify",
        "CONSIDER: Optional improvement",
    ],
)
```

### Condition Types

- **Level conditions**: `"very_high"`, `"high"`, `"elevated"`, `"moderate"`, `"normal"`, `"low"`, `"very_low"`
- **Ratio conditions**: `"> 0.6"`, `"< 0.3"` (for `*_ratio` metrics)
- **Any**: `"any"` (always matches)
- **Dependency context**: `"upstream_anomaly"`, `"chain_degraded"`, `"dependencies_healthy"` (for `_dependency_context`)

**Important**: Unknown conditions will cause the pattern to NOT match (fail-closed behavior for safety).

### Available Metrics for Conditions

| Metric Name | Variable | Description | Lower is Better? |
|-------------|----------|-------------|------------------|
| `request_rate` | Traffic volume | Requests per second | No |
| `application_latency` | Response time | Total request processing time (ms) | Context-dependent* |
| `error_rate` | Error percentage | Fraction of failed requests (0.0-1.0) | Yes |
| `dependency_latency` | External call time | Time spent calling external services (ms) | Yes |
| `database_latency` | DB query time | Time spent on database operations (ms) | Yes |
| `dependency_latency_ratio` | Ratio | `dependency_latency / application_latency` | - |
| `db_latency_ratio` | Ratio | `database_latency / application_latency` | - |

*`application_latency` is NOT in the "lower is better" list because low latency combined with high errors indicates fast-fail scenarios (e.g., requests rejected before processing).

---

## Exception Enrichment

When error-related anomalies are detected, the inference engine automatically queries OpenTelemetry exception metrics to identify which exception types are causing the errors.

### How It Works

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Anomaly         │     │  Exception       │     │  Enriched        │
│  Detection       │────▶│  Enrichment      │────▶│  Result          │
│  (IF + Patterns) │     │  Service         │     │  + Exception     │
└──────────────────┘     └──────────────────┘     │  Context         │
                                │                 └──────────────────┘
                                │
                                ▼
                         ┌──────────────────┐
                         │  VictoriaMetrics │
                         │  (events_total)  │
                         └──────────────────┘
```

### Time-Aligned Queries

Exception queries are aligned with the anomaly detection window:

- **end_time** = anomaly timestamp
- **start_time** = anomaly timestamp - lookback_minutes (default: 5)
- **Query**: `sum(rate(events_total{service_name="X"}[5m])) by (exception_type)`

This ensures the exception data matches the exact time window where the anomaly was detected.

### When Enrichment Triggers

Enrichment is performed when ALL of the following are true:

1. **Severity is HIGH or CRITICAL**
2. **Error-related anomaly detected** (pattern name contains "error", "failure", "outage")
3. **Error rate > 1%** in current metrics

### Output Format

```json
{
  "exception_context": {
    "service_name": "search",
    "timestamp": "2024-01-15T10:30:00",
    "total_exception_rate": 0.35,
    "exception_count": 3,
    "top_exceptions": [
      {
        "type": "Smartbox\\Search\\R2D2\\Exception\\R2D2Exception",
        "short_name": "R2D2Exception",
        "rate": 0.217,
        "percentage": 62.0
      },
      {
        "type": "Smartbox\\Search\\Exception\\UserInputException",
        "short_name": "UserInputException",
        "rate": 0.083,
        "percentage": 23.7
      }
    ],
    "query_successful": true
  }
}
```

### Configuration

Exception enrichment is enabled by default. It can be disabled via config:

```json
{
  "inference": {
    "exception_enrichment_enabled": false
  }
}
```

---

### Step-by-Step: Adding a New Pattern

1. **Open the file**:
   ```bash
   code smartbox_anomaly/detection/interpretations.py
   ```

2. **Find the appropriate dictionary** (around line 266 for main patterns):
   ```python
   MULTIVARIATE_PATTERNS: Final[dict[str, PatternDefinition]] = {
       # ... existing patterns ...
   }
   ```

3. **Add your pattern** before the closing brace:
   ```python
   MULTIVARIATE_PATTERNS: Final[dict[str, PatternDefinition]] = {
       # ... existing patterns ...

       "cache_miss_storm": PatternDefinition(
           name="cache_miss_storm",
           conditions={
               "request_rate": "high",
               "database_latency": "high",
               "db_latency_ratio": "> 0.8",
           },
           message_template=(
               "Cache miss storm: {request_rate:.1f} req/s with "
               "{database_latency:.0f}ms DB latency ({db_latency_ratio:.0%} of total)"
           ),
           severity="high",
           interpretation=(
               "High traffic with excessive database load suggests cache failures. "
               "Database is handling requests that should be cached."
           ),
           recommended_actions=[
               "CHECK: Cache service health (Redis/Memcached)",
               "CHECK: Cache hit rate metrics",
               "VERIFY: Cache TTL hasn't expired for hot keys",
               "CONSIDER: Implementing cache warming",
           ],
       ),
   }
   ```

4. **Test the pattern** by running inference with verbose mode:
   ```bash
   uv run python inference.py -v
   ```

### Step-by-Step: Modifying an Existing Pattern

1. **Find the pattern** in `interpretations.py` (use search)

2. **Modify the fields**:
   - `conditions` - Change when the pattern matches
   - `severity` - Adjust alert priority (`low`, `medium`, `high`, `critical`)
   - `message_template` - Update the description (use `{metric_name}` placeholders)
   - `interpretation` - Change the explanation
   - `recommended_actions` - Update action items

3. **Example - Making "elevated_errors" less sensitive**:
   ```python
   # Before: triggers on "high" (> p90)
   "elevated_errors": PatternDefinition(
       conditions={
           "error_rate": "high",  # > p90
           ...
       },
       severity="high",
   )

   # After: only triggers on "very_high" (> p95)
   "elevated_errors": PatternDefinition(
       conditions={
           "error_rate": "very_high",  # > p95
           ...
       },
       severity="medium",  # Also reduced severity
   )
   ```

### Adding Metric-Specific Interpretations

To add interpretation messages for individual metrics (not patterns), update `METRIC_INTERPRETATIONS`:

```python
# Location: interpretations.py, line 57-258

METRIC_INTERPRETATIONS: Final[dict[str, dict[str, MetricInterpretation]]] = {
    MetricName.REQUEST_RATE: {
        "high": MetricInterpretation(
            message_template="Traffic spike: {value:.1f} req/s ({percentile:.0f}th percentile)",
            possible_causes=[
                "Marketing campaign",
                "Bot activity",
                "DDoS attack",
            ],
            checks=[
                "Check traffic sources",
                "Review user-agent patterns",
            ],
        ),
        "low": MetricInterpretation(
            message_template="Traffic drop: {value:.1f} req/s",
            possible_causes=["Upstream outage", "DNS issues"],
            checks=["Check upstream services"],
        ),
    },
    # Add interpretations for other metrics...
}
```

### Adding Recommendation Rules

To add severity-specific recommendations, update `RECOMMENDATION_RULES`:

```python
# Location: interpretations.py, line 773-847

RECOMMENDATION_RULES: Final[dict[tuple[str, str], list[str]]] = {
    # Key: (pattern_name, severity)
    ("my_new_pattern", "critical"): [
        "IMMEDIATE: First critical action",
        "CHECK: Verification step",
        "PREPARE: Contingency action",
    ],
    ("my_new_pattern", "high"): [
        "CHECK: First high-priority action",
        "MONITOR: Observation step",
    ],
}
```

### Recommendation Priority Prefixes

Use these prefixes for consistent ordering (highest to lowest priority):

| Prefix | Priority | Use For |
|--------|----------|---------|
| `IMMEDIATE` | 1 | Actions needed right now |
| `VERIFY` | 2 | Confirm the situation |
| `CHECK` | 3 | Investigate specific things |
| `INVESTIGATE` | 4 | Deeper analysis |
| `CORRELATE` | 5 | Cross-reference data |
| `TIMELINE` | 6 | Check recent changes |
| `ASSESS` | 7 | Evaluate impact |
| `IDENTIFY` | 8 | Find specific causes |
| `FOCUS` | 9 | Narrow down investigation |
| `CONSIDER` | 10 | Optional improvements |
| `PREPARE` | 11 | Contingency planning |
| `DECIDE` | 12 | Make a choice |
| `REVIEW` | 13 | Post-incident review |
| `MONITOR` | 14 | Ongoing observation |

---

## Summary

| Aspect | Isolation Forest | Pattern Matching | Dependency Patterns |
|--------|------------------|------------------|---------------------|
| Type | ML (unsupervised) | Rule-based | Rule-based + context |
| Role | Trigger | Interpreter | Root cause analysis |
| Learns from | Training data | Expert knowledge | Service topology |
| Output | Anomaly signals | Named pattern | Cascade analysis |
| Adapts | To data distribution | Manually maintained | Config-driven |
| Explains | "This is unusual" | "This is X because Y" | "Root cause is service Z" |

**Best practice**: The sequential pipeline (IF → Pattern → Cascade) provides:
1. **Novel detection** from Isolation Forest
2. **Actionable interpretation** from Pattern Matching
3. **Root cause identification** from Dependency Analysis
