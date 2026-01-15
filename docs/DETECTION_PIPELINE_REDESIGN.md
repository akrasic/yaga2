# Detection Pipeline Redesign: Sequential IF → Pattern Interpretation

**Status**: IMPLEMENTED

## Executive Summary

**Problem**: The previous detection pipeline ran Isolation Forest (IF) and Pattern Matching in parallel, creating duplicate alerts for the same issue that required post-hoc consolidation.

**Solution**: Restructured to a sequential pipeline where IF acts as an anomaly trigger and patterns interpret the IF output, producing a single, actionable alert.

**Benefit**: Cleaner alerts, less noise, better SRE experience, and more efficient processing.

---

## Current Architecture

### Detection Flow (Parallel)

```
detect() method in detector.py
│
├─→ _detect_univariate_anomalies()     → IF anomalies (per-metric)
├─→ _detect_multivariate_anomalies()   → IF multivariate + Pattern anomalies
├─→ _detect_service_patterns()         → Threshold-based patterns
├─→ _detect_statistical_correlations() → Ratio-based correlations
│
└─→ _correlate_anomalies()             → Consolidation (workaround)
         │
         ▼
    Final anomalies dict
```

### Problems

1. **Duplicate Detection**: IF and patterns both fire on high latency, creating two anomalies for one issue.

2. **Consolidation Complexity**: `_correlate_anomalies()` (150+ lines) exists solely to merge duplicates.

3. **Inconsistent Interpretation**: Sometimes we get IF output, sometimes pattern output, sometimes both.

4. **Wasted Processing**: Patterns run on ALL metrics, not just anomalous ones.

5. **Alert Noise**: SREs see multiple signals for one problem.

### Example: Current Output (Redundant)

```json
{
  "anomalies": {
    "application_latency_isolation": {
      "type": "ml_isolation",
      "severity": "high",
      "score": -0.35,
      "description": "Latency degradation: 450ms (92nd percentile)",
      "detection_method": "isolation_forest"
    },
    "traffic_surge_degrading": {
      "type": "multivariate_pattern",
      "severity": "high",
      "description": "Traffic surge causing slowdown: 850 req/s driving latency to 450ms",
      "detection_method": "named_pattern_matching"
    }
  }
}
```

Both describe the SAME issue. After consolidation, we merge them, but this is a workaround.

---

## Proposed Architecture

### Detection Flow (Sequential)

```
detect() method - REDESIGNED
│
├─→ Phase 1: IF Detection (Trigger)
│   ├─→ _detect_univariate_anomalies()
│   └─→ _detect_multivariate_anomalies() [IF only, no patterns]
│
│   Output: List of anomalous metrics with scores
│   Example: [("application_latency", -0.35, "high"), ("request_rate", -0.22, "high")]
│
├─→ Phase 2: Pattern Interpretation (Interpreter)
│   └─→ _interpret_anomalies()
│       ├─→ Match IF output against pattern conditions
│       ├─→ If match: Return interpreted alert with actions
│       └─→ If no match: Return raw IF alert as "unknown"
│
└─→ Output: Single interpreted anomaly OR unknown anomaly
```

### Key Changes

| Component | Current | Proposed |
|-----------|---------|----------|
| IF Detection | Produces final anomalies | Produces trigger signals |
| Pattern Matching | Independent detector | Interprets IF signals |
| Consolidation | Required (workaround) | Not needed |
| Output | Multiple anomalies | Single interpreted anomaly |

---

## Detailed Design

### Phase 1: IF Detection (Trigger)

The IF models remain unchanged. They produce a list of **triggered metrics**:

```python
@dataclass
class AnomalySignal:
    """Output from IF detection - a trigger signal."""
    metric_name: str
    score: float                    # IF decision score (negative = anomalous)
    direction: str                  # "high" or "low"
    value: float                    # Current metric value
    percentile: float               # Position relative to training data
    deviation_sigma: float          # Standard deviations from mean


def _detect_if_signals(self, metrics: dict[str, float]) -> list[AnomalySignal]:
    """
    Phase 1: Run IF models to identify anomalous metrics.

    Returns list of signals for metrics flagged as anomalous.
    Does NOT produce final anomalies - just triggers.
    """
    signals = []

    for metric_name in self.CORE_METRICS:
        model_key = f"{metric_name}_isolation"
        if model_key not in self.models:
            continue

        # Run IF prediction
        prediction = model.predict(scaled_value)
        if prediction == -1:  # Anomaly
            score = model.decision_function(scaled_value)
            signals.append(AnomalySignal(
                metric_name=metric_name,
                score=score,
                direction="high" if value > stats.mean else "low",
                value=value,
                percentile=self._estimate_percentile(value, stats),
                deviation_sigma=(value - stats.mean) / stats.std,
            ))

    return signals
```

### Phase 2: Pattern Interpretation

Patterns now **interpret IF signals** rather than detecting independently:

```python
def _interpret_anomalies(
    self,
    signals: list[AnomalySignal],
    metrics: dict[str, float]
) -> dict[str, Any]:
    """
    Phase 2: Interpret IF signals using pattern definitions.

    Args:
        signals: Anomaly signals from IF detection
        metrics: Full metrics dict (for ratio calculations)

    Returns:
        Single interpreted anomaly, or raw IF anomaly if no pattern matches
    """
    if not signals:
        return {}

    # Convert IF signals to metric levels for pattern matching
    metric_levels = self._signals_to_levels(signals, metrics)

    # Try to match a named pattern
    matched_pattern = self._find_matching_pattern(metric_levels, metrics)

    if matched_pattern:
        # Pattern matched - return interpreted anomaly
        return self._build_interpreted_anomaly(matched_pattern, signals, metrics)
    else:
        # No pattern match - return raw IF anomaly as "unknown"
        return self._build_unknown_anomaly(signals, metrics)


def _signals_to_levels(
    self,
    signals: list[AnomalySignal],
    metrics: dict[str, float]
) -> dict[str, str]:
    """
    Convert IF signals to metric levels for pattern matching.

    IF signals tell us WHICH metrics are anomalous.
    Levels tell us HOW they're anomalous (high/low/normal).
    """
    levels = {}

    # Metrics with IF signals
    for signal in signals:
        if signal.percentile > 95:
            levels[signal.metric_name] = "very_high"
        elif signal.percentile > 90:
            levels[signal.metric_name] = "high"
        elif signal.percentile < 10:
            levels[signal.metric_name] = "very_low"
        elif signal.percentile < 25:
            levels[signal.metric_name] = "low"
        else:
            levels[signal.metric_name] = "normal"  # Edge case

    # Metrics WITHOUT IF signals are "normal"
    for metric in self.CORE_METRICS:
        if metric not in levels:
            levels[metric] = "normal"

    return levels
```

### Pattern Matching Against IF Output

```python
def _find_matching_pattern(
    self,
    metric_levels: dict[str, str],
    metrics: dict[str, float]
) -> PatternDefinition | None:
    """
    Find a pattern that matches the current IF output.

    Patterns are checked in priority order (critical first).
    """
    # Calculate ratios for ratio-based conditions
    ratios = self._calculate_metric_ratios(metrics)

    # Check patterns in priority order
    pattern_priority = [
        # Critical patterns first
        "traffic_surge_failing",
        "error_rate_critical",
        "fast_rejection",
        "traffic_cliff",
        # High severity
        "traffic_surge_degrading",
        "elevated_errors",
        "database_bottleneck",
        "downstream_cascade",
        "internal_bottleneck",
        # Medium/Low
        "traffic_surge_healthy",
        "database_degradation",
    ]

    for pattern_name in pattern_priority:
        pattern_def = get_pattern_definition(pattern_name)
        if pattern_def and self._pattern_matches(metric_levels, ratios, pattern_def.conditions):
            return pattern_def

    return None  # No pattern matches
```

### Building Final Output

The output format matches `docs/INFERENCE_API_PAYLOAD.md` specification.

```python
def _build_interpreted_anomaly(
    self,
    pattern: PatternDefinition,
    signals: list[AnomalySignal],
    metrics: dict[str, float]
) -> dict[str, Any]:
    """
    Build an interpreted anomaly from a matched pattern.

    Combines IF detection evidence with pattern interpretation.
    Output format matches docs/INFERENCE_API_PAYLOAD.md specification.
    """
    # Use worst IF score for severity
    worst_score = min(s.score for s in signals)

    # Severity from pattern, but can be escalated by IF score
    severity = pattern.severity
    if worst_score < -0.6 and severity != "critical":
        severity = "critical"  # IF says it's very bad

    # Build detection_signals array (API-compatible format)
    detection_signals = [
        {
            "method": "isolation_forest",
            "type": "ml_isolation",
            "severity": Thresholds.score_to_severity(s.score).value,
            "score": s.score,
            "direction": s.direction,
            "percentile": s.percentile,
        }
        for s in signals
    ]
    detection_signals.append({
        "method": "named_pattern_matching",
        "type": "multivariate_pattern",
        "severity": severity,
        "score": worst_score,
        "pattern": pattern.name,
    })

    return {
        pattern.name: {
            "type": "consolidated",  # API-compatible type
            "root_metric": root_metric,
            "severity": severity,
            "confidence": round(confidence, 2),
            "score": worst_score,
            "signal_count": len(detection_signals),
            "description": pattern.message_template.format(**metrics, **self._get_format_values(metrics)),
            "interpretation": pattern.interpretation,
            "pattern_name": pattern.name,
            "value": primary_signal.value,
            "detection_method": "isolation_forest + pattern_interpretation",
            "detection_signals": detection_signals,  # Array format per API spec
            "recommended_actions": pattern.recommended_actions,
            "contributing_metrics": [s.metric_name for s in signals],
            "metric_values": {m: metrics.get(m, 0) for m in self.CORE_METRICS},
        }
    }


def _build_unknown_anomaly(
    self,
    signals: list[AnomalySignal],
    metrics: dict[str, float]
) -> dict[str, Any]:
    """
    Build an anomaly for IF detections that don't match any pattern.

    This is the fallback - we detected something but don't know what pattern it is.
    Output format matches docs/INFERENCE_API_PAYLOAD.md specification.
    """
    # Primary signal is the most severe
    primary = min(signals, key=lambda s: s.score)
    severity = Thresholds.score_to_severity(primary.score)

    # Generate anomaly name based on root metric (per API naming convention)
    if root_metric == "application_latency":
        anomaly_name = "latency_anomaly"
    elif root_metric == "error_rate":
        anomaly_name = "error_rate_anomaly"
    elif root_metric == "request_rate":
        anomaly_name = "traffic_anomaly"
    else:
        anomaly_name = f"{root_metric}_anomaly"

    # Build detection_signals array (API-compatible format)
    detection_signals = [
        {
            "method": "isolation_forest",
            "type": "ml_isolation",
            "severity": Thresholds.score_to_severity(s.score).value,
            "score": s.score,
            "direction": s.direction,
            "percentile": s.percentile,
        }
        for s in signals
    ]

    return {
        anomaly_name: {
            "type": "ml_isolation" if len(signals) == 1 else "consolidated",
            "root_metric": root_metric,
            "severity": severity.value,
            "confidence": round(confidence, 2),
            "score": primary.score,
            "signal_count": len(detection_signals),
            "description": (
                f"Unusual behavior detected in {len(signals)} metric(s): "
                f"{', '.join(s.metric_name for s in signals)}"
            ),
            "interpretation": (
                "Isolation Forest detected anomalous behavior that doesn't match "
                "any known pattern. This may be a novel issue requiring investigation."
            ),
            "detection_method": "isolation_forest",
            "value": primary.value,
            "detection_signals": detection_signals,  # Array format per API spec
            "recommended_actions": [
                "INVESTIGATE: Review the flagged metrics for unusual values",
                "CORRELATE: Check for recent deployments or configuration changes",
                "COMPARE: Review metric trends over the past hour",
            ],
            "possible_causes": all_causes[:5],
            "checks": all_checks[:5],
            "contributing_metrics": [s.metric_name for s in signals],
            "metric_values": {m: metrics.get(m, 0) for m in self.CORE_METRICS},
        }
    }
```

---

## Files to Modify

### 1. `smartbox_anomaly/detection/detector.py`

| Change | Lines | Description |
|--------|-------|-------------|
| Add `AnomalySignal` dataclass | New | Data structure for IF output |
| Rename `_detect_univariate_anomalies` | 678-757 | → `_detect_if_signals` (returns signals, not anomalies) |
| Modify `_detect_multivariate_anomalies` | 811-869 | Remove pattern detection, IF only |
| Remove `_detect_named_multivariate_patterns` | 871-915 | Moved to interpretation phase |
| Add `_interpret_anomalies` | New | Phase 2: Pattern interpretation |
| Add `_signals_to_levels` | New | Convert IF signals to levels |
| Add `_find_matching_pattern` | New | Pattern matching logic |
| Add `_build_interpreted_anomaly` | New | Build interpreted output |
| Add `_build_unknown_anomaly` | New | Fallback for unmatched |
| Remove `_correlate_anomalies` | 274-445 | No longer needed |
| Simplify `detect()` | 201-272 | New sequential flow |

### 2. `smartbox_anomaly/detection/interpretations.py`

| Change | Lines | Description |
|--------|-------|-------------|
| Add pattern priority | New | Define pattern matching order |
| No structural changes | - | Patterns remain the same |

### 3. `inference.py`

| Change | Lines | Description |
|--------|-------|-------------|
| Simplify result processing | 309-347 | Single anomaly format |
| Update `_format_time_aware_alert_json` | 415-508 | Handle new format |

---

## Output Format Comparison

### Current Output (Multiple Anomalies)

```json
{
  "anomalies": {
    "application_latency_isolation": { ... },
    "traffic_surge_degrading": { ... }
  },
  "anomaly_count": 2
}
```

### Proposed Output (Single Interpreted Anomaly)

```json
{
  "anomalies": {
    "traffic_surge_degrading": {
      "type": "interpreted",
      "pattern_name": "traffic_surge_degrading",
      "severity": "high",
      "score": -0.35,
      "description": "Traffic surge causing slowdown: 850 req/s driving latency to 450ms",
      "interpretation": "Service is slowing under load but not failing - approaching capacity.",
      "detection_method": "isolation_forest + pattern_interpretation",

      "triggered_by": {
        "application_latency": {
          "score": -0.35,
          "direction": "high",
          "percentile": 92.0
        },
        "request_rate": {
          "score": -0.22,
          "direction": "high",
          "percentile": 88.0
        }
      },

      "recommended_actions": [
        "IMMEDIATE: Scale horizontally if possible",
        "CHECK: Resource bottlenecks (CPU, memory, connections)",
        "MONITOR: Error rate for signs of impending failure"
      ],
      "metric_values": {
        "request_rate": 850.0,
        "application_latency": 450.0,
        "error_rate": 0.015,
        "dependency_latency": 120.0,
        "database_latency": 80.0
      }
    }
  },
  "anomaly_count": 1
}
```

### Unknown Anomaly (No Pattern Match)

```json
{
  "anomalies": {
    "unknown_anomaly": {
      "type": "unknown",
      "severity": "medium",
      "score": -0.28,
      "description": "Unusual behavior detected in 2 metric(s): dependency_latency, database_latency",
      "interpretation": "Isolation Forest detected anomalous behavior that doesn't match any known pattern.",
      "detection_method": "isolation_forest",

      "signals": [
        {"metric": "dependency_latency", "score": -0.28, "direction": "high", "percentile": 94.0},
        {"metric": "database_latency", "score": -0.19, "direction": "low", "percentile": 8.0}
      ],

      "recommended_actions": [
        "INVESTIGATE: Check the flagged metrics for unusual values",
        "CORRELATE: Look for recent deployments or changes"
      ]
    }
  },
  "anomaly_count": 1
}
```

---

## Migration Strategy

### Phase 1: Add New Methods (Non-Breaking)

1. Add `AnomalySignal` dataclass
2. Add `_detect_if_signals()` (parallel to existing)
3. Add `_interpret_anomalies()` and helpers
4. Add feature flag to enable new pipeline

### Phase 2: Switch Pipeline

1. Update `detect()` to use new flow when flag enabled
2. Test with existing services
3. Compare output quality

### Phase 3: Remove Legacy

1. Remove `_correlate_anomalies()`
2. Remove pattern detection from `_detect_multivariate_anomalies()`
3. Remove feature flag, make new pipeline default

---

## Edge Cases

### 1. No IF Anomalies Detected

```
IF Detection → No signals
Pattern Interpretation → Not called
Output → {} (empty, no anomaly)
```

### 2. IF Detects, No Pattern Matches

```
IF Detection → [application_latency: high]
Pattern Interpretation → No match
Output → "unknown_anomaly" with investigation guidance
```

### 3. Multiple IF Signals, One Pattern Matches

```
IF Detection → [app_latency: high, request_rate: high, error_rate: normal]
Pattern Interpretation → Matches "traffic_surge_degrading"
Output → Single "traffic_surge_degrading" anomaly
```

### 4. Multiple IF Signals Could Match Multiple Patterns

```
IF Detection → [app_latency: high, request_rate: high, error_rate: high]
Pattern Interpretation → Could match "traffic_surge_failing" OR "error_rate_critical"
Resolution → First match in priority order wins ("traffic_surge_failing")
```

---

## Benefits Summary

| Metric | Current | Proposed |
|--------|---------|----------|
| Anomalies per issue | 2-4 | 1 |
| Alert noise | High (duplicates) | Low (single alert) |
| Explainability | Mixed signals | Clear: trigger + interpretation |
| Processing efficiency | All patterns on all metrics | Patterns only on IF-flagged |
| Unknown anomaly handling | Lost in noise | Explicit "unknown" type |
| Code complexity | High (consolidation) | Lower (sequential) |

---

## Open Questions

1. **Pattern Priority**: Should pattern priority be configurable per service?

2. **Multiple Pattern Matches**: Should we return ALL matching patterns or just the highest priority?

3. **IF Score Escalation**: Should a very negative IF score override pattern severity?

4. **Unknown Anomaly Threshold**: How many unknown anomalies before we should add a new pattern?

---

## Next Steps

1. Review this design document
2. Discuss open questions
3. Approve approach
4. Implement Phase 1 (non-breaking additions)
5. Test and validate
6. Implement Phases 2-3

---

## Appendix A: Pattern Priority Order

```python
PATTERN_PRIORITY = [
    # Critical - immediate user impact
    "traffic_surge_failing",
    "error_rate_critical",
    "fast_rejection",
    "traffic_cliff",
    "reduced_traffic_with_errors",

    # Dependency-aware patterns (checked before generic patterns)
    "upstream_cascade",
    "dependency_chain_degradation",

    # High - significant degradation
    "traffic_surge_degrading",
    "elevated_errors",
    "database_bottleneck",
    "downstream_cascade",
    "internal_bottleneck",
    "internal_latency_issue",
    "latency_spike_recent",
    "fast_failure",
    "partial_rejection",

    # Medium - noticeable issues
    "database_degradation",
    "resource_contention",
    "performance_degradation",

    # Low - informational
    "traffic_surge_healthy",
]
```

---

## Appendix B: Dependency-Aware Cascade Detection

The detection pipeline supports cross-service dependency analysis to identify cascade failures.

### Overview

When Service A is slow because it depends on B which depends on C, and C has an anomaly, Service A's detection output identifies "cascade from C" rather than just "high latency."

### Two-Pass Inference

The inference engine uses a two-pass approach:

```
Pass 1: Detect anomalies for all services (no dependency context)
        → Build map of {service: anomaly_status}

Pass 2: For services with latency anomalies:
        → Build DependencyContext from Pass 1 results
        → Re-run detection with cascade analysis
        → Pattern matching now has upstream status
```

### Dependency Configuration

Service dependencies are configured in `config.json`:

```json
{
  "dependencies": {
    "graph": {
      "mobile-api": ["booking", "search", "shire-api"],
      "booking": ["search", "vms", "r2d2"],
      "vms": ["titan"]
    },
    "cascade_detection": {
      "enabled": true,
      "max_depth": 5
    }
  }
}
```

### Dependency-Aware Patterns

| Pattern | Conditions | Description |
|---------|------------|-------------|
| `upstream_cascade` | high latency + upstream has anomaly | Root cause in dependency |
| `dependency_chain_degradation` | high latency + multiple deps affected | Chain of services degraded |
| `internal_latency_issue` | high latency + all deps healthy | Issue is internal to service |

### Cascade Analysis Output

When a cascade is detected, the anomaly includes `cascade_analysis`:

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
      "confidence": 0.85,
      "propagation_path": [
        {"service": "titan", "has_anomaly": true, "anomaly_type": "database_bottleneck"},
        {"service": "vms", "has_anomaly": true, "anomaly_type": "downstream_cascade"}
      ]
    },
    "recommended_actions": [
      "FOCUS: Investigate titan first - it is the root cause",
      "CHECK: titan's database_bottleneck anomaly"
    ]
  }
}
```

### Example Flow

```
Scenario: booking slow due to vms → titan failure

Config: booking → [vms], vms → [titan]

Pass 1:
  titan: "database_bottleneck" (root cause)
  vms: "downstream_cascade" (waiting on titan)
  booking: high latency (waiting on vms)

Pass 2 (booking with dependency context):
  DependencyContext: {vms: anomaly, titan: anomaly}
  find_root_cause() → titan
  Pattern match: "upstream_cascade"

Output:
  booking gets "upstream_cascade" pattern with root_cause_service: "titan"
```
