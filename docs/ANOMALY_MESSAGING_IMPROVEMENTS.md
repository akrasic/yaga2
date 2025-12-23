# Anomaly Detection Messaging Improvements

## Executive Summary

The current anomaly detection system correctly identifies anomalies but produces messages that are:
- Too generic to be actionable
- Missing causal context
- Not differentiating between similar symptoms with different root causes

This proposal enhances the semantic richness of anomaly messages to enable faster incident response.

---

## 1. Enhanced Univariate Anomaly Messages

### Current Problem
```python
# Current: Generic, no context
"Unusual request_rate: 250.00"
```

### Proposed Enhancement

```python
# New structure with direction, comparison, and interpretation
METRIC_INTERPRETATIONS = {
    "request_rate": {
        "high": {
            "message": "Traffic spike: {value:.1f} req/s ({percentile:.0f}th percentile, {deviation:.1f}σ above normal)",
            "interpretations": [
                "Possible causes: marketing campaign, viral content, bot activity, upstream retry storm",
                "Check: referrer patterns, geographic distribution, user-agent diversity"
            ],
            "severity_modifier": lambda v, p95: "critical" if v > p95 * 3 else "high" if v > p95 * 2 else "medium"
        },
        "low": {
            "message": "Traffic drop: {value:.1f} req/s ({percentile:.0f}th percentile, {deviation:.1f}σ below normal)",
            "interpretations": [
                "Possible causes: upstream outage, DNS issues, load balancer misconfiguration, deployment blocking traffic",
                "Check: upstream service health, DNS resolution, recent deployments"
            ],
            "severity_modifier": lambda v, p5: "critical" if v < p5 * 0.5 else "high"
        }
    },
    "application_latency": {
        "high": {
            "message": "Latency degradation: {value:.0f}ms ({percentile:.0f}th percentile, normally {mean:.0f}ms)",
            "interpretations": [
                "Possible causes: resource exhaustion, GC pressure, lock contention, downstream slowness",
                "Check: CPU/memory utilization, GC logs, database query times, external API latencies"
            ]
        }
    },
    "error_rate": {
        "high": {
            "message": "Error rate elevated: {value:.2%} (normally {mean:.2%}, threshold {p95:.2%})",
            "interpretations": [
                "Possible causes: deployment regression, downstream failure, capacity exhaustion, data corruption",
                "Check: error logs by type, recent deployments, downstream health, resource limits"
            ],
            "severity_modifier": lambda v, _: "critical" if v > 0.10 else "high" if v > 0.05 else "medium"
        }
    },
    "client_latency": {
        "high": {
            "message": "External dependency slow: {value:.0f}ms (p90: {p90:.0f}ms)",
            "interpretations": [
                "Possible causes: third-party API degradation, network issues, DNS latency",
                "Check: specific client endpoints, network path, third-party status pages"
            ]
        },
        "activated": {  # Special case: normally zero
            "message": "External calls detected: {value:.0f}ms (service normally makes no external calls)",
            "interpretations": [
                "Possible causes: new code path activated, fallback triggered, configuration change",
                "Check: recent deployments, feature flags, fallback logic"
            ]
        }
    },
    "database_latency": {
        "high": {
            "message": "Database response degraded: {value:.0f}ms (p90: {p90:.0f}ms)",
            "interpretations": [
                "Possible causes: missing index, table lock, connection pool exhaustion, replication lag",
                "Check: slow query logs, connection pool metrics, replication status"
            ]
        }
    }
}
```

### Implementation Pattern

```python
def _generate_univariate_message(self, metric_name: str, value: float,
                                  stats: TrainingStatistics) -> dict:
    """Generate semantically rich univariate anomaly message."""

    # Determine direction
    direction = "high" if value > stats.mean else "low"

    # Special case: zero-normal metrics that become non-zero
    if metric_name in self.ZERO_NORMAL_METRICS and stats.is_zero_dominant and value > 0:
        direction = "activated"

    # Calculate context
    deviation = (value - stats.mean) / (stats.std + 1e-8)
    percentile = self._estimate_percentile(value, stats)

    # Get interpretation template
    interpretations = METRIC_INTERPRETATIONS.get(metric_name, {}).get(direction, {})

    return {
        "description": interpretations.get("message", f"Unusual {metric_name}: {value}").format(
            value=value,
            percentile=percentile,
            deviation=abs(deviation),
            mean=stats.mean,
            p90=stats.p90,
            p95=stats.p95
        ),
        "direction": direction,
        "deviation_sigma": deviation,
        "percentile": percentile,
        "possible_causes": interpretations.get("interpretations", []),
        "comparison": {
            "current": value,
            "normal_mean": stats.mean,
            "normal_p95": stats.p95,
            "deviation_factor": value / (stats.mean + 1e-8)
        }
    }
```

---

## 2. Enhanced Multivariate Pattern Detection

### Current Problem
```python
# Current: Just lists metrics, no pattern identification
"Unusual combination of metrics detected (5 metrics: request_rate, application_latency, ...)"
```

### Proposed: Named Multivariate Patterns

```python
MULTIVARIATE_PATTERNS = {
    "traffic_surge_handling_well": {
        "conditions": {
            "request_rate": "high",      # > p90
            "application_latency": "normal",  # within p75
            "error_rate": "normal"       # within p75
        },
        "message": "Traffic surge absorbed successfully: {request_rate:.1f} req/s with stable latency ({application_latency:.0f}ms) and errors ({error_rate:.2%})",
        "severity": "low",
        "interpretation": "System is handling increased load well - no action needed, but monitor for capacity limits",
        "recommended_action": "Monitor for sustained load; consider proactive scaling if traffic continues"
    },

    "traffic_surge_degrading": {
        "conditions": {
            "request_rate": "high",
            "application_latency": "high",
            "error_rate": "normal"
        },
        "message": "Traffic surge causing slowdown: {request_rate:.1f} req/s driving latency to {application_latency:.0f}ms",
        "severity": "high",
        "interpretation": "Service is slowing under load but not failing - approaching capacity",
        "recommended_action": "Scale horizontally; check for resource bottlenecks (CPU, memory, connections)"
    },

    "traffic_surge_failing": {
        "conditions": {
            "request_rate": "high",
            "application_latency": "high",
            "error_rate": "high"
        },
        "message": "Traffic surge overwhelming service: {request_rate:.1f} req/s with {application_latency:.0f}ms latency and {error_rate:.2%} errors",
        "severity": "critical",
        "interpretation": "Service at or beyond capacity - users affected",
        "recommended_action": "Immediate scaling required; consider enabling rate limiting or circuit breakers"
    },

    "silent_degradation": {
        "conditions": {
            "request_rate": "normal",
            "application_latency": "high",
            "error_rate": "normal"
        },
        "message": "Silent performance degradation: latency {application_latency:.0f}ms at normal traffic ({request_rate:.1f} req/s)",
        "severity": "high",
        "interpretation": "Something changed internally - not load-related",
        "recommended_action": "Check recent deployments, database performance, external dependencies, GC behavior"
    },

    "partial_outage": {
        "conditions": {
            "request_rate": "normal",
            "application_latency": "normal",
            "error_rate": "high"
        },
        "message": "Partial outage: {error_rate:.2%} error rate at normal traffic and latency",
        "severity": "critical",
        "interpretation": "Specific code path or dependency failing - not a capacity issue",
        "recommended_action": "Check error logs for specific exception types; identify affected endpoints"
    },

    "fast_failure_mode": {
        "conditions": {
            "request_rate": "any",
            "application_latency": "low",  # Below p25
            "error_rate": "high"
        },
        "message": "Fast-fail mode: {error_rate:.2%} errors with unusually low latency ({application_latency:.0f}ms)",
        "severity": "critical",
        "interpretation": "Service failing quickly without processing - likely upstream rejection or circuit breaker",
        "recommended_action": "Check circuit breaker states, upstream service health, authentication/authorization"
    },

    "downstream_cascade": {
        "conditions": {
            "client_latency": "high",
            "application_latency": "high",
            "client_latency_ratio": "> 0.6"  # client_latency/app_latency
        },
        "message": "Downstream cascade: external calls ({client_latency:.0f}ms) causing {client_latency_ratio:.0%} of total latency",
        "severity": "high",
        "interpretation": "External dependency is the bottleneck",
        "recommended_action": "Check third-party status; consider circuit breaker or fallback; increase timeout awareness"
    },

    "database_bottleneck": {
        "conditions": {
            "database_latency": "high",
            "application_latency": "high",
            "db_latency_ratio": "> 0.5"
        },
        "message": "Database bottleneck: DB queries ({database_latency:.0f}ms) causing {db_latency_ratio:.0%} of total latency",
        "severity": "high",
        "interpretation": "Database is the constraint",
        "recommended_action": "Check slow query logs, connection pool, index usage, replication lag"
    },

    "traffic_cliff": {
        "conditions": {
            "request_rate": "very_low",  # Below p10
            "error_rate": "any"
        },
        "message": "Traffic cliff: only {request_rate:.1f} req/s (normally {normal_rate:.1f} req/s)",
        "severity": "critical",
        "interpretation": "Major traffic loss - likely upstream issue or routing problem",
        "recommended_action": "Check load balancer health, DNS resolution, upstream services, recent deployments"
    }
}
```

### Implementation

```python
def _detect_multivariate_patterns(self, metrics: dict[str, float]) -> dict[str, Any]:
    """Detect named multivariate patterns with semantic meaning."""

    detected_patterns = {}

    for pattern_name, pattern_config in MULTIVARIATE_PATTERNS.items():
        if self._pattern_matches(metrics, pattern_config["conditions"]):
            # Calculate ratios for message formatting
            format_values = {
                **metrics,
                "client_latency_ratio": metrics.get("client_latency", 0) / (metrics.get("application_latency", 1) + 1e-8),
                "db_latency_ratio": metrics.get("database_latency", 0) / (metrics.get("application_latency", 1) + 1e-8),
                "normal_rate": self.training_statistics.get("request_rate", {}).mean
            }

            detected_patterns[pattern_name] = {
                "type": "multivariate_pattern",
                "pattern_name": pattern_name,
                "severity": pattern_config["severity"],
                "description": pattern_config["message"].format(**format_values),
                "interpretation": pattern_config["interpretation"],
                "recommended_action": pattern_config["recommended_action"],
                "contributing_metrics": list(pattern_config["conditions"].keys()),
                "metric_values": {k: metrics.get(k, 0) for k in pattern_config["conditions"].keys()}
            }

    return detected_patterns

def _pattern_matches(self, metrics: dict, conditions: dict) -> bool:
    """Check if metrics match pattern conditions."""
    for metric, condition in conditions.items():
        if metric.endswith("_ratio"):
            continue  # Ratios are calculated, not direct metrics

        stats = self.training_statistics.get(metric)
        if not stats:
            continue

        value = metrics.get(metric, 0)

        if condition == "high" and value <= stats.p90:
            return False
        elif condition == "low" and value >= stats.p25:
            return False
        elif condition == "very_low" and value >= stats.p10:
            return False
        elif condition == "normal" and (value > stats.p75 or value < stats.p25):
            return False

    # Check ratio conditions
    if "client_latency_ratio" in conditions:
        ratio = metrics.get("client_latency", 0) / (metrics.get("application_latency", 1) + 1e-8)
        threshold = float(conditions["client_latency_ratio"].split()[-1])
        if ratio <= threshold:
            return False

    if "db_latency_ratio" in conditions:
        ratio = metrics.get("database_latency", 0) / (metrics.get("application_latency", 1) + 1e-8)
        threshold = float(conditions["db_latency_ratio"].split()[-1])
        if ratio <= threshold:
            return False

    return True
```

---

## 3. Enhanced Service Pattern Detection

### Current Cascading Failure Issue

The current logic:
```python
# Current: error_rate > 5% AND latency < median = "cascading failure"
```

This is ambiguous. Low latency + high errors could mean:
1. **True cascade**: Upstream failing, requests rejected quickly
2. **Circuit breaker active**: Intentional fast-fail
3. **Auth/rate limiting**: Rejected before processing
4. **Partial outage**: Some endpoints failing fast

### Proposed: Differentiated Fast-Fail Patterns

```python
def _detect_fast_fail_patterns(self, metrics: dict[str, float]) -> dict[str, Any]:
    """Detect and differentiate fast-fail scenarios."""

    patterns = {}

    app_latency = metrics.get("application_latency", 0)
    error_rate = metrics.get("error_rate", 0)
    request_rate = metrics.get("request_rate", 0)

    lat_stats = self.training_statistics.get("application_latency")
    req_stats = self.training_statistics.get("request_rate")

    if not lat_stats or error_rate <= 0.05:
        return patterns

    is_fast_fail = app_latency < lat_stats.p25
    is_very_fast = app_latency < lat_stats.p10

    if not is_fast_fail:
        return patterns

    # Differentiate based on additional signals

    if is_very_fast and error_rate > 0.20:
        # Extremely fast failures with high error rate
        patterns["circuit_breaker_open"] = {
            "type": "pattern",
            "severity": "critical",
            "description": f"Circuit breaker likely OPEN: {error_rate:.1%} errors at {app_latency:.0f}ms (requests not reaching backend)",
            "interpretation": "Service is rejecting requests before processing - circuit breaker or upstream rejection",
            "evidence": {
                "latency_vs_p10": f"{app_latency:.0f}ms vs p10={lat_stats.p10:.0f}ms",
                "error_rate": f"{error_rate:.1%}"
            },
            "recommended_action": "Check circuit breaker dashboards; verify upstream service health; look for 503/429 status codes"
        }

    elif is_fast_fail and request_rate < req_stats.p25:
        # Fast failures with reduced traffic
        patterns["upstream_rejection"] = {
            "type": "pattern",
            "severity": "critical",
            "description": f"Upstream rejection pattern: {error_rate:.1%} errors, {app_latency:.0f}ms latency, traffic down to {request_rate:.1f} req/s",
            "interpretation": "Traffic reduced AND fast failures - likely rejected at load balancer or upstream",
            "recommended_action": "Check load balancer health; verify routing configuration; check for deployment issues"
        }

    elif is_fast_fail and error_rate > 0.05 and error_rate < 0.20:
        # Moderate fast failures
        patterns["partial_fast_fail"] = {
            "type": "pattern",
            "severity": "high",
            "description": f"Partial fast-fail: {error_rate:.1%} errors failing quickly ({app_latency:.0f}ms)",
            "interpretation": "Some requests failing before full processing - specific code path or validation",
            "recommended_action": "Check error logs for specific endpoints; look for validation errors or auth failures"
        }

    else:
        # Generic fast-fail
        patterns["cascading_failure"] = {
            "type": "pattern",
            "severity": "critical",
            "description": f"Cascading failure mode: {error_rate:.1%} errors with {app_latency:.0f}ms latency",
            "interpretation": "Service failing quickly - downstream dependency likely unavailable",
            "recommended_action": "Check downstream service health; review circuit breaker states; check connection pools"
        }

    return patterns
```

---

## 4. Contextual Severity Adjustment

### Problem
Static severity doesn't account for:
- Time of day (night issues less impactful)
- Service criticality
- Duration of anomaly
- Rate of change

### Proposed: Dynamic Severity

```python
@dataclass
class SeverityContext:
    base_severity: str
    adjusted_severity: str
    adjustment_reasons: list[str]
    confidence: float

def _calculate_contextual_severity(
    self,
    anomaly_type: str,
    base_severity: str,
    metrics: dict[str, float],
    timestamp: datetime
) -> SeverityContext:
    """Calculate severity with contextual adjustments."""

    adjustments = []
    severity_score = {"low": 1, "medium": 2, "high": 3, "critical": 4}[base_severity]

    # Time-based adjustment
    hour = timestamp.hour
    is_off_hours = hour < 6 or hour > 22
    is_weekend = timestamp.weekday() >= 5

    if is_off_hours or is_weekend:
        # Only downgrade if not user-facing critical
        if anomaly_type not in ["system_overload", "traffic_surge_failing"]:
            severity_score = max(1, severity_score - 1)
            adjustments.append(f"Downgraded: off-hours ({timestamp.strftime('%H:%M')})")

    # Error rate escalation
    error_rate = metrics.get("error_rate", 0)
    if error_rate > 0.10:
        severity_score = min(4, severity_score + 1)
        adjustments.append(f"Escalated: error rate {error_rate:.1%} > 10%")

    # Traffic impact multiplier
    request_rate = metrics.get("request_rate", 0)
    req_stats = self.training_statistics.get("request_rate")
    if req_stats and request_rate > req_stats.p99:
        severity_score = min(4, severity_score + 1)
        adjustments.append(f"Escalated: traffic at p99+ ({request_rate:.1f} req/s)")

    # Map back to severity string
    adjusted = {1: "low", 2: "medium", 3: "high", 4: "critical"}[severity_score]

    return SeverityContext(
        base_severity=base_severity,
        adjusted_severity=adjusted,
        adjustment_reasons=adjustments,
        confidence=0.9 if not adjustments else 0.8
    )
```

---

## 5. Actionable Recommendations Engine

### Proposed: Context-Aware Recommendations

```python
RECOMMENDATION_RULES = {
    # Pattern-based recommendations
    ("traffic_surge_failing", "critical"): [
        "IMMEDIATE: Enable auto-scaling or manually scale horizontally",
        "IMMEDIATE: Consider activating rate limiting to protect backend",
        "INVESTIGATE: Check if this is legitimate traffic or potential attack",
        "PREPARE: Have rollback ready if recent deployment is cause"
    ],

    ("database_bottleneck", "high"): [
        "INVESTIGATE: Run EXPLAIN on recent slow queries",
        "CHECK: Database connection pool utilization",
        "CHECK: Replication lag if using read replicas",
        "CONSIDER: Query result caching if appropriate"
    ],

    ("circuit_breaker_open", "critical"): [
        "VERIFY: Which circuit breaker is open (check dashboard)",
        "CHECK: Health of protected downstream service",
        "ASSESS: Is this protecting the system or causing user impact?",
        "DECIDE: Manual circuit breaker reset vs wait for auto-recovery"
    ],

    # Metric-specific recommendations
    ("error_rate_high", "critical"): [
        "IMMEDIATE: Check error logs for exception stack traces",
        "CORRELATE: Identify if errors are from specific endpoints",
        "TIMELINE: Was there a recent deployment? (last 30 min)",
        "VERIFY: Are dependent services healthy?"
    ],

    ("latency_high", "high"): [
        "CHECK: CPU and memory utilization",
        "CHECK: Database and external API response times",
        "PROFILE: Enable request tracing if not already active",
        "COMPARE: Latency breakdown (app vs db vs client)"
    ]
}

def _generate_recommendations(
    self,
    anomaly_type: str,
    severity: str,
    metrics: dict[str, float],
    pattern_data: dict
) -> list[str]:
    """Generate prioritized, actionable recommendations."""

    recommendations = []

    # Get pattern-specific recommendations
    key = (anomaly_type, severity)
    if key in RECOMMENDATION_RULES:
        recommendations.extend(RECOMMENDATION_RULES[key])

    # Add metric-specific recommendations
    for metric, value in metrics.items():
        stats = self.training_statistics.get(metric)
        if stats and value > stats.p95:
            metric_key = (f"{metric}_high", severity)
            if metric_key in RECOMMENDATION_RULES:
                recommendations.extend(RECOMMENDATION_RULES[metric_key])

    # Add contextual recommendations based on ratios
    if pattern_data.get("client_latency_ratio", 0) > 0.6:
        recommendations.append("FOCUS: External dependency is primary bottleneck - investigate third-party status")

    if pattern_data.get("db_latency_ratio", 0) > 0.5:
        recommendations.append("FOCUS: Database is primary bottleneck - check slow query logs and connection pool")

    # Deduplicate and prioritize (IMMEDIATE > CHECK > INVESTIGATE > CONSIDER)
    priority_order = ["IMMEDIATE", "VERIFY", "CHECK", "INVESTIGATE", "CORRELATE", "TIMELINE", "ASSESS", "FOCUS", "CONSIDER", "PREPARE", "DECIDE"]

    seen = set()
    unique_recs = []
    for rec in recommendations:
        if rec not in seen:
            seen.add(rec)
            unique_recs.append(rec)

    # Sort by priority
    def priority_key(rec):
        for i, prefix in enumerate(priority_order):
            if rec.startswith(prefix):
                return i
        return len(priority_order)

    return sorted(unique_recs, key=priority_key)[:5]  # Top 5 recommendations
```

---

## 6. Enhanced Output Structure

### Proposed Anomaly Payload

```python
{
    "anomaly_id": "fp_booking_traffic_surge_failing_20241217_143022",
    "service": "booking",
    "timestamp": "2024-12-17T14:30:22Z",
    "time_period": "business_hours",

    "classification": {
        "type": "multivariate_pattern",
        "pattern_name": "traffic_surge_failing",
        "detection_method": "pattern_matching + isolation_forest"
    },

    "severity": {
        "level": "critical",
        "base_level": "high",
        "adjustments": ["Escalated: error rate 12.3% > 10%"],
        "confidence": 0.92
    },

    "description": {
        "summary": "Traffic surge overwhelming service: 450 req/s with 2340ms latency and 12.3% errors",
        "interpretation": "Service at or beyond capacity - users affected",
        "impact": "Immediate service degradation likely affecting users"
    },

    "metrics": {
        "current": {
            "request_rate": 450.2,
            "application_latency": 2340,
            "error_rate": 0.123,
            "client_latency": 120,
            "database_latency": 890
        },
        "comparison": {
            "request_rate": {"current": 450.2, "mean": 180.5, "p95": 320.0, "percentile": 99.2, "deviation_sigma": 4.2},
            "application_latency": {"current": 2340, "mean": 450, "p95": 1200, "percentile": 99.8, "deviation_sigma": 5.1},
            "error_rate": {"current": 0.123, "mean": 0.008, "p95": 0.025, "percentile": 99.9, "deviation_sigma": 8.3}
        },
        "ratios": {
            "db_latency_ratio": 0.38,
            "client_latency_ratio": 0.05
        }
    },

    "root_cause_indicators": [
        {"factor": "request_rate", "contribution": "high", "detail": "Traffic 2.5x above p95"},
        {"factor": "application_latency", "contribution": "high", "detail": "Latency 2x above p95"},
        {"factor": "error_rate", "contribution": "critical", "detail": "Errors 5x above p95"}
    ],

    "recommendations": [
        "IMMEDIATE: Enable auto-scaling or manually scale horizontally",
        "IMMEDIATE: Consider activating rate limiting to protect backend",
        "INVESTIGATE: Check if this is legitimate traffic or potential attack",
        "CHECK: Database connection pool utilization",
        "PREPARE: Have rollback ready if recent deployment is cause"
    ],

    "fingerprinting": {
        "fingerprint_id": "fp_booking_traffic_surge_failing",
        "incident_id": "inc_abc123",
        "incident_action": "CONTINUE",
        "occurrence_count": 3,
        "first_seen": "2024-12-17T14:25:00Z",
        "incident_duration_minutes": 5
    }
}
```

---

## 7. Implementation Roadmap

### Phase 1: Message Enhancement (Low Risk)
1. Add `METRIC_INTERPRETATIONS` dictionary
2. Modify `_detect_univariate_anomalies` to use enhanced messages
3. Add direction and comparison context to all messages

### Phase 2: Multivariate Patterns (Medium Risk)
1. Implement `MULTIVARIATE_PATTERNS` dictionary
2. Add `_detect_multivariate_patterns` method
3. Replace generic multivariate message with pattern-specific messages

### Phase 3: Fast-Fail Differentiation (Medium Risk)
1. Implement `_detect_fast_fail_patterns`
2. Replace simple cascading failure detection
3. Add circuit breaker and upstream rejection patterns

### Phase 4: Recommendations Engine (Low Risk)
1. Implement `RECOMMENDATION_RULES` dictionary
2. Add `_generate_recommendations` method
3. Include recommendations in anomaly payload

### Phase 5: Contextual Severity (Low Risk)
1. Implement `_calculate_contextual_severity`
2. Add severity adjustment reasoning to payload
3. Include time-of-day and service criticality factors

---

## 8. Testing Strategy

### Unit Tests
```python
def test_traffic_surge_failing_pattern():
    """Test that high traffic + high latency + high errors = traffic_surge_failing"""
    metrics = {
        "request_rate": 500,  # Way above p95
        "application_latency": 3000,  # Way above p95
        "error_rate": 0.15  # 15% errors
    }

    detector = SmartboxAnomalyDetector("test_service")
    detector.training_statistics = create_mock_stats()

    patterns = detector._detect_multivariate_patterns(metrics)

    assert "traffic_surge_failing" in patterns
    assert patterns["traffic_surge_failing"]["severity"] == "critical"
    assert "scale" in patterns["traffic_surge_failing"]["recommended_action"].lower()

def test_circuit_breaker_detection():
    """Test that very fast failures are identified as circuit breaker"""
    metrics = {
        "request_rate": 200,
        "application_latency": 5,  # Extremely fast - below p10
        "error_rate": 0.30  # 30% errors
    }

    detector = SmartboxAnomalyDetector("test_service")
    detector.training_statistics = create_mock_stats(latency_p10=50)

    patterns = detector._detect_fast_fail_patterns(metrics)

    assert "circuit_breaker_open" in patterns
```

### Integration Tests
- Test full detection pipeline with realistic metric combinations
- Verify fingerprinting works with new pattern names
- Ensure API payloads match expected schema

---

## Summary

These improvements transform anomaly messages from:

**Before**: "Unusual combination of metrics detected"

**After**: "Traffic surge overwhelming service: 450 req/s with 2340ms latency and 12.3% errors. Service at or beyond capacity - users affected. IMMEDIATE: Enable auto-scaling or manually scale horizontally"

The key principles:
1. **Name the pattern** - Give anomalies semantic names operators recognize
2. **Explain the meaning** - What does this combination actually indicate?
3. **Provide context** - How does current compare to normal?
4. **Suggest actions** - What should the operator do RIGHT NOW?
5. **Differentiate similar symptoms** - Fast failures aren't all the same
