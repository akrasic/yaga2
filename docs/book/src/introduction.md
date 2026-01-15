# Introduction

Yaga2 is an ML-based anomaly detection system for monitoring service health in production environments. This guide explains how the system detects anomalies, evaluates their operational significance, and manages incident lifecycles.

## What Problem Does Yaga2 Solve?

Modern microservice architectures generate massive amounts of metrics data. Manually setting static thresholds for every service and metric combination is:

- **Impractical**: Hundreds of services × multiple metrics × different time periods = thousands of thresholds
- **Brittle**: Traffic patterns change seasonally, after deployments, and as services evolve
- **Noisy**: Static thresholds either miss real issues or generate alert fatigue

Yaga2 addresses these challenges by:

1. **Learning normal behavior** from historical data using machine learning
2. **Detecting deviations** that represent genuine anomalies, not just threshold breaches
3. **Evaluating operational impact** to determine if anomalies actually matter
4. **Managing alert lifecycle** to reduce noise and prevent duplicate alerts

## Core Concepts

### What is Anomaly Detection?

Anomaly detection identifies data points that deviate significantly from expected patterns. Unlike threshold-based alerting (e.g., "alert if latency > 500ms"), anomaly detection learns what "normal" looks like for each service and flags when behavior deviates from that baseline.

**Example:**
- Service A normally has 200ms latency → 350ms is anomalous
- Service B normally has 800ms latency → 350ms is actually *better* than normal
- A static 500ms threshold would miss the anomaly for Service A and false-alarm for Service B

### What is an SLO?

**SLO (Service Level Objective)** is a target level of service reliability. It defines what "good enough" means for your service.

Common SLOs include:
- **Latency SLO**: "99% of requests complete in under 500ms"
- **Availability SLO**: "Service is available 99.9% of the time"
- **Error Rate SLO**: "Less than 0.5% of requests result in errors"

**Why SLOs Matter for Anomaly Detection:**

An anomaly might be statistically significant but operationally irrelevant:
- ML detects latency at 280ms (unusual compared to baseline of 150ms)
- But SLO says 500ms is acceptable
- Result: The anomaly is real but doesn't warrant immediate action

Yaga2 combines ML detection with SLO evaluation to answer two questions:
1. **Is this unusual?** (ML detection)
2. **Does it matter?** (SLO evaluation)

### What is Isolation Forest?

**Isolation Forest** is the machine learning algorithm at the heart of Yaga2's anomaly detection. It's an "unsupervised" algorithm, meaning it doesn't need labeled examples of anomalies—it learns what's normal from your data and identifies deviations.

The key insight: **Anomalies are easier to isolate than normal points.**

Think of it like a game of "20 questions" for data points:
- Normal data points are similar to many others, requiring many questions to identify uniquely
- Anomalies are outliers that can be identified with just a few questions

We'll explore Isolation Forest in detail in the [Detection Layer](./detection/isolation-forest.md) section.

### What is an Incident?

An **incident** is a tracked occurrence of an anomaly pattern. Yaga2 doesn't just detect anomalies—it tracks them over time to:

- **Confirm** anomalies aren't transient glitches (require 2+ consecutive detections)
- **Track duration** of ongoing issues
- **Prevent duplicates** by recognizing the same issue across detection cycles
- **Manage resolution** with grace periods to avoid flapping

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Yaga2 Pipeline                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌───────────────┐                                                         │
│   │ VictoriaMetrics│    Collect 5 core metrics every 2-3 minutes            │
│   │   (Metrics)    │    • request_rate    • dependency_latency                  │
│   └───────┬───────┘    • app_latency      • db_latency                      │
│           │            • error_rate                                         │
│           ▼                                                                 │
│   ┌───────────────────────────────────────────────────────────────────┐     │
│   │                     1. DETECTION LAYER                            │     │
│   │  ┌─────────────────────────┐    ┌─────────────────────────────┐   │     │
│   │  │    Isolation Forest     │    │    Pattern Matching         │   │     │
│   │  │    (ML Detection)       │───▶│    (Interpretation)         │   │     │
│   │  │                         │    │                             │   │     │
│   │  │ "Is this unusual?"      │    │ "What does it mean?"        │   │     │
│   │  └─────────────────────────┘    └─────────────────────────────┘   │     │
│   │                                                                   │     │
│   │  Output: anomaly detected, severity, pattern name                 │     │
│   └───────────────────────────────────────────────────────────────────┘     │
│           │                                                                 │
│           ▼                                                                 │
│   ┌───────────────────────────────────────────────────────────────────┐     │
│   │                     2. SLO EVALUATION LAYER                       │     │
│   │                                                                   │     │
│   │  Compare metrics against operational thresholds:                  │     │
│   │  • Latency vs SLO targets (acceptable/warning/critical)           │     │
│   │  • Error rate vs SLO targets                                      │     │
│   │  • Database latency vs baseline ratio                             │     │
│   │  • Request rate for surge/cliff                                   │     │
│   │                                                                   │     │
│   │  "Does this anomaly matter operationally?"                        │     │
│   │                                                                   │     │
│   │  Output: adjusted severity (critical/high/low), SLO status        │     │
│   └───────────────────────────────────────────────────────────────────┘     │
│           │                                                                 │
│           ▼                                                                 │
│   ┌───────────────────────────────────────────────────────────────────┐     │
│   │                     3. INCIDENT LIFECYCLE                         │     │
│   │                                                                   │     │
│   │  SUSPECTED ──▶ OPEN ──▶ RECOVERING ──▶ CLOSED                     │     │
│   │   (wait)     (alert)    (grace)       (resolve)                   │     │
│   │                                                                   │     │
│   │  • Confirmation: 2 consecutive detections before alerting         │     │
│   │  • Grace period: 3 cycles without detection before closing        │     │
│   │  • Fingerprinting: Track same issue across time                   │     │
│   │                                                                   │     │
│   │  "Should we alert now, or wait for confirmation?"                 │     │
│   └───────────────────────────────────────────────────────────────────┘     │
│           │                                                                 │
│           ▼                                                                 │
│   ┌───────────────┐                                                         │
│   │  Web API      │    Alert payload sent for confirmed incidents           │
│   │  Dashboard    │    Resolution payload sent when incidents close         │
│   └───────────────┘                                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Time-Aware Detection

Service behavior varies significantly by time:

| Time Period | Typical Behavior |
|-------------|------------------|
| Business hours (Mon-Fri 8-18) | High traffic, tight latency requirements |
| Evening (Mon-Fri 18-22) | Moderate traffic |
| Night (22-06) | Low traffic, batch jobs |
| Weekend | Different patterns entirely |

A 3 AM traffic level that would be alarming at 3 PM is completely normal at night. Yaga2 trains **separate models for each time period** to avoid false positives from expected behavioral differences.

## The Five Metrics

Yaga2 monitors five core metrics for each service:

| Metric | What It Measures | Why It Matters |
|--------|------------------|----------------|
| **request_rate** | Requests per second | Traffic volume - sudden drops or spikes indicate problems |
| **application_latency** | Server processing time (ms) | User experience - slow responses frustrate users |
| **dependency_latency** | External dependency call time (ms) | Downstream issues - if a dependency is slow, you'll be slow |
| **database_latency** | Database query time (ms) | Database health - often the bottleneck |
| **error_rate** | Failed requests (0-1 ratio) | Reliability - errors directly impact users |

## How to Use This Guide

| If You Want To... | Read... |
|-------------------|---------|
| Understand how ML detection works | [Isolation Forest](./detection/isolation-forest.md) |
| Learn about named patterns | [Pattern Matching](./detection/pattern-matching.md) |
| Configure SLO thresholds | [SLO Evaluation](./slo/README.md) |
| Understand alert timing | [Incident Lifecycle](./incidents/README.md) |
| Troubleshoot issues | [Troubleshooting](./reference/troubleshooting.md) |
| Quick reference | [Decision Matrix](./reference/decision-matrix.md) |

## Quick Reference

### Severity Levels

| Severity | Meaning | Action Required |
|----------|---------|-----------------|
| **critical** | SLO breached, users impacted | Immediate response |
| **high** | Approaching SLO limits | Investigate promptly |
| **low** | Anomaly detected but within SLO | Monitor, no action |

### Alert Flow Summary

```
Anomaly detected → SUSPECTED (no alert, wait for confirmation)
                        │
                        ▼ (detected again)
                    OPEN (alert sent!)
                        │
                        ▼ (not detected)
                  RECOVERING (grace period)
                        │
                        ▼ (still not detected)
                    CLOSED (resolution sent)
```
