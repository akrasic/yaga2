# Severity Adjustment

The final step of SLO evaluation: adjusting ML-assigned severity based on operational impact.

## Core Principle

**SLO status determines final severity, not ML confidence.**

| SLO Status | Final Severity | Rationale |
|------------|----------------|-----------|
| `ok` | `low` | Anomaly detected but operationally acceptable |
| `warning` | `high` | Approaching limits, should investigate |
| `breached` | `critical` | SLO exceeded, requires action |

## Adjustment Matrix

| ML Severity | SLO Status | Final Severity | Example |
|-------------|------------|----------------|---------|
| critical | ok | **low** | 280ms latency (unusual but < 300ms SLO) |
| critical | warning | high | 750ms latency (approaching 800ms SLO) |
| critical | breached | critical | 1200ms latency (> 1000ms SLO) |
| high | ok | **low** | Same principle |
| high | warning | high | No change needed |
| high | breached | critical | Escalated |
| medium | ok | **low** | Minor anomaly, within SLO |
| medium | warning | high | Escalated due to SLO proximity |
| low | breached | critical | Even low anomaly escalated if SLO breached |

## Key Behavior Change (v1.3.1)

**Before v1.3.1**: SLO `ok` adjusted to `medium`
**After v1.3.1**: SLO `ok` adjusts to `low`

This ensures "operationally acceptable" consistently means "low priority."

## Adjustment Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Severity Adjustment                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ML Detection                SLO Evaluation                    │
│   ────────────               ───────────────                    │
│   severity: critical         slo_status: ok                     │
│                                                                 │
│          │                          │                           │
│          └──────────┬───────────────┘                           │
│                     │                                           │
│                     ▼                                           │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  Rule: If slo_status == "ok" → severity = "low"         │   │
│   │  Rule: If slo_status == "warning" → severity >= "high"  │   │
│   │  Rule: If slo_status == "breached" → severity = "critical│   │
│   └─────────────────────────────────────────────────────────┘   │
│                     │                                           │
│                     ▼                                           │
│                                                                 │
│   Final Output:                                                 │
│   ─────────────                                                 │
│   overall_severity: low                                         │
│   slo_evaluation:                                               │
│     original_severity: critical                                 │
│     adjusted_severity: low                                      │
│     severity_changed: true                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Output Example

```json
{
  "overall_severity": "low",
  "slo_evaluation": {
    "original_severity": "critical",
    "adjusted_severity": "low",
    "severity_changed": true,
    "slo_status": "ok",
    "slo_proximity": 0.56,
    "operational_impact": "informational",
    "explanation": "Severity adjusted from critical to low based on SLO evaluation. Anomaly detected but metrics within acceptable SLO thresholds (latency: 280ms < 300ms, errors: 0.10% < 0.50%)."
  }
}
```

## Operational Impact Levels

| Level | Meaning | Action |
|-------|---------|--------|
| `none` | No anomaly or fully normal | No action |
| `informational` | Anomaly noted but acceptable | Log only |
| `actionable` | Approaching limits | Investigate |
| `critical` | SLO breached | Immediate action |

## Configuration Options

```json
{
  "slos": {
    "enabled": true,
    "allow_downgrade_to_informational": true,
    "require_slo_breach_for_critical": true
  }
}
```

| Option | Default | Effect |
|--------|---------|--------|
| `enabled` | true | Enable/disable SLO evaluation |
| `allow_downgrade_to_informational` | true | Allow high/critical → low when SLO ok |
| `require_slo_breach_for_critical` | true | Only allow critical if SLO breached |

## root `overall_severity`

**Important (v1.3.1)**: The root-level `overall_severity` field now correctly reflects the SLO-adjusted value.

```json
{
  "overall_severity": "low",          // ← SLO-adjusted value
  "slo_evaluation": {
    "original_severity": "critical",  // ← ML-assigned value
    "adjusted_severity": "low"        // ← Same as overall_severity
  }
}
```

Previously, `overall_severity` could show `critical` while `adjusted_severity` showed `low`.

## Alert Suppression

When SLO evaluation results in `low` severity:
- Alert is still generated (for logging/visibility)
- Alert may be filtered by downstream systems (e.g., skip PagerDuty)
- Dashboard shows alert with low priority indicator

To completely suppress alerts below a threshold, use downstream alert filtering rules.
