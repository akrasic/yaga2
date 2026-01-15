# Pattern Matching

Pattern matching interprets ML signals by comparing metric combinations against known operational scenarios. This chapter explains what pattern matching is, how it works, and provides detailed examples of each pattern type.

## What is Pattern Matching?

While Isolation Forest answers **"Is this unusual?"**, pattern matching answers **"What does this unusual behavior mean?"**

Pattern matching is a **rule-based interpretation layer** that recognizes known operational scenarios by examining how metrics relate to each other.

### Why Pattern Matching?

ML detection tells you something is anomalous, but not what it means:

```
ML Output:
  application_latency: anomalous (score: -0.35)
  request_rate: anomalous (score: -0.28)
  error_rate: normal

Human question: "So... is this bad? What should I do?"
```

Pattern matching adds semantic meaning:

```
Pattern Match: traffic_surge_degrading

Interpretation: "You have a traffic surge (3x normal) that's causing
  latency degradation (450ms vs 150ms normal). Errors are still normal,
  so you're approaching capacity but not failing yet."

Recommended Actions:
  1. Scale horizontally if possible
  2. Check resource bottlenecks (CPU, connections)
  3. Monitor error rate for signs of impending failure
```

### ML vs Pattern Matching

| Aspect | Isolation Forest | Pattern Matching |
|--------|------------------|------------------|
| **Approach** | Statistical (unsupervised ML) | Rule-based (expert knowledge) |
| **Question answered** | "Is this unusual?" | "What does it mean?" |
| **Learns from** | Historical data | Domain expertise |
| **Catches** | Novel/unknown issues | Known operational scenarios |
| **Output** | Anomaly score | Named pattern + recommendations |
| **Adapts** | Automatically via training | Manually via pattern definitions |

### How They Work Together

```
┌─────────────────────────────────────────────────────────────────────┐
│                   Sequential Detection Pipeline                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Current Metrics                                                   │
│   ┌────────────────────────────────────┐                           │
│   │ request_rate: 850 req/s           │                           │
│   │ application_latency: 450ms         │                           │
│   │ error_rate: 1.5%                   │                           │
│   │ database_latency: 15ms             │                           │
│   └─────────────────┬──────────────────┘                           │
│                     │                                               │
│                     ▼                                               │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  Phase 1: Isolation Forest Detection                        │   │
│   │                                                             │   │
│   │  For each metric, compute anomaly score:                    │   │
│   │    request_rate: score=-0.28, direction=high               │   │
│   │    application_latency: score=-0.35, direction=high        │   │
│   │    error_rate: score=-0.05, direction=normal               │   │
│   │    database_latency: score=0.1, direction=normal           │   │
│   │                                                             │   │
│   │  Output: List of AnomalySignal objects                     │   │
│   └─────────────────────────┬───────────────────────────────────┘   │
│                             │                                       │
│                             ▼                                       │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  Phase 2: Pattern Interpretation                            │   │
│   │                                                             │   │
│   │  Convert signals to levels:                                 │   │
│   │    request_rate: high                                      │   │
│   │    application_latency: high                               │   │
│   │    error_rate: normal                                      │   │
│   │    database_latency: normal                                │   │
│   │                                                             │   │
│   │  Match against patterns:                                    │   │
│   │    traffic_surge_failing? No (errors not high)             │   │
│   │    traffic_surge_degrading? ✓ YES                          │   │
│   │                                                             │   │
│   │  Output: Interpreted anomaly with recommendations          │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## How Pattern Matching Works

### Step 1: Convert IF Signals to Metric Levels

Each metric is classified based on its Isolation Forest signal:

| Percentile Range | Level |
|------------------|-------|
| > 95th | `very_high` |
| 90th - 95th | `high` |
| 80th - 90th | `elevated` |
| 70th - 80th | `moderate` |
| 10th - 70th | `normal` |
| 5th - 10th | `low` |
| < 5th | `very_low` |
| No IF signal | `normal` |

**Example:**
```
IF says: latency is at 92nd percentile (score: -0.35)
Level: high (between 90th-95th)
```

### Step 2: Handle "Lower is Better" Metrics

Some metrics are better when low:

| Metric | Lower is Better? | Why |
|--------|------------------|-----|
| `error_rate` | Yes | 0% errors is ideal |
| `database_latency` | Yes | Faster DB is better |
| `dependency_latency` | Yes | Faster dependencies is better |
| `application_latency` | No* | Low + high errors = fast-fail |

*Application latency is NOT in "lower is better" because low latency with high errors indicates fast failures (requests rejected before processing).

When IF flags a "lower is better" metric as anomalously low, it's treated as `normal`:

```
IF says: database_latency is unusually low (faster than normal)
Pattern matching: Treat as "normal" (this is good, not a problem)
```

### Step 3: Match Against Pattern Conditions

Each pattern has conditions that must be met:

```python
"traffic_surge_degrading": {
    "conditions": {
        "request_rate": "high",           # Must be high
        "application_latency": "high",    # Must be high
        "error_rate": "normal",           # Must be normal
    }
}
```

The pattern matches only if **all conditions** are satisfied.

### Step 4: Generate Interpreted Output

When a pattern matches, generate:
- Human-readable description
- Semantic interpretation
- Recommended actions
- Contributing metrics

## Named Patterns - Detailed Reference

### Traffic Patterns

These patterns relate to changes in traffic volume (request rate).

#### `traffic_surge_healthy`

```
┌─────────────────────────────────────────────────────────────────────┐
│  PATTERN: traffic_surge_healthy                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Conditions:                                                        │
│    request_rate: high                                              │
│    application_latency: normal                                     │
│    error_rate: normal                                              │
│                                                                     │
│  Severity: low                                                      │
│                                                                     │
│  What It Means:                                                     │
│    Traffic has increased significantly (e.g., 3x normal) but       │
│    the service is handling it gracefully. Latency and errors       │
│    remain normal - your system has headroom.                       │
│                                                                     │
│  Possible Causes:                                                   │
│    • Marketing campaign launched                                   │
│    • Viral content / social media mention                          │
│    • Seasonal peak (holiday shopping)                              │
│    • Organic growth                                                │
│                                                                     │
│  Recommended Actions:                                               │
│    • Monitor for signs of degradation                              │
│    • Verify traffic is legitimate (not bot/attack)                 │
│    • Consider pre-emptive scaling if trend continues               │
│                                                                     │
│  Example Scenario:                                                  │
│    Black Friday starts. Traffic jumps from 100 req/s to 350 req/s. │
│    Latency stays at 150ms, errors at 0.1%.                         │
│    System is handling the surge well.                              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### `traffic_surge_degrading`

```
┌─────────────────────────────────────────────────────────────────────┐
│  PATTERN: traffic_surge_degrading                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Conditions:                                                        │
│    request_rate: high                                              │
│    application_latency: high                                       │
│    error_rate: normal                                              │
│                                                                     │
│  Severity: high                                                     │
│                                                                     │
│  What It Means:                                                     │
│    Traffic surge is causing performance degradation. Users are     │
│    experiencing slower responses, but requests are still           │
│    completing successfully. You're approaching capacity.           │
│                                                                     │
│  Why Errors Are Still Normal:                                       │
│    The system is slowing down to cope with load rather than        │
│    failing. This is often due to:                                  │
│    • Thread pools filling up (requests queue)                      │
│    • Connection pools near exhaustion                              │
│    • CPU at high utilization but not 100%                          │
│                                                                     │
│  Recommended Actions:                                               │
│    • IMMEDIATE: Scale horizontally if possible                     │
│    • CHECK: Resource utilization (CPU, memory, connections)        │
│    • MONITOR: Error rate for signs of impending failure            │
│    • CONSIDER: Enable request throttling to protect backend        │
│                                                                     │
│  Example Scenario:                                                  │
│    Traffic: 850 req/s (normally 200 req/s)                         │
│    Latency: 450ms (normally 120ms)                                 │
│    Errors: 0.5% (normal)                                           │
│                                                                     │
│    Users notice slowness but can still complete transactions.      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### `traffic_surge_failing`

```
┌─────────────────────────────────────────────────────────────────────┐
│  PATTERN: traffic_surge_failing                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Conditions:                                                        │
│    request_rate: high                                              │
│    application_latency: high                                       │
│    error_rate: high                                                │
│                                                                     │
│  Severity: critical                                                 │
│                                                                     │
│  What It Means:                                                     │
│    Service is at or beyond capacity. The traffic surge has         │
│    overwhelmed the system - both latency and errors are elevated.  │
│    Users are actively being impacted.                              │
│                                                                     │
│  Why This Is Critical:                                              │
│    High errors + high latency = users are both waiting AND         │
│    failing. This is the worst user experience - slow failures.     │
│                                                                     │
│  Recommended Actions:                                               │
│    • IMMEDIATE: Scale horizontally or add capacity                 │
│    • IMMEDIATE: Enable rate limiting to protect backend            │
│    • INVESTIGATE: Confirm traffic is legitimate vs attack          │
│    • PREPARE: Have rollback ready if recent deployment caused it   │
│                                                                     │
│  Example Scenario:                                                  │
│    Traffic: 1200 req/s (normally 200 req/s)                        │
│    Latency: 2500ms (normally 120ms)                                │
│    Errors: 15%                                                     │
│                                                                     │
│    System is collapsing under load. Many users seeing errors,      │
│    others waiting forever for timeouts.                            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### `traffic_cliff`

```
┌─────────────────────────────────────────────────────────────────────┐
│  PATTERN: traffic_cliff                                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Conditions:                                                        │
│    request_rate: very_low (sudden drop to <10% of normal)          │
│                                                                     │
│  Severity: critical                                                 │
│                                                                     │
│  What It Means:                                                     │
│    Traffic has suddenly dropped to near-zero. This often           │
│    indicates an upstream problem preventing requests from          │
│    reaching the service.                                           │
│                                                                     │
│  Common Causes:                                                     │
│    • DNS failure (users can't resolve hostname)                    │
│    • Load balancer misconfigured (routing to wrong backend)        │
│    • Firewall rule blocking traffic                                │
│    • CDN/edge failure                                              │
│    • Upstream service failure                                      │
│    • Network partition                                             │
│                                                                     │
│  Why Not Low Latency Pattern:                                       │
│    The few requests getting through might show normal latency,     │
│    but the problem is that NO requests are arriving.               │
│                                                                     │
│  Recommended Actions:                                               │
│    • IMMEDIATE: Check upstream services and load balancers         │
│    • CHECK: DNS resolution from multiple locations                 │
│    • VERIFY: Network connectivity and firewall rules               │
│    • CORRELATE: Check if other services are also affected          │
│                                                                     │
│  Example Scenario:                                                  │
│    Traffic: 5 req/s (normally 200 req/s) - 97.5% drop             │
│                                                                     │
│    At 2:15 PM, traffic suddenly drops from 200 req/s to 5 req/s.  │
│    Investigation reveals DNS TTL expired and DNS provider had      │
│    an outage, causing resolution failures.                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Error Patterns

These patterns relate to error rate anomalies.

#### `error_rate_elevated`

```
┌─────────────────────────────────────────────────────────────────────┐
│  PATTERN: error_rate_elevated                                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Conditions:                                                        │
│    request_rate: normal                                            │
│    application_latency: normal                                     │
│    error_rate: high                                                │
│                                                                     │
│  Severity: high                                                     │
│                                                                     │
│  What It Means:                                                     │
│    Error rate has increased significantly, but traffic and         │
│    latency are normal. This usually indicates a specific code      │
│    path or endpoint is failing, not a systemic issue.              │
│                                                                     │
│  Common Causes:                                                     │
│    • Bug in a specific endpoint                                    │
│    • External dependency failing for certain operations            │
│    • Bad input data causing validation failures                    │
│    • Feature flag enabled broken code                              │
│    • Database constraint violations                                │
│                                                                     │
│  Why Latency Is Normal:                                             │
│    Most requests succeed normally - only a subset is failing.      │
│    The average latency doesn't change much because successes       │
│    dominate the metrics.                                           │
│                                                                     │
│  Recommended Actions:                                               │
│    • INVESTIGATE: Check exception types in exception_context       │
│    • CHECK: Recent deployments or configuration changes            │
│    • IDENTIFY: Which endpoints/operations are failing              │
│    • VERIFY: External dependency health                            │
│                                                                     │
│  Example Scenario:                                                  │
│    Traffic: 150 req/s (normal)                                     │
│    Latency: 120ms (normal)                                         │
│    Errors: 3.5% (normally 0.1%)                                    │
│                                                                     │
│    Exception breakdown shows 85% of errors are                     │
│    "PaymentGatewayException" - the payment provider is having      │
│    issues, but only checkout flow is affected.                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### `error_rate_critical`

```
┌─────────────────────────────────────────────────────────────────────┐
│  PATTERN: error_rate_critical                                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Conditions:                                                        │
│    request_rate: normal                                            │
│    application_latency: normal                                     │
│    error_rate: very_high (>5%)                                     │
│                                                                     │
│  Severity: critical                                                 │
│                                                                     │
│  What It Means:                                                     │
│    A significant portion of requests are failing. While the        │
│    service responds quickly (suggesting infrastructure is OK),     │
│    business logic is failing at a high rate.                       │
│                                                                     │
│  Common Causes:                                                     │
│    • Critical bug in recent deployment                             │
│    • Database schema mismatch after migration                      │
│    • External API contract changed                                 │
│    • Authentication/authorization failure                          │
│    • Missing configuration or secrets                              │
│                                                                     │
│  Recommended Actions:                                               │
│    • IMMEDIATE: Check recent deployments (rollback candidate)      │
│    • INVESTIGATE: Exception breakdown for root cause               │
│    • CHECK: Database connectivity and schema                       │
│    • VERIFY: All required configuration/secrets present            │
│                                                                     │
│  Example Scenario:                                                  │
│    Traffic: 150 req/s (normal)                                     │
│    Latency: 80ms (normal - fast errors)                           │
│    Errors: 25%                                                     │
│                                                                     │
│    Exception breakdown: 100% "DatabaseSchemaException"             │
│    Recent deployment added a required column, but migration        │
│    didn't run in production.                                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Latency Patterns

These patterns relate to response time anomalies.

#### `latency_spike_recent`

```
┌─────────────────────────────────────────────────────────────────────┐
│  PATTERN: latency_spike_recent                                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Conditions:                                                        │
│    request_rate: normal                                            │
│    application_latency: high                                       │
│    error_rate: normal                                              │
│                                                                     │
│  Severity: high                                                     │
│                                                                     │
│  What It Means:                                                     │
│    Latency increased without traffic change. Something changed     │
│    in the system - this is NOT a capacity issue (traffic is        │
│    normal). Likely a recent deployment, config change, or          │
│    dependency degradation.                                         │
│                                                                     │
│  Why Traffic Being Normal Matters:                                  │
│    If traffic were high, latency increase would make sense         │
│    (capacity). With normal traffic, the slowdown is internal.      │
│                                                                     │
│  Common Causes:                                                     │
│    • Recent deployment introduced inefficiency                     │
│    • Database query plan regression                                │
│    • New logging/tracing overhead                                  │
│    • Downstream service slower                                     │
│    • GC pressure from memory leak                                  │
│    • Missing cache (cold cache after restart)                      │
│                                                                     │
│  Recommended Actions:                                               │
│    • IMMEDIATE: Check deployments in last 2 hours                  │
│    • CHECK: Configuration changes (feature flags, settings)        │
│    • CHECK: External dependency response times                     │
│    • CHECK: Database query performance                             │
│    • CHECK: GC behavior and memory pressure                        │
│                                                                     │
│  Example Scenario:                                                  │
│    Traffic: 150 req/s (normal)                                     │
│    Latency: 450ms (normally 150ms)                                 │
│    Errors: 0.1% (normal)                                           │
│                                                                     │
│    Timeline shows latency jumped at 10:30 AM.                      │
│    Deployment log shows release v2.3.4 deployed at 10:28 AM.       │
│    New release added detailed audit logging that's synchronous.    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### `internal_latency_issue`

```
┌─────────────────────────────────────────────────────────────────────┐
│  PATTERN: internal_latency_issue                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Conditions:                                                        │
│    application_latency: high                                       │
│    dependency_latency: normal                                          │
│    database_latency: normal                                        │
│                                                                     │
│  Severity: high                                                     │
│                                                                     │
│  What It Means:                                                     │
│    The service itself is slow, but all its dependencies            │
│    (database, external services) are fast. The problem is          │
│    INTERNAL to the application code.                               │
│                                                                     │
│  Why This Is Significant:                                           │
│    Eliminates external causes. The fix must be in this service,    │
│    not upstream/downstream.                                        │
│                                                                     │
│  Common Causes:                                                     │
│    • CPU-intensive computation                                     │
│    • Memory allocation/GC pressure                                 │
│    • Lock contention / thread starvation                           │
│    • Inefficient algorithm (O(n²) loop)                           │
│    • Synchronous I/O blocking event loop                           │
│    • Missing async/await causing serialization                     │
│                                                                     │
│  Recommended Actions:                                               │
│    • PROFILE: CPU profiling to find hot paths                      │
│    • CHECK: Thread dumps for lock contention                       │
│    • CHECK: Memory usage and GC logs                               │
│    • REVIEW: Recent code changes for algorithmic issues            │
│                                                                     │
│  Example Scenario:                                                  │
│    App latency: 800ms                                              │
│    Client latency: 50ms (fast external calls)                      │
│    DB latency: 10ms (fast queries)                                 │
│                                                                     │
│    Where's the extra 740ms? Profiling reveals a nested loop        │
│    iterating over large collections - O(n²) complexity.            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### `database_bottleneck`

```
┌─────────────────────────────────────────────────────────────────────┐
│  PATTERN: database_bottleneck                                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Conditions:                                                        │
│    database_latency: high                                          │
│    application_latency: high                                       │
│    db_latency_ratio: > 0.7 (DB is >70% of total latency)          │
│                                                                     │
│  Severity: high                                                     │
│                                                                     │
│  What It Means:                                                     │
│    Database operations dominate response time. The database        │
│    is the bottleneck - fixing app code won't help much.            │
│                                                                     │
│  Why The Ratio Matters:                                             │
│    If DB is 15ms out of 500ms total, DB isn't the problem.         │
│    If DB is 400ms out of 500ms total, DB IS the problem.           │
│                                                                     │
│  Common Causes:                                                     │
│    • Missing database index                                        │
│    • Query plan regression (statistics stale)                      │
│    • Lock contention in database                                   │
│    • Database server under-provisioned                             │
│    • N+1 query pattern                                             │
│    • Full table scans                                              │
│                                                                     │
│  Recommended Actions:                                               │
│    • INVESTIGATE: Slow query logs                                  │
│    • CHECK: Missing indexes on frequently queried columns          │
│    • CHECK: Database CPU and I/O metrics                           │
│    • ANALYZE: Query execution plans                                │
│    • CONSIDER: Query caching or read replicas                      │
│                                                                     │
│  Example Scenario:                                                  │
│    App latency: 650ms                                              │
│    DB latency: 520ms (80% of total)                                │
│    Client latency: 10ms                                            │
│                                                                     │
│    Slow query log shows: SELECT * FROM orders WHERE                │
│    customer_id = ? taking 500ms. Missing index on customer_id.     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### `downstream_cascade`

```
┌─────────────────────────────────────────────────────────────────────┐
│  PATTERN: downstream_cascade                                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Conditions:                                                        │
│    dependency_latency: high                                            │
│    application_latency: high                                       │
│    dependency_latency_ratio: > 0.6 (external calls >60% of total)      │
│                                                                     │
│  Severity: high                                                     │
│                                                                     │
│  What It Means:                                                     │
│    External dependency calls are dominating response time.         │
│    The service you're looking at is slow BECAUSE something         │
│    it depends on is slow.                                          │
│                                                                     │
│  Why This Pattern Exists:                                           │
│    Helps you avoid blaming the wrong service. If booking           │
│    is slow because search is slow, fix search, not booking.        │
│                                                                     │
│  What To Look At:                                                   │
│    • service_graph_context shows downstream call breakdown         │
│    • Identify which downstream service is slowest                  │
│    • That service is likely the root cause                         │
│                                                                     │
│  Recommended Actions:                                               │
│    • FOCUS: Investigate the slow downstream service                │
│    • CHECK: service_graph_context for detailed breakdown           │
│    • CORRELATE: Does downstream service also have anomalies?       │
│    • CONSIDER: Circuit breaker to fail fast                        │
│                                                                     │
│  Example Scenario:                                                  │
│    App latency: 1200ms                                             │
│    Client latency: 950ms (79% of total)                            │
│    DB latency: 15ms                                                │
│                                                                     │
│    service_graph_context shows:                                    │
│      search-api: 850ms avg (highest)                               │
│      payment-api: 50ms avg                                         │
│      inventory-api: 40ms avg                                       │
│                                                                     │
│    Root cause: search-api is slow, causing booking to be slow.     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Fast-Fail Patterns

These patterns indicate requests failing before normal processing.

#### `fast_rejection`

```
┌─────────────────────────────────────────────────────────────────────┐
│  PATTERN: fast_rejection                                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Conditions:                                                        │
│    application_latency: very_low                                   │
│    error_rate: very_high                                           │
│                                                                     │
│  Severity: critical                                                 │
│                                                                     │
│  What It Means:                                                     │
│    Requests are failing VERY fast. They're being rejected          │
│    before reaching normal application logic. Something is          │
│    blocking requests at the entry point.                           │
│                                                                     │
│  Why Low Latency + High Errors Is Bad:                              │
│    Normal errors take time (processing, then failure).             │
│    Fast errors mean immediate rejection - no processing at all.    │
│                                                                     │
│  Common Causes:                                                     │
│    • Circuit breaker OPEN (rejecting all requests)                 │
│    • Rate limiter active (over quota)                              │
│    • Authentication/authorization failure                          │
│    • Invalid API key or expired token                              │
│    • TLS certificate mismatch                                      │
│    • Connection pool exhausted (immediate rejection)               │
│                                                                     │
│  Recommended Actions:                                               │
│    • IMMEDIATE: Check circuit breaker status                       │
│    • CHECK: Rate limiter configuration and current rate            │
│    • CHECK: Authentication service health                          │
│    • VERIFY: API keys and certificates are valid                   │
│                                                                     │
│  Example Scenario:                                                  │
│    Latency: 5ms (normally 150ms)                                   │
│    Errors: 95%                                                     │
│                                                                     │
│    5ms is just enough time to check auth and reject.               │
│    Auth service is down, causing all requests to fail at           │
│    the middleware layer before reaching business logic.            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### `fast_failure`

```
┌─────────────────────────────────────────────────────────────────────┐
│  PATTERN: fast_failure                                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Conditions:                                                        │
│    application_latency: low                                        │
│    error_rate: high                                                │
│                                                                     │
│  Severity: critical                                                 │
│                                                                     │
│  What It Means:                                                     │
│    Requests are failing quickly - faster than normal processing    │
│    would take. Similar to fast_rejection but with slightly         │
│    more processing time (maybe some validation runs).              │
│                                                                     │
│  The Latency-Error Relationship:                                    │
│    Normal request: Validate → Process → DB → External → Return     │
│    Fast failure: Validate → Fail (skip processing)                 │
│                                                                     │
│  Common Causes:                                                     │
│    • Early validation failure                                      │
│    • Feature flag disabled critical path                           │
│    • Missing required configuration                                │
│    • Database connection failure (fail after connect attempt)      │
│                                                                     │
│  Recommended Actions:                                               │
│    • CHECK: Application logs for error patterns                    │
│    • CHECK: Required configuration and feature flags               │
│    • CHECK: Database and cache connectivity                        │
│    • REVIEW: Recent deployments for breaking changes               │
│                                                                     │
│  Example Scenario:                                                  │
│    Latency: 25ms (normally 200ms)                                  │
│    Errors: 80%                                                     │
│                                                                     │
│    Logs show: "Redis connection failed"                            │
│    Redis cluster is unreachable. Service tries to connect          │
│    (25ms timeout), fails, returns error.                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Cascade Patterns

These patterns relate to dependency failures propagating.

#### `upstream_cascade`

```
┌─────────────────────────────────────────────────────────────────────┐
│  PATTERN: upstream_cascade                                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Conditions:                                                        │
│    application_latency: high                                       │
│    _dependency_context: upstream_anomaly                           │
│                                                                     │
│  Severity: high                                                     │
│                                                                     │
│  What It Means:                                                     │
│    This service is slow, AND at least one of its upstream          │
│    dependencies also has an anomaly. The root cause is likely      │
│    in the upstream service, not here.                              │
│                                                                     │
│  How Cascade Detection Works:                                       │
│    1. Yaga2 runs detection for all services (Pass 1)               │
│    2. For services with latency anomalies, check dependencies      │
│    3. If a dependency has an anomaly, it's a cascade               │
│    4. Trace the chain to find the root cause service               │
│                                                                     │
│  Why This Matters:                                                  │
│    Without cascade detection, you'd get 5 alerts for 5 services    │
│    when only 1 service (the root cause) needs fixing.              │
│                                                                     │
│  Example Dependency Chain:                                          │
│    mobile-api → booking → vms → titan                              │
│                                                                     │
│    If titan has a database issue, all 4 services will be slow.     │
│    Cascade detection identifies titan as root cause.               │
│                                                                     │
│  cascade_analysis Output:                                           │
│    {                                                                │
│      "is_cascade": true,                                           │
│      "root_cause_service": "titan",                                │
│      "affected_chain": ["titan", "vms", "booking", "mobile-api"],  │
│      "cascade_type": "upstream_cascade"                            │
│    }                                                                │
│                                                                     │
│  Recommended Actions:                                               │
│    • FOCUS: Investigate root_cause_service first                   │
│    • IGNORE: Don't fix downstream services (they'll recover)       │
│    • CORRELATE: Check root cause service's anomaly for details     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Pattern Matching in Action

### Complete Example: Traffic Surge

```
Input Metrics at 2:30 PM:
─────────────────────────────────────────────────────────────
  request_rate: 850 req/s (baseline: 200 req/s)
  application_latency: 420ms (baseline: 120ms)
  error_rate: 0.5% (baseline: 0.1%)
  database_latency: 15ms (baseline: 12ms)
  dependency_latency: 45ms (baseline: 40ms)

Phase 1: Isolation Forest
─────────────────────────────────────────────────────────────
  request_rate:        score=-0.42  percentile=96  → very_high
  application_latency: score=-0.35  percentile=94  → high
  error_rate:          score=-0.08  percentile=72  → normal
  database_latency:    score=0.05   percentile=55  → normal
  dependency_latency:      score=0.02   percentile=52  → normal

Phase 2: Level Classification
─────────────────────────────────────────────────────────────
  request_rate: very_high (>95th percentile)
  application_latency: high (90-95th percentile)
  error_rate: normal (errors not anomalous)
  database_latency: normal (not flagged by IF)
  dependency_latency: normal (not flagged by IF)

Phase 3: Pattern Matching
─────────────────────────────────────────────────────────────
  Checking: traffic_surge_failing
    Conditions: request_rate=high ✓, latency=high ✓, errors=high ✗
    Result: NO MATCH (errors are normal)

  Checking: traffic_surge_degrading
    Conditions: request_rate=high ✓, latency=high ✓, errors=normal ✓
    Result: MATCH ✓

Output:
─────────────────────────────────────────────────────────────
{
  "pattern_name": "traffic_surge_degrading",
  "severity": "high",
  "description": "Traffic surge causing slowdown: 850 req/s driving
    latency to 420ms (errors stable at 0.50%)",
  "interpretation": "Service is slowing under load but not failing -
    approaching capacity. Users experiencing degraded performance but
    requests are completing.",
  "recommended_actions": [
    "IMMEDIATE: Scale horizontally if possible",
    "CHECK: Resource bottlenecks (CPU, memory, connections)",
    "MONITOR: Error rate for signs of impending failure",
    "CONSIDER: Enable request throttling to protect backend"
  ],
  "metric_levels": {
    "request_rate": "very_high",
    "application_latency": "high",
    "error_rate": "normal"
  }
}
```

## Adding New Patterns

To add a new pattern, edit `smartbox_anomaly/detection/interpretations.py`:

```python
MULTIVARIATE_PATTERNS["cache_miss_storm"] = PatternDefinition(
    name="cache_miss_storm",
    conditions={
        "request_rate": "normal",
        "database_latency": "high",
        "db_latency_ratio": "> 0.8",
    },
    severity="high",
    message_template=(
        "Cache miss storm: {database_latency:.0f}ms DB latency "
        "({db_latency_ratio:.0%} of total response time)"
    ),
    interpretation=(
        "Database is handling requests that should be cached. "
        "Cache service may be down or TTL expired for hot keys."
    ),
    recommended_actions=[
        "CHECK: Cache service health (Redis/Memcached)",
        "CHECK: Cache hit rate metrics",
        "VERIFY: Cache TTL hasn't expired for hot keys",
        "CONSIDER: Implementing cache warming",
    ],
)
```

### Pattern Conditions

| Condition Type | Example | Description |
|----------------|---------|-------------|
| Level | `"request_rate": "high"` | Metric must be at this level |
| Ratio | `"db_latency_ratio": "> 0.7"` | Ratio must exceed threshold |
| Any | `"error_rate": "any"` | Matches any level |
| Dependency | `"_dependency_context": "upstream_anomaly"` | Cascade detection |

### Recommendation Prefixes

Use these prefixes for consistent action ordering:

| Prefix | Priority | Use For |
|--------|----------|---------|
| `IMMEDIATE` | 1 | Actions needed right now |
| `CHECK` | 2 | Things to investigate |
| `INVESTIGATE` | 3 | Deeper analysis needed |
| `MONITOR` | 4 | Ongoing observation |
| `CONSIDER` | 5 | Optional improvements |

## Further Reading

- [Isolation Forest](./isolation-forest.md) - How ML detection works
- [Detection Pipeline](./pipeline.md) - End-to-end detection flow
- [API Payload Reference](../reference/api-payload.md) - Output format
