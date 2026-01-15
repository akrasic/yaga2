"""
Semantic anomaly interpretations and pattern definitions.

This module provides human-readable interpretations for anomaly patterns,
enabling operators to quickly understand what's happening and take action.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from smartbox_anomaly.core.constants import MetricName

# =============================================================================
# Data Classes for Interpretations
# =============================================================================


@dataclass
class MetricInterpretation:
    """Interpretation template for a metric anomaly."""

    message_template: str
    possible_causes: list[str]
    checks: list[str]
    severity_thresholds: dict[str, float] = field(default_factory=dict)


@dataclass
class PatternDefinition:
    """Definition of a multivariate anomaly pattern.

    Attributes:
        name: Pattern identifier (e.g., 'request_rate_surge_healthy')
        conditions: Dict mapping metric names to required levels
        message_template: Format string for description
        severity: Default severity level
        interpretation: What this pattern means operationally
        recommended_actions: List of suggested actions
        severity_locked: If True, IF score cannot override severity.
            Use for "healthy" patterns that should never escalate.
        priority: Pattern matching priority (higher = checked first).
            Default 0. Use higher values for more specific patterns.
    """

    name: str
    conditions: dict[str, str]
    message_template: str
    severity: str
    interpretation: str
    recommended_actions: list[str]
    severity_locked: bool = False
    priority: int = 0


@dataclass
class SeverityContext:
    """Contextual severity information."""

    base_severity: str
    adjusted_severity: str
    adjustment_reasons: list[str]
    confidence: float


# =============================================================================
# Metric Interpretations - Direction-Aware Messages
# =============================================================================


METRIC_INTERPRETATIONS: Final[dict[str, dict[str, MetricInterpretation]]] = {
    MetricName.REQUEST_RATE: {
        "high": MetricInterpretation(
            message_template=(
                "Traffic spike: {value:.1f} req/s "
                "({percentile:.0f}th percentile, {deviation:.1f}σ above normal)"  # noqa: RUF001
            ),
            possible_causes=[
                "Marketing campaign or promotional event",
                "Viral content driving organic traffic",
                "Bot or crawler activity",
                "Upstream service retry storm",
                "DDoS or traffic amplification attack",
            ],
            checks=[
                "Check referrer patterns and traffic sources",
                "Analyze geographic distribution of requests",
                "Review user-agent diversity",
                "Check for unusual URL patterns",
            ],
            severity_thresholds={"critical": 3.0, "high": 2.0, "medium": 1.5},
        ),
        "low": MetricInterpretation(
            message_template=(
                "Traffic drop: {value:.1f} req/s "
                "({percentile:.0f}th percentile, {deviation:.1f}σ below normal)"  # noqa: RUF001
            ),
            possible_causes=[
                "Upstream service outage blocking traffic",
                "DNS resolution issues",
                "Load balancer misconfiguration",
                "Recent deployment blocking requests",
                "Network routing issues",
            ],
            checks=[
                "Verify upstream service health",
                "Check DNS resolution and propagation",
                "Review load balancer logs and health checks",
                "Check for recent infrastructure changes",
            ],
            severity_thresholds={"critical": 0.3, "high": 0.5},
        ),
    },
    MetricName.APPLICATION_LATENCY: {
        "high": MetricInterpretation(
            message_template=(
                "Latency degradation: {value:.0f}ms "
                "({percentile:.0f}th percentile, normally {mean:.0f}ms)"
            ),
            possible_causes=[
                "Resource exhaustion (CPU, memory, threads)",
                "Garbage collection pressure",
                "Lock contention or thread pool saturation",
                "Downstream service slowness",
                "Database query performance degradation",
            ],
            checks=[
                "Check CPU and memory utilization",
                "Review GC logs for long pauses",
                "Monitor database query times",
                "Check external API response times",
                "Review thread pool metrics",
            ],
        ),
        "low": MetricInterpretation(
            message_template=(
                "Unusually fast responses: {value:.0f}ms "
                "(normally {mean:.0f}ms, {deviation:.1f}σ below)"  # noqa: RUF001
            ),
            possible_causes=[
                "Cache hit rate increase",
                "Early termination or error responses",
                "Load balancer short-circuiting requests",
                "Feature flag disabling heavy processing",
            ],
            checks=[
                "Check cache hit rates",
                "Review response codes distribution",
                "Verify request processing completeness",
            ],
        ),
    },
    MetricName.ERROR_RATE: {
        "high": MetricInterpretation(
            message_template=(
                "Error rate elevated: {value:.2%} "
                "(normally {mean:.2%}, threshold {p95:.2%})"
            ),
            possible_causes=[
                "Deployment regression or bug",
                "Downstream service failure",
                "Capacity exhaustion",
                "Data corruption or validation failures",
                "Authentication/authorization issues",
            ],
            checks=[
                "Check error logs for exception stack traces",
                "Identify affected endpoints",
                "Review recent deployments (last 30 min)",
                "Verify dependent service health",
                "Check for data integrity issues",
            ],
            severity_thresholds={"critical": 0.10, "high": 0.05, "medium": 0.02},
        ),
        "low": MetricInterpretation(
            message_template=(
                "Error rate dropped: {value:.2%} "
                "(normally {mean:.2%}) - verify this is genuine recovery"
            ),
            possible_causes=[
                "Recovery from previous incident (expected)",
                "Error logging or reporting may be broken",
                "Errors being silently swallowed in code",
                "Traffic pattern changed (fewer error-prone requests)",
                "Circuit breaker preventing errors from reaching this service",
            ],
            checks=[
                "Verify error logging pipeline is functioning",
                "Check if this follows a recent incident (expected recovery)",
                "Review application logs for silent failures or swallowed exceptions",
                "Confirm monitoring and alerting is working",
            ],
        ),
    },
    MetricName.DEPENDENCY_LATENCY: {
        "high": MetricInterpretation(
            message_template=(
                "External dependency slow: {value:.0f}ms (p90: {p90:.0f}ms)"
            ),
            possible_causes=[
                "Third-party API degradation",
                "Network latency to external services",
                "DNS resolution slowness",
                "TLS handshake issues",
                "External rate limiting",
            ],
            checks=[
                "Check specific external endpoint latencies",
                "Review third-party status pages",
                "Monitor network path metrics",
                "Check DNS resolution times",
            ],
        ),
        "activated": MetricInterpretation(
            message_template=(
                "External calls detected: {value:.0f}ms "
                "(service normally makes no external calls)"
            ),
            possible_causes=[
                "New code path activated by feature flag",
                "Fallback logic triggered",
                "Configuration change enabling external calls",
                "Cache miss forcing external fetch",
            ],
            checks=[
                "Review recent deployments",
                "Check feature flag states",
                "Monitor fallback trigger rates",
                "Review cache hit/miss ratios",
            ],
        ),
    },
    MetricName.DATABASE_LATENCY: {
        "high": MetricInterpretation(
            message_template=(
                "Database response degraded: {value:.0f}ms (p90: {p90:.0f}ms)"
            ),
            possible_causes=[
                "Missing or inefficient index",
                "Table lock contention",
                "Connection pool exhaustion",
                "Replication lag",
                "Query plan regression",
                "Database resource saturation",
            ],
            checks=[
                "Review slow query logs",
                "Check connection pool utilization",
                "Monitor replication status",
                "Analyze query execution plans",
                "Check database CPU/memory/disk",
            ],
        ),
        "activated": MetricInterpretation(
            message_template=(
                "Database calls detected: {value:.0f}ms "
                "(service normally has no database access)"
            ),
            possible_causes=[
                "New feature requiring database access",
                "Cache bypass triggered",
                "Fallback to database from failed cache",
                "Configuration change",
            ],
            checks=[
                "Review recent deployments",
                "Check cache service health",
                "Monitor cache hit rates",
            ],
        ),
    },
}


# =============================================================================
# Multivariate Pattern Definitions
# =============================================================================


MULTIVARIATE_PATTERNS: Final[dict[str, PatternDefinition]] = {
    "request_rate_surge_healthy": PatternDefinition(
        name="request_rate_surge_healthy",
        conditions={
            "request_rate": "high",
            "application_latency": "normal",
            "error_rate": "normal",
        },
        message_template=(
            "Traffic surge absorbed successfully: {request_rate:.1f} req/s "
            "with stable latency ({application_latency:.0f}ms) and errors ({error_rate:.2%})"
        ),
        severity="low",
        interpretation=(
            "System is handling increased load well - no immediate action needed, "
            "but monitor for sustained load approaching capacity limits"
        ),
        recommended_actions=[
            "MONITOR: Watch for sustained load approaching capacity",
            "CONSIDER: Proactive scaling if traffic continues trending up",
            "VERIFY: Confirm traffic is legitimate (not bot/attack)",
        ],
        severity_locked=True,  # Healthy state - IF score should not escalate severity
    ),
    "request_rate_surge_degrading": PatternDefinition(
        name="request_rate_surge_degrading",
        conditions={
            "request_rate": "high",
            "application_latency": "high",
            "error_rate": "normal",
        },
        message_template=(
            "Traffic surge causing slowdown: {request_rate:.1f} req/s "
            "driving latency to {application_latency:.0f}ms (errors stable at {error_rate:.2%})"
        ),
        severity="high",
        interpretation=(
            "Service is slowing under load but not failing - approaching capacity. "
            "Users experiencing degraded performance but requests are completing."
        ),
        recommended_actions=[
            "IMMEDIATE: Scale horizontally if possible",
            "CHECK: Resource bottlenecks (CPU, memory, connections)",
            "MONITOR: Error rate for signs of impending failure",
            "CONSIDER: Enable request throttling to protect backend",
        ],
    ),
    "request_rate_surge_failing": PatternDefinition(
        name="request_rate_surge_failing",
        conditions={
            "request_rate": "high",
            "application_latency": "high",
            "error_rate": "high",
        },
        message_template=(
            "Traffic surge overwhelming service: {request_rate:.1f} req/s "
            "with {application_latency:.0f}ms latency and {error_rate:.2%} errors"
        ),
        severity="critical",
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
    "latency_spike_recent": PatternDefinition(
        name="latency_spike_recent",
        conditions={
            "request_rate": "normal",
            "application_latency": "high",
            "error_rate": "normal",
            # NOTE: "latency_change": "recent_increase" was removed - not implemented
            # This pattern now matches any high latency with normal traffic/errors
            # Consider implementing trend detection in the future to distinguish
            # recent spikes from sustained high latency (see KNOWN_ISSUES.md)
        },
        message_template=(
            "Latency spike detected: {application_latency:.0f}ms "
            "at normal traffic ({request_rate:.1f} req/s), errors stable"
        ),
        severity="high",
        interpretation=(
            "Latency recently increased without traffic change - something changed. "
            "Not a capacity issue; likely recent deployment, config change, or dependency degradation. "
            "Focus investigation on what changed in the last 1-2 hours."
        ),
        recommended_actions=[
            "IMMEDIATE: Check deployments in last 2 hours",
            "CHECK: Configuration changes (feature flags, settings)",
            "CHECK: External dependency response times",
            "CHECK: Database query performance",
            "CHECK: GC behavior and memory pressure",
        ],
    ),
    "error_rate_elevated": PatternDefinition(
        name="error_rate_elevated",
        conditions={
            "request_rate": "normal",
            "application_latency": "normal",
            "error_rate": "high",
        },
        message_template=(
            "Elevated error rate: {error_rate:.2%} errors "
            "at normal traffic ({request_rate:.1f} req/s) and latency ({application_latency:.0f}ms)"
        ),
        severity="high",
        interpretation=(
            "Error rate above typical baseline while other metrics remain stable. "
            "This could indicate validation failures, specific endpoint issues, "
            "rate limiting, or dependency problems affecting a subset of requests."
        ),
        recommended_actions=[
            "CHECK: Error logs for specific exception types and affected endpoints",
            "IDENTIFY: Are errors concentrated on specific operations or spread across service?",
            "CHECK: Dependent service health and response codes",
            "REVIEW: Recent deployments or configuration changes",
            "VERIFY: Is this a temporary spike or sustained elevation?",
        ],
    ),
    "error_rate_critical": PatternDefinition(
        name="error_rate_critical",
        conditions={
            "request_rate": "normal",
            "application_latency": "normal",
            "error_rate": "very_high",
        },
        message_template=(
            "Critical error rate: {error_rate:.2%} errors "
            "at normal traffic ({request_rate:.1f} req/s) and latency ({application_latency:.0f}ms)"
        ),
        severity="critical",
        interpretation=(
            "Significant portion of requests failing while service is otherwise responsive. "
            "Likely a specific code path, feature, or dependency is broken. "
            "Not a capacity issue - targeted investigation needed."
        ),
        recommended_actions=[
            "IMMEDIATE: Check error logs for specific exception types",
            "IDENTIFY: Which endpoints or operations are failing",
            "CHECK: Dependent service health for partial failures",
            "REVIEW: Recent deployments affecting specific features",
        ],
    ),
    "fast_failure_mode": PatternDefinition(
        name="fast_failure_mode",
        conditions={
            "request_rate": "any",
            "application_latency": "very_low",
            "error_rate": "high",
        },
        message_template=(
            "Fast-fail mode: {error_rate:.2%} errors with unusually low latency "
            "({application_latency:.0f}ms) - requests rejected without processing"
        ),
        severity="critical",
        interpretation=(
            "Service failing quickly without full processing - likely upstream rejection, "
            "circuit breaker activation, or authentication/authorization failures."
        ),
        recommended_actions=[
            "CHECK: Circuit breaker states in dashboard",
            "CHECK: Upstream service health",
            "CHECK: Authentication/authorization service",
            "REVIEW: Response status codes (look for 401, 403, 429, 503)",
        ],
    ),
    "dependency_latency_cascade": PatternDefinition(
        name="dependency_latency_cascade",
        conditions={
            "dependency_latency": "high",
            "application_latency": "high",
            "dependency_latency_ratio": "> 0.6",
        },
        message_template=(
            "Downstream cascade: external calls ({dependency_latency:.0f}ms) "
            "causing {dependency_latency_ratio:.0%} of total latency ({application_latency:.0f}ms)"
        ),
        severity="high",
        interpretation=(
            "External dependency is the primary bottleneck - your service is waiting on others. "
            "Local optimization won't help until dependency is addressed."
        ),
        recommended_actions=[
            "CHECK: Third-party service status pages",
            "CONSIDER: Circuit breaker to fail fast if dependency is unhealthy",
            "CONSIDER: Fallback or degraded mode if available",
            "REVIEW: Timeout settings for external calls",
        ],
    ),
    "database_latency_degraded": PatternDefinition(
        name="database_latency_degraded",
        conditions={
            "database_latency": "high",
            "application_latency": "normal",
            "error_rate": "normal",
        },
        message_template=(
            "Database degradation: DB queries slow ({database_latency:.0f}ms) "
            "but application still responding normally ({application_latency:.0f}ms)"
        ),
        severity="medium",
        interpretation=(
            "Database is responding slowly but the application is compensating - "
            "possibly through caching, connection pooling, or the slow queries are on non-critical paths. "
            "This is an early warning; if database latency increases further, application impact is likely."
        ),
        recommended_actions=[
            "INVESTIGATE: Slow query logs to identify problematic queries",
            "CHECK: Database server resource utilization (CPU, memory, I/O)",
            "CHECK: Connection pool health and wait times",
            "CHECK: Replication lag if using read replicas",
            "MONITOR: Watch for application latency increase as leading indicator",
            "CONSIDER: Proactive query optimization before user impact",
        ],
    ),
    "database_latency_bottleneck": PatternDefinition(
        name="database_latency_bottleneck",
        conditions={
            "database_latency": "high",
            "application_latency": "high",
            "db_latency_ratio": "> 0.7",  # 70% threshold - avoids false positives for DB-heavy services
        },
        message_template=(
            "Database bottleneck: DB queries ({database_latency:.0f}ms) "
            "causing {db_latency_ratio:.0%} of total latency ({application_latency:.0f}ms)"
        ),
        severity="high",
        interpretation=(
            "Database is the dominant constraint on performance - {db_latency_ratio:.0%} of request time "
            "is spent waiting for database responses. This is significantly higher than typical."
        ),
        recommended_actions=[
            "CHECK: Slow query logs for problematic queries (>100ms)",
            "CHECK: Database connection pool utilization and wait times",
            "CHECK: Database server resource utilization (CPU, I/O)",
            "ANALYZE: Query execution plans for N+1 queries or missing indexes",
            "CONSIDER: Query result caching or read replicas if appropriate",
        ],
    ),
    "request_rate_cliff": PatternDefinition(
        name="request_rate_cliff",
        conditions={
            "request_rate": "very_low",
            "request_rate_change": "sudden_drop",  # Distinguishes from normal low-traffic periods
        },
        message_template=(
            "Traffic significantly below expected: {request_rate:.1f} req/s "
            "(expected ~{expected_rate:.1f} req/s for this time, {drop_percent:.0f}% below)"
        ),
        severity="critical",
        interpretation=(
            "Traffic significantly lower than expected for this time period. "
            "If this is outside normal low-traffic hours, investigate immediately. "
            "Could indicate upstream issues, routing problems, DNS failure, or clients unable to connect."
        ),
        recommended_actions=[
            "VERIFY: Is this expected low-traffic period? (Check time of day/week)",
            "CHECK: Load balancer health and routing",
            "CHECK: DNS resolution and propagation",
            "CHECK: Upstream services sending traffic",
            "VERIFY: Service is actually reachable from outside",
            "CHECK: Recent infrastructure or network changes",
        ],
    ),
    "application_latency_bottleneck": PatternDefinition(
        name="application_latency_bottleneck",
        conditions={
            "request_rate": "normal",
            "application_latency": "high",
            "error_rate": "normal",
            "dependency_latency": "normal",
            "database_latency": "normal",
        },
        message_template=(
            "Internal bottleneck: high latency ({application_latency:.0f}ms) "
            "with normal external dependencies - application processing is slow"
        ),
        severity="high",
        interpretation=(
            "High latency not caused by external dependencies or database. "
            "The application itself is the bottleneck. Likely causes: "
            "CPU saturation, memory pressure, GC pauses, thread pool exhaustion, or lock contention."
        ),
        recommended_actions=[
            "CHECK: CPU utilization and throttling (Kubernetes limits?)",
            "CHECK: Memory pressure and GC activity",
            "CHECK: Thread pool saturation and queue depths",
            "PROFILE: Enable request tracing to identify slow code paths",
            "CHECK: Recent deployments that might have introduced inefficiency",
        ],
    ),
}


# =============================================================================
# Fast-Fail Pattern Definitions (Specialized)
# =============================================================================


FAST_FAIL_PATTERNS: Final[dict[str, PatternDefinition]] = {
    "error_rate_fast_rejection": PatternDefinition(
        name="error_rate_fast_rejection",
        conditions={
            "application_latency": "very_low",
            "error_rate": "very_high",
        },
        message_template=(
            "Requests being rejected rapidly: {error_rate:.1%} errors "
            "at {application_latency:.0f}ms (failing before full processing)"
        ),
        severity="critical",
        interpretation=(
            "Requests failing very quickly without full processing. "
            "Could be circuit breaker, rate limiting, auth failure, or application errors. "
            "Check HTTP status codes to determine specific cause."
        ),
        recommended_actions=[
            "IMMEDIATE: Check HTTP status codes (429=rate limit, 401/403=auth, 503=circuit breaker, 500=app error)",
            "CHECK: Circuit breaker dashboard",
            "CHECK: Rate limiter metrics",
            "CHECK: Auth service health",
            "CHECK: Recent deployments for regression",
        ],
    ),
    "reduced_traffic_with_errors": PatternDefinition(
        name="reduced_traffic_with_errors",
        conditions={
            "request_rate": "low",
            "application_latency": "very_low",
            "error_rate": "high",
        },
        message_template=(
            "Reduced traffic with fast failures: {error_rate:.1%} errors, "
            "{application_latency:.0f}ms latency, only {request_rate:.1f} req/s reaching service"
        ),
        severity="critical",
        interpretation=(
            "Traffic below expected AND requests failing quickly. "
            "Requests may be rejected before reaching this service, "
            "or clients may be unable to connect. Multiple causes possible."
        ),
        recommended_actions=[
            "CHECK: Load balancer health and error logs",
            "CHECK: DNS resolution from client perspective",
            "CHECK: Ingress controller and routing",
            "VERIFY: Service is reachable from outside",
            "CHECK: Client-side error rates if available",
        ],
    ),
    "error_rate_partial_rejection": PatternDefinition(
        name="error_rate_partial_rejection",
        conditions={
            "application_latency": "low",
            "error_rate": "moderate",
        },
        message_template=(
            "Partial rejection: {error_rate:.1%} errors failing quickly "
            "({application_latency:.0f}ms) - specific operations being rejected"
        ),
        severity="high",
        interpretation=(
            "Some requests failing before full processing. "
            "Likely validation errors, authentication failures, or specific code path issues. "
            "Check error response details to identify affected operations."
        ),
        recommended_actions=[
            "CHECK: Error logs for specific error types and affected endpoints",
            "IDENTIFY: Which operations or endpoints are failing",
            "CHECK: Authentication/authorization errors (401, 403)",
            "REVIEW: Input validation failures (400)",
            "CHECK: Recent deployments affecting specific features",
        ],
    ),
    "error_rate_fast_failure": PatternDefinition(
        name="error_rate_fast_failure",
        conditions={
            "application_latency": "low",
            "error_rate": "high",
        },
        message_template=(
            "Fast failure mode: {error_rate:.1%} errors "
            "with {application_latency:.0f}ms latency"
        ),
        severity="critical",
        interpretation=(
            "Requests failing quickly without full processing. "
            "Investigate error logs and HTTP status codes to determine root cause. "
            "Could indicate dependency issues, configuration problems, or code errors."
        ),
        recommended_actions=[
            "IMMEDIATE: Check error logs for exception types and stack traces",
            "CHECK: HTTP response status codes distribution",
            "CHECK: Downstream dependency health",
            "CHECK: Recent deployments or configuration changes",
            "CHECK: Circuit breaker and connection pool states",
        ],
    ),
}


# =============================================================================
# Additional Operational Patterns
# =============================================================================
#
# NOTE: These patterns are NOT ACTIVE because they are not in PATTERN_PRIORITY
# (see detector.py). They are preserved here as documentation of future
# functionality that requires trend detection and state tracking.
#
# Patterns with unimplemented conditions:
# - gradual_degradation: requires latency_trend, error_trend (trend detection)
# - recovery_in_progress: requires error_trend, previous_state (state tracking)
# - flapping_service: requires state_changes (state change counting)
# - suspicious_fast_responses: HAS valid conditions, could be enabled
#
# To enable a pattern: add it to PATTERN_PRIORITY in detector.py
# =============================================================================


OPERATIONAL_PATTERNS: Final[dict[str, PatternDefinition]] = {
    # DISABLED: Requires trend detection (not implemented)
    "gradual_degradation": PatternDefinition(
        name="gradual_degradation",
        conditions={
            # UNIMPLEMENTED: These conditions require trend detection over time
            "latency_trend": "increasing",
            "error_trend": "increasing",
        },
        message_template=(
            "Gradual degradation detected: latency trending up {latency_trend_percent:.0f}%, "
            "errors trending up {error_trend_percent:.0f}% over last {trend_window_hours} hours"
        ),
        severity="medium",
        interpretation=(
            "Service slowly degrading over time. Individual metrics may be within thresholds "
            "but the trend indicates a developing problem. Often indicates resource leaks, "
            "growing datasets, or accumulating technical debt."
        ),
        recommended_actions=[
            "INVESTIGATE: Memory usage trends (possible leak)",
            "CHECK: Database table sizes and query performance",
            "CHECK: Log volume growth",
            "REVIEW: Resource utilization trends",
            "CONSIDER: Proactive scaling or optimization",
        ],
    ),
    # DISABLED: Requires state tracking (not implemented)
    "recovery_in_progress": PatternDefinition(
        name="recovery_in_progress",
        conditions={
            # UNIMPLEMENTED: These conditions require historical state tracking
            "error_trend": "decreasing",
            "previous_state": "was_degraded",
        },
        message_template=(
            "Recovery in progress: errors down from {peak_error_rate:.1%} to {current_error_rate:.1%}, "
            "latency improving from {peak_latency:.0f}ms to {current_latency:.0f}ms"
        ),
        severity="low",
        interpretation=(
            "Service appears to be recovering from a degraded state. "
            "Continue monitoring to confirm full recovery."
        ),
        recommended_actions=[
            "MONITOR: Confirm error rate continues to decrease",
            "VERIFY: All affected endpoints returning to normal",
            "DOCUMENT: Root cause if identified",
            "REVIEW: What triggered recovery (auto-healing, manual fix, traffic change)",
        ],
        severity_locked=True,  # Recovery state - IF score should not escalate severity
    ),
    # DISABLED: Requires state change counting (not implemented)
    "flapping_service": PatternDefinition(
        name="flapping_service",
        conditions={
            # UNIMPLEMENTED: Requires tracking state transitions over time
            "state_changes": "frequent",
        },
        message_template=(
            "Service flapping: {state_change_count} state changes in last {window_minutes} minutes "
            "(alternating between healthy and degraded)"
        ),
        severity="high",
        interpretation=(
            "Service unstable - alternating between healthy and unhealthy states. "
            "Often indicates resource limits being hit intermittently, unstable dependency, "
            "or threshold set too close to normal operating range."
        ),
        recommended_actions=[
            "CHECK: Resource utilization at time of each degradation",
            "CHECK: Dependency health and response time variability",
            "REVIEW: Alert thresholds may be too sensitive",
            "INVESTIGATE: What changes between healthy and unhealthy periods",
            "CONSIDER: Adding hysteresis to prevent alert noise",
        ],
    ),
    # COULD BE ENABLED: Has valid conditions, just not in PATTERN_PRIORITY
    "suspicious_fast_responses": PatternDefinition(
        name="suspicious_fast_responses",
        conditions={
            "application_latency": "very_low",
            "error_rate": "elevated",
        },
        message_template=(
            "Suspicious fast responses: {application_latency:.0f}ms latency "
            "(normally {normal_latency:.0f}ms) with {error_rate:.1%} errors"
        ),
        severity="high",
        interpretation=(
            "Unusually fast responses combined with elevated errors indicates "
            "requests are failing before full processing. Fast failures are often "
            "error responses, rejections, or short-circuited requests."
        ),
        recommended_actions=[
            "CHECK: HTTP status code distribution (fast 4xx/5xx responses)",
            "CHECK: Are requests being fully processed or rejected early?",
            "VERIFY: Response bodies contain expected data",
            "CHECK: Circuit breakers, rate limiters, auth service",
        ],
    ),
}


# =============================================================================
# Recommendation Rules
# =============================================================================


RECOMMENDATION_RULES: Final[dict[tuple[str, str], list[str]]] = {
    # Critical pattern recommendations
    ("request_rate_surge_failing", "critical"): [
        "IMMEDIATE: Enable auto-scaling or manually scale horizontally",
        "IMMEDIATE: Consider activating rate limiting to protect backend",
        "INVESTIGATE: Check if this is legitimate traffic or potential attack",
        "PREPARE: Have rollback ready if recent deployment is cause",
    ],
    ("error_rate_critical", "critical"): [
        "IMMEDIATE: Check error logs for exception stack traces",
        "CORRELATE: Identify if errors are from specific endpoints",
        "TIMELINE: Was there a recent deployment? (check last 30 min)",
        "VERIFY: Are dependent services healthy?",
    ],
    ("circuit_breaker_open", "critical"): [
        "VERIFY: Which circuit breaker is open (check dashboard)",
        "CHECK: Health of protected downstream service",
        "ASSESS: Is this protecting the system or causing user impact?",
        "DECIDE: Manual circuit breaker reset vs wait for auto-recovery",
    ],
    ("request_rate_cliff", "critical"): [
        "IMMEDIATE: Verify service is reachable externally",
        "CHECK: Load balancer and DNS health",
        "CHECK: Upstream services and routing",
        "INVESTIGATE: Recent infrastructure changes",
    ],
    # High severity recommendations
    ("database_latency_bottleneck", "high"): [
        "INVESTIGATE: Run EXPLAIN on recent slow queries",
        "CHECK: Database connection pool utilization",
        "CHECK: Replication lag if using read replicas",
        "CONSIDER: Query result caching if appropriate",
    ],
    ("dependency_latency_cascade", "high"): [
        "CHECK: Third-party service status pages",
        "CONSIDER: Circuit breaker or timeout adjustments",
        "CONSIDER: Fallback mode if available",
        "REVIEW: Retry policies and backoff settings",
    ],
    # Metric-specific recommendations
    ("error_rate_elevated", "critical"): [
        "IMMEDIATE: Check error logs for exception types",
        "CORRELATE: Identify affected endpoints",
        "TIMELINE: Check for recent deployments",
        "VERIFY: Dependent service health",
    ],
    ("error_rate_elevated", "high"): [
        "CHECK: Error logs for patterns",
        "IDENTIFY: Most frequent error types",
        "REVIEW: Recent code or config changes",
    ],
    ("latency_elevated", "high"): [
        "CHECK: CPU and memory utilization",
        "CHECK: Database and external API response times",
        "PROFILE: Enable request tracing if not active",
        "COMPARE: Latency breakdown (app vs db vs client)",
    ],
    ("traffic_surge", "high"): [
        "VERIFY: Traffic source (legitimate vs attack)",
        "MONITOR: System resource utilization",
        "CONSIDER: Scaling if sustained",
        "CHECK: Rate limiting thresholds",
    ],
    # NOTE: ("request_rate_cliff", "critical") is defined above at line 793
    # Removed duplicate entry that was here (SRE Review 2026-01-15)
}


# =============================================================================
# Dependency-Aware Pattern Definitions
# =============================================================================


DEPENDENCY_AWARE_PATTERNS: Final[dict[str, PatternDefinition]] = {
    "upstream_cascade": PatternDefinition(
        name="upstream_cascade",
        conditions={
            "application_latency": "high",
            "dependency_latency": "high",
            "_dependency_context": "upstream_anomaly",
        },
        message_template=(
            "Cascade from upstream dependency: {root_cause_service} is degraded "
            "({root_cause_anomaly_type}), affecting this service via "
            "{affected_chain_length} service(s) in chain. "
            "Current latency: {application_latency:.0f}ms"
        ),
        severity="high",
        interpretation=(
            "This service's latency is caused by an upstream dependency failure. "
            "The root cause is {root_cause_service}, not this service. "
            "Fixing {root_cause_service} should resolve this issue."
        ),
        recommended_actions=[
            "FOCUS: Investigate {root_cause_service} first - it is the root cause",
            "CHECK: {root_cause_service} status page and recent deployments",
            "MONITOR: Watch for {root_cause_service} recovery before taking action here",
            "CONSIDER: Enabling circuit breaker if not already active",
            "FALLBACK: Activate degraded mode if available for dependency calls",
        ],
    ),
    "dependency_chain_degradation": PatternDefinition(
        name="dependency_chain_degradation",
        conditions={
            "application_latency": "high",
            "_dependency_context": "chain_degraded",
        },
        message_template=(
            "Dependency chain degradation: {affected_chain_length} services affected "
            "({affected_services}). Root cause: {root_cause_service}"
        ),
        severity="high",
        interpretation=(
            "Multiple services in the dependency chain are degraded. "
            "This typically indicates a cascading failure originating from "
            "{root_cause_service}. The degradation is propagating through the chain."
        ),
        recommended_actions=[
            "IMMEDIATE: Focus on {root_cause_service} - fixing it will resolve the cascade",
            "CHECK: All affected services for circuit breaker status",
            "VERIFY: Network connectivity between services in the chain",
            "CONSIDER: Implementing bulkhead pattern to isolate failures",
            "MONITOR: Recovery should propagate back through the chain",
        ],
    ),
    "internal_latency_issue": PatternDefinition(
        name="internal_latency_issue",
        conditions={
            "application_latency": "high",
            "_dependency_context": "dependencies_healthy",
        },
        message_template=(
            "Internal latency issue: latency elevated to {application_latency:.0f}ms "
            "with all dependencies healthy. This indicates an internal service problem."
        ),
        severity="high",
        interpretation=(
            "This service is experiencing latency issues but all its dependencies "
            "are responding normally. The problem is internal to this service - "
            "likely resource exhaustion, code issues, or configuration problems."
        ),
        recommended_actions=[
            "FOCUS: Issue is internal to this service, not dependencies",
            "CHECK: CPU, memory, and GC metrics for this service",
            "CHECK: Thread pool saturation and connection pool health",
            "REVIEW: Recent deployments to this service specifically",
            "PROFILE: Enable request tracing to identify slow code paths",
        ],
    ),
}


# =============================================================================
# Pattern Name Aliases (Backward Compatibility)
# =============================================================================
# Old pattern names are mapped to new standardized names following the
# {metric}_{state}_{modifier} convention. These aliases enable a gradual
# migration without breaking existing integrations.


PATTERN_ALIASES: Final[dict[str, str]] = {
    # Traffic patterns → request_rate patterns
    "traffic_surge_healthy": "request_rate_surge_healthy",
    "traffic_surge_degrading": "request_rate_surge_degrading",
    "traffic_surge_failing": "request_rate_surge_failing",
    "traffic_cliff": "request_rate_cliff",
    # Error patterns
    "elevated_errors": "error_rate_elevated",
    "fast_rejection": "error_rate_fast_rejection",
    "fast_failure": "error_rate_fast_failure",
    "partial_rejection": "error_rate_partial_rejection",
    # Latency patterns
    "downstream_cascade": "dependency_latency_cascade",
    "internal_bottleneck": "application_latency_bottleneck",
    "database_bottleneck": "database_latency_bottleneck",
    "database_degradation": "database_latency_degraded",
}


def normalize_pattern_name(pattern_name: str) -> str:
    """Normalize a pattern name using aliases for backward compatibility.

    Args:
        pattern_name: The original pattern name (may be old or new)

    Returns:
        The normalized (new) pattern name
    """
    return PATTERN_ALIASES.get(pattern_name, pattern_name)


# =============================================================================
# Business Impact Templates
# =============================================================================


BUSINESS_IMPACT_MAP: Final[dict[tuple[str, str], str]] = {
    ("critical", "pattern"): "Immediate service degradation affecting users",
    ("critical", "multivariate"): "Multiple metrics abnormal - investigate immediately",
    ("critical", "fast_fail"): "Service rejecting requests - user impact likely",
    ("high", "correlation"): "Dependency or infrastructure issue affecting performance",
    ("high", "ml_isolation"): "Significant deviation from normal behavior",
    ("high", "pattern"): "Service degraded - user experience impacted",
    ("medium", "threshold"): "Metric exceeding expected range",
    ("medium", "correlation"): "Unusual relationship between metrics",
    ("medium", "ml_isolation"): "Moderate deviation detected",
    ("low", "pattern"): "Minor anomaly - monitor for escalation",
    ("low", "ml_isolation"): "Small deviation from baseline",
}


# =============================================================================
# Helper Functions
# =============================================================================


def get_metric_interpretation(
    metric_name: str, direction: str
) -> MetricInterpretation | None:
    """Get interpretation for a metric anomaly."""
    metric_interps = METRIC_INTERPRETATIONS.get(metric_name, {})
    return metric_interps.get(direction)


def get_pattern_definition(pattern_name: str) -> PatternDefinition | None:
    """Get definition for a multivariate pattern.

    Supports both old and new pattern names via alias normalization.
    """
    # Normalize old pattern names to new convention
    normalized_name = normalize_pattern_name(pattern_name)
    return (
        DEPENDENCY_AWARE_PATTERNS.get(normalized_name)
        or MULTIVARIATE_PATTERNS.get(normalized_name)
        or FAST_FAIL_PATTERNS.get(normalized_name)
        or OPERATIONAL_PATTERNS.get(normalized_name)
    )


def get_recommendations(
    anomaly_type: str, severity: str, max_count: int = 5
) -> list[str]:
    """Get prioritized recommendations for an anomaly."""
    recommendations = RECOMMENDATION_RULES.get((anomaly_type, severity), [])

    # Sort by priority prefix
    priority_order = [
        "IMMEDIATE",
        "VERIFY",
        "CHECK",
        "INVESTIGATE",
        "CORRELATE",
        "TIMELINE",
        "ASSESS",
        "IDENTIFY",
        "FOCUS",
        "CONSIDER",
        "PREPARE",
        "DECIDE",
        "REVIEW",
        "MONITOR",
    ]

    def priority_key(rec: str) -> int:
        for i, prefix in enumerate(priority_order):
            if rec.startswith(prefix):
                return i
        return len(priority_order)

    sorted_recs = sorted(recommendations, key=priority_key)
    return sorted_recs[:max_count]


def get_business_impact(severity: str, anomaly_type: str) -> str:
    """Get business impact description."""
    return BUSINESS_IMPACT_MAP.get(
        (severity, anomaly_type), "Anomalous behavior detected - monitor for escalation"
    )
