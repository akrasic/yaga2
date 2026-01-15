# Advanced Anomaly Detection Improvements

## Executive Summary

This proposal outlines next-generation improvements for the anomaly detection system, focusing on:
1. **Predictive Detection** - Catch issues before they impact users
2. **Root Cause Analysis** - Automatically identify why anomalies occur
3. **Cross-Service Intelligence** - Understand cascading effects across the system
4. **Operational Intelligence** - Learn from past incidents to improve response
5. **Alert Quality** - Reduce noise while increasing actionability

---

## 1. Predictive Anomaly Detection

### Current Limitation
The system detects anomalies *after* they occur. By the time an alert fires, users are already affected.

### Proposed: Forecasting-Based Early Warning

```python
@dataclass
class PredictiveAlert:
    """Alert generated before anomaly threshold is crossed."""

    metric_name: str
    current_value: float
    predicted_value: float  # Forecasted value
    prediction_horizon: int  # Minutes ahead
    probability_of_breach: float  # 0-1
    estimated_time_to_breach: int | None  # Minutes
    trend_direction: str  # "increasing", "decreasing", "stable"
    trend_acceleration: float  # Rate of change of the rate of change
    confidence_interval: tuple[float, float]  # 95% CI


class PredictiveDetector:
    """Forecast-based anomaly prediction using time series analysis."""

    def __init__(self, service_name: str, horizon_minutes: int = 15):
        self.service_name = service_name
        self.horizon = horizon_minutes
        self.models: dict[str, Any] = {}  # Prophet or ARIMA models per metric
        self.history_buffer: dict[str, deque] = {}  # Rolling window of recent values

    def update(self, timestamp: datetime, metrics: dict[str, float]) -> None:
        """Update history buffer with new observation."""
        for metric, value in metrics.items():
            if metric not in self.history_buffer:
                self.history_buffer[metric] = deque(maxlen=1440)  # 24 hours at 1-min resolution
            self.history_buffer[metric].append((timestamp, value))

    def predict(self, metric_name: str) -> PredictiveAlert | None:
        """Generate predictive alert if breach likely within horizon."""
        history = self.history_buffer.get(metric_name)
        if not history or len(history) < 60:  # Need at least 1 hour
            return None

        # Fit lightweight model (e.g., Holt-Winters or simple ARIMA)
        forecast, ci_lower, ci_upper = self._forecast(history, self.horizon)

        # Check if forecast crosses threshold
        threshold = self.thresholds.get(f"{metric_name}_p95", float("inf"))

        if forecast > threshold:
            # Calculate time to breach using linear interpolation
            current = history[-1][1]
            rate = (forecast - current) / self.horizon
            time_to_breach = int((threshold - current) / rate) if rate > 0 else None

            return PredictiveAlert(
                metric_name=metric_name,
                current_value=current,
                predicted_value=forecast,
                prediction_horizon=self.horizon,
                probability_of_breach=self._calculate_breach_probability(forecast, threshold, ci_upper),
                estimated_time_to_breach=time_to_breach,
                trend_direction=self._classify_trend(history),
                trend_acceleration=self._calculate_acceleration(history),
                confidence_interval=(ci_lower, ci_upper),
            )
        return None

    def _classify_trend(self, history: deque) -> str:
        """Classify trend direction from recent history."""
        if len(history) < 10:
            return "stable"

        recent = [v for _, v in list(history)[-10:]]
        slope = np.polyfit(range(len(recent)), recent, 1)[0]

        if slope > 0.05 * np.mean(recent):
            return "increasing"
        elif slope < -0.05 * np.mean(recent):
            return "decreasing"
        return "stable"
```

### Use Cases

| Scenario | Current Behavior | With Prediction |
|----------|-----------------|-----------------|
| Traffic ramp-up | Alert at breach | "Traffic increasing 15%/min, will hit capacity in ~8 minutes" |
| Memory leak | Alert at OOM | "Memory usage trend will exhaust heap in ~45 minutes" |
| Gradual latency degradation | Alert at threshold | "Latency trending up, will breach SLO in ~12 minutes" |

---

## 2. Automated Root Cause Analysis

### Current Limitation
System identifies *what* is anomalous but operators must determine *why*.

### Proposed: Causal Inference Engine

```python
@dataclass
class RootCauseHypothesis:
    """A hypothesis about what caused an anomaly."""

    cause_type: str  # "deployment", "dependency", "capacity", "data", "external"
    description: str
    confidence: float  # 0-1
    evidence: list[str]
    suggested_investigation: list[str]
    related_changes: list[dict]  # Recent deployments, config changes
    similar_incidents: list[str]  # Past incident IDs with similar pattern


class RootCauseAnalyzer:
    """Automated root cause analysis using causal inference."""

    CAUSE_SIGNATURES = {
        "deployment_regression": {
            "pattern": {
                "error_rate": "sudden_increase",
                "latency": "step_change",
                "request_rate": "stable",
            },
            "temporal": "coincides_with_deployment",
            "description": "Deployment likely introduced regression",
            "investigation": [
                "CHECK: Diff between current and previous deployment",
                "CHECK: Error logs for new exception types",
                "VERIFY: Feature flags that changed",
                "CONSIDER: Immediate rollback if user-impacting",
            ],
        },
        "capacity_exhaustion": {
            "pattern": {
                "request_rate": "gradual_increase",
                "latency": "gradual_increase",
                "error_rate": "late_increase",
            },
            "temporal": "follows_traffic_growth",
            "description": "Service approaching or exceeding capacity limits",
            "investigation": [
                "CHECK: CPU/memory utilization trends",
                "CHECK: Connection pool saturation",
                "CHECK: Thread pool queue depths",
                "SCALE: Add capacity if utilization > 80%",
            ],
        },
        "dependency_failure": {
            "pattern": {
                "dependency_latency": "spike_or_timeout",
                "error_rate": "sudden_increase",
                "request_rate": "stable_or_decreasing",
            },
            "temporal": "correlates_with_dependency_metrics",
            "description": "Downstream dependency is failing or degraded",
            "investigation": [
                "CHECK: Dependency service health dashboards",
                "CHECK: Network connectivity to dependency",
                "VERIFY: Circuit breaker states",
                "FALLBACK: Enable degraded mode if available",
            ],
        },
        "database_issue": {
            "pattern": {
                "database_latency": "dominant_contributor",
                "application_latency": "high",
                "error_rate": "variable",
            },
            "temporal": "correlates_with_db_metrics",
            "description": "Database is the bottleneck",
            "investigation": [
                "CHECK: Slow query logs (queries > 100ms)",
                "CHECK: Lock contention and deadlocks",
                "CHECK: Replication lag",
                "CHECK: Connection pool exhaustion",
                "ANALYZE: Query plans for recent slow queries",
            ],
        },
        "traffic_anomaly": {
            "pattern": {
                "request_rate": "sudden_spike_or_drop",
                "latency": "variable",
                "error_rate": "variable",
            },
            "temporal": "no_internal_correlation",
            "description": "Unusual traffic pattern (possible attack, bot, or upstream issue)",
            "investigation": [
                "CHECK: Traffic source distribution (IPs, user agents)",
                "CHECK: Geographic distribution of requests",
                "CHECK: Request patterns (URLs, methods)",
                "VERIFY: Upstream service health if traffic dropped",
            ],
        },
        "resource_contention": {
            "pattern": {
                "latency": "high",
                "error_rate": "low",
                "cpu_or_memory": "high",
            },
            "temporal": "gradual_onset",
            "description": "Internal resource contention (CPU, memory, threads)",
            "investigation": [
                "PROFILE: CPU flame graph",
                "CHECK: GC logs for long pauses",
                "CHECK: Thread dumps for contention",
                "CHECK: Memory allocation patterns",
            ],
        },
    }

    def analyze(
        self,
        anomalies: dict[str, Any],
        metrics: dict[str, float],
        metric_history: dict[str, list[tuple[datetime, float]]],
        recent_deployments: list[dict],
        dependency_health: dict[str, str],
    ) -> list[RootCauseHypothesis]:
        """Generate ranked hypotheses about anomaly root cause."""

        hypotheses = []

        # Analyze pattern signature
        pattern_signature = self._extract_pattern_signature(metric_history)

        # Check each cause signature
        for cause_type, signature in self.CAUSE_SIGNATURES.items():
            confidence = self._calculate_signature_match(pattern_signature, signature["pattern"])

            if confidence > 0.5:
                # Enhance confidence with temporal correlation
                if self._check_temporal_correlation(cause_type, metric_history, recent_deployments):
                    confidence = min(0.95, confidence + 0.2)

                # Build evidence list
                evidence = self._gather_evidence(cause_type, anomalies, metrics, recent_deployments)

                hypotheses.append(RootCauseHypothesis(
                    cause_type=cause_type,
                    description=signature["description"],
                    confidence=confidence,
                    evidence=evidence,
                    suggested_investigation=signature["investigation"],
                    related_changes=self._find_related_changes(cause_type, recent_deployments),
                    similar_incidents=self._find_similar_incidents(pattern_signature),
                ))

        # Rank by confidence
        return sorted(hypotheses, key=lambda h: h.confidence, reverse=True)

    def _extract_pattern_signature(
        self, metric_history: dict[str, list[tuple[datetime, float]]]
    ) -> dict[str, str]:
        """Extract the pattern signature from metric history."""
        signature = {}

        for metric, history in metric_history.items():
            if len(history) < 10:
                continue

            values = [v for _, v in history]
            recent = values[-10:]
            baseline = values[:-10] if len(values) > 20 else values[:len(values)//2]

            # Classify the pattern
            recent_mean = np.mean(recent)
            baseline_mean = np.mean(baseline)
            baseline_std = np.std(baseline)

            # Sudden vs gradual
            first_half = np.mean(recent[:5])
            second_half = np.mean(recent[5:])

            if recent_mean > baseline_mean + 3 * baseline_std:
                if abs(first_half - second_half) < baseline_std:
                    signature[metric] = "step_change"
                else:
                    signature[metric] = "sudden_increase"
            elif recent_mean > baseline_mean + 2 * baseline_std:
                signature[metric] = "gradual_increase"
            elif recent_mean < baseline_mean - 2 * baseline_std:
                signature[metric] = "sudden_decrease"
            else:
                signature[metric] = "stable"

        return signature
```

### Integration with Change Management

```python
class ChangeCorrelator:
    """Correlate anomalies with recent changes."""

    def __init__(self, change_sources: list[ChangeSource]):
        """
        change_sources: List of adapters for deployment systems, config management, etc.
        Examples: GitHubAdapter, ArgoCD adapter, Kubernetes events, feature flag systems
        """
        self.sources = change_sources

    def find_correlated_changes(
        self,
        anomaly_timestamp: datetime,
        service_name: str,
        lookback_minutes: int = 60,
    ) -> list[CorrelatedChange]:
        """Find changes that might have caused the anomaly."""

        changes = []
        window_start = anomaly_timestamp - timedelta(minutes=lookback_minutes)

        for source in self.sources:
            recent_changes = source.get_changes(
                service=service_name,
                start_time=window_start,
                end_time=anomaly_timestamp,
            )

            for change in recent_changes:
                correlation_score = self._calculate_correlation(
                    change_time=change.timestamp,
                    anomaly_time=anomaly_timestamp,
                    change_type=change.type,
                )

                if correlation_score > 0.3:
                    changes.append(CorrelatedChange(
                        change=change,
                        correlation_score=correlation_score,
                        time_delta_minutes=(anomaly_timestamp - change.timestamp).total_seconds() / 60,
                    ))

        return sorted(changes, key=lambda c: c.correlation_score, reverse=True)
```

---

## 3. Cross-Service Anomaly Correlation

### Current Limitation
Each service is analyzed independently. Cascading failures across services are not automatically correlated.

### Proposed: Service Dependency Graph Analysis

```python
@dataclass
class CrossServiceAnomaly:
    """Anomaly that spans multiple services."""

    primary_service: str  # Likely root cause service
    affected_services: list[str]
    propagation_path: list[str]  # Order of impact
    blast_radius: int  # Number of affected services
    estimated_user_impact: str  # "none", "partial", "major", "complete"
    common_cause: str | None


class ServiceDependencyAnalyzer:
    """Analyze anomalies across service dependency graph."""

    def __init__(self, dependency_graph: dict[str, list[str]]):
        """
        dependency_graph: service -> list of services it depends on
        Example: {"booking": ["database", "payment", "inventory"]}
        """
        self.graph = dependency_graph
        self.reverse_graph = self._build_reverse_graph()

    def correlate_anomalies(
        self,
        service_anomalies: dict[str, dict[str, Any]],  # service -> anomalies
        timestamps: dict[str, datetime],  # service -> first anomaly time
    ) -> CrossServiceAnomaly | None:
        """Identify if anomalies across services are related."""

        if len(service_anomalies) < 2:
            return None

        # Find temporal ordering
        ordered_services = sorted(timestamps.keys(), key=lambda s: timestamps[s])

        # Check if propagation follows dependency graph
        propagation_path = self._trace_propagation(ordered_services)

        if propagation_path:
            primary = propagation_path[0]

            # Calculate blast radius
            affected = self._calculate_affected_services(primary)

            return CrossServiceAnomaly(
                primary_service=primary,
                affected_services=list(service_anomalies.keys()),
                propagation_path=propagation_path,
                blast_radius=len(affected),
                estimated_user_impact=self._estimate_user_impact(affected),
                common_cause=self._identify_common_cause(service_anomalies),
            )

        return None

    def _trace_propagation(self, ordered_services: list[str]) -> list[str] | None:
        """Trace how anomaly propagated through dependency graph."""

        if not ordered_services:
            return None

        path = [ordered_services[0]]

        for service in ordered_services[1:]:
            # Check if this service depends on any service in the path
            dependencies = self.graph.get(service, [])
            if any(dep in path for dep in dependencies):
                path.append(service)
            # Or if any service in the path depends on this service
            elif service in self.graph.get(path[-1], []):
                path.insert(0, service)  # This is actually upstream

        return path if len(path) > 1 else None

    def _calculate_affected_services(self, root_service: str) -> set[str]:
        """Calculate all services that could be affected by root service failure."""

        affected = set()
        to_process = [root_service]

        while to_process:
            current = to_process.pop()
            if current in affected:
                continue
            affected.add(current)

            # Add all services that depend on this one
            dependents = self.reverse_graph.get(current, [])
            to_process.extend(dependents)

        return affected

    def generate_incident_summary(
        self, cross_service_anomaly: CrossServiceAnomaly
    ) -> dict[str, Any]:
        """Generate comprehensive incident summary."""

        return {
            "title": f"Cross-service incident originating from {cross_service_anomaly.primary_service}",
            "severity": "critical" if cross_service_anomaly.blast_radius > 3 else "high",
            "summary": (
                f"Anomaly detected in {cross_service_anomaly.primary_service} has propagated to "
                f"{len(cross_service_anomaly.affected_services)} services. "
                f"Estimated user impact: {cross_service_anomaly.estimated_user_impact}."
            ),
            "propagation": " → ".join(cross_service_anomaly.propagation_path),
            "recommended_actions": [
                f"FOCUS: Investigate {cross_service_anomaly.primary_service} first (likely root cause)",
                f"ASSESS: Confirm blast radius of {cross_service_anomaly.blast_radius} services",
                "COMMUNICATE: Update status page if user-facing",
                "MITIGATE: Consider circuit breakers on affected dependencies",
            ],
            "affected_services": cross_service_anomaly.affected_services,
        }
```

---

## 4. Incident Learning System

### Current Limitation
System doesn't learn from past incidents. Each new incident starts from scratch.

### Proposed: Incident Similarity and Pattern Learning

```python
@dataclass
class IncidentFingerprint:
    """Unique fingerprint of an incident for similarity matching."""

    metric_pattern: dict[str, str]  # metric -> pattern type
    time_of_day: str  # business_hours, evening, night, weekend
    day_of_week: int
    duration_minutes: int
    affected_metrics: list[str]
    severity_progression: list[str]  # How severity changed over time
    resolution_type: str  # "auto_resolved", "manual_fix", "rollback", "scaling"


class IncidentLearningSystem:
    """Learn from past incidents to improve detection and response."""

    def __init__(self, incident_store: IncidentStore):
        self.store = incident_store
        self.similarity_model = self._build_similarity_model()

    def find_similar_incidents(
        self,
        current_fingerprint: IncidentFingerprint,
        top_k: int = 5,
    ) -> list[SimilarIncident]:
        """Find historically similar incidents."""

        historical = self.store.get_all_fingerprints()

        similarities = []
        for hist_id, hist_fp in historical.items():
            score = self._calculate_similarity(current_fingerprint, hist_fp)
            if score > 0.6:
                incident = self.store.get_incident(hist_id)
                similarities.append(SimilarIncident(
                    incident_id=hist_id,
                    similarity_score=score,
                    resolution=incident.resolution,
                    time_to_resolution=incident.duration,
                    what_worked=incident.effective_actions,
                    what_didnt_work=incident.ineffective_actions,
                ))

        return sorted(similarities, key=lambda s: s.similarity_score, reverse=True)[:top_k]

    def suggest_resolution(
        self,
        current_anomalies: dict[str, Any],
        similar_incidents: list[SimilarIncident],
    ) -> ResolutionSuggestion:
        """Suggest resolution based on similar past incidents."""

        if not similar_incidents:
            return ResolutionSuggestion(
                confidence=0.3,
                suggested_actions=["No similar incidents found - manual investigation required"],
                estimated_resolution_time=None,
            )

        # Aggregate successful actions from similar incidents
        action_success_rate = defaultdict(lambda: {"success": 0, "total": 0})
        resolution_times = []

        for incident in similar_incidents:
            weight = incident.similarity_score
            resolution_times.append(incident.time_to_resolution)

            for action in incident.what_worked:
                action_success_rate[action]["success"] += weight
                action_success_rate[action]["total"] += weight

            for action in incident.what_didnt_work:
                action_success_rate[action]["total"] += weight

        # Rank actions by success rate
        ranked_actions = sorted(
            action_success_rate.items(),
            key=lambda x: x[1]["success"] / (x[1]["total"] + 0.1),
            reverse=True,
        )

        return ResolutionSuggestion(
            confidence=similar_incidents[0].similarity_score,
            suggested_actions=[
                f"{action} (worked in {stats['success']/stats['total']:.0%} of similar incidents)"
                for action, stats in ranked_actions[:5]
            ],
            estimated_resolution_time=timedelta(minutes=np.median(resolution_times)),
            similar_incident_ids=[i.incident_id for i in similar_incidents[:3]],
        )

    def record_resolution(
        self,
        incident_id: str,
        resolution: IncidentResolution,
    ) -> None:
        """Record how an incident was resolved for future learning."""

        self.store.update_incident(incident_id, {
            "resolution": resolution.resolution_type,
            "effective_actions": resolution.effective_actions,
            "ineffective_actions": resolution.ineffective_actions,
            "duration": resolution.duration,
            "root_cause": resolution.root_cause,
            "prevention_measures": resolution.prevention_measures,
        })

        # Retrain similarity model periodically
        if self.store.count() % 100 == 0:
            self.similarity_model = self._build_similarity_model()
```

### Runbook Automation

```python
class RunbookAutomation:
    """Automatically execute runbook steps based on anomaly type."""

    AUTOMATED_RUNBOOKS = {
        "circuit_breaker_open": [
            RunbookStep(
                name="Check downstream health",
                action="http_health_check",
                params={"target": "{downstream_service}"},
                timeout_seconds=10,
            ),
            RunbookStep(
                name="Check circuit breaker metrics",
                action="prometheus_query",
                params={"query": "circuit_breaker_state{service='{service}'}"},
                timeout_seconds=5,
            ),
            RunbookStep(
                name="Gather recent errors",
                action="log_query",
                params={"query": "level:error service:{service}", "limit": 100},
                timeout_seconds=30,
            ),
        ],
        "database_bottleneck": [
            RunbookStep(
                name="Get slow queries",
                action="db_slow_query_log",
                params={"threshold_ms": 100, "limit": 20},
                timeout_seconds=15,
            ),
            RunbookStep(
                name="Check connection pool",
                action="prometheus_query",
                params={"query": "db_connection_pool_usage{service='{service}'}"},
                timeout_seconds=5,
            ),
            RunbookStep(
                name="Check replication lag",
                action="db_replication_status",
                params={},
                timeout_seconds=10,
            ),
        ],
    }

    async def execute_runbook(
        self,
        anomaly_type: str,
        context: dict[str, Any],
    ) -> RunbookResult:
        """Execute automated runbook and gather diagnostic data."""

        runbook = self.AUTOMATED_RUNBOOKS.get(anomaly_type)
        if not runbook:
            return RunbookResult(success=False, message="No runbook defined")

        results = []
        for step in runbook:
            try:
                # Interpolate parameters
                params = {k: v.format(**context) for k, v in step.params.items()}

                # Execute step
                result = await self._execute_step(step.action, params, step.timeout_seconds)
                results.append(StepResult(
                    step_name=step.name,
                    success=True,
                    output=result,
                ))
            except Exception as e:
                results.append(StepResult(
                    step_name=step.name,
                    success=False,
                    error=str(e),
                ))

        return RunbookResult(
            success=all(r.success for r in results),
            steps=results,
            diagnostic_summary=self._summarize_diagnostics(results),
        )
```

---

## 5. Alert Quality Improvements

### Current Limitation
Alert fatigue from noisy or redundant alerts.

### Proposed: Intelligent Alert Management

```python
class AlertQualityManager:
    """Manage alert quality to reduce noise and improve actionability."""

    def __init__(self):
        self.alert_history = deque(maxlen=10000)
        self.suppression_rules = []
        self.correlation_window = timedelta(minutes=5)

    def should_alert(
        self,
        anomaly: dict[str, Any],
        service: str,
    ) -> AlertDecision:
        """Decide whether to send an alert."""

        # 1. Check for duplicate/related recent alerts
        related = self._find_related_alerts(anomaly, service)
        if related:
            return AlertDecision(
                should_send=False,
                reason="correlated_with_existing",
                related_alert_id=related[0].id,
                action="append_to_existing",
            )

        # 2. Check suppression rules
        for rule in self.suppression_rules:
            if rule.matches(anomaly, service):
                return AlertDecision(
                    should_send=False,
                    reason="suppressed",
                    suppression_rule=rule.name,
                )

        # 3. Check if this is a flapping alert
        if self._is_flapping(service, anomaly.get("type")):
            return AlertDecision(
                should_send=False,
                reason="flapping_detected",
                action="aggregate",
            )

        # 4. Calculate alert score
        score = self._calculate_alert_score(anomaly)

        if score < 0.5:
            return AlertDecision(
                should_send=False,
                reason="low_confidence",
                score=score,
            )

        return AlertDecision(
            should_send=True,
            priority=self._calculate_priority(anomaly, score),
            routing=self._determine_routing(anomaly, service),
        )

    def _is_flapping(self, service: str, anomaly_type: str) -> bool:
        """Detect if alert is flapping (rapidly switching state)."""

        recent = [
            a for a in self.alert_history
            if a.service == service
            and a.type == anomaly_type
            and a.timestamp > datetime.now() - timedelta(minutes=30)
        ]

        if len(recent) < 4:
            return False

        # Check for alternating resolve/fire pattern
        states = [a.state for a in sorted(recent, key=lambda a: a.timestamp)]
        transitions = sum(1 for i in range(1, len(states)) if states[i] != states[i-1])

        return transitions >= 4  # 4+ state changes in 30 min = flapping

    def _calculate_alert_score(self, anomaly: dict[str, Any]) -> float:
        """Calculate overall alert quality score."""

        score = 0.5  # Base score

        # Higher severity = higher score
        severity_boost = {
            "critical": 0.3,
            "high": 0.2,
            "medium": 0.1,
            "low": 0.0,
        }
        score += severity_boost.get(anomaly.get("severity", "medium"), 0)

        # Strong ML confidence = higher score
        ml_score = anomaly.get("score", 0)
        if ml_score < -0.5:
            score += 0.2

        # Named pattern match = higher score (vs generic multivariate)
        if anomaly.get("pattern_name"):
            score += 0.1

        # Multiple detection methods agreeing = higher score
        # (This would need to be passed in)

        return min(1.0, score)


class AlertAggregator:
    """Aggregate related alerts into coherent incidents."""

    def aggregate(
        self,
        alerts: list[Alert],
        time_window: timedelta = timedelta(minutes=10),
    ) -> list[AggregatedIncident]:
        """Group related alerts into incidents."""

        incidents = []
        used_alerts = set()

        for alert in sorted(alerts, key=lambda a: a.timestamp):
            if alert.id in used_alerts:
                continue

            # Find all related alerts
            related = self._find_related(alert, alerts, time_window)
            used_alerts.update(a.id for a in related)

            # Create aggregated incident
            incident = AggregatedIncident(
                primary_alert=alert,
                related_alerts=related,
                services=list(set(a.service for a in [alert] + related)),
                summary=self._generate_summary([alert] + related),
                priority=max(a.priority for a in [alert] + related),
            )
            incidents.append(incident)

        return incidents

    def _generate_summary(self, alerts: list[Alert]) -> str:
        """Generate human-readable summary of aggregated alerts."""

        services = list(set(a.service for a in alerts))
        types = list(set(a.anomaly_type for a in alerts))

        if len(services) == 1:
            return f"{services[0]}: {', '.join(types)}"
        else:
            return f"Multi-service incident affecting {', '.join(services)}: {types[0]}"
```

---

## 6. Adaptive Thresholds

### Current Limitation
Static thresholds don't account for expected variation.

### Proposed: Dynamic Threshold Learning

```python
class AdaptiveThresholdManager:
    """Dynamically adjust thresholds based on observed patterns."""

    def __init__(self, service_name: str):
        self.service = service_name
        self.baseline_windows = {
            "hourly": deque(maxlen=24*7),  # 1 week of hourly data
            "daily": deque(maxlen=30),      # 1 month of daily data
        }
        self.seasonality_model = None

    def update_baseline(self, timestamp: datetime, metrics: dict[str, float]) -> None:
        """Update rolling baseline with new observation."""

        hour = timestamp.hour
        day = timestamp.weekday()

        self.baseline_windows["hourly"].append({
            "hour": hour,
            "day": day,
            "metrics": metrics,
            "timestamp": timestamp,
        })

    def get_adaptive_threshold(
        self,
        metric_name: str,
        current_time: datetime,
        base_threshold: float,
    ) -> AdaptiveThreshold:
        """Get context-aware threshold for current time."""

        hour = current_time.hour
        day = current_time.weekday()

        # Find similar time periods
        similar_periods = [
            obs for obs in self.baseline_windows["hourly"]
            if obs["hour"] == hour and obs["day"] == day
        ]

        if len(similar_periods) < 3:
            return AdaptiveThreshold(
                value=base_threshold,
                confidence=0.5,
                adjustment_reason="insufficient_baseline_data",
            )

        # Calculate expected value and variance for this time period
        values = [obs["metrics"].get(metric_name, 0) for obs in similar_periods]
        expected = np.mean(values)
        variance = np.std(values)

        # Adjust threshold based on expected value
        # If current time typically has higher values, raise threshold
        adjustment_factor = expected / (base_threshold + 1e-8)

        if adjustment_factor > 1.5:
            adjusted_threshold = base_threshold * adjustment_factor * 0.8  # Some buffer
            reason = f"Historically higher at {hour}:00 on {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][day]}"
        elif adjustment_factor < 0.5:
            adjusted_threshold = base_threshold * 0.7  # More sensitive during low periods
            reason = f"Historically lower at {hour}:00 - using tighter threshold"
        else:
            adjusted_threshold = base_threshold
            reason = "Within normal range for time period"

        return AdaptiveThreshold(
            value=adjusted_threshold,
            confidence=min(0.95, 0.5 + len(similar_periods) * 0.05),
            adjustment_reason=reason,
            expected_value=expected,
            expected_variance=variance,
        )
```

---

## 7. SLO Integration

### Proposed: SLO-Aware Anomaly Detection

```python
@dataclass
class SLODefinition:
    """Service Level Objective definition."""

    name: str
    metric: str
    target: float  # e.g., 99.9 for 99.9% availability
    window: timedelta  # e.g., 30 days
    burn_rate_thresholds: dict[str, float]  # e.g., {"critical": 14.4, "high": 6, "medium": 1}


class SLOAwareDetector:
    """Detect anomalies in context of SLO burn rate."""

    def __init__(self, slos: list[SLODefinition]):
        self.slos = {slo.name: slo for slo in slos}
        self.error_budgets = {}

    def calculate_burn_rate(
        self,
        slo_name: str,
        current_error_rate: float,
        window_minutes: int = 60,
    ) -> BurnRateResult:
        """Calculate how fast error budget is being consumed."""

        slo = self.slos.get(slo_name)
        if not slo:
            return None

        # Error budget = 1 - SLO target (e.g., 0.1% for 99.9% SLO)
        error_budget = 1 - (slo.target / 100)

        # Burn rate = actual error rate / allowed error rate
        # If burn rate = 1, we're consuming budget at exactly the sustainable rate
        # If burn rate = 10, we're consuming 10x faster than sustainable
        burn_rate = current_error_rate / error_budget

        # Calculate time until budget exhaustion at current rate
        remaining_budget = self._get_remaining_budget(slo_name)
        hours_until_exhaustion = (remaining_budget / current_error_rate) if current_error_rate > 0 else float("inf")

        # Determine severity based on burn rate
        severity = "low"
        for sev, threshold in sorted(slo.burn_rate_thresholds.items(), key=lambda x: x[1], reverse=True):
            if burn_rate >= threshold:
                severity = sev
                break

        return BurnRateResult(
            slo_name=slo_name,
            burn_rate=burn_rate,
            severity=severity,
            remaining_budget_percent=remaining_budget * 100,
            hours_until_exhaustion=hours_until_exhaustion,
            message=self._generate_burn_rate_message(slo_name, burn_rate, hours_until_exhaustion),
        )

    def _generate_burn_rate_message(
        self,
        slo_name: str,
        burn_rate: float,
        hours_until_exhaustion: float,
    ) -> str:
        """Generate actionable message about burn rate."""

        if burn_rate < 1:
            return f"SLO healthy: consuming error budget at {burn_rate:.1f}x sustainable rate"
        elif burn_rate < 6:
            return f"SLO warning: consuming budget at {burn_rate:.1f}x rate, ~{hours_until_exhaustion:.0f}h until exhaustion"
        elif burn_rate < 14.4:
            return f"SLO at risk: {burn_rate:.1f}x burn rate, budget exhausted in ~{hours_until_exhaustion:.1f}h if sustained"
        else:
            return f"SLO critical: {burn_rate:.1f}x burn rate, immediate action required"
```

---

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
- [ ] Implement `AdaptiveThresholdManager`
- [ ] Add SLO definitions and `SLOAwareDetector`
- [ ] Integrate alert aggregation

### Phase 2: Intelligence (Weeks 3-4)
- [ ] Implement `RootCauseAnalyzer`
- [ ] Add `ChangeCorrelator` with deployment system integration
- [ ] Build incident similarity matching

### Phase 3: Prediction (Weeks 5-6)
- [ ] Implement `PredictiveDetector` with time series forecasting
- [ ] Add trend analysis and early warning
- [ ] Integrate with alerting system

### Phase 4: Cross-Service (Weeks 7-8)
- [ ] Build service dependency graph
- [ ] Implement `ServiceDependencyAnalyzer`
- [ ] Add cross-service correlation

### Phase 5: Automation (Weeks 9-10)
- [ ] Define automated runbooks
- [ ] Implement `RunbookAutomation`
- [ ] Add feedback loop for resolution learning

---

## Expected Impact

| Metric | Current | Expected |
|--------|---------|----------|
| Mean Time to Detect (MTTD) | ~5 min | ~2 min (with prediction: -10 min before impact) |
| Mean Time to Resolution (MTTR) | ~45 min | ~20 min |
| False Positive Rate | ~15% | ~5% |
| Alert Noise | High | Low (with aggregation) |
| Root Cause Identification | Manual | 70% automated |
| Cross-Service Correlation | Manual | Automated |
