"""
Enrichment module for anomaly diagnosis.

This module provides tools to enrich anomaly alerts with contextual information,
helping operators understand the root cause of detected issues.

Enrichment types:
- Exception enrichment: Shows which exceptions are occurring during error spikes
- Service graph enrichment: Shows downstream service calls during latency issues
- Envoy enrichment: Shows edge/ingress metrics from Envoy proxy
"""

from __future__ import annotations

from smartbox_anomaly.enrichment.envoy import (
    EnvoyEnrichmentService,
    EnvoyLatencyPercentiles,
    EnvoyMetricsContext,
    EnvoyRequestRates,
)
from smartbox_anomaly.enrichment.exceptions import (
    ExceptionBreakdown,
    ExceptionEnrichmentService,
    ExceptionSummary,
)
from smartbox_anomaly.enrichment.service_graph import (
    RouteSummary,
    ServiceGraphBreakdown,
    ServiceGraphEnrichmentService,
)

__all__ = [
    # Envoy enrichment
    "EnvoyEnrichmentService",
    "EnvoyLatencyPercentiles",
    "EnvoyMetricsContext",
    "EnvoyRequestRates",
    # Exception enrichment
    "ExceptionBreakdown",
    "ExceptionEnrichmentService",
    "ExceptionSummary",
    # Service graph enrichment
    "RouteSummary",
    "ServiceGraphBreakdown",
    "ServiceGraphEnrichmentService",
]
