"""
Enrichment module for anomaly diagnosis.

This module provides tools to enrich anomaly alerts with contextual information,
helping operators understand the root cause of detected issues.

Enrichment types:
- Exception enrichment: Shows which exceptions are occurring during error spikes
- Service graph enrichment: Shows downstream service calls during latency issues
"""

from __future__ import annotations

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
    # Exception enrichment
    "ExceptionBreakdown",
    "ExceptionEnrichmentService",
    "ExceptionSummary",
    # Service graph enrichment
    "RouteSummary",
    "ServiceGraphBreakdown",
    "ServiceGraphEnrichmentService",
]
