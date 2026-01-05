"""
SLO-aware severity evaluation module.

This module provides a post-processing layer that adjusts anomaly severity
based on operational SLO thresholds, not just statistical deviation.
"""

from smartbox_anomaly.slo.evaluator import SLOEvaluator, SLOEvaluationResult

__all__ = [
    "SLOEvaluator",
    "SLOEvaluationResult",
]
