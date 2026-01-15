"""
Data quality analysis utilities for time series metrics.

This module provides tools for analyzing the quality of metric data
fetched from VictoriaMetrics, including gap detection, coverage analysis,
and quality scoring.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TimeGap:
    """Represents a gap in time series data."""

    start: datetime
    end: datetime
    duration_minutes: float

    @property
    def duration_str(self) -> str:
        """Human-readable duration string."""
        if self.duration_minutes < 60:
            return f"{self.duration_minutes:.0f}m"
        elif self.duration_minutes < 1440:
            hours = self.duration_minutes / 60
            return f"{hours:.1f}h"
        else:
            days = self.duration_minutes / 1440
            return f"{days:.1f}d"


@dataclass
class DataQualityReport:
    """Report on time series data quality."""

    metric_name: str
    start_time: datetime
    end_time: datetime
    expected_points: int
    actual_points: int
    coverage_percent: float
    gaps: list[TimeGap] = field(default_factory=list)
    max_gap_minutes: float = 0.0
    null_count: int = 0
    inf_count: int = 0
    negative_count: int = 0
    quality_grade: str = "A"
    quality_score: int = 100
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "metric_name": self.metric_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "expected_points": self.expected_points,
            "actual_points": self.actual_points,
            "coverage_percent": round(self.coverage_percent, 2),
            "gap_count": len(self.gaps),
            "max_gap_minutes": round(self.max_gap_minutes, 1),
            "null_count": self.null_count,
            "inf_count": self.inf_count,
            "negative_count": self.negative_count,
            "quality_grade": self.quality_grade,
            "quality_score": self.quality_score,
            "issues": self.issues,
        }

    def log_summary(self) -> None:
        """Log a summary of the quality report."""
        if self.quality_grade in ("A", "B"):
            log_fn = logger.info
        elif self.quality_grade == "C":
            log_fn = logger.warning
        else:
            log_fn = logger.error

        log_fn(
            "Data quality for %s: Grade %s (score %d/100) - "
            "%.1f%% coverage, %d gaps (max %.0fm), %d nulls",
            self.metric_name,
            self.quality_grade,
            self.quality_score,
            self.coverage_percent,
            len(self.gaps),
            self.max_gap_minutes,
            self.null_count,
        )

        for issue in self.issues:
            logger.warning("  Issue: %s", issue)


def detect_time_gaps(
    timestamps: pd.DatetimeIndex | pd.Series,
    expected_interval_minutes: float = 5.0,
    gap_threshold_factor: float = 2.0,
) -> list[TimeGap]:
    """Detect gaps in a time series.

    A gap is detected when the interval between consecutive timestamps
    exceeds the expected interval by the threshold factor.

    Args:
        timestamps: Sorted timestamps to analyze.
        expected_interval_minutes: Expected interval between points (default: 5m).
        gap_threshold_factor: Multiplier for expected interval to consider a gap.

    Returns:
        List of TimeGap objects representing detected gaps.
    """
    if len(timestamps) < 2:
        return []

    gaps = []
    timestamps_sorted = pd.to_datetime(timestamps).sort_values()
    threshold_td = timedelta(minutes=expected_interval_minutes * gap_threshold_factor)

    for i in range(1, len(timestamps_sorted)):
        prev_ts = timestamps_sorted[i - 1]
        curr_ts = timestamps_sorted[i]
        delta = curr_ts - prev_ts

        if delta > threshold_td:
            gap = TimeGap(
                start=prev_ts.to_pydatetime(),
                end=curr_ts.to_pydatetime(),
                duration_minutes=delta.total_seconds() / 60,
            )
            gaps.append(gap)

    return gaps


def analyze_data_quality(
    df: pd.DataFrame,
    metric_name: str,
    expected_interval_minutes: float = 5.0,
    lookback_days: int = 30,
) -> DataQualityReport:
    """Analyze the quality of a metric's time series data.

    Args:
        df: DataFrame with DatetimeIndex containing the metric data.
        metric_name: Name of the metric being analyzed.
        expected_interval_minutes: Expected interval between data points.
        lookback_days: Number of days the data should cover.

    Returns:
        DataQualityReport with quality metrics and issues.
    """
    if df.empty or metric_name not in df.columns:
        return DataQualityReport(
            metric_name=metric_name,
            start_time=datetime.now() - timedelta(days=lookback_days),
            end_time=datetime.now(),
            expected_points=int(lookback_days * 24 * 60 / expected_interval_minutes),
            actual_points=0,
            coverage_percent=0.0,
            quality_grade="F",
            quality_score=0,
            issues=["No data available"],
        )

    # Basic statistics
    series = df[metric_name]
    start_time = df.index.min()
    end_time = df.index.max()

    # Calculate expected vs actual points
    expected_points = int(lookback_days * 24 * 60 / expected_interval_minutes)
    actual_points = len(series.dropna())
    coverage_percent = (actual_points / expected_points) * 100 if expected_points > 0 else 0

    # Detect gaps
    gaps = detect_time_gaps(df.index, expected_interval_minutes)
    max_gap_minutes = max((g.duration_minutes for g in gaps), default=0.0)

    # Count data quality issues
    null_count = int(series.isna().sum())
    inf_count = int(np.isinf(series.replace([np.nan], 0)).sum())
    negative_count = int((series < 0).sum()) if metric_name != "error_rate" else 0

    # Build issues list
    issues = []
    if coverage_percent < 50:
        issues.append(f"Very low coverage: {coverage_percent:.1f}%")
    elif coverage_percent < 80:
        issues.append(f"Low coverage: {coverage_percent:.1f}%")

    if max_gap_minutes > 60:
        issues.append(f"Large gap detected: {max_gap_minutes:.0f} minutes")

    if len(gaps) > 10:
        issues.append(f"Many gaps in data: {len(gaps)} gaps detected")

    if null_count > actual_points * 0.1:
        issues.append(f"High null rate: {null_count} null values ({null_count/actual_points*100:.1f}%)")

    if inf_count > 0:
        issues.append(f"Invalid values: {inf_count} infinite values")

    if negative_count > 0 and metric_name in ("request_rate", "application_latency", "dependency_latency", "database_latency"):
        issues.append(f"Invalid values: {negative_count} negative values")

    # Calculate quality score
    score = 100

    # Coverage penalties
    if coverage_percent < 50:
        score -= 40
    elif coverage_percent < 70:
        score -= 25
    elif coverage_percent < 90:
        score -= 10

    # Gap penalties
    if max_gap_minutes > 120:  # > 2 hours
        score -= 20
    elif max_gap_minutes > 60:  # > 1 hour
        score -= 10
    elif max_gap_minutes > 30:  # > 30 min
        score -= 5

    # Gap count penalties
    if len(gaps) > 20:
        score -= 15
    elif len(gaps) > 10:
        score -= 10
    elif len(gaps) > 5:
        score -= 5

    # Data quality penalties
    if null_count > actual_points * 0.1:
        score -= 10
    if inf_count > 0:
        score -= 15
    if negative_count > 0:
        score -= 10

    score = max(0, score)

    # Assign grade
    if score >= 90:
        grade = "A"
    elif score >= 75:
        grade = "B"
    elif score >= 60:
        grade = "C"
    elif score >= 40:
        grade = "D"
    else:
        grade = "F"

    return DataQualityReport(
        metric_name=metric_name,
        start_time=start_time.to_pydatetime() if hasattr(start_time, 'to_pydatetime') else start_time,
        end_time=end_time.to_pydatetime() if hasattr(end_time, 'to_pydatetime') else end_time,
        expected_points=expected_points,
        actual_points=actual_points,
        coverage_percent=coverage_percent,
        gaps=gaps,
        max_gap_minutes=max_gap_minutes,
        null_count=null_count,
        inf_count=inf_count,
        negative_count=negative_count,
        quality_grade=grade,
        quality_score=score,
        issues=issues,
    )


def analyze_combined_data_quality(
    df: pd.DataFrame,
    lookback_days: int = 30,
    expected_interval_minutes: float = 5.0,
) -> dict[str, Any]:
    """Analyze quality of a combined metrics DataFrame.

    Args:
        df: DataFrame with all metrics.
        lookback_days: Number of days the data should cover.
        expected_interval_minutes: Expected interval between data points.

    Returns:
        Dictionary with per-metric reports and overall summary.
    """
    reports = {}
    overall_score = 0
    metric_count = 0

    for col in df.columns:
        if col in ("timestamp", "index"):
            continue

        report = analyze_data_quality(
            df,
            metric_name=col,
            expected_interval_minutes=expected_interval_minutes,
            lookback_days=lookback_days,
        )
        reports[col] = report
        overall_score += report.quality_score
        metric_count += 1

    avg_score = overall_score / metric_count if metric_count > 0 else 0

    if avg_score >= 90:
        overall_grade = "A"
    elif avg_score >= 75:
        overall_grade = "B"
    elif avg_score >= 60:
        overall_grade = "C"
    elif avg_score >= 40:
        overall_grade = "D"
    else:
        overall_grade = "F"

    return {
        "metrics": {name: report.to_dict() for name, report in reports.items()},
        "overall_score": round(avg_score, 1),
        "overall_grade": overall_grade,
        "metric_count": metric_count,
        "row_count": len(df),
    }
