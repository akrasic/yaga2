"""Training run storage for tracking model training history.

This module stores training run metadata in SQLite for visualization
in the admin dashboard. It tracks:
- Training status (RUNNING, COMPLETED, FAILED)
- Validation results per time period
- Data quality metrics
- Timing information

The storage uses the same database as fingerprinting (anomaly_state.db)
for operational simplicity.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import traceback
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Generator

from smartbox_anomaly.core.config import get_config

logger = logging.getLogger(__name__)


# =============================================================================
# Constants and Enums
# =============================================================================


class TrainingRunStatus(str, Enum):
    """Status of a training run."""
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ValidationStatus(str, Enum):
    """Overall validation status for a training run."""
    PASSED = "PASSED"      # All periods pass validation
    WARNING = "WARNING"    # Some periods have issues but training completed
    FAILED = "FAILED"      # Critical validation failures


SCHEMA_VERSION = "1.0_training_runs"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS training_runs (
    run_id TEXT PRIMARY KEY,
    service_name TEXT NOT NULL,
    model_variant TEXT NOT NULL DEFAULT 'baseline',
    started_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP,
    status TEXT NOT NULL DEFAULT 'RUNNING',

    -- Data metrics
    total_data_points INTEGER,
    training_start_date TEXT,
    training_end_date TEXT,

    -- Model metrics
    time_periods_trained INTEGER,
    feature_count INTEGER,
    explainability_metrics INTEGER,

    -- Validation summary
    validation_status TEXT,
    periods_passed INTEGER DEFAULT 0,
    periods_warned INTEGER DEFAULT 0,
    periods_failed INTEGER DEFAULT 0,

    -- Detailed results (JSON)
    validation_details TEXT,
    data_quality_report TEXT,
    model_metadata TEXT,

    -- Timing
    duration_seconds REAL,

    -- Error info (if failed)
    error_message TEXT,
    error_traceback TEXT,

    CHECK (status IN ('RUNNING', 'COMPLETED', 'FAILED'))
)
"""

CREATE_INDEXES_SQL = [
    """CREATE INDEX IF NOT EXISTS idx_training_service
       ON training_runs(service_name, started_at DESC)""",
    """CREATE INDEX IF NOT EXISTS idx_training_status
       ON training_runs(status, started_at DESC)""",
    """CREATE INDEX IF NOT EXISTS idx_training_validation
       ON training_runs(validation_status, started_at DESC)""",
]


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class TrainingRunRecord:
    """Record of a single training run."""
    run_id: str
    service_name: str
    model_variant: str
    started_at: datetime
    completed_at: datetime | None = None
    status: TrainingRunStatus = TrainingRunStatus.RUNNING

    # Data metrics
    total_data_points: int | None = None
    training_start_date: str | None = None
    training_end_date: str | None = None

    # Model metrics
    time_periods_trained: int | None = None
    feature_count: int | None = None
    explainability_metrics: int | None = None

    # Validation summary
    validation_status: ValidationStatus | None = None
    periods_passed: int = 0
    periods_warned: int = 0
    periods_failed: int = 0

    # Detailed results
    validation_details: dict[str, Any] = field(default_factory=dict)
    data_quality_report: dict[str, Any] = field(default_factory=dict)
    model_metadata: dict[str, Any] = field(default_factory=dict)

    # Timing
    duration_seconds: float | None = None

    # Error info
    error_message: str | None = None
    error_traceback: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "run_id": self.run_id,
            "service_name": self.service_name,
            "model_variant": self.model_variant,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status.value if self.status else None,
            "total_data_points": self.total_data_points,
            "training_start_date": self.training_start_date,
            "training_end_date": self.training_end_date,
            "time_periods_trained": self.time_periods_trained,
            "feature_count": self.feature_count,
            "explainability_metrics": self.explainability_metrics,
            "validation_status": self.validation_status.value if self.validation_status else None,
            "periods_passed": self.periods_passed,
            "periods_warned": self.periods_warned,
            "periods_failed": self.periods_failed,
            "validation_details": self.validation_details,
            "data_quality_report": self.data_quality_report,
            "model_metadata": self.model_metadata,
            "duration_seconds": self.duration_seconds,
            "error_message": self.error_message,
            "error_traceback": self.error_traceback,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> TrainingRunRecord:
        """Create record from database row."""
        return cls(
            run_id=row["run_id"],
            service_name=row["service_name"],
            model_variant=row["model_variant"],
            started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
            completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
            status=TrainingRunStatus(row["status"]) if row["status"] else None,
            total_data_points=row["total_data_points"],
            training_start_date=row["training_start_date"],
            training_end_date=row["training_end_date"],
            time_periods_trained=row["time_periods_trained"],
            feature_count=row["feature_count"],
            explainability_metrics=row["explainability_metrics"],
            validation_status=ValidationStatus(row["validation_status"]) if row["validation_status"] else None,
            periods_passed=row["periods_passed"] or 0,
            periods_warned=row["periods_warned"] or 0,
            periods_failed=row["periods_failed"] or 0,
            validation_details=json.loads(row["validation_details"]) if row["validation_details"] else {},
            data_quality_report=json.loads(row["data_quality_report"]) if row["data_quality_report"] else {},
            model_metadata=json.loads(row["model_metadata"]) if row["model_metadata"] else {},
            duration_seconds=row["duration_seconds"],
            error_message=row["error_message"],
            error_traceback=row["error_traceback"],
        )


# =============================================================================
# Training Run Storage
# =============================================================================


class TrainingRunStorage:
    """Storage for training run history.

    Provides methods to:
    - Start a training run (creates RUNNING record)
    - Complete a training run (updates with results)
    - Fail a training run (records error)
    - Query training history

    Example:
        >>> storage = TrainingRunStorage()
        >>> run_id = storage.start_run("booking", "baseline")
        >>> # ... training happens ...
        >>> storage.complete_run(run_id, validation_results, metadata)
    """

    def __init__(self, db_path: str | None = None) -> None:
        """Initialize storage.

        Args:
            db_path: Path to SQLite database. Defaults to config value.
        """
        self._config = get_config().fingerprinting
        self.db_path = db_path or self._config.db_path
        self._lock = threading.Lock()
        self._init_database()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get database connection with row factory."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_database(self) -> None:
        """Initialize database schema."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(CREATE_TABLE_SQL)
                for index_sql in CREATE_INDEXES_SQL:
                    cursor.execute(index_sql)
                conn.commit()
                logger.debug("Training runs database initialized at %s", self.db_path)

    def start_run(
        self,
        service_name: str,
        model_variant: str = "baseline",
    ) -> str:
        """Start a new training run.

        Args:
            service_name: Name of the service being trained.
            model_variant: Model variant (baseline, holiday, etc.).

        Returns:
            Unique run_id for this training run.
        """
        run_id = f"train_{uuid.uuid4().hex[:12]}"
        started_at = datetime.now()

        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO training_runs (run_id, service_name, model_variant, started_at, status)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (run_id, service_name, model_variant, started_at.isoformat(), TrainingRunStatus.RUNNING.value),
                )
                conn.commit()

        logger.info("Started training run %s for %s (%s)", run_id, service_name, model_variant)
        return run_id

    def complete_run(
        self,
        run_id: str,
        validation_results: dict[str, Any],
        metadata: dict[str, Any],
        data_quality: dict[str, Any] | None = None,
    ) -> None:
        """Complete a training run with results.

        Args:
            run_id: The run ID from start_run().
            validation_results: Per-period validation results.
            metadata: Model metadata (time_periods, data_points, etc.).
            data_quality: Optional data quality report.
        """
        completed_at = datetime.now()

        # Calculate validation summary
        periods_passed = 0
        periods_warned = 0
        periods_failed = 0

        for period, result in validation_results.items():
            # Skip the summary entry - it's metadata, not a time period
            if period == "_summary":
                continue
            if isinstance(result, dict):
                if result.get("passed", False) or result.get("enhanced_passed", False):
                    if result.get("anomaly_rate", 0) > 0.10:  # High anomaly rate = warning
                        periods_warned += 1
                    else:
                        periods_passed += 1
                elif result.get("status") == "insufficient_data":
                    periods_warned += 1
                else:
                    periods_failed += 1

        # Determine overall validation status
        if periods_failed > 0:
            validation_status = ValidationStatus.FAILED
        elif periods_warned > 0:
            validation_status = ValidationStatus.WARNING
        else:
            validation_status = ValidationStatus.PASSED

        # Extract metrics from metadata
        time_periods = metadata.get("time_periods", [])

        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Get started_at to calculate duration
                cursor.execute("SELECT started_at FROM training_runs WHERE run_id = ?", (run_id,))
                row = cursor.fetchone()
                if row:
                    started_at = datetime.fromisoformat(row["started_at"])
                    duration_seconds = (completed_at - started_at).total_seconds()
                else:
                    duration_seconds = None

                cursor.execute(
                    """
                    UPDATE training_runs SET
                        completed_at = ?,
                        status = ?,
                        total_data_points = ?,
                        training_start_date = ?,
                        training_end_date = ?,
                        time_periods_trained = ?,
                        feature_count = ?,
                        explainability_metrics = ?,
                        validation_status = ?,
                        periods_passed = ?,
                        periods_warned = ?,
                        periods_failed = ?,
                        validation_details = ?,
                        data_quality_report = ?,
                        model_metadata = ?,
                        duration_seconds = ?
                    WHERE run_id = ?
                    """,
                    (
                        completed_at.isoformat(),
                        TrainingRunStatus.COMPLETED.value,
                        metadata.get("data_points"),
                        metadata.get("training_start"),
                        metadata.get("training_end"),
                        len(time_periods) if time_periods else None,
                        metadata.get("feature_count"),
                        metadata.get("total_explainability_metrics"),
                        validation_status.value,
                        periods_passed,
                        periods_warned,
                        periods_failed,
                        json.dumps(validation_results),
                        json.dumps(data_quality) if data_quality else None,
                        json.dumps(metadata),
                        duration_seconds,
                        run_id,
                    ),
                )
                conn.commit()

        logger.info(
            "Completed training run %s: %s (%d passed, %d warned, %d failed)",
            run_id,
            validation_status.value,
            periods_passed,
            periods_warned,
            periods_failed,
        )

    def fail_run(
        self,
        run_id: str,
        error_message: str,
        error_tb: str | None = None,
    ) -> None:
        """Mark a training run as failed.

        Args:
            run_id: The run ID from start_run().
            error_message: Error message describing the failure.
            error_tb: Optional traceback string.
        """
        completed_at = datetime.now()

        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Get started_at to calculate duration
                cursor.execute("SELECT started_at FROM training_runs WHERE run_id = ?", (run_id,))
                row = cursor.fetchone()
                if row:
                    started_at = datetime.fromisoformat(row["started_at"])
                    duration_seconds = (completed_at - started_at).total_seconds()
                else:
                    duration_seconds = None

                cursor.execute(
                    """
                    UPDATE training_runs SET
                        completed_at = ?,
                        status = ?,
                        validation_status = ?,
                        error_message = ?,
                        error_traceback = ?,
                        duration_seconds = ?
                    WHERE run_id = ?
                    """,
                    (
                        completed_at.isoformat(),
                        TrainingRunStatus.FAILED.value,
                        ValidationStatus.FAILED.value,
                        error_message,
                        error_tb,
                        duration_seconds,
                        run_id,
                    ),
                )
                conn.commit()

        logger.error("Training run %s failed: %s", run_id, error_message)

    def get_run(self, run_id: str) -> TrainingRunRecord | None:
        """Get a specific training run by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM training_runs WHERE run_id = ?", (run_id,))
            row = cursor.fetchone()
            return TrainingRunRecord.from_row(row) if row else None

    def get_runs_for_service(
        self,
        service_name: str,
        limit: int = 20,
    ) -> list[TrainingRunRecord]:
        """Get recent training runs for a service.

        Args:
            service_name: Service name to filter by.
            limit: Maximum number of runs to return.

        Returns:
            List of training runs, most recent first.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM training_runs
                WHERE service_name = ?
                ORDER BY started_at DESC
                LIMIT ?
                """,
                (service_name, limit),
            )
            return [TrainingRunRecord.from_row(row) for row in cursor.fetchall()]

    def get_recent_runs(
        self,
        limit: int = 50,
        status_filter: TrainingRunStatus | None = None,
        validation_filter: ValidationStatus | None = None,
    ) -> list[TrainingRunRecord]:
        """Get recent training runs across all services.

        Args:
            limit: Maximum number of runs to return.
            status_filter: Optional filter by training status.
            validation_filter: Optional filter by validation status.

        Returns:
            List of training runs, most recent first.
        """
        query = "SELECT * FROM training_runs WHERE 1=1"
        params: list[Any] = []

        if status_filter:
            query += " AND status = ?"
            params.append(status_filter.value)

        if validation_filter:
            query += " AND validation_status = ?"
            params.append(validation_filter.value)

        query += " ORDER BY started_at DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [TrainingRunRecord.from_row(row) for row in cursor.fetchall()]

    def get_latest_run_per_service(self) -> dict[str, TrainingRunRecord]:
        """Get the most recent training run for each service.

        Returns:
            Dict mapping service name to latest training run.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM training_runs t1
                WHERE started_at = (
                    SELECT MAX(started_at) FROM training_runs t2
                    WHERE t2.service_name = t1.service_name
                )
                ORDER BY service_name
                """
            )
            return {row["service_name"]: TrainingRunRecord.from_row(row) for row in cursor.fetchall()}

    def get_summary_stats(self) -> dict[str, Any]:
        """Get summary statistics for all training runs.

        Returns:
            Dict with counts by status and validation status.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Count by status
            cursor.execute(
                """
                SELECT status, COUNT(*) as count
                FROM training_runs
                GROUP BY status
                """
            )
            status_counts = {row["status"]: row["count"] for row in cursor.fetchall()}

            # Count by validation status
            cursor.execute(
                """
                SELECT validation_status, COUNT(*) as count
                FROM training_runs
                WHERE validation_status IS NOT NULL
                GROUP BY validation_status
                """
            )
            validation_counts = {row["validation_status"]: row["count"] for row in cursor.fetchall()}

            # Recent failures
            cursor.execute(
                """
                SELECT service_name, error_message, started_at
                FROM training_runs
                WHERE status = 'FAILED'
                ORDER BY started_at DESC
                LIMIT 5
                """
            )
            recent_failures = [
                {"service": row["service_name"], "error": row["error_message"], "at": row["started_at"]}
                for row in cursor.fetchall()
            ]

            # Total runs
            cursor.execute("SELECT COUNT(*) as total FROM training_runs")
            total = cursor.fetchone()["total"]

            return {
                "total_runs": total,
                "by_status": status_counts,
                "by_validation": validation_counts,
                "recent_failures": recent_failures,
            }

    def cleanup_old_runs(self, max_age_days: int = 30) -> int:
        """Remove training runs older than max_age_days.

        Args:
            max_age_days: Remove runs older than this.

        Returns:
            Number of runs deleted.
        """
        from datetime import timedelta
        cutoff = datetime.now() - timedelta(days=max_age_days)

        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM training_runs WHERE started_at < ?",
                    (cutoff.isoformat(),),
                )
                deleted = cursor.rowcount
                conn.commit()

        if deleted > 0:
            logger.info("Cleaned up %d training runs older than %d days", deleted, max_age_days)

        return deleted
