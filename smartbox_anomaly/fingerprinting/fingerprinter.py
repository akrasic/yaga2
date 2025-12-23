"""
Stateful anomaly fingerprinting system with incident lifecycle management.

This module provides incident tracking with two-level identity:
- Fingerprint ID: Content-based, persistent pattern identity
- Incident ID: Unique occurrence instance identity

Features:
- Temporal incident separation (same pattern, different occurrences)
- Model-agnostic fingerprinting (no model name in fingerprint ID)
- Enhanced state tracking and analytics
- Thread-safe database operations
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any

from smartbox_anomaly.core.config import FingerprintingConfig, get_config
from smartbox_anomaly.core.constants import IncidentAction
from smartbox_anomaly.core.exceptions import DatabaseError
from smartbox_anomaly.core.logging import EventType, get_logger, log_event
from smartbox_anomaly.core.protocols import IncidentTracker
from smartbox_anomaly.core.utils import (
    calculate_duration_minutes,
    generate_anomaly_name,
    generate_fingerprint_id,
    generate_incident_id,
    now_iso,
    parse_service_model,
)

logger = get_logger(__name__)


# =============================================================================
# Database Schema
# =============================================================================

SCHEMA_VERSION = "2.1_refactored"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS anomaly_incidents (
    fingerprint_id TEXT NOT NULL,
    incident_id TEXT PRIMARY KEY,
    service_name TEXT NOT NULL,
    anomaly_name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'OPEN',
    severity TEXT NOT NULL,
    first_seen TIMESTAMP NOT NULL,
    last_updated TIMESTAMP NOT NULL,
    resolved_at TIMESTAMP NULL,
    occurrence_count INTEGER NOT NULL DEFAULT 1,
    current_value REAL,
    threshold_value REAL,
    confidence_score REAL,
    detection_method TEXT,
    description TEXT,
    detected_by_model TEXT,
    metadata TEXT,

    CHECK (status IN ('OPEN', 'CLOSED')),
    CHECK (occurrence_count > 0)
)
"""

CREATE_INDEXES_SQL = [
    """CREATE INDEX IF NOT EXISTS idx_fingerprint_status
       ON anomaly_incidents(fingerprint_id, status)""",
    """CREATE INDEX IF NOT EXISTS idx_service_timeline
       ON anomaly_incidents(service_name, first_seen DESC)""",
    """CREATE INDEX IF NOT EXISTS idx_incident_lookup
       ON anomaly_incidents(incident_id)""",
    """CREATE INDEX IF NOT EXISTS idx_open_incidents
       ON anomaly_incidents(status, last_updated DESC)
       WHERE status = 'OPEN'""",
]


# =============================================================================
# Anomaly Fingerprinter
# =============================================================================


class AnomalyFingerprinter(IncidentTracker):
    """Enhanced stateful anomaly fingerprinting system.

    Implements the IncidentTracker protocol for incident lifecycle management.

    Example:
        >>> fingerprinter = AnomalyFingerprinter()
        >>> result = fingerprinter.process_anomalies(
        ...     "booking_evening_hours",
        ...     {"anomalies": [{"type": "threshold", "severity": "high"}]}
        ... )
        >>> print(result["fingerprinting"]["overall_action"])
    """

    def __init__(
        self,
        db_path: str | None = None,
        config: FingerprintingConfig | None = None,
    ) -> None:
        """Initialize the fingerprinter.

        Args:
            db_path: Path to SQLite database. Defaults to config value.
            config: Optional configuration. If not provided, uses global config.
        """
        self._config = config or get_config().fingerprinting
        self.db_path = db_path or self._config.db_path
        self._lock = threading.Lock()
        self._init_database()

    @contextmanager
    def _get_connection(self) -> Iterator[sqlite3.Connection]:
        """Get a database connection with row factory configured."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_database(self) -> None:
        """Initialize SQLite database with schema and indexes."""
        try:
            with self._get_connection() as conn:
                conn.execute(CREATE_TABLE_SQL)
                for index_sql in CREATE_INDEXES_SQL:
                    conn.execute(index_sql)
                self._migrate_legacy_schema(conn)
                conn.commit()
        except sqlite3.Error as e:
            raise DatabaseError("initialization", db_path=self.db_path, cause=e) from e

    def _migrate_legacy_schema(self, conn: sqlite3.Connection) -> None:
        """Migrate from legacy anomaly_state table if it exists."""
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='anomaly_state'"
        )

        if cursor.fetchone():
            logger.info("Migrating legacy anomaly_state table to new schema...")

            conn.execute("""
                INSERT INTO anomaly_incidents (
                    fingerprint_id, incident_id, service_name, anomaly_name,
                    status, severity, first_seen, last_updated, occurrence_count,
                    current_value, threshold_value, confidence_score,
                    detection_method, description, metadata
                )
                SELECT
                    id as fingerprint_id,
                    'incident_' || substr(hex(randomblob(6)), 1, 12) as incident_id,
                    service_name, anomaly_name, 'OPEN' as status, severity,
                    first_seen, last_updated, occurrence_count,
                    current_value, threshold_value, confidence_score,
                    detection_method, description, metadata
                FROM anomaly_state
            """)

            conn.execute("ALTER TABLE anomaly_state RENAME TO anomaly_state_backup")
            logger.info("Legacy data migrated. Backup: anomaly_state_backup")

    # =========================================================================
    # Core Processing
    # =========================================================================

    def process_anomalies(
        self,
        full_service_name: str,
        anomaly_result: dict[str, Any],
        current_metrics: dict[str, Any] | None = None,
        timestamp: datetime | None = None,
    ) -> dict[str, Any]:
        """Process anomaly detection result with incident tracking.

        Args:
            full_service_name: Full service name like "booking_evening_hours".
            anomaly_result: Detection result with anomalies.
            current_metrics: Optional current metric values.
            timestamp: Optional timestamp for this detection.

        Returns:
            Enhanced payload with fingerprinting and incident tracking.
        """
        if timestamp is None:
            timestamp = datetime.now()

        if current_metrics is None:
            current_metrics = anomaly_result.get("current_metrics", {})

        service_name, model_name = parse_service_model(full_service_name)
        current_anomalies = self._normalize_anomalies(anomaly_result.get("anomalies", []))

        with self._lock:
            existing_incidents = self._get_open_incidents_by_service(service_name)
            enhanced_anomalies, processed_fingerprints = self._process_current_anomalies(
                current_anomalies, service_name, model_name, existing_incidents, timestamp
            )
            resolved_incidents = self._process_resolutions(
                existing_incidents, processed_fingerprints, timestamp
            )

            return self._build_enhanced_payload(
                anomaly_result,
                enhanced_anomalies,
                resolved_incidents,
                service_name,
                model_name,
                timestamp,
            )

    def _normalize_anomalies(self, anomalies: Any) -> list[dict[str, Any]]:
        """Normalize anomalies to list format."""
        if isinstance(anomalies, dict):
            return list(anomalies.values())
        if isinstance(anomalies, list):
            return [a for a in anomalies if isinstance(a, dict)]
        return []

    def _process_current_anomalies(
        self,
        anomalies: list[dict[str, Any]],
        service_name: str,
        model_name: str,
        existing_incidents: dict[str, dict],
        timestamp: datetime,
    ) -> tuple[list[dict[str, Any]], set[str]]:
        """Process currently detected anomalies."""
        enhanced = []
        processed_fps = set()

        for i, anomaly_data in enumerate(anomalies):
            anomaly_name = generate_anomaly_name(anomaly_data, i)
            fp_id = generate_fingerprint_id(service_name, anomaly_name)
            processed_fps.add(fp_id)

            existing = existing_incidents.get(fp_id)
            details = self._extract_anomaly_details(anomaly_name, anomaly_data, model_name)

            enhanced_anomaly = anomaly_data.copy()

            if existing:
                enhanced_anomaly.update(
                    self._continue_incident(existing, details, fp_id, anomaly_name, model_name, timestamp)
                )
            else:
                enhanced_anomaly.update(
                    self._create_incident(service_name, anomaly_name, details, fp_id, model_name, timestamp)
                )

            enhanced.append(enhanced_anomaly)

        return enhanced, processed_fps

    def _create_incident(
        self,
        service_name: str,
        anomaly_name: str,
        details: dict[str, Any],
        fp_id: str,
        model_name: str,
        timestamp: datetime,
    ) -> dict[str, Any]:
        """Create a new incident."""
        incident_id = generate_incident_id()

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO anomaly_incidents
                (fingerprint_id, incident_id, service_name, anomaly_name, status,
                 severity, first_seen, last_updated, occurrence_count, current_value,
                 threshold_value, confidence_score, detection_method, description,
                 detected_by_model, metadata)
                VALUES (?, ?, ?, ?, 'OPEN', ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fp_id, incident_id, service_name, anomaly_name,
                    details["severity"], timestamp, timestamp,
                    details["current_value"], details["threshold_value"],
                    details["confidence_score"], details["detection_method"],
                    details["description"], details["detected_by_model"],
                    details["metadata"],
                ),
            )
            conn.commit()

        log_event(logger, 20, EventType.INCIDENT_CREATED, service_name,
                  f"New incident {incident_id}", fingerprint=fp_id)

        return {
            "fingerprint_id": fp_id,
            "incident_id": incident_id,
            "anomaly_name": anomaly_name,
            "fingerprint_action": IncidentAction.CREATE.value,
            "incident_action": IncidentAction.CREATE.value,
            "occurrence_count": 1,
            "first_seen": timestamp.isoformat(),
            "last_updated": timestamp.isoformat(),
            "incident_duration_minutes": 0,
            "detected_by_model": model_name,
        }

    def _continue_incident(
        self,
        existing: dict[str, Any],
        details: dict[str, Any],
        fp_id: str,
        anomaly_name: str,
        model_name: str,
        timestamp: datetime,
    ) -> dict[str, Any]:
        """Continue an existing incident."""
        new_count = existing["occurrence_count"] + 1
        severity_changed = existing["severity"] != details["severity"]

        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE anomaly_incidents
                SET severity = ?, last_updated = ?, occurrence_count = ?,
                    current_value = ?, threshold_value = ?, confidence_score = ?,
                    description = ?, detected_by_model = ?, metadata = ?
                WHERE incident_id = ?
                """,
                (
                    details["severity"], timestamp, new_count,
                    details["current_value"], details["threshold_value"],
                    details["confidence_score"], details["description"],
                    details["detected_by_model"], details["metadata"],
                    existing["incident_id"],
                ),
            )
            conn.commit()

        log_event(logger, 20, EventType.INCIDENT_CONTINUED, existing["service_name"],
                  f"Updated incident {existing['incident_id']}", count=new_count)

        result = {
            "fingerprint_id": fp_id,
            "incident_id": existing["incident_id"],
            "anomaly_name": anomaly_name,
            "fingerprint_action": IncidentAction.UPDATE.value,
            "incident_action": IncidentAction.CONTINUE.value,
            "occurrence_count": new_count,
            "first_seen": existing["first_seen"],
            "last_updated": timestamp.isoformat(),
            "incident_duration_minutes": calculate_duration_minutes(existing["first_seen"], timestamp),
            "detected_by_model": model_name,
        }

        if severity_changed:
            result.update({
                "severity_changed": True,
                "previous_severity": existing["severity"],
                "severity_changed_at": timestamp.isoformat(),
            })

        return result

    def _process_resolutions(
        self,
        existing_incidents: dict[str, dict],
        processed_fps: set[str],
        timestamp: datetime,
    ) -> list[dict[str, Any]]:
        """Process resolved incidents (no longer detected)."""
        resolved = []

        for fp_id, incident in existing_incidents.items():
            if fp_id not in processed_fps:
                duration = calculate_duration_minutes(incident["first_seen"], timestamp)
                self._close_incident(incident["incident_id"], timestamp)

                log_event(logger, 20, EventType.INCIDENT_RESOLVED, incident["service_name"],
                          f"Resolved {incident['incident_id']}", duration_min=duration)

                resolved.append({
                    "fingerprint_id": fp_id,
                    "incident_id": incident["incident_id"],
                    "anomaly_name": incident["anomaly_name"],
                    "fingerprint_action": IncidentAction.RESOLVE.value,
                    "incident_action": IncidentAction.CLOSE.value,
                    "final_severity": incident["severity"],
                    "resolved_at": timestamp.isoformat(),
                    "total_occurrences": incident["occurrence_count"],
                    "incident_duration_minutes": duration,
                    "first_seen": incident["first_seen"],
                    "service_name": incident["service_name"],
                    "last_detected_by_model": incident.get("detected_by_model", "unknown"),
                })

        return resolved

    def _close_incident(self, incident_id: str, timestamp: datetime) -> None:
        """Close an incident in the database."""
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE anomaly_incidents SET status = 'CLOSED', resolved_at = ? WHERE incident_id = ?",
                (timestamp, incident_id),
            )
            conn.commit()

    def _build_enhanced_payload(
        self,
        original: dict[str, Any],
        anomalies: list[dict[str, Any]],
        resolved: list[dict[str, Any]],
        service_name: str,
        model_name: str,
        timestamp: datetime,
    ) -> dict[str, Any]:
        """Build the enhanced result payload."""
        payload = original.copy()
        payload["anomalies"] = anomalies

        creates = sum(1 for a in anomalies if a.get("incident_action") == IncidentAction.CREATE.value)
        continues = sum(1 for a in anomalies if a.get("incident_action") == IncidentAction.CONTINUE.value)

        payload["fingerprinting"] = {
            "service_name": service_name,
            "model_name": model_name,
            "timestamp": timestamp.isoformat(),
            "action_summary": {
                "incident_creates": creates,
                "incident_continues": continues,
                "incident_closes": len(resolved),
            },
            "overall_action": self._determine_overall_action(anomalies, resolved),
            "resolved_incidents": resolved,
            "total_open_incidents": len(anomalies),
            "detection_context": {
                "model_used": model_name,
                "inference_timestamp": timestamp.isoformat(),
            },
        }

        payload["service_name"] = service_name
        payload["model_name"] = model_name

        return payload

    def _determine_overall_action(
        self, anomalies: list[dict], resolved: list[dict]
    ) -> str:
        """Determine the overall action for the payload."""
        total = len(anomalies) + len(resolved)
        if total == 0:
            return IncidentAction.NO_CHANGE.value

        creates = sum(1 for a in anomalies if a.get("incident_action") == IncidentAction.CREATE.value)
        continues = sum(1 for a in anomalies if a.get("incident_action") == IncidentAction.CONTINUE.value)

        if total == 1:
            if creates:
                return IncidentAction.CREATE.value
            if continues:
                return IncidentAction.UPDATE.value
            if resolved:
                return IncidentAction.RESOLVE.value

        return IncidentAction.MIXED.value

    def _extract_anomaly_details(
        self, anomaly_name: str, data: dict[str, Any], model: str
    ) -> dict[str, Any]:
        """Extract details from anomaly data."""
        return {
            "severity": data.get("severity", "medium"),
            "current_value": data.get("value", data.get("actual_value")),
            "threshold_value": data.get("threshold", data.get("threshold_value")),
            "confidence_score": data.get("score", data.get("confidence_score")),
            "detection_method": data.get("detection_method", "unknown"),
            "description": data.get("description", f"Anomaly detected: {anomaly_name}"),
            "detected_by_model": model,
            "metadata": json.dumps({
                "type": data.get("type"),
                "business_impact": data.get("business_impact"),
                "feature_contributions": data.get("feature_contributions"),
                "comparison_data": data.get("comparison_data"),
                "detection_context": {"model": model, "timestamp": now_iso()},
            }),
        }

    # =========================================================================
    # Query Methods
    # =========================================================================

    def _get_open_incidents_by_service(self, service_name: str) -> dict[str, dict]:
        """Get all open incidents for a service, keyed by fingerprint_id."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """SELECT * FROM anomaly_incidents
                   WHERE service_name = ? AND status = 'OPEN'
                   ORDER BY first_seen DESC""",
                (service_name,),
            )
            return {row["fingerprint_id"]: dict(row) for row in cursor.fetchall()}

    def get_open_incidents(
        self, service_name: str | None = None
    ) -> dict[str, list[dict]]:
        """Get all open incidents, optionally filtered by service."""
        with self._get_connection() as conn:
            if service_name:
                cursor = conn.execute(
                    """SELECT * FROM anomaly_incidents
                       WHERE service_name = ? AND status = 'OPEN'
                       ORDER BY first_seen DESC""",
                    (service_name,),
                )
            else:
                cursor = conn.execute(
                    """SELECT * FROM anomaly_incidents
                       WHERE status = 'OPEN'
                       ORDER BY service_name, first_seen DESC"""
                )

            result: dict[str, list[dict]] = {}
            for row in cursor.fetchall():
                svc = row["service_name"]
                if svc not in result:
                    result[svc] = []
                result[svc].append(dict(row))
            return result

    def get_incident_by_id(self, incident_id: str) -> dict[str, Any] | None:
        """Get a specific incident by ID."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM anomaly_incidents WHERE incident_id = ?",
                (incident_id,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_service_incidents(
        self,
        service_name: str,
        include_closed: bool = False,
        limit: int = 100,
    ) -> list[dict]:
        """Get incidents for a service."""
        with self._get_connection() as conn:
            if include_closed:
                cursor = conn.execute(
                    """SELECT * FROM anomaly_incidents
                       WHERE service_name = ?
                       ORDER BY first_seen DESC LIMIT ?""",
                    (service_name, limit),
                )
            else:
                cursor = conn.execute(
                    """SELECT * FROM anomaly_incidents
                       WHERE service_name = ? AND status = 'OPEN'
                       ORDER BY first_seen DESC LIMIT ?""",
                    (service_name, limit),
                )
            return [dict(row) for row in cursor.fetchall()]

    def get_pattern_history(self, fingerprint_id: str, limit: int = 50) -> list[dict]:
        """Get incident history for a specific anomaly pattern."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """SELECT * FROM anomaly_incidents
                   WHERE fingerprint_id = ?
                   ORDER BY first_seen DESC LIMIT ?""",
                (fingerprint_id, limit),
            )
            return [dict(row) for row in cursor.fetchall()]

    # =========================================================================
    # Statistics & Analytics
    # =========================================================================

    def get_statistics(self) -> dict[str, Any]:
        """Get system statistics with incident analytics."""
        with self._get_connection() as conn:
            total_open = conn.execute(
                "SELECT COUNT(*) as c FROM anomaly_incidents WHERE status = 'OPEN'"
            ).fetchone()["c"]

            total_all = conn.execute(
                "SELECT COUNT(*) as c FROM anomaly_incidents"
            ).fetchone()["c"]

            by_service = [
                dict(r) for r in conn.execute(
                    """SELECT service_name, COUNT(*) as count
                       FROM anomaly_incidents WHERE status = 'OPEN'
                       GROUP BY service_name ORDER BY count DESC"""
                ).fetchall()
            ]

            by_severity = {
                r["severity"]: r["count"]
                for r in conn.execute(
                    """SELECT severity, COUNT(*) as count
                       FROM anomaly_incidents WHERE status = 'OPEN'
                       GROUP BY severity"""
                ).fetchall()
            }

            frequent = [
                dict(r) for r in conn.execute(
                    """SELECT fingerprint_id, anomaly_name, COUNT(*) as occurrences,
                       AVG(CASE WHEN status = 'CLOSED' THEN
                           (julianday(resolved_at) - julianday(first_seen)) * 24 * 60
                       ELSE NULL END) as avg_duration_minutes
                       FROM anomaly_incidents
                       GROUP BY fingerprint_id, anomaly_name
                       HAVING occurrences > 1
                       ORDER BY occurrences DESC LIMIT 10"""
                ).fetchall()
            ]

            longest = conn.execute(
                """SELECT service_name, anomaly_name, incident_id, first_seen,
                   (julianday('now') - julianday(first_seen)) * 24 * 60 as duration_minutes
                   FROM anomaly_incidents WHERE status = 'OPEN'
                   ORDER BY first_seen ASC LIMIT 1"""
            ).fetchone()

            return {
                "total_open_incidents": total_open,
                "total_all_incidents": total_all,
                "total_closed_incidents": total_all - total_open,
                "open_incidents_by_service": by_service,
                "open_incidents_by_severity": by_severity,
                "frequent_patterns": frequent,
                "longest_running_incident": dict(longest) if longest else None,
                "database_path": self.db_path,
                "schema_version": SCHEMA_VERSION,
            }

    def get_analytics_summary(self, days: int = 7) -> dict[str, Any]:
        """Get analytics summary for the last N days."""
        start = (datetime.now() - timedelta(days=days)).isoformat()

        with self._get_connection() as conn:
            created = conn.execute(
                "SELECT COUNT(*) as c FROM anomaly_incidents WHERE first_seen >= ?",
                (start,),
            ).fetchone()["c"]

            resolved = conn.execute(
                """SELECT COUNT(*) as c FROM anomaly_incidents
                   WHERE resolved_at >= ? AND status = 'CLOSED'""",
                (start,),
            ).fetchone()["c"]

            avg_duration = conn.execute(
                """SELECT AVG((julianday(resolved_at) - julianday(first_seen)) * 24 * 60)
                   FROM anomaly_incidents
                   WHERE resolved_at >= ? AND status = 'CLOSED'""",
                (start,),
            ).fetchone()[0] or 0

            return {
                "period_days": days,
                "incidents_created": created,
                "incidents_resolved": resolved,
                "average_resolution_minutes": round(avg_duration, 2),
                "net_incident_change": created - resolved,
            }

    # =========================================================================
    # Maintenance
    # =========================================================================

    def cleanup_old_incidents(
        self,
        max_age_hours: int = 72,
        status: str = "CLOSED",
    ) -> int:
        """Clean up old incidents."""
        cutoff = (datetime.now() - timedelta(hours=max_age_hours)).isoformat()

        with self._get_connection() as conn:
            if status == "CLOSED":
                cursor = conn.execute(
                    "DELETE FROM anomaly_incidents WHERE status = 'CLOSED' AND resolved_at < ?",
                    (cutoff,),
                )
            else:
                cursor = conn.execute(
                    "DELETE FROM anomaly_incidents WHERE last_updated < ?",
                    (cutoff,),
                )
            deleted = cursor.rowcount
            conn.commit()

        if deleted > 0:
            logger.info(f"Cleaned up {deleted} old incidents (status: {status}, age > {max_age_hours}h)")

        return deleted


# =============================================================================
# Factory Function
# =============================================================================


def create_fingerprinter(db_path: str | None = None) -> AnomalyFingerprinter:
    """Create a ready-to-use anomaly fingerprinter.

    Args:
        db_path: Optional database path. Defaults to config value.

    Returns:
        Configured AnomalyFingerprinter instance.
    """
    return AnomalyFingerprinter(db_path=db_path)
