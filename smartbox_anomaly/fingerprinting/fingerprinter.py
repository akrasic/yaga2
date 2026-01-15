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
import math
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from smartbox_anomaly.slo.evaluator import SLOEvaluator

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

SCHEMA_VERSION = "2.2_cycle_based"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS anomaly_incidents (
    fingerprint_id TEXT NOT NULL,
    incident_id TEXT PRIMARY KEY,
    service_name TEXT NOT NULL,
    anomaly_name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'SUSPECTED',
    severity TEXT NOT NULL,
    first_seen TIMESTAMP NOT NULL,
    last_updated TIMESTAMP NOT NULL,
    resolved_at TIMESTAMP NULL,
    occurrence_count INTEGER NOT NULL DEFAULT 1,
    consecutive_detections INTEGER NOT NULL DEFAULT 1,
    missed_cycles INTEGER NOT NULL DEFAULT 0,
    current_value REAL,
    threshold_value REAL,
    confidence_score REAL,
    detection_method TEXT,
    description TEXT,
    detected_by_model TEXT,
    metadata TEXT,

    CHECK (status IN ('SUSPECTED', 'OPEN', 'RECOVERING', 'CLOSED')),
    CHECK (occurrence_count > 0),
    CHECK (consecutive_detections >= 0),
    CHECK (missed_cycles >= 0)
)
"""

CREATE_INDEXES_SQL = [
    """CREATE INDEX IF NOT EXISTS idx_fingerprint_status
       ON anomaly_incidents(fingerprint_id, status)""",
    """CREATE INDEX IF NOT EXISTS idx_service_timeline
       ON anomaly_incidents(service_name, first_seen DESC)""",
    """CREATE INDEX IF NOT EXISTS idx_incident_lookup
       ON anomaly_incidents(incident_id)""",
    """CREATE INDEX IF NOT EXISTS idx_active_incidents
       ON anomaly_incidents(status, last_updated DESC)
       WHERE status IN ('SUSPECTED', 'OPEN', 'RECOVERING')""",
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
        """Migrate from legacy schemas to current schema."""
        # Check for very old anomaly_state table
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='anomaly_state'"
        )

        if cursor.fetchone():
            logger.info("Migrating legacy anomaly_state table to new schema...")

            conn.execute("""
                INSERT INTO anomaly_incidents (
                    fingerprint_id, incident_id, service_name, anomaly_name,
                    status, severity, first_seen, last_updated, occurrence_count,
                    consecutive_detections, missed_cycles,
                    current_value, threshold_value, confidence_score,
                    detection_method, description, metadata
                )
                SELECT
                    id as fingerprint_id,
                    'incident_' || substr(hex(randomblob(6)), 1, 12) as incident_id,
                    service_name, anomaly_name, 'OPEN' as status, severity,
                    first_seen, last_updated, occurrence_count,
                    occurrence_count as consecutive_detections, 0 as missed_cycles,
                    current_value, threshold_value, confidence_score,
                    detection_method, description, metadata
                FROM anomaly_state
            """)

            conn.execute("ALTER TABLE anomaly_state RENAME TO anomaly_state_backup")
            logger.info("Legacy data migrated. Backup: anomaly_state_backup")

        # Migrate from schema 2.1 to 2.2 (add cycle tracking columns)
        self._migrate_to_cycle_based_schema(conn)

    def _migrate_to_cycle_based_schema(self, conn: sqlite3.Connection) -> None:
        """Migrate existing anomaly_incidents to cycle-based schema (2.1 → 2.2)."""
        # Check if we need to add the new columns
        cursor = conn.execute("PRAGMA table_info(anomaly_incidents)")
        columns = {row[1] for row in cursor.fetchall()}

        migrations_needed = []

        if "consecutive_detections" not in columns:
            migrations_needed.append(
                "ALTER TABLE anomaly_incidents ADD COLUMN consecutive_detections INTEGER NOT NULL DEFAULT 1"
            )

        if "missed_cycles" not in columns:
            migrations_needed.append(
                "ALTER TABLE anomaly_incidents ADD COLUMN missed_cycles INTEGER NOT NULL DEFAULT 0"
            )

        if migrations_needed:
            logger.info("Migrating to cycle-based schema (v2.2)...")

            for sql in migrations_needed:
                try:
                    conn.execute(sql)
                except sqlite3.OperationalError as e:
                    # Column might already exist
                    if "duplicate column" not in str(e).lower():
                        raise

            # Update existing OPEN incidents to have proper cycle counts
            # Assume existing OPEN incidents are confirmed (set consecutive_detections = confirmation threshold)
            conn.execute("""
                UPDATE anomaly_incidents
                SET consecutive_detections = CASE
                    WHEN status = 'OPEN' THEN occurrence_count
                    ELSE 1
                END,
                missed_cycles = 0
                WHERE consecutive_detections = 1 AND occurrence_count > 1
            """)

            # Migrate status values: old 'OPEN' stays 'OPEN', old 'CLOSED' stays 'CLOSED'
            # New incidents will use SUSPECTED/RECOVERING states
            logger.info("Cycle-based schema migration complete")

    # =========================================================================
    # Core Processing
    # =========================================================================

    def process_anomalies(
        self,
        full_service_name: str,
        anomaly_result: dict[str, Any],
        current_metrics: dict[str, Any] | None = None,
        timestamp: datetime | None = None,
        slo_evaluator: "SLOEvaluator | None" = None,
        training_statistics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Process anomaly detection result with cycle-based incident tracking.

        Implements a state machine for incident lifecycle:
        - SUSPECTED: Initial detection, waiting for confirmation (N cycles)
        - OPEN: Confirmed incident, alerts are sent
        - RECOVERING: Not detected for 1-2 cycles, may resolve soon
        - CLOSED: Not detected for N cycles, incident resolved

        Args:
            full_service_name: Full service name like "booking_evening_hours".
            anomaly_result: Detection result with anomalies.
            current_metrics: Optional current metric values.
            timestamp: Optional timestamp for this detection.
            slo_evaluator: Optional SLO evaluator for building resolution context.
            training_statistics: Optional training statistics for baseline comparison.

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
            # Get all active incidents (SUSPECTED, OPEN, RECOVERING)
            active_incidents = self._get_active_incidents_by_service(service_name)

            # Process current anomalies - may create, continue, or confirm incidents
            # Also returns stale_resolutions for incidents that were auto-closed due to time gap
            enhanced_anomalies, processed_fingerprints, newly_confirmed, stale_resolutions = (
                self._process_current_anomalies(
                    current_anomalies, service_name, model_name, active_incidents, timestamp
                )
            )

            # Process resolutions for incidents not seen this cycle
            # Pass metrics and SLO evaluator for building resolution context
            resolved_incidents = self._process_resolutions(
                active_incidents,
                processed_fingerprints,
                timestamp,
                current_metrics=current_metrics,
                slo_evaluator=slo_evaluator,
                training_statistics=training_statistics,
                time_period=model_name,  # model_name is the time period
            )

            # Merge stale resolutions into resolved_incidents so they get sent to the API
            all_resolved = stale_resolutions + resolved_incidents

            return self._build_enhanced_payload(
                anomaly_result,
                enhanced_anomalies,
                all_resolved,
                service_name,
                model_name,
                timestamp,
                newly_confirmed=newly_confirmed,
            )

    def _normalize_anomalies(self, anomalies: Any) -> list[dict[str, Any]]:
        """Normalize anomalies to list format, preserving anomaly names.

        When anomalies come as a dict (key=anomaly_name, value=anomaly_data),
        the key is preserved as '_anomaly_key' in the data so it can be used
        for naming instead of re-deriving from description heuristics.
        """
        if isinstance(anomalies, dict):
            # Preserve the key (anomaly name) in the data
            return [
                {**data, "_anomaly_key": name}
                for name, data in anomalies.items()
            ]
        if isinstance(anomalies, list):
            return [a for a in anomalies if isinstance(a, dict)]
        return []

    def _process_current_anomalies(
        self,
        anomalies: list[dict[str, Any]],
        service_name: str,
        model_name: str,
        active_incidents: dict[str, dict],
        timestamp: datetime,
    ) -> tuple[list[dict[str, Any]], set[str], list[dict[str, Any]], list[dict[str, Any]]]:
        """Process currently detected anomalies with cycle-based state machine.

        Returns:
            Tuple of (enhanced_anomalies, processed_fingerprints, newly_confirmed_incidents, stale_resolutions)
        """
        enhanced = []
        processed_fps = set()
        newly_confirmed = []  # Incidents that just transitioned to OPEN
        stale_resolutions = []  # Stale incidents that were auto-closed

        for i, anomaly_data in enumerate(anomalies):
            anomaly_name = generate_anomaly_name(anomaly_data, i)
            fp_id = generate_fingerprint_id(service_name, anomaly_name)
            processed_fps.add(fp_id)

            existing = active_incidents.get(fp_id)
            details = self._extract_anomaly_details(anomaly_name, anomaly_data, model_name)

            enhanced_anomaly = anomaly_data.copy()

            if existing:
                # Check if incident is stale (time gap exceeds separation threshold)
                if self._is_incident_stale(existing, timestamp):
                    # Close stale incident and create new one
                    resolution = self._close_stale_incident(existing, timestamp)
                    stale_resolutions.append(resolution)
                    result = self._create_incident(
                        service_name, anomaly_name, details, fp_id, model_name, timestamp
                    )
                    enhanced_anomaly.update(result)
                else:
                    # Continue/confirm existing incident
                    result, is_newly_confirmed = self._continue_incident(
                        existing, details, fp_id, anomaly_name, model_name, timestamp
                    )
                    enhanced_anomaly.update(result)
                    if is_newly_confirmed:
                        newly_confirmed.append(result)
            else:
                # Check if there's a recently-closed incident we should reopen
                recently_closed = self._get_recently_closed_incident(
                    fp_id, timestamp, self._config.incident_separation_minutes
                )
                if recently_closed:
                    # Reopen the recently-closed incident instead of creating new one
                    result = self._reopen_incident(
                        recently_closed, details, fp_id, anomaly_name, model_name, timestamp
                    )
                    enhanced_anomaly.update(result)
                else:
                    # Create new suspected incident
                    enhanced_anomaly.update(
                        self._create_incident(service_name, anomaly_name, details, fp_id, model_name, timestamp)
                    )

            enhanced.append(enhanced_anomaly)

        return enhanced, processed_fps, newly_confirmed, stale_resolutions

    def _is_incident_stale(self, incident: dict[str, Any], current_time: datetime) -> bool:
        """Check if an incident is stale based on time gap."""
        last_updated = incident.get("last_updated")
        if not last_updated:
            return False

        minutes_since_update = calculate_duration_minutes(last_updated, current_time)
        return minutes_since_update > self._config.incident_separation_minutes

    def _close_stale_incident(
        self, incident: dict[str, Any], timestamp: datetime
    ) -> dict[str, Any]:
        """Close a stale incident with auto-resolution reason.

        Returns resolution data so it can be sent to the API.
        """
        duration = calculate_duration_minutes(incident["first_seen"], timestamp)

        with self._get_connection() as conn:
            conn.execute(
                """UPDATE anomaly_incidents
                   SET status = 'CLOSED', resolved_at = ?,
                       metadata = json_set(COALESCE(metadata, '{}'), '$.resolution_reason', 'auto_stale')
                   WHERE incident_id = ?""",
                (timestamp, incident["incident_id"]),
            )
            conn.commit()

        log_event(
            logger, 20, EventType.INCIDENT_RESOLVED, incident["service_name"],
            f"Auto-closed stale incident {incident['incident_id']} (gap > {self._config.incident_separation_minutes}min)",
            fingerprint=incident["fingerprint_id"],
        )

        # Return resolution data for API notification
        return {
            "fingerprint_id": incident["fingerprint_id"],
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
            "resolution_reason": "auto_stale",
        }

    def _get_recently_closed_incident(
        self,
        fingerprint_id: str,
        current_time: datetime,
        max_age_minutes: int,
    ) -> dict[str, Any] | None:
        """Find a recently-closed incident with the same fingerprint.

        This allows reopening incidents that were briefly resolved but the anomaly
        reappeared within the incident_separation_minutes threshold.

        Args:
            fingerprint_id: The fingerprint ID to search for.
            current_time: Current timestamp.
            max_age_minutes: Maximum age (in minutes) since resolution to consider.

        Returns:
            The recently-closed incident dict if found, None otherwise.
        """
        cutoff = current_time - timedelta(minutes=max_age_minutes)

        with self._get_connection() as conn:
            cursor = conn.execute(
                """SELECT * FROM anomaly_incidents
                   WHERE fingerprint_id = ?
                     AND status = 'CLOSED'
                     AND resolved_at >= ?
                   ORDER BY resolved_at DESC
                   LIMIT 1""",
                (fingerprint_id, cutoff),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def _reopen_incident(
        self,
        closed_incident: dict[str, Any],
        details: dict[str, Any],
        fp_id: str,
        anomaly_name: str,
        model_name: str,
        timestamp: datetime,
    ) -> dict[str, Any]:
        """Reopen a recently-closed incident.

        Instead of creating a new incident, we reopen the existing one,
        preserving incident history and avoiding fragmentation.

        The incident transitions back to OPEN status (not SUSPECTED) since
        it was previously confirmed.
        """
        new_count = closed_incident["occurrence_count"] + 1

        with self._get_connection() as conn:
            conn.execute(
                """UPDATE anomaly_incidents
                   SET status = 'OPEN',
                       resolved_at = NULL,
                       last_updated = ?,
                       occurrence_count = ?,
                       consecutive_detections = 1,
                       missed_cycles = 0,
                       severity = ?,
                       current_value = ?,
                       threshold_value = ?,
                       confidence_score = ?,
                       description = ?,
                       detected_by_model = ?,
                       metadata = json_set(COALESCE(metadata, '{}'), '$.reopened_at', ?)
                   WHERE incident_id = ?""",
                (
                    timestamp,
                    new_count,
                    details["severity"],
                    details["current_value"],
                    details["threshold_value"],
                    details["confidence_score"],
                    details["description"],
                    details["detected_by_model"],
                    timestamp.isoformat(),
                    closed_incident["incident_id"],
                ),
            )
            conn.commit()

        log_event(
            logger, 20, EventType.INCIDENT_CONTINUED, closed_incident["service_name"],
            f"Reopened incident {closed_incident['incident_id']} (was closed {calculate_duration_minutes(closed_incident['resolved_at'], timestamp):.0f} min ago)",
            fingerprint=fp_id,
        )

        return {
            "fingerprint_id": fp_id,
            "incident_id": closed_incident["incident_id"],
            "anomaly_name": anomaly_name,
            "status": "OPEN",
            "previous_status": "CLOSED",
            "fingerprint_action": IncidentAction.UPDATE.value,
            "incident_action": "REOPEN",  # New action type for tracking
            "occurrence_count": new_count,
            "consecutive_detections": 1,
            "first_seen": closed_incident["first_seen"],
            "last_updated": timestamp.isoformat(),
            "incident_duration_minutes": calculate_duration_minutes(
                closed_incident["first_seen"], timestamp
            ),
            "detected_by_model": model_name,
            "is_confirmed": True,  # Was previously confirmed
            "reopened": True,
            "time_since_closure_minutes": calculate_duration_minutes(
                closed_incident["resolved_at"], timestamp
            ),
        }

    def _create_incident(
        self,
        service_name: str,
        anomaly_name: str,
        details: dict[str, Any],
        fp_id: str,
        model_name: str,
        timestamp: datetime,
    ) -> dict[str, Any]:
        """Create a new incident in SUSPECTED state.

        Incidents start as SUSPECTED and transition to OPEN after
        confirmation_cycles consecutive detections.
        """
        incident_id = generate_incident_id()

        # Start in SUSPECTED state - will transition to OPEN after confirmation
        initial_status = "SUSPECTED"

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO anomaly_incidents
                (fingerprint_id, incident_id, service_name, anomaly_name, status,
                 severity, first_seen, last_updated, occurrence_count,
                 consecutive_detections, missed_cycles, current_value,
                 threshold_value, confidence_score, detection_method, description,
                 detected_by_model, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, 1, 0, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fp_id, incident_id, service_name, anomaly_name, initial_status,
                    details["severity"], timestamp, timestamp,
                    details["current_value"], details["threshold_value"],
                    details["confidence_score"], details["detection_method"],
                    details["description"], details["detected_by_model"],
                    details["metadata"],
                ),
            )
            conn.commit()

        log_event(logger, 20, EventType.INCIDENT_CREATED, service_name,
                  f"New suspected incident {incident_id} (awaiting confirmation)", fingerprint=fp_id)

        return {
            "fingerprint_id": fp_id,
            "incident_id": incident_id,
            "anomaly_name": anomaly_name,
            "status": initial_status,
            "fingerprint_action": IncidentAction.CREATE.value,
            "incident_action": IncidentAction.CREATE.value,
            "occurrence_count": 1,
            "consecutive_detections": 1,
            "confirmation_pending": True,  # Signal that this needs more cycles to confirm
            "cycles_to_confirm": self._config.confirmation_cycles - 1,
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
    ) -> tuple[dict[str, Any], bool]:
        """Continue an existing incident with state machine transitions.

        State transitions:
        - SUSPECTED + detected → increment consecutive_detections
          - If consecutive_detections >= confirmation_cycles → OPEN (newly confirmed!)
        - RECOVERING + detected → OPEN (resume, reset missed_cycles)
        - OPEN + detected → OPEN (continue, reset missed_cycles)

        Returns:
            Tuple of (result_dict, is_newly_confirmed)
        """
        current_status = existing.get("status", "OPEN")
        new_count = existing["occurrence_count"] + 1
        consecutive = existing.get("consecutive_detections", 1) + 1
        severity_changed = existing["severity"] != details["severity"]

        # Determine new status based on state machine
        is_newly_confirmed = False
        new_status = current_status

        if current_status == "SUSPECTED":
            # Check if we've reached confirmation threshold
            if consecutive >= self._config.confirmation_cycles:
                new_status = "OPEN"
                is_newly_confirmed = True
                log_event(
                    logger, 20, EventType.INCIDENT_CREATED, existing["service_name"],
                    f"Incident {existing['incident_id']} CONFIRMED after {consecutive} cycles",
                    fingerprint=fp_id,
                )
            else:
                log_event(
                    logger, 10, EventType.INCIDENT_CONTINUED, existing["service_name"],
                    f"Suspected incident {existing['incident_id']} ({consecutive}/{self._config.confirmation_cycles} cycles)",
                    fingerprint=fp_id,
                )
        elif current_status == "RECOVERING":
            # Anomaly reappeared - back to OPEN
            new_status = "OPEN"
            log_event(
                logger, 20, EventType.INCIDENT_CONTINUED, existing["service_name"],
                f"Incident {existing['incident_id']} resumed from RECOVERING",
                fingerprint=fp_id,
            )
        else:
            # Already OPEN, just continue
            log_event(
                logger, 20, EventType.INCIDENT_CONTINUED, existing["service_name"],
                f"Updated incident {existing['incident_id']}", count=new_count,
            )

        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE anomaly_incidents
                SET status = ?, severity = ?, last_updated = ?, occurrence_count = ?,
                    consecutive_detections = ?, missed_cycles = 0,
                    current_value = ?, threshold_value = ?, confidence_score = ?,
                    description = ?, detected_by_model = ?, metadata = ?
                WHERE incident_id = ?
                """,
                (
                    new_status, details["severity"], timestamp, new_count,
                    consecutive,
                    details["current_value"], details["threshold_value"],
                    details["confidence_score"], details["description"],
                    details["detected_by_model"], details["metadata"],
                    existing["incident_id"],
                ),
            )
            conn.commit()

        result = {
            "fingerprint_id": fp_id,
            "incident_id": existing["incident_id"],
            "anomaly_name": anomaly_name,
            "status": new_status,
            "previous_status": current_status,
            "fingerprint_action": IncidentAction.UPDATE.value,
            "incident_action": IncidentAction.CONTINUE.value,
            "occurrence_count": new_count,
            "consecutive_detections": consecutive,
            "first_seen": existing["first_seen"],
            "last_updated": timestamp.isoformat(),
            "incident_duration_minutes": calculate_duration_minutes(existing["first_seen"], timestamp),
            "detected_by_model": model_name,
            "is_confirmed": new_status == "OPEN",
            "newly_confirmed": is_newly_confirmed,
        }

        if is_newly_confirmed:
            result["confirmation_pending"] = False

        if severity_changed:
            result.update({
                "severity_changed": True,
                "previous_severity": existing["severity"],
                "severity_changed_at": timestamp.isoformat(),
            })

        return result, is_newly_confirmed

    def _process_resolutions(
        self,
        active_incidents: dict[str, dict],
        processed_fps: set[str],
        timestamp: datetime,
        current_metrics: dict[str, Any] | None = None,
        slo_evaluator: "SLOEvaluator | None" = None,
        training_statistics: dict[str, Any] | None = None,
        time_period: str = "business_hours",
    ) -> list[dict[str, Any]]:
        """Process resolution grace period for incidents not detected this cycle.

        State transitions for undetected incidents:
        - SUSPECTED + not detected → increment missed_cycles
          - If missed_cycles >= grace → silently close (no alert was sent)
        - OPEN + not detected → RECOVERING, increment missed_cycles
        - RECOVERING + not detected → increment missed_cycles
          - If missed_cycles >= grace → CLOSED (send resolution)

        Only returns resolutions for OPEN/RECOVERING incidents that are actually closed,
        since SUSPECTED incidents never had an alert sent.

        Args:
            active_incidents: Dict of active incidents keyed by fingerprint_id.
            processed_fps: Set of fingerprint IDs detected this cycle.
            timestamp: Current detection timestamp.
            current_metrics: Optional current metric values for resolution context.
            slo_evaluator: Optional SLO evaluator for building resolution context.
            training_statistics: Optional training statistics for baseline comparison.
            time_period: Current time period (e.g., "business_hours").

        Returns:
            List of resolution dicts for incidents that were closed.
        """
        resolved = []

        for fp_id, incident in active_incidents.items():
            if fp_id not in processed_fps:
                # Incident was not detected this cycle
                current_status = incident.get("status", "OPEN")
                missed_cycles = incident.get("missed_cycles", 0) + 1
                duration = calculate_duration_minutes(incident["first_seen"], timestamp)

                if current_status == "SUSPECTED":
                    # SUSPECTED incidents: silently close if grace exceeded, no resolution alert
                    if missed_cycles >= self._config.resolution_grace_cycles:
                        self._close_incident(incident["incident_id"], timestamp, reason="suspected_expired")
                        log_event(
                            logger, 10, EventType.INCIDENT_RESOLVED, incident["service_name"],
                            f"Suspected incident {incident['incident_id']} expired (never confirmed)",
                            fingerprint=fp_id,
                        )
                    else:
                        self._increment_missed_cycles(incident["incident_id"], missed_cycles)

                elif current_status in ("OPEN", "RECOVERING"):
                    # Confirmed incidents: use grace period before closing
                    if missed_cycles >= self._config.resolution_grace_cycles:
                        # Grace period exceeded - close the incident
                        self._close_incident(incident["incident_id"], timestamp, reason="resolved")

                        log_event(
                            logger, 20, EventType.INCIDENT_RESOLVED, incident["service_name"],
                            f"Resolved {incident['incident_id']} after {missed_cycles} cycles without detection",
                            duration_min=duration,
                        )

                        # Build resolution context with metrics and SLO evaluation
                        resolution_context = self._build_resolution_context(
                            metrics=current_metrics,
                            slo_evaluator=slo_evaluator,
                            training_stats=training_statistics,
                            service_name=incident["service_name"],
                            timestamp=timestamp,
                            time_period=time_period,
                        )

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
                            "missed_cycles_before_close": missed_cycles,
                            "resolution_context": resolution_context,
                        })
                    else:
                        # Still within grace period - transition to RECOVERING
                        new_status = "RECOVERING" if current_status == "OPEN" else current_status
                        self._transition_to_recovering(
                            incident["incident_id"], new_status, missed_cycles
                        )

                        log_event(
                            logger, 10, EventType.INCIDENT_CONTINUED, incident["service_name"],
                            f"Incident {incident['incident_id']} → {new_status} "
                            f"({missed_cycles}/{self._config.resolution_grace_cycles} missed cycles)",
                            fingerprint=fp_id,
                        )

        return resolved

    def _build_resolution_context(
        self,
        metrics: dict[str, Any] | None,
        slo_evaluator: "SLOEvaluator | None",
        training_stats: dict[str, Any] | None,
        service_name: str,
        timestamp: datetime,
        time_period: str,
    ) -> dict[str, Any] | None:
        """Build resolution context with metrics, SLO evaluation, and baseline comparison.

        This provides comprehensive context about the service state at resolution time,
        enabling Web UI to show WHY an incident was closed and what "healthy" looks like.

        Args:
            metrics: Current metric values at resolution time.
            slo_evaluator: Optional SLO evaluator for SLO status.
            training_stats: Optional training statistics for baseline comparison.
            service_name: Service name for SLO config lookup.
            timestamp: Resolution timestamp.
            time_period: Current time period for SLO evaluation.

        Returns:
            Resolution context dict or None if no metrics available.
        """
        if not metrics:
            return None

        context: dict[str, Any] = {
            "metrics_at_resolution": metrics,
            "health_summary": {
                "all_metrics_normal": True,
                "slo_compliant": True,
                "summary": "",
            },
        }

        # Add SLO evaluation if evaluator is available
        if slo_evaluator:
            try:
                slo_result = slo_evaluator.evaluate_metrics(
                    metrics=metrics,
                    service_name=service_name,
                    original_severity="none",  # No anomaly at resolution
                    timestamp=timestamp,
                    time_period=time_period,
                )
                context["slo_evaluation"] = slo_result.to_dict()
                context["health_summary"]["slo_compliant"] = slo_result.slo_status == "ok"
            except Exception as e:
                logger.warning(f"Failed to evaluate SLO for resolution context: {e}")

        # Add baseline comparison if training statistics available
        if training_stats:
            context["comparison_to_baseline"] = self._build_baseline_comparison(
                metrics, training_stats
            )
            # Update all_metrics_normal based on comparison
            comparison = context.get("comparison_to_baseline", {})
            if comparison:
                all_normal = all(
                    m.get("status") in ("normal", "low")
                    for m in comparison.values()
                )
                context["health_summary"]["all_metrics_normal"] = all_normal

        # Build summary message
        context["health_summary"]["summary"] = self._build_health_summary_message(context)

        return context

    def _build_baseline_comparison(
        self,
        metrics: dict[str, Any],
        training_stats: dict[str, Any],
    ) -> dict[str, Any]:
        """Build comparison of current metrics to training baseline.

        Args:
            metrics: Current metric values.
            training_stats: Training statistics with mean, std, etc.

        Returns:
            Per-metric comparison dict.
        """
        comparison = {}

        for metric_name, current_value in metrics.items():
            if not isinstance(current_value, (int, float)):
                continue

            stats = training_stats.get(metric_name, {})
            if not stats:
                continue

            mean = stats.get("mean", stats.get("training_mean"))
            std = stats.get("std", stats.get("training_std"))

            if mean is None:
                continue

            # Calculate deviation in sigma units
            deviation_sigma = 0.0
            if std and std > 0:
                deviation_sigma = (current_value - mean) / std

            # Estimate percentile position (assuming roughly normal distribution)
            percentile = 50.0
            if std and std > 0:
                # Use error function for percentile approximation
                percentile = 50.0 * (1 + math.erf(deviation_sigma / math.sqrt(2)))
                percentile = max(0.0, min(100.0, percentile))

            # Determine status based on deviation (check higher thresholds first)
            status = "normal"
            if abs(deviation_sigma) > 3:
                status = "high" if deviation_sigma > 0 else "very_low"
            elif abs(deviation_sigma) > 2:
                status = "elevated" if deviation_sigma > 0 else "low"

            comparison[metric_name] = {
                "current": current_value,
                "training_mean": mean,
                "training_std": std,
                "deviation_sigma": round(deviation_sigma, 2),
                "percentile_estimate": round(percentile, 1),
                "status": status,
            }

        return comparison

    def _build_health_summary_message(self, context: dict[str, Any]) -> str:
        """Build human-readable health summary message.

        Args:
            context: Resolution context dict with metrics and evaluations.

        Returns:
            Human-readable summary string.
        """
        parts = []

        metrics = context.get("metrics_at_resolution", {})
        slo_eval = context.get("slo_evaluation", {})

        # Check SLO status
        slo_status = slo_eval.get("slo_status", "unknown")
        if slo_status == "ok":
            parts.append("All metrics within acceptable SLO thresholds.")
        elif slo_status == "warning":
            parts.append("Some metrics approaching SLO thresholds.")
        elif slo_status == "breached":
            parts.append("Warning: Some metrics may still be elevated.")

        # Add specific metric context
        latency_eval = slo_eval.get("latency_evaluation", {})
        if latency_eval:
            latency_val = latency_eval.get("value")
            latency_acceptable = latency_eval.get("threshold_acceptable")
            if latency_val is not None and latency_acceptable is not None:
                parts.append(f"Latency {latency_val:.0f}ms (acceptable < {latency_acceptable}ms).")

        error_eval = slo_eval.get("error_rate_evaluation", {})
        if error_eval:
            error_pct = error_eval.get("value_percent")
            error_acceptable = error_eval.get("threshold_acceptable")
            if error_pct and error_acceptable is not None:
                parts.append(f"Errors {error_pct} (acceptable < {error_acceptable * 100:.1f}%).")

        if not parts:
            # Fallback if no SLO evaluation
            if metrics:
                latency = metrics.get("application_latency")
                error_rate = metrics.get("error_rate")
                if latency is not None:
                    parts.append(f"Latency: {latency:.0f}ms.")
                if error_rate is not None:
                    parts.append(f"Error rate: {error_rate * 100:.2f}%.")

        return " ".join(parts) if parts else "Service metrics at resolution time."

    def _close_incident(self, incident_id: str, timestamp: datetime, reason: str = "resolved") -> None:
        """Close an incident in the database."""
        with self._get_connection() as conn:
            conn.execute(
                """UPDATE anomaly_incidents
                   SET status = 'CLOSED', resolved_at = ?,
                       metadata = json_set(COALESCE(metadata, '{}'), '$.resolution_reason', ?)
                   WHERE incident_id = ?""",
                (timestamp, reason, incident_id),
            )
            conn.commit()

    def _increment_missed_cycles(self, incident_id: str, missed_cycles: int) -> None:
        """Increment missed cycles counter for an incident."""
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE anomaly_incidents SET missed_cycles = ? WHERE incident_id = ?",
                (missed_cycles, incident_id),
            )
            conn.commit()

    def _transition_to_recovering(
        self, incident_id: str, new_status: str, missed_cycles: int
    ) -> None:
        """Transition an incident to RECOVERING state."""
        with self._get_connection() as conn:
            conn.execute(
                """UPDATE anomaly_incidents
                   SET status = ?, missed_cycles = ?, consecutive_detections = 0
                   WHERE incident_id = ?""",
                (new_status, missed_cycles, incident_id),
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
        newly_confirmed: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Build the enhanced result payload with cycle-based tracking info."""
        payload = original.copy()
        payload["anomalies"] = anomalies

        if newly_confirmed is None:
            newly_confirmed = []

        # Count by status
        suspected = sum(1 for a in anomalies if a.get("status") == "SUSPECTED")
        confirmed = sum(1 for a in anomalies if a.get("status") == "OPEN")
        recovering = sum(1 for a in anomalies if a.get("status") == "RECOVERING")

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
                "newly_confirmed": len(newly_confirmed),
            },
            "status_summary": {
                "suspected": suspected,  # Awaiting confirmation
                "confirmed": confirmed,   # Alerts being sent
                "recovering": recovering, # May resolve soon
            },
            "overall_action": self._determine_overall_action(anomalies, resolved, newly_confirmed),
            "resolved_incidents": resolved,
            "newly_confirmed_incidents": newly_confirmed,
            "total_active_incidents": len(anomalies),
            "total_alerting_incidents": confirmed,  # Only OPEN incidents trigger alerts
            "detection_context": {
                "model_used": model_name,
                "inference_timestamp": timestamp.isoformat(),
                "confirmation_cycles": self._config.confirmation_cycles,
                "resolution_grace_cycles": self._config.resolution_grace_cycles,
            },
        }

        payload["service_name"] = service_name
        payload["model_name"] = model_name

        return payload

    def _determine_overall_action(
        self,
        anomalies: list[dict],
        resolved: list[dict],
        newly_confirmed: list[dict] | None = None,
    ) -> str:
        """Determine the overall action for the payload."""
        if newly_confirmed is None:
            newly_confirmed = []

        total = len(anomalies) + len(resolved) + len(newly_confirmed)
        if total == 0:
            return IncidentAction.NO_CHANGE.value

        # Prioritize newly confirmed (these are the most important for alerting)
        if newly_confirmed:
            return "CONFIRMED"  # New state: incidents just confirmed and ready to alert

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
                "root_metric": data.get("root_metric"),
                "business_impact": data.get("business_impact"),
                "feature_contributions": data.get("feature_contributions"),
                "comparison_data": data.get("comparison_data"),
                "detection_context": {"model": model, "timestamp": now_iso()},
            }),
        }

    # =========================================================================
    # Query Methods
    # =========================================================================

    def _get_active_incidents_by_service(self, service_name: str) -> dict[str, dict]:
        """Get all active incidents for a service, keyed by fingerprint_id.

        Active incidents are those in SUSPECTED, OPEN, or RECOVERING state.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """SELECT * FROM anomaly_incidents
                   WHERE service_name = ? AND status IN ('SUSPECTED', 'OPEN', 'RECOVERING')
                   ORDER BY first_seen DESC""",
                (service_name,),
            )
            return {row["fingerprint_id"]: dict(row) for row in cursor.fetchall()}

    def _get_open_incidents_by_service(self, service_name: str) -> dict[str, dict]:
        """Get all open incidents for a service, keyed by fingerprint_id.

        Deprecated: Use _get_active_incidents_by_service instead.
        Kept for backward compatibility - returns OPEN status only.
        """
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
