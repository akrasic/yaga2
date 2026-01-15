#!/usr/bin/env python3
"""
Reset incident database for production deployment.

This script:
1. Closes all active incidents (SUSPECTED, OPEN, RECOVERING)
2. Generates resolution payloads for the observability API
3. Optionally cleans up old closed incidents
4. Optionally truncates the entire database

Usage:
    # Preview what would be closed (dry-run)
    uv run python scripts/reset_incidents.py --dry-run

    # Close active incidents and generate resolutions JSON
    uv run python scripts/reset_incidents.py --close-active

    # Close active and clean up old closed incidents (> 24h)
    uv run python scripts/reset_incidents.py --close-active --cleanup-hours 24

    # Full reset - truncate entire database
    uv run python scripts/reset_incidents.py --truncate

    # Use custom database path
    uv run python scripts/reset_incidents.py --db-path /path/to/db --close-active
"""

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path


def get_active_incidents(conn: sqlite3.Connection) -> list[dict]:
    """Get all active incidents (SUSPECTED, OPEN, RECOVERING)."""
    conn.row_factory = sqlite3.Row
    cursor = conn.execute("""
        SELECT * FROM anomaly_incidents
        WHERE status IN ('SUSPECTED', 'OPEN', 'RECOVERING')
        ORDER BY service_name, first_seen
    """)
    return [dict(row) for row in cursor.fetchall()]


def close_incidents(
    conn: sqlite3.Connection,
    incidents: list[dict],
    reason: str = "system_reset",
) -> list[dict]:
    """Close incidents and return resolution payloads for API notification."""
    timestamp = datetime.now()
    resolutions = []

    for incident in incidents:
        # Only generate resolution payloads for confirmed incidents (OPEN/RECOVERING)
        # SUSPECTED incidents never had an alert sent, so no resolution needed
        if incident["status"] in ("OPEN", "RECOVERING"):
            duration = (
                timestamp - datetime.fromisoformat(incident["first_seen"])
            ).total_seconds() / 60

            resolutions.append({
                "alert_type": "incident_resolved",
                "service": incident["service_name"],
                "timestamp": timestamp.isoformat(),
                "incident_id": incident["incident_id"],
                "fingerprint_id": incident["fingerprint_id"],
                "anomaly_name": incident["anomaly_name"],
                "model_type": "incident_resolution",
                "resolution_details": {
                    "final_severity": incident["severity"],
                    "total_occurrences": incident["occurrence_count"],
                    "incident_duration_minutes": int(duration),
                    "first_seen": incident["first_seen"],
                    "last_detected_by_model": incident.get("detected_by_model", "unknown"),
                    "resolution_reason": reason,
                },
            })

        # Update database
        conn.execute(
            """UPDATE anomaly_incidents
               SET status = 'CLOSED',
                   resolved_at = ?,
                   metadata = json_set(COALESCE(metadata, '{}'), '$.resolution_reason', ?)
               WHERE incident_id = ?""",
            (timestamp, reason, incident["incident_id"]),
        )

    conn.commit()
    return resolutions


def cleanup_old_incidents(conn: sqlite3.Connection, hours: int) -> int:
    """Delete closed incidents older than specified hours."""
    cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
    cursor = conn.execute(
        "DELETE FROM anomaly_incidents WHERE status = 'CLOSED' AND resolved_at < ?",
        (cutoff,),
    )
    deleted = cursor.rowcount
    conn.commit()
    return deleted


def truncate_database(conn: sqlite3.Connection) -> int:
    """Delete all incidents from the database."""
    cursor = conn.execute("SELECT COUNT(*) FROM anomaly_incidents")
    count = cursor.fetchone()[0]
    conn.execute("DELETE FROM anomaly_incidents")
    conn.commit()
    return count


def print_summary(incidents: list[dict]) -> None:
    """Print summary of active incidents."""
    by_status = {}
    by_service = {}

    for inc in incidents:
        status = inc["status"]
        service = inc["service_name"]
        by_status[status] = by_status.get(status, 0) + 1
        by_service[service] = by_service.get(service, 0) + 1

    print("\n=== Active Incidents Summary ===")
    print(f"Total: {len(incidents)}")
    print("\nBy Status:")
    for status, count in sorted(by_status.items()):
        print(f"  {status}: {count}")
    print("\nBy Service:")
    for service, count in sorted(by_service.items()):
        print(f"  {service}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description="Reset incident database for production deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--db-path",
        default="./anomaly_state.db",
        help="Path to SQLite database (default: ./anomaly_state.db)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be done without making changes",
    )
    parser.add_argument(
        "--close-active",
        action="store_true",
        help="Close all active incidents (SUSPECTED, OPEN, RECOVERING)",
    )
    parser.add_argument(
        "--cleanup-hours",
        type=int,
        metavar="HOURS",
        help="Also delete closed incidents older than HOURS",
    )
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="Delete ALL incidents (full database reset)",
    )
    parser.add_argument(
        "--output",
        "-o",
        metavar="FILE",
        help="Write resolution payloads to JSON file (for API notification)",
    )
    parser.add_argument(
        "--reason",
        default="system_reset",
        help="Resolution reason to record (default: system_reset)",
    )

    args = parser.parse_args()

    # Validate arguments
    if not any([args.dry_run, args.close_active, args.truncate, args.cleanup_hours]):
        parser.error("Must specify one of: --dry-run, --close-active, --truncate, --cleanup-hours")

    db_path = Path(args.db_path)
    if not db_path.exists():
        print(f"Error: Database not found: {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(db_path)

    try:
        # Get current state
        active_incidents = get_active_incidents(conn)

        if args.dry_run:
            print("=== DRY RUN - No changes will be made ===")
            print_summary(active_incidents)

            if active_incidents:
                print("\nIncidents that would be closed:")
                for inc in active_incidents:
                    print(f"  [{inc['status']}] {inc['service_name']}: {inc['anomaly_name']} "
                          f"(id: {inc['incident_id'][:20]}...)")

            # Count what would be cleaned up
            if args.cleanup_hours:
                cutoff = (datetime.now() - timedelta(hours=args.cleanup_hours)).isoformat()
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM anomaly_incidents WHERE status = 'CLOSED' AND resolved_at < ?",
                    (cutoff,),
                )
                old_count = cursor.fetchone()[0]
                print(f"\nWould delete {old_count} closed incidents older than {args.cleanup_hours} hours")

            if args.truncate:
                cursor = conn.execute("SELECT COUNT(*) FROM anomaly_incidents")
                total = cursor.fetchone()[0]
                print(f"\nWould delete ALL {total} incidents (truncate)")

            return

        # Perform requested actions
        resolutions = []

        if args.truncate:
            print("=== TRUNCATING DATABASE ===")
            # First close active incidents to generate resolutions
            if active_incidents:
                resolutions = close_incidents(conn, active_incidents, args.reason)
                print(f"Generated {len(resolutions)} resolution payloads")

            count = truncate_database(conn)
            print(f"Deleted {count} incidents")

        elif args.close_active:
            print("=== CLOSING ACTIVE INCIDENTS ===")
            print_summary(active_incidents)

            if active_incidents:
                resolutions = close_incidents(conn, active_incidents, args.reason)
                print(f"\nClosed {len(active_incidents)} incidents")
                print(f"Generated {len(resolutions)} resolution payloads (for OPEN/RECOVERING only)")
            else:
                print("\nNo active incidents to close")

        if args.cleanup_hours:
            print(f"\n=== CLEANING UP INCIDENTS OLDER THAN {args.cleanup_hours} HOURS ===")
            deleted = cleanup_old_incidents(conn, args.cleanup_hours)
            print(f"Deleted {deleted} old closed incidents")

        # Output resolutions
        if resolutions:
            if args.output:
                output_path = Path(args.output)
                with open(output_path, "w") as f:
                    json.dump({"resolutions": resolutions, "schema_version": "1.0.0"}, f, indent=2)
                print(f"\nResolution payloads written to: {output_path}")
            else:
                print("\n=== RESOLUTION PAYLOADS (send to /api/incidents/resolve) ===")
                print(json.dumps({"resolutions": resolutions}, indent=2))

        print("\n=== DONE ===")

        # Show final state
        cursor = conn.execute("SELECT status, COUNT(*) FROM anomaly_incidents GROUP BY status")
        print("\nFinal database state:")
        for row in cursor.fetchall():
            print(f"  {row[0]}: {row[1]}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
