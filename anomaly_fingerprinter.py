import sqlite3
import hashlib
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnomalyFingerprinter:
    """
    Enhanced stateful anomaly fingerprinting system with incident ID support.
    
    Tracks anomaly lifecycle with two-level identity:
    - Fingerprint ID: Content-based, persistent pattern identity
    - Incident ID: Unique occurrence instance identity
    
    Features:
    - Temporal incident separation (same pattern, different occurrences)
    - Model-agnostic fingerprinting (no model name in fingerprint ID)
    - Enhanced state tracking and analytics
    """
    
    def __init__(self, db_path: str = "./anomaly_state.db"):
        self.db_path = db_path
        self.lock = threading.Lock()  # Thread safety for concurrent access
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with enhanced schema for incident tracking"""
        with sqlite3.connect(self.db_path) as conn:
            # Create enhanced table with incident ID support
            conn.execute('''
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
                    metadata TEXT,  -- JSON string for additional data
                    
                    -- Constraints
                    CHECK (status IN ('OPEN', 'CLOSED')),
                    CHECK (occurrence_count > 0)
                )
            ''')
            
            # Create optimized indexes for common queries
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_fingerprint_status 
                ON anomaly_incidents(fingerprint_id, status)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_service_timeline 
                ON anomaly_incidents(service_name, first_seen DESC)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_incident_lookup
                ON anomaly_incidents(incident_id)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_open_incidents
                ON anomaly_incidents(status, last_updated DESC) 
                WHERE status = 'OPEN'
            ''')
            
            # Migration: Handle existing anomaly_state table if it exists
            self._migrate_legacy_schema(conn)
            
            conn.commit()
    
    def _migrate_legacy_schema(self, conn: sqlite3.Connection):
        """Migrate from legacy anomaly_state table to new schema"""
        # Check if legacy table exists
        cursor = conn.execute('''
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='anomaly_state'
        ''')
        
        if cursor.fetchone():
            logger.info("Migrating legacy anomaly_state table to new schema...")
            
            # Migrate existing data to new schema with generated incident IDs
            conn.execute('''
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
            ''')
            
            # Rename legacy table for backup
            conn.execute('ALTER TABLE anomaly_state RENAME TO anomaly_state_backup')
            logger.info("Legacy data migrated successfully. Backup table: anomaly_state_backup")
    
    def generate_incident_id(self) -> str:
        """Generate unique incident ID for this occurrence"""
        return f"incident_{uuid.uuid4().hex[:12]}"
    
    def _generate_fingerprint_id(self, service_name: str, anomaly_name: str) -> str:
        """
        Generate deterministic fingerprint ID for anomaly pattern.
        
        Note: Model name is NOT included to enable cross-temporal tracking.
        Same anomaly pattern detected by different models = same fingerprint.
        """
        content = f"{service_name}_{anomaly_name}"
        hash_obj = hashlib.sha256(content.encode())
        return f"anomaly_{hash_obj.hexdigest()[:12]}"
    
    def _parse_service_model(self, full_service_name: str) -> Tuple[str, str]:
        """
        Parse full service name into service and model components.
        
        Examples:
        "booking_evening_hours" -> ("booking", "evening_hours")
        "fa5_business_hours" -> ("fa5", "business_hours") 
        "mobile-api_weekend_night" -> ("mobile-api", "weekend_night")
        """
        # Enhanced parsing for 5-period model approach
        time_periods = ['business_hours', 'evening_hours', 'night_hours', 
                       'weekend_day', 'weekend_night', 'weekend']  # Keep legacy weekend
        
        for period in time_periods:
            if full_service_name.endswith(f'_{period}'):
                service_name = full_service_name[:-len(f'_{period}')]
                return service_name, period
        
        # Fallback for unexpected formats
        parts = full_service_name.rsplit('_', 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        else:
            return full_service_name, "unknown"
    
    def _generate_anomaly_name(self, anomaly_data: Dict, index: int) -> str:
        """Generate consistent anomaly name from anomaly data"""
        try:
            # Extract meaningful name from anomaly data
            anomaly_type = anomaly_data.get('type', 'unknown')
            detection_method = anomaly_data.get('detection_method', 'unknown')
            
            # Create descriptive name
            if anomaly_type == 'multivariate':
                return f"multivariate_{detection_method}"
            elif anomaly_type in ['ml_isolation', 'threshold', 'pattern', 'correlation']:
                return f"{anomaly_type}_{detection_method}"
            else:
                # Fallback to index-based naming
                return f"anomaly_{index}_{anomaly_type}"
        except Exception:
            # Safe fallback if anomaly_data is malformed
            return f"anomaly_{index}_unknown"
    
    def _determine_overall_action(self, enhanced_anomalies: List[Dict], 
                                 resolved_incidents: List[Dict]) -> str:
        """Determine overall action for the payload"""
        total_actions = len(enhanced_anomalies) + len(resolved_incidents)
        
        if total_actions == 0:
            return "NO_CHANGE"
        
        creates = len([a for a in enhanced_anomalies if a.get('incident_action') == 'CREATE'])
        continues = len([a for a in enhanced_anomalies if a.get('incident_action') == 'CONTINUE'])
        resolves = len(resolved_incidents)
        
        if total_actions == 1:
            if creates > 0:
                return "CREATE"
            elif continues > 0:
                return "UPDATE"
            elif resolves > 0:
                return "RESOLVE"
        
        return "MIXED"
    
    def _extract_anomaly_details(self, anomaly_name: str, anomaly_data: Dict, 
                                detected_by_model: str) -> Dict:
        """Extract relevant details from anomaly data with model context"""
        return {
            'severity': anomaly_data.get('severity', 'medium'),
            'current_value': anomaly_data.get('value', anomaly_data.get('actual_value')),
            'threshold_value': anomaly_data.get('threshold', anomaly_data.get('threshold_value')),
            'confidence_score': anomaly_data.get('score', anomaly_data.get('confidence_score')),
            'detection_method': anomaly_data.get('detection_method', 'unknown'),
            'description': anomaly_data.get('description', f'Anomaly detected: {anomaly_name}'),
            'detected_by_model': detected_by_model,
            'metadata': json.dumps({
                'type': anomaly_data.get('type'),
                'business_impact': anomaly_data.get('business_impact'),
                'feature_contributions': anomaly_data.get('feature_contributions'),
                'comparison_data': anomaly_data.get('comparison_data'),
                'detection_context': {
                    'model': detected_by_model,
                    'timestamp': datetime.now().isoformat()
                }
            })
        }
    
    def process_anomalies(self, 
                         full_service_name: str,
                         anomaly_result: Dict,
                         current_metrics: Optional[Dict] = None,
                         timestamp: Optional[datetime] = None) -> Dict:
        """
        Process anomaly detection result with enhanced incident tracking.
        
        Enhanced features:
        - Two-level identity: fingerprint IDs + incident IDs
        - Model-agnostic fingerprinting
        - Temporal incident separation
        - Cross-model state tracking
        
        Args:
            full_service_name: Full service name like "booking_evening_hours"
            anomaly_result: Rich anomaly detection result
            current_metrics: Current metric values
            timestamp: Timestamp for this detection run
            
        Returns:
            Enhanced payload with fingerprinting and incident tracking
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Extract current_metrics from anomaly_result if not provided
        if current_metrics is None:
            current_metrics = anomaly_result.get('current_metrics', {})
        
        # Parse service and model names
        service_name, model_name = self._parse_service_model(full_service_name)
        
        # Extract current anomalies - handle both list and dict formats
        current_anomalies = anomaly_result.get('anomalies', [])
        if isinstance(current_anomalies, dict):
            current_anomalies = [current_anomalies[key] for key in current_anomalies.keys()]
        
        with self.lock:
            # Get existing open incidents for this service
            existing_incidents = self._get_open_incidents_by_service(service_name)
            
            # Process current anomalies and determine actions
            enhanced_anomalies = []
            processed_fingerprints = set()
            
            # Process currently detected anomalies
            for i, anomaly_data in enumerate(current_anomalies):
                if not isinstance(anomaly_data, dict):
                    continue
                
                # Generate fingerprint ID (model-agnostic)
                anomaly_name = self._generate_anomaly_name(anomaly_data, i)
                fingerprint_id = self._generate_fingerprint_id(service_name, anomaly_name)
                processed_fingerprints.add(fingerprint_id)
                
                # Check for existing open incident with this fingerprint
                existing_incident = existing_incidents.get(fingerprint_id)
                anomaly_details = self._extract_anomaly_details(
                    anomaly_name, anomaly_data, model_name
                )
                
                # Create enhanced anomaly with incident tracking
                enhanced_anomaly = anomaly_data.copy()  # Preserve all original data
                
                if existing_incident:
                    # UPDATE existing incident (CONTINUE)
                    incident_info = self._update_existing_incident(
                        existing_incident, anomaly_details, timestamp
                    )
                    
                    enhanced_anomaly.update({
                        'fingerprint_id': fingerprint_id,
                        'incident_id': existing_incident['incident_id'],
                        'anomaly_name': anomaly_name,
                        'fingerprint_action': 'UPDATE',  # Pattern level
                        'incident_action': 'CONTINUE',   # Instance level
                        'occurrence_count': incident_info['occurrence_count'],
                        'first_seen': existing_incident['first_seen'],
                        'last_updated': timestamp.isoformat(),
                        'incident_duration_minutes': self._calculate_duration_minutes(
                            existing_incident['first_seen'], timestamp
                        ),
                        'detected_by_model': model_name
                    })
                    
                    # Add severity change info if applicable
                    if existing_incident['severity'] != anomaly_details['severity']:
                        enhanced_anomaly.update({
                            'severity_changed': True,
                            'previous_severity': existing_incident['severity'],
                            'severity_changed_at': timestamp.isoformat()
                        })
                else:
                    # CREATE new incident
                    incident_id = self.generate_incident_id()
                    self._create_new_incident(
                        fingerprint_id, incident_id, service_name, anomaly_name,
                        anomaly_details, timestamp
                    )
                    
                    enhanced_anomaly.update({
                        'fingerprint_id': fingerprint_id,
                        'incident_id': incident_id,
                        'anomaly_name': anomaly_name,
                        'fingerprint_action': 'CREATE',  # Could be UPDATE if pattern seen before
                        'incident_action': 'CREATE',     # Always CREATE for new incident
                        'occurrence_count': 1,
                        'first_seen': timestamp.isoformat(),
                        'last_updated': timestamp.isoformat(),
                        'incident_duration_minutes': 0,
                        'detected_by_model': model_name
                    })
                
                enhanced_anomalies.append(enhanced_anomaly)
            
            # Process resolved incidents (fingerprints no longer detected)
            resolved_incidents = []
            for fingerprint_id, existing_incident in existing_incidents.items():
                if fingerprint_id not in processed_fingerprints:
                    incident_duration = self._calculate_duration_minutes(
                        existing_incident['first_seen'], timestamp
                    )
                    
                    # Close incident in database
                    self._close_incident(existing_incident['incident_id'], timestamp)
                    
                    logger.info(
                        f"✅ Resolved incident: {existing_incident['incident_id']} "
                        f"(fingerprint: {fingerprint_id}, duration: {incident_duration}m)"
                    )
                    
                    resolved_incidents.append({
                        'fingerprint_id': fingerprint_id,
                        'incident_id': existing_incident['incident_id'],
                        'anomaly_name': existing_incident['anomaly_name'],
                        'fingerprint_action': 'RESOLVE',
                        'incident_action': 'CLOSE',
                        'final_severity': existing_incident['severity'],
                        'resolved_at': timestamp.isoformat(),
                        'total_occurrences': existing_incident['occurrence_count'],
                        'incident_duration_minutes': incident_duration,
                        'first_seen': existing_incident['first_seen'],
                        'service_name': existing_incident['service_name'],
                        'last_detected_by_model': existing_incident.get('detected_by_model', 'unknown')
                    })
            
            # Create enhanced payload that preserves everything
            enhanced_payload = anomaly_result.copy()  # Start with original payload
            
            # Update anomalies with enhanced versions
            enhanced_payload['anomalies'] = enhanced_anomalies
            
            # Add enhanced fingerprinting summary
            enhanced_payload['fingerprinting'] = {
                'service_name': service_name,
                'model_name': model_name,
                'timestamp': timestamp.isoformat(),
                'action_summary': {
                    'incident_creates': len([a for a in enhanced_anomalies 
                                           if a.get('incident_action') == 'CREATE']),
                    'incident_continues': len([a for a in enhanced_anomalies 
                                             if a.get('incident_action') == 'CONTINUE']),
                    'incident_closes': len(resolved_incidents)
                },
                'overall_action': self._determine_overall_action(enhanced_anomalies, resolved_incidents),
                'resolved_incidents': resolved_incidents,
                'total_open_incidents': len(enhanced_anomalies),
                'detection_context': {
                    'model_used': model_name,
                    'inference_timestamp': timestamp.isoformat()
                }
            }
            
            # Add backward compatibility fields
            enhanced_payload['service_name'] = service_name
            enhanced_payload['model_name'] = model_name
            
            return enhanced_payload
    
    def _get_open_incidents_by_service(self, service_name: str) -> Dict[str, Dict]:
        """Get all open incidents for a service, keyed by fingerprint_id"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute('''
                SELECT * FROM anomaly_incidents 
                WHERE service_name = ? AND status = 'OPEN'
                ORDER BY first_seen DESC
            ''', (service_name,))
            
            incidents = {}
            for row in cursor.fetchall():
                incidents[row['fingerprint_id']] = dict(row)
            
            return incidents
    
    def _create_new_incident(self, 
                           fingerprint_id: str,
                           incident_id: str,
                           service_name: str, 
                           anomaly_name: str,
                           anomaly_details: Dict,
                           timestamp: datetime) -> None:
        """Create new incident in database"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO anomaly_incidents 
                (fingerprint_id, incident_id, service_name, anomaly_name, status, 
                 severity, first_seen, last_updated, occurrence_count, current_value, 
                 threshold_value, confidence_score, detection_method, description, 
                 detected_by_model, metadata)
                VALUES (?, ?, ?, ?, 'OPEN', ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                fingerprint_id, incident_id, service_name, anomaly_name,
                anomaly_details['severity'], timestamp, timestamp,
                anomaly_details['current_value'], anomaly_details['threshold_value'],
                anomaly_details['confidence_score'], anomaly_details['detection_method'],
                anomaly_details['description'], anomaly_details['detected_by_model'],
                anomaly_details['metadata']
            ))
            conn.commit()
        
        logger.info(
            f"🆕 Created new incident: {incident_id} "
            f"(fingerprint: {fingerprint_id}, service: {service_name})"
        )
    
    def _update_existing_incident(self,
                                existing_incident: Dict,
                                anomaly_details: Dict,
                                timestamp: datetime) -> Dict:
        """Update existing incident and return updated info"""
        
        new_occurrence_count = existing_incident['occurrence_count'] + 1
        severity_changed = existing_incident['severity'] != anomaly_details['severity']
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE anomaly_incidents 
                SET severity = ?, last_updated = ?, occurrence_count = ?,
                    current_value = ?, threshold_value = ?, confidence_score = ?,
                    description = ?, detected_by_model = ?, metadata = ?
                WHERE incident_id = ?
            ''', (
                anomaly_details['severity'], timestamp, new_occurrence_count,
                anomaly_details['current_value'], anomaly_details['threshold_value'],
                anomaly_details['confidence_score'], anomaly_details['description'],
                anomaly_details['detected_by_model'], anomaly_details['metadata'],
                existing_incident['incident_id']
            ))
            conn.commit()
        
        if severity_changed:
            logger.info(
                f"📈 Severity changed for {existing_incident['incident_id']}: "
                f"{existing_incident['severity']} → {anomaly_details['severity']}"
            )
        else:
            logger.info(
                f"🔄 Updated incident: {existing_incident['incident_id']} "
                f"(count: {new_occurrence_count})"
            )
        
        return {
            'occurrence_count': new_occurrence_count,
            'severity_changed': severity_changed,
            'previous_severity': existing_incident['severity'] if severity_changed else None
        }
    
    def _close_incident(self, incident_id: str, timestamp: datetime) -> None:
        """Close an incident in the database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE anomaly_incidents 
                SET status = 'CLOSED', resolved_at = ?
                WHERE incident_id = ?
            ''', (timestamp, incident_id))
            conn.commit()
    
    def _calculate_duration_minutes(self, first_seen: str, current_time: datetime) -> int:
        """Calculate duration in minutes between first seen and current time"""
        try:
            if isinstance(first_seen, str):
                first_seen_dt = datetime.fromisoformat(first_seen.replace('Z', '+00:00'))
            else:
                first_seen_dt = first_seen
            
            duration = current_time - first_seen_dt.replace(tzinfo=None)
            return int(duration.total_seconds() / 60)
        except Exception:
            return 0
    
    # Enhanced Public API methods
    
    def get_service_incidents(self, service_name: str, 
                            include_closed: bool = False,
                            limit: int = 100) -> List[Dict]:
        """Get incidents for a service with enhanced filtering"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            if include_closed:
                query = '''
                    SELECT * FROM anomaly_incidents 
                    WHERE service_name = ?
                    ORDER BY first_seen DESC
                    LIMIT ?
                '''
                params = (service_name, limit)
            else:
                query = '''
                    SELECT * FROM anomaly_incidents 
                    WHERE service_name = ? AND status = 'OPEN'
                    ORDER BY first_seen DESC
                    LIMIT ?
                '''
                params = (service_name, limit)
            
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_incident_by_id(self, incident_id: str) -> Optional[Dict]:
        """Get specific incident by ID"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute('''
                SELECT * FROM anomaly_incidents 
                WHERE incident_id = ?
            ''', (incident_id,))
            
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def get_pattern_history(self, fingerprint_id: str, limit: int = 50) -> List[Dict]:
        """Get incident history for a specific anomaly pattern"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute('''
                SELECT * FROM anomaly_incidents 
                WHERE fingerprint_id = ?
                ORDER BY first_seen DESC
                LIMIT ?
            ''', (fingerprint_id, limit))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_all_open_incidents(self) -> Dict[str, List[Dict]]:
        """Get all open incidents grouped by service"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute('''
                SELECT * FROM anomaly_incidents 
                WHERE status = 'OPEN'
                ORDER BY service_name, first_seen DESC
            ''')
            
            incidents_by_service = {}
            for row in cursor.fetchall():
                service_name = row['service_name']
                if service_name not in incidents_by_service:
                    incidents_by_service[service_name] = []
                incidents_by_service[service_name].append(dict(row))
            
            return incidents_by_service
    
    def cleanup_old_incidents(self, max_age_hours: int = 72, 
                             status: str = 'CLOSED') -> int:
        """Clean up old incidents (default: closed incidents older than 72 hours)"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        with sqlite3.connect(self.db_path) as conn:
            if status == 'CLOSED':
                cursor = conn.execute('''
                    DELETE FROM anomaly_incidents 
                    WHERE status = 'CLOSED' AND resolved_at < ?
                ''', (cutoff_time.isoformat(),))
            else:
                # Clean up any status older than cutoff
                cursor = conn.execute('''
                    DELETE FROM anomaly_incidents 
                    WHERE last_updated < ?
                ''', (cutoff_time.isoformat(),))
            
            deleted_count = cursor.rowcount
            conn.commit()
        
        if deleted_count > 0:
            logger.info(
                f"🧹 Cleaned up {deleted_count} old incidents "
                f"(status: {status}, older than {max_age_hours}h)"
            )
        
        return deleted_count
    
    def get_statistics(self) -> Dict:
        """Get enhanced system statistics with incident analytics"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Basic counts
            total_open_cursor = conn.execute('''
                SELECT COUNT(*) as count FROM anomaly_incidents WHERE status = 'OPEN'
            ''')
            total_open = total_open_cursor.fetchone()['count']
            
            total_all_cursor = conn.execute('''
                SELECT COUNT(*) as count FROM anomaly_incidents
            ''')
            total_all = total_all_cursor.fetchone()['count']
            
            # Open incidents by service
            service_cursor = conn.execute('''
                SELECT service_name, COUNT(*) as count 
                FROM anomaly_incidents 
                WHERE status = 'OPEN'
                GROUP BY service_name
                ORDER BY count DESC
            ''')
            by_service = [dict(row) for row in service_cursor.fetchall()]
            
            # Incidents by severity
            severity_cursor = conn.execute('''
                SELECT severity, COUNT(*) as count 
                FROM anomaly_incidents 
                WHERE status = 'OPEN'
                GROUP BY severity
            ''')
            by_severity = {row['severity']: row['count'] for row in severity_cursor.fetchall()}
            
            # Pattern frequency (fingerprint analysis)
            pattern_cursor = conn.execute('''
                SELECT fingerprint_id, anomaly_name, COUNT(*) as occurrences,
                       AVG(CASE WHEN status = 'CLOSED' THEN 
                           (julianday(resolved_at) - julianday(first_seen)) * 24 * 60 
                           ELSE NULL END) as avg_duration_minutes
                FROM anomaly_incidents 
                GROUP BY fingerprint_id, anomaly_name
                HAVING occurrences > 1
                ORDER BY occurrences DESC
                LIMIT 10
            ''')
            frequent_patterns = [dict(row) for row in pattern_cursor.fetchall()]
            
            # Longest running open incident
            longest_cursor = conn.execute('''
                SELECT service_name, anomaly_name, incident_id, first_seen,
                       (julianday('now') - julianday(first_seen)) * 24 * 60 as duration_minutes
                FROM anomaly_incidents 
                WHERE status = 'OPEN'
                ORDER BY first_seen ASC 
                LIMIT 1
            ''')
            longest_incident = longest_cursor.fetchone()
            
            return {
                'total_open_incidents': total_open,
                'total_all_incidents': total_all,
                'total_closed_incidents': total_all - total_open,
                'open_incidents_by_service': by_service,
                'open_incidents_by_severity': by_severity,
                'frequent_patterns': frequent_patterns,
                'longest_running_incident': dict(longest_incident) if longest_incident else None,
                'database_path': self.db_path,
                'schema_version': '2.0_enhanced_incident_tracking'
            }
    
    def get_analytics_summary(self, days: int = 7) -> Dict:
        """Get analytics summary for the last N days"""
        start_date = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Incidents created in period
            created_cursor = conn.execute('''
                SELECT COUNT(*) as count
                FROM anomaly_incidents 
                WHERE first_seen >= ?
            ''', (start_date.isoformat(),))
            incidents_created = created_cursor.fetchone()['count']
            
            # Incidents resolved in period
            resolved_cursor = conn.execute('''
                SELECT COUNT(*) as count
                FROM anomaly_incidents 
                WHERE resolved_at >= ? AND status = 'CLOSED'
            ''', (start_date.isoformat(),))
            incidents_resolved = resolved_cursor.fetchone()['count']
            
            # Average incident duration for resolved incidents
            duration_cursor = conn.execute('''
                SELECT AVG((julianday(resolved_at) - julianday(first_seen)) * 24 * 60) as avg_duration
                FROM anomaly_incidents 
                WHERE resolved_at >= ? AND status = 'CLOSED'
            ''', (start_date.isoformat(),))
            avg_duration = duration_cursor.fetchone()['avg_duration'] or 0
            
            return {
                'period_days': days,
                'incidents_created': incidents_created,
                'incidents_resolved': incidents_resolved,
                'average_resolution_minutes': round(avg_duration, 2),
                'net_incident_change': incidents_created - incidents_resolved
            }


# Convenience function for easy integration
def create_fingerprinter(db_path: str = "./anomaly_state.db") -> AnomalyFingerprinter:
    """Create a ready-to-use enhanced anomaly fingerprinter"""
    return AnomalyFingerprinter(db_path=db_path)


# Example usage and testing with enhanced features
if __name__ == "__main__":
    # Initialize enhanced fingerprinter
    fingerprinter = create_fingerprinter()
    
    # Example 1: First detection run (creates new incident)
    print("=== Run 1: Initial Detection (New Incident) ===")
    anomaly_result_1 = {
        'anomalies': [{
            'type': 'multivariate',
            'severity': 'high',
            'value': 1200.0,
            'threshold': 800.0,
            'score': -0.85,
            'description': 'Application latency significantly elevated',
            'detection_method': 'enhanced_isolation_forest'
        }],
        'current_metrics': {
            'request_rate': 150.0,
            'application_latency': 1200.0,
            'database_latency': 200.0,
            'error_rate': 0.02
        }
    }
    
    payload_1 = fingerprinter.process_anomalies(
        full_service_name="booking_evening_hours",
        anomaly_result=anomaly_result_1
    )
    
    print(f"Action: {payload_1['fingerprinting']['overall_action']}")
    print(f"Incident ID: {payload_1['anomalies'][0]['incident_id']}")
    print(f"Fingerprint ID: {payload_1['anomalies'][0]['fingerprint_id']}")
    
    # Example 2: Same anomaly persists (continues same incident)
    print("\n=== Run 2: Same Anomaly Persists (Continue Incident) ===")
    payload_2 = fingerprinter.process_anomalies(
        full_service_name="booking_night_hours",  # Different model, same service
        anomaly_result=anomaly_result_1
    )
    
    print(f"Action: {payload_2['fingerprinting']['overall_action']}")
    print(f"Incident Action: {payload_2['anomalies'][0]['incident_action']}")
    print(f"Same Incident ID: {payload_2['anomalies'][0]['incident_id']}")
    print(f"Occurrence count: {payload_2['anomalies'][0]['occurrence_count']}")
    
    # Example 3: Resolution (incident closes)
    print("\n=== Run 3: Resolution (Close Incident) ===")
    payload_3 = fingerprinter.process_anomalies(
        full_service_name="booking_business_hours",
        anomaly_result={'anomalies': []}  # No anomalies
    )
    
    print(f"Action: {payload_3['fingerprinting']['overall_action']}")
    if payload_3['fingerprinting']['resolved_incidents']:
        resolved = payload_3['fingerprinting']['resolved_incidents'][0]
        print(f"Closed Incident: {resolved['incident_id']}")
        print(f"Duration: {resolved['incident_duration_minutes']} minutes")
    
    # Example 4: New occurrence of same pattern (new incident)
    print("\n=== Run 4: New Occurrence of Same Pattern (New Incident) ===")
    payload_4 = fingerprinter.process_anomalies(
        full_service_name="booking_evening_hours",
        anomaly_result=anomaly_result_1  # Same anomaly pattern
    )
    
    print(f"Action: {payload_4['fingerprinting']['overall_action']}")
    print(f"New Incident ID: {payload_4['anomalies'][0]['incident_id']}")
    print(f"Same Fingerprint ID: {payload_4['anomalies'][0]['fingerprint_id']}")
    
    # Example 5: Enhanced statistics
    print("\n=== Enhanced System Statistics ===")
    stats = fingerprinter.get_statistics()
    print(f"Total open incidents: {stats['total_open_incidents']}")
    print(f"Total all incidents: {stats['total_all_incidents']}")
    print(f"By severity: {stats['open_incidents_by_severity']}")
    
    analytics = fingerprinter.get_analytics_summary(days=1)
    print(f"Analytics (24h): {analytics['incidents_created']} created, {analytics['incidents_resolved']} resolved")
