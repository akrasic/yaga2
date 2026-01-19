"""
Enhanced Production-Grade ML Inference Pipeline for Smartbox Anomaly Detection
Now supports explainable anomaly detection with rich context and recommendations
"""

import json
import logging
import argparse
import requests
from datetime import datetime
from typing import Dict, List, Optional, Any

import urllib3

# Import from smartbox_anomaly package (primary source)
from smartbox_anomaly.core import (
    get_config,
    ObservabilityConfig,
    # TypedDicts for type safety
    FingerprintingStats,
    InferenceResultDict,
)
from smartbox_anomaly.detection import TimeAwareAnomalyDetector
from smartbox_anomaly.fingerprinting import AnomalyFingerprinter

# Import inference components from new module structure
from smartbox_anomaly.inference import (
    SmartboxMLInferencePipeline,
    AnomalyResult,
    ServiceInferenceResult,
    EnhancedModelManager,
    EnhancedAnomalyDetectionEngine,
    EnhancedResultsProcessor,
    EnhancedTimeAwareDetector,
)

# Import VictoriaMetrics client (still uses root-level stub for backward compatibility)
from vmclient import VictoriaMetricsClient, InferenceMetrics

# Disable urllib3 warnings for cleaner output
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Reduce urllib3 logging level to avoid connection pool warnings
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)


def main():
    """Main execution function with integrated anomaly fingerprinting"""
    parser = argparse.ArgumentParser(
        description="Smartbox ML Anomaly Detection - Enhanced Production Inference Pipeline"
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output with progress information and explanations'
    )
    parser.add_argument(
        '--fingerprint-db',
        type=str,
        default="/app/data/anomaly_state.db",
        help='Path to fingerprinting database (default: /app/data/anomaly_state.db)'
    )

    args = parser.parse_args()

    # Set logging level based on verbose flag
    if not args.verbose:
        logging.getLogger().setLevel(logging.WARNING)
        logging.getLogger(__name__).setLevel(logging.WARNING)

    if args.verbose:
        logger.info("Smartbox Enhanced ML Anomaly Detection - Production Inference Pipeline")
        logger.info("With Explainable AI, Rich Context, and Anomaly Fingerprinting")
        logger.info("=" * 80)

    try:
        # Initialize pipeline and fingerprinting
        pipeline = SmartboxMLInferencePipeline(verbose=args.verbose)
        fingerprinter = _initialize_fingerprinting(args.fingerprint_db, args.verbose)

        # Health check VictoriaMetrics before proceeding
        if not _check_victoria_metrics_health(pipeline, args.verbose):
            return

        # Get and display system status
        status = pipeline.get_system_status()
        _display_system_status(status, fingerprinter, args.verbose)

        if status['status'] == 'no_models':
            _handle_no_models_error(args.verbose)
            return

        # Run inference with fingerprinting
        if args.verbose:
            logger.info("Starting Enhanced ML Inference...")
            logger.info("Using explainable time-aware anomaly detection with fingerprinting")

        results = pipeline.run_enhanced_time_aware_inference()
        fingerprinted_results, fp_stats = _apply_fingerprinting(
            results, fingerprinter, args.verbose, pipeline=pipeline
        )

        # Process and output results (now includes resolved incidents)
        _update_results_processor(pipeline, fingerprinted_results, fp_stats)
        _display_execution_summary(fingerprinted_results, fp_stats, args.verbose)

        pipeline.results_processor.output_anomalies()
        _send_to_observability_service(pipeline.results_processor.detected_anomalies, fp_stats, args.verbose)

        if args.verbose:
            logger.info("Enhanced inference completed!")
            _display_tips(status, fp_stats)

    except Exception as e:
        _handle_pipeline_error(e, args.verbose)


def _initialize_fingerprinting(db_path: str, verbose: bool) -> Optional['AnomalyFingerprinter']:
    """Initialize anomaly fingerprinting with error handling"""
    try:
        import os

        # Ensure the directory exists
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            # If Docker data directory doesn't exist, fall back to current directory
            if db_dir == "/app/data":
                db_path = "./anomaly_state.db"
                if verbose:
                    logger.info(f"Docker data directory not found, using local path: {db_path}")
            else:
                os.makedirs(db_dir, exist_ok=True)

        fingerprinter = AnomalyFingerprinter(db_path=db_path)

        if verbose:
            logger.info(f"Anomaly fingerprinting enabled (db: {db_path})")
            _display_fingerprinting_stats(fingerprinter)

        return fingerprinter

    except ImportError:
        if verbose:
            logger.warning("Fingerprinting module not available - continuing without fingerprinting")
        return None
    except Exception as e:
        if verbose:
            logger.warning(f"Failed to initialize fingerprinting: {e}")
        return None


def _check_victoria_metrics_health(
    pipeline: 'SmartboxMLInferencePipeline',
    verbose: bool,
    max_latency_ms: float = 5000
) -> bool:
    """Check VictoriaMetrics health before running inference.

    Args:
        pipeline: The inference pipeline with vm_client.
        verbose: Whether to log detailed output.
        max_latency_ms: Maximum acceptable response time in milliseconds.

    Returns:
        True if healthy and inference should proceed, False to skip inference.
    """
    if not hasattr(pipeline, 'vm_client') or pipeline.vm_client is None:
        if verbose:
            logger.warning("VictoriaMetrics client not available - skipping health check")
        return True  # Proceed anyway if no client

    try:
        is_healthy, details = pipeline.vm_client.health_check(max_latency_ms)

        if is_healthy:
            if verbose:
                logger.info(f"VictoriaMetrics health check passed (latency: {details['latency_ms']}ms)")
            return True

        # Health check failed
        error_msg = details.get('error', 'Unknown error')

        if details.get('circuit_breaker_open'):
            logger.warning(f"VictoriaMetrics circuit breaker is open - skipping inference")
            logger.warning("Too many recent failures. Will retry after circuit breaker timeout.")
        elif details.get('latency_ms') and details['latency_ms'] > max_latency_ms:
            logger.warning(
                f"VictoriaMetrics responding slowly ({details['latency_ms']:.0f}ms > {max_latency_ms:.0f}ms threshold) - skipping inference"
            )
            logger.warning("High latency may cause timeouts during metrics collection.")
        else:
            logger.warning(f"VictoriaMetrics health check failed: {error_msg} - skipping inference")

        if verbose:
            logger.info("Inference skipped to avoid incomplete or stale results.")
            logger.info("Next scheduled run will retry automatically.")

        return False

    except Exception as e:
        logger.error(f"VictoriaMetrics health check error: {e} - skipping inference")
        return False


def _display_fingerprinting_stats(fingerprinter: 'AnomalyFingerprinter') -> None:
    """Display current fingerprinting statistics"""
    fp_stats = fingerprinter.get_statistics()
    if fp_stats['total_open_incidents'] > 0:
        logger.info(f"Active incidents: {fp_stats['total_open_incidents']}")
        logger.info(f"Services with incidents: {len(fp_stats['open_incidents_by_service'])}")

        # Show severity breakdown
        by_severity = fp_stats['open_incidents_by_severity']
        if by_severity:
            severity_summary = ", ".join([f"{sev}: {count}" for sev, count in by_severity.items()])
            logger.info(f"By severity: {severity_summary}")
    else:
        logger.info("No active incidents in database")


def _display_system_status(status: Dict[str, Any], fingerprinter: Optional['AnomalyFingerprinter'], verbose: bool) -> None:
    """Display comprehensive system status"""
    if not verbose:
        return

    logger.info(f"System Status: {status['status']}")
    logger.info(f"Available Services: {status['available_services']}")
    logger.info(f"Explainable Services: {status['explainable_services']} ({status['explainability_coverage']})")
    logger.info(f"Services: {', '.join(status['services'][:5])}{'...' if len(status['services']) > 5 else ''}")
    logger.info(f"VictoriaMetrics: {status['vm_status']}")

    # Show feature status
    features = status['features']
    logger.info("Enhanced Features:")
    for feature, enabled in features.items():
        status_icon = "enabled" if enabled else "disabled"
        feature_name = feature.replace('_', ' ').title()
        logger.info(f"  {feature_name}: {status_icon}")

    # Add fingerprinting feature status
    fingerprinting_status = "enabled" if fingerprinter else "disabled"
    logger.info(f"  Anomaly Fingerprinting: {fingerprinting_status}")


def _handle_no_models_error(verbose: bool) -> None:
    """Handle the case when no models are found"""
    if verbose:
        logger.error("No trained models found. Please run the training pipeline first.")
        logger.info("   uv run main.py")
    else:
        error_result = {
            "error": "no_models",
            "message": "No trained models found. Please run the training pipeline first.",
            "timestamp": datetime.now().isoformat()
        }
        print(json.dumps(error_result, indent=2))


def _apply_fingerprinting(
    results: dict[str, InferenceResultDict],
    fingerprinter: Optional['AnomalyFingerprinter'],
    verbose: bool,
    pipeline: Optional['SmartboxMLInferencePipeline'] = None,
) -> tuple[dict[str, InferenceResultDict], FingerprintingStats]:
    """Apply fingerprinting to inference results.

    Args:
        results: Raw inference results by service.
        fingerprinter: Optional fingerprinter for incident tracking.
        verbose: Whether to log verbose output.
        pipeline: Optional pipeline for SLO evaluator access.

    Returns:
        Tuple of (fingerprinted_results, fingerprinting_stats).
    """
    if not fingerprinter:
        # Return results unchanged with empty stats
        return results, {
            'enhanced_services': 0,
            'creates': 0,
            'updates': 0,
            'resolves': 0,
            'fingerprinting_errors': 0,
            'resolved_incidents': []
        }

    # Extract SLO evaluator from pipeline if available
    slo_evaluator = pipeline.slo_evaluator if pipeline else None

    fingerprinted_results = {}
    fp_stats = {
        'enhanced_services': 0,
        'creates': 0,
        'updates': 0,
        'resolves': 0,
        'fingerprinting_errors': 0
    }

    # Collect all resolved incidents across all services
    all_resolved_incidents = []
    resolution_summary = {}  # Track by service

    for service_name, result in results.items():
        if isinstance(result, dict) and not result.get('error'):
            try:
                full_service_name = _determine_full_service_name(service_name, result)

                if verbose:
                    logger.info(f"Processing {service_name} as {full_service_name}")

                enhanced_result = fingerprinter.process_anomalies(
                    full_service_name=full_service_name,
                    anomaly_result=result,
                    slo_evaluator=slo_evaluator,
                )

                fingerprinted_results[service_name] = enhanced_result
                fp_stats['enhanced_services'] += 1

                # Collect fingerprinting statistics
                fingerprinting_data = enhanced_result.get('fingerprinting', {})
                action_summary = fingerprinting_data.get('action_summary', {})

                fp_stats['creates'] += action_summary.get('incident_creates', 0)
                fp_stats['updates'] += action_summary.get('incident_continues', 0)
                fp_stats['resolves'] += action_summary.get('incident_closes', 0)

                # NEW: Collect resolved incidents for API notification
                resolved_incidents = fingerprinting_data.get('resolved_incidents', [])
                all_resolved_incidents.extend(resolved_incidents)

                # Track resolutions by service for summary
                if resolved_incidents:
                    resolution_summary[service_name] = len(resolved_incidents)

                if verbose and fingerprinting_data.get('overall_action') != 'NO_CHANGE':
                    overall_action = fingerprinting_data.get('overall_action', 'UNKNOWN')
                    logger.info(f"{service_name}: {overall_action} fingerprinting action")

            except (KeyError, ValueError, TypeError) as e:
                # Data structure errors during fingerprinting - recoverable
                logger.warning(
                    f"Fingerprinting failed for {service_name}: {e}",
                    exc_info=verbose  # Full stack trace only in verbose mode
                )
                fingerprinted_results[service_name] = result
                fp_stats['fingerprinting_errors'] += 1
            except Exception as e:
                # Unexpected error - log with full stack trace always
                logger.error(
                    f"Unexpected fingerprinting error for {service_name}: {e}",
                    exc_info=True
                )
                fingerprinted_results[service_name] = result
                fp_stats['fingerprinting_errors'] += 1
        else:
            fingerprinted_results[service_name] = result

    # Store resolved incidents and summary for API notification
    fp_stats['resolved_incidents'] = all_resolved_incidents
    fp_stats['resolution_summary'] = resolution_summary

    # Enhanced logging for multiple resolutions
    if verbose and all_resolved_incidents:
        total_resolved = len(all_resolved_incidents)
        services_count = len(resolution_summary)
        logger.info(f"Batch resolution: {total_resolved} incidents across {services_count} services")

        for service, count in resolution_summary.items():
            logger.info(f"  {service}: {count} incidents resolved")

    return fingerprinted_results, fp_stats


def _determine_full_service_name(service_name: str, result: Dict) -> str:
    """Determine the full service name including model period"""
    # Check if service_name already includes model info
    # Using 3-period model: business_hours, evening_hours, night_hours (all 7 days)
    model_periods = ['business_hours', 'evening_hours', 'night_hours']
    if '_' in service_name and any(model in service_name for model in model_periods):
        return service_name

    # Determine model from result or use default
    model_name = 'evening_hours'  # Default

    if 'time_period' in result and result['time_period']:
        model_name = result['time_period']
    elif 'model_type' in result and 'time_aware' in result.get('model_type', ''):
        # Infer from current time
        current_hour = datetime.now().hour
        if 8 <= current_hour < 18:
            model_name = 'business_hours'
        elif 22 <= current_hour or current_hour < 6:
            model_name = 'night_hours'
        else:
            model_name = 'evening_hours'

    return f"{service_name}_{model_name}"


def _generate_correlation_id() -> str:
    """Generate a unique correlation ID for grouped anomalies."""
    import uuid
    return f"corr_{uuid.uuid4().hex[:12]}"


def _select_primary_anomaly(
    anomalies: dict[str, dict],
    selection_strategy: str = "highest_confidence"
) -> tuple[str, dict]:
    """Select the primary anomaly from a group based on selection strategy.

    Args:
        anomalies: Dict of anomaly_name -> anomaly_data
        selection_strategy: How to select primary:
            - "highest_confidence": Anomaly with highest confidence score
            - "highest_severity": Anomaly with highest severity level
            - "named_pattern_first": Prefer named patterns over generic ML detections

    Returns:
        Tuple of (primary_name, primary_data)
    """
    if not anomalies:
        return ("", {})

    severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1, "none": 0}

    def get_score(name: str, anomaly: dict) -> tuple:
        """Return a sort key for anomaly ranking."""
        confidence = anomaly.get("confidence", anomaly.get("confidence_score", 0.5))
        severity = anomaly.get("severity", "low")
        severity_score = severity_order.get(severity.lower(), 0)

        # Named patterns (not ending in _anomaly) are preferred
        is_named_pattern = not name.endswith("_anomaly") and "_high" not in name and "_low" not in name

        if selection_strategy == "highest_confidence":
            return (confidence, severity_score, is_named_pattern)
        elif selection_strategy == "highest_severity":
            return (severity_score, confidence, is_named_pattern)
        else:  # named_pattern_first
            return (is_named_pattern, severity_score, confidence)

    # Sort anomalies by score (highest first)
    sorted_anomalies = sorted(
        anomalies.items(),
        key=lambda x: get_score(x[0], x[1]),
        reverse=True
    )

    return sorted_anomalies[0]


def _correlate_service_anomalies(
    anomalies: dict[str, dict],
    correlation_config: "CorrelationConfig"
) -> dict[str, dict]:
    """Correlate multiple anomalies for a single service into a single alert.

    When multiple anomalies are detected for the same service, this function
    groups them into a single correlated alert with:
    - A primary anomaly (selected based on configuration)
    - Contributing anomalies listed for context
    - A correlation ID for tracking

    Args:
        anomalies: Dict of anomaly_name -> anomaly_data for a single service
        correlation_config: Configuration for correlation behavior

    Returns:
        Dict with single correlated anomaly, or original anomalies if correlation
        doesn't apply (e.g., only one anomaly, or correlation disabled)
    """
    from smartbox_anomaly.core.config import CorrelationConfig

    # Don't correlate if fewer than min_anomalies_to_correlate
    if len(anomalies) < correlation_config.min_anomalies_to_correlate:
        return anomalies

    # Select primary anomaly
    primary_name, primary_data = _select_primary_anomaly(
        anomalies, correlation_config.primary_selection
    )

    # Build list of contributing anomalies (excluding primary)
    contributing = []
    for name, data in anomalies.items():
        if name != primary_name:
            contributing.append({
                "name": name,
                "severity": data.get("severity", "unknown"),
                "confidence": data.get("confidence", data.get("confidence_score", 0.5)),
                "pattern_name": data.get("pattern_name", name),
            })

    # Sort contributing by severity then confidence
    severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1, "none": 0}
    contributing.sort(
        key=lambda x: (severity_order.get(x["severity"], 0), x["confidence"]),
        reverse=True
    )

    # Build correlated anomaly
    correlation_id = _generate_correlation_id()

    correlated_anomaly = {
        **primary_data,
        "correlation": {
            "correlation_id": correlation_id,
            "is_correlated": True,
            "primary_anomaly": primary_name,
            "anomaly_count": len(anomalies),
            "contributing_anomalies": contributing,
        },
        # Update description to reflect correlation
        "description": (
            f"{primary_data.get('description', 'Multiple anomalies detected')} "
            f"(correlated with {len(contributing)} other anomalies)"
        ),
    }

    # Return with primary anomaly name as key, correlation info embedded
    return {primary_name: correlated_anomaly}


def _update_results_processor(
    pipeline: 'SmartboxMLInferencePipeline',
    results: dict[str, InferenceResultDict],
    fp_stats: FingerprintingStats,
) -> None:
    """Update the results processor with both active anomalies and resolved incidents.

    Formats results using the API payload specification before storing.
    Only sends CONFIRMED anomalies (status=OPEN or RECOVERING) to avoid creating
    orphaned incidents in the web API for unconfirmed SUSPECTED anomalies.

    Also filters based on alerting configuration:
    - severity_threshold: Only anomalies at or above this severity are sent
    - non_alerting_patterns: Patterns that are logged but never sent to API
    """
    formatted_payloads = []

    # Get alerting configuration
    config = get_config()
    alerting_config = config.alerting

    # Add ACTIVE anomalies - only CONFIRMED ones (not SUSPECTED) that pass alerting filter
    for service_name, result in results.items():
        if isinstance(result, dict) and not result.get('error'):
            anomalies = result.get('anomalies', {})

            if isinstance(anomalies, dict):
                # Filter to only confirmed anomalies (OPEN or RECOVERING status)
                # SUSPECTED anomalies should not be sent to web API
                confirmed_anomalies = {
                    name: anomaly for name, anomaly in anomalies.items()
                    if anomaly.get('is_confirmed', False) or
                       anomaly.get('status') in ('OPEN', 'RECOVERING')
                }

                # Apply alerting filter (severity threshold and non-alerting patterns)
                alertable_anomalies = {}
                for name, anomaly in confirmed_anomalies.items():
                    severity = anomaly.get('severity', 'low')
                    pattern_name = anomaly.get('pattern_name') or name

                    if alerting_config.should_alert(severity, pattern_name):
                        alertable_anomalies[name] = anomaly
                    elif alerting_config.log_below_threshold:
                        # Log the filtered anomaly for analytics
                        logger.info(
                            f"[FILTERED] {service_name}:{pattern_name} "
                            f"(severity={severity}) - below threshold or non-alerting pattern"
                        )

                if alertable_anomalies:
                    # Apply correlation if enabled (groups multiple anomalies into single alert)
                    correlation_config = alerting_config.correlation
                    if correlation_config.enabled:
                        original_count = len(alertable_anomalies)
                        alertable_anomalies = _correlate_service_anomalies(
                            alertable_anomalies, correlation_config
                        )
                        if len(alertable_anomalies) < original_count:
                            logger.info(
                                f"[CORRELATED] {service_name}: {original_count} anomalies → "
                                f"1 correlated alert"
                            )

                    # Create a copy of result with only alertable anomalies
                    filtered_result = {**result, 'anomalies': alertable_anomalies}
                    filtered_result['anomaly_count'] = len(alertable_anomalies)
                    # Format using the API spec formatter
                    formatted_payload = pipeline.results_processor._format_time_aware_alert_json(filtered_result)
                    formatted_payloads.append(formatted_payload)
            elif isinstance(anomalies, list) and anomalies:
                # Legacy list format - filter confirmed anomalies
                confirmed_anomalies = [
                    a for a in anomalies
                    if a.get('is_confirmed', False) or
                       a.get('status') in ('OPEN', 'RECOVERING')
                ]

                # Apply alerting filter for list format
                alertable_anomalies = []
                for anomaly in confirmed_anomalies:
                    severity = anomaly.get('severity', 'low')
                    pattern_name = anomaly.get('pattern_name') or anomaly.get('type', '')

                    if alerting_config.should_alert(severity, pattern_name):
                        alertable_anomalies.append(anomaly)
                    elif alerting_config.log_below_threshold:
                        logger.info(
                            f"[FILTERED] {service_name}:{pattern_name} "
                            f"(severity={severity}) - below threshold or non-alerting pattern"
                        )

                if alertable_anomalies:
                    filtered_result = {**result, 'anomalies': alertable_anomalies}
                    filtered_result['anomaly_count'] = len(alertable_anomalies)
                    formatted_payload = pipeline.results_processor._format_time_aware_alert_json(filtered_result)
                    formatted_payloads.append(formatted_payload)

    # Add RESOLVED incidents as special payloads
    resolved_incidents = fp_stats.get('resolved_incidents', [])
    for resolved_incident in resolved_incidents:
        resolution_payload = {
            "alert_type": "incident_resolved",
            "service_name": resolved_incident['service_name'],
            "timestamp": resolved_incident['resolved_at'],
            "incident_id": resolved_incident['incident_id'],
            "fingerprint_id": resolved_incident['fingerprint_id'],
            "anomaly_name": resolved_incident['anomaly_name'],
            "resolution_details": {
                "final_severity": resolved_incident['final_severity'],
                "total_occurrences": resolved_incident['total_occurrences'],
                "incident_duration_minutes": resolved_incident['incident_duration_minutes'],
                "first_seen": resolved_incident['first_seen'],
                "last_detected_by_model": resolved_incident.get('last_detected_by_model', 'unknown')
            },
            "model_type": "incident_resolution"
        }
        formatted_payloads.append(resolution_payload)

    pipeline.results_processor.detected_anomalies = formatted_payloads


def _display_execution_summary(
    results: dict[str, InferenceResultDict],
    fp_stats: FingerprintingStats,
    verbose: bool,
) -> None:
    """Display comprehensive execution summary"""
    if not verbose:
        return

    logger.info("Execution Summary:")
    logger.info(f"Services Evaluated: {len(results)}")

    # Analyze results
    total_anomalies = 0
    services_with_anomalies = 0
    successful_services = 0
    explainable_alerts = 0

    for result in results.values():
        if isinstance(result, dict) and not result.get('error'):
            successful_services += 1

            # Count anomalies (handle both original and fingerprinted formats)
            anomalies = result.get('anomalies', [])
            if isinstance(anomalies, dict):
                anomaly_count = len(anomalies)
            else:
                anomaly_count = len(anomalies) if hasattr(anomalies, '__len__') else 0

            total_anomalies += anomaly_count

            if anomaly_count > 0:
                services_with_anomalies += 1

                # Check if alert has explainability features
                if (result.get('explainable', False) or
                    'historical_context' in result or
                    'explanation' in result):
                    explainable_alerts += 1

    logger.info(f"Total Anomalies: {total_anomalies}")
    logger.info(f"Services with Anomalies: {services_with_anomalies}")
    logger.info(f"Explainable Alerts: {explainable_alerts}/{services_with_anomalies}")
    logger.info(f"Successful Services: {successful_services}/{len(results)}")

    # Fingerprinting summary
    if fp_stats['enhanced_services'] > 0:
        logger.info("Fingerprinting Summary:")
        logger.info(f"Enhanced Services: {fp_stats['enhanced_services']}/{len(results)}")
        logger.info(f"Lifecycle Actions: {fp_stats['creates']} creates, "
              f"{fp_stats['updates']} updates, "
              f"{fp_stats['resolves']} resolves")

        # Show resolution details if any
        resolved_incidents = fp_stats.get('resolved_incidents', [])
        if resolved_incidents:
            resolution_summary = fp_stats.get('resolution_summary', {})
            logger.info(f"Resolved Incidents: {len(resolved_incidents)} total")
            for service, count in resolution_summary.items():
                logger.info(f"  {service}: {count} resolved")

        if fp_stats['fingerprinting_errors'] > 0:
            logger.info(f"Fingerprinting Errors: {fp_stats['fingerprinting_errors']}")


def _send_to_observability_service(
    detected_anomalies: list[dict[str, Any]],
    fp_stats: FingerprintingStats,
    verbose: bool,
) -> None:
    """Send anomalies and resolutions to appropriate observability service endpoints"""

    # Load observability config
    config = get_config()
    obs_config = config.observability

    # Check if API integration is enabled
    if not obs_config.enabled:
        if verbose:
            logger.info("Observability API integration is disabled in config")
        return

    # Separate current anomalies from resolved incidents
    active_anomalies = [item for item in detected_anomalies
                       if item.get('alert_type') != 'incident_resolved']

    resolved_incidents = [item for item in detected_anomalies
                         if item.get('alert_type') == 'incident_resolved']

    # Send active anomalies to the main anomalies endpoint
    if active_anomalies:
        _send_active_anomalies(active_anomalies, obs_config, verbose)

    # Send resolved incidents to a dedicated resolutions endpoint
    if resolved_incidents:
        _send_resolved_incidents(resolved_incidents, obs_config, verbose)

    # Summary logging
    if active_anomalies or resolved_incidents:
        if verbose:
            logger.info(f"Mixed batch sent: {len(active_anomalies)} active anomalies, {len(resolved_incidents)} incident resolutions")
        else:
            logger.info(f"Sent {len(active_anomalies)} active anomalies and {len(resolved_incidents)} resolutions")
    else:
        logger.info("No anomalies or resolutions to send to observability service")


def _send_active_anomalies(anomalies: List[Dict], obs_config: ObservabilityConfig, verbose: bool) -> None:
    """Send active anomalies to the main anomalies endpoint"""
    try:
        r = requests.post(
            obs_config.anomalies_url,
            json=anomalies,
            timeout=obs_config.request_timeout_seconds
        )
        r.raise_for_status()

        if verbose:
            fingerprinted_count = len([a for a in anomalies if 'fingerprinting' in a])
            logger.info(f"Sent {len(anomalies)} active anomalies to {obs_config.anomalies_endpoint} — status {r.status_code}")
            if fingerprinted_count > 0:
                logger.info(f"  {fingerprinted_count} with fingerprinting data")
        else:
            logger.info(f"Sent {len(anomalies)} active anomalies — status {r.status_code}")

    except requests.exceptions.Timeout as e:
        logger.error(f"Timeout sending active anomalies to {obs_config.anomalies_url}: {e}")
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error sending active anomalies to {obs_config.anomalies_url}: {e}")
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error sending active anomalies to {obs_config.anomalies_url}: {e}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed sending active anomalies to {obs_config.anomalies_url}: {e}")


def _send_resolved_incidents(resolutions: List[Dict], obs_config: ObservabilityConfig, verbose: bool) -> None:
    """Send resolved incidents to the dedicated resolutions endpoint"""
    try:
        r = requests.post(
            obs_config.resolutions_url,
            json=resolutions,
            timeout=obs_config.request_timeout_seconds
        )
        r.raise_for_status()

        if verbose:
            logger.info(f"Sent {len(resolutions)} incident resolutions to {obs_config.resolutions_endpoint} — status {r.status_code}")
            # Log some details about what was resolved
            for resolution in resolutions[:3]:  # Show first 3
                service = resolution.get('service_name', resolution.get('service', 'unknown'))
                incident_id = resolution['incident_id']
                duration = resolution['resolution_details']['incident_duration_minutes']
                logger.info(f"  Resolved {service}/{incident_id} after {duration}min")
            if len(resolutions) > 3:
                logger.info(f"  ... and {len(resolutions) - 3} more")
        else:
            logger.info(f"Sent {len(resolutions)} incident resolutions — status {r.status_code}")

    except requests.exceptions.Timeout as e:
        logger.error(f"Timeout sending incident resolutions to {obs_config.resolutions_url}: {e}")
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error sending incident resolutions to {obs_config.resolutions_url}: {e}")
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error sending incident resolutions to {obs_config.resolutions_url}: {e}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed sending incident resolutions to {obs_config.resolutions_url}: {e}")


def _display_tips(status: Dict, fp_stats: Dict) -> None:
    """Display helpful tips based on system status"""
    tips = []

    if status['explainable_services'] < status['available_services']:
        missing_explainable = status['available_services'] - status['explainable_services']
        tips.append(f"{missing_explainable} services could benefit from retraining with explainability features")

    if fp_stats['fingerprinting_errors'] > 0:
        tips.append("Some fingerprinting errors occurred - check service name formats")

    if tips:
        logger.info("Tips:")
        for tip in tips:
            logger.info(f"  • {tip}")


def _handle_pipeline_error(e: Exception, verbose: bool) -> None:
    """Handle pipeline execution errors"""
    logger.error(f"Enhanced pipeline execution failed: {e}")
    if verbose:
        logger.error(f"Pipeline failed: {e}")
    else:
        error_result = {
            "error": "pipeline_failed",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }
        print(json.dumps(error_result, indent=2))


if __name__ == "__main__":
    main()
