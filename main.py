#!/usr/bin/env python3
"""CLI for training Smartbox anomaly detection models.

This script provides a command-line interface for training, validating,
promoting, and rolling back anomaly detection models. It uses the training
pipeline from smartbox_anomaly.training.

Example usage:
    # Train all services (trains to staging, auto-promotes if all pass)
    python main.py

    # Train specific service
    python main.py --service booking

    # Train with specific date range
    python main.py --start-date 2025-08-01 --end-date 2025-10-31

    # Train directly to production (bypasses staging)
    python main.py --direct

    # Manually promote staging to production
    python main.py --promote

    # Rollback to previous production models
    python main.py --rollback
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Final

from smartbox_anomaly.training import (
    SmartboxTrainingPipeline,
    parse_date,
    check_disk_space,
    create_backup,
    rollback_from_backup,
    cleanup_directory,
    promote_models,
    format_bytes,
)

__all__ = ["main"]

# =============================================================================
# Constants
# =============================================================================

# Exit codes for different failure modes
EXIT_SUCCESS: Final[int] = 0
EXIT_TRAINING_FAILURE: Final[int] = 1
EXIT_CONFIG_ERROR: Final[int] = 2
EXIT_DISK_SPACE_ERROR: Final[int] = 3
EXIT_BACKUP_ERROR: Final[int] = 4
EXIT_PROMOTION_ERROR: Final[int] = 5
EXIT_ROLLBACK_ERROR: Final[int] = 6

# Result status strings returned by the training pipeline
# Note: These should ideally be an enum in the training module
RESULT_STATUS_SUCCESS: Final[str] = "success"
RESULT_STATUS_ERROR: Final[str] = "error"

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Path Validation
# =============================================================================


def _validate_directory_path(path_str: str) -> Path:
    """Validate and normalize a directory path argument.

    Args:
        path_str: Raw path string from user input or environment.

    Returns:
        Normalized, resolved Path object.

    Raises:
        argparse.ArgumentTypeError: If path contains suspicious patterns.
    """
    # Expand user home directory (~) and resolve to absolute path
    path = Path(path_str).expanduser().resolve()

    # Basic security checks
    path_str_resolved = str(path)

    # Check for null bytes (path injection)
    if "\x00" in path_str:
        raise argparse.ArgumentTypeError(f"Invalid path (contains null byte): {path_str}")

    # Warn about paths outside current working tree (but allow them)
    cwd = Path.cwd().resolve()
    try:
        path.relative_to(cwd)
    except ValueError:
        # Path is outside CWD - this is allowed but worth logging
        logger.debug("Path is outside current directory: %s", path)

    return path


def main() -> int:
    """Main entry point for the training CLI."""
    parser = argparse.ArgumentParser(
        description="Train anomaly detection models for Smartbox services",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all services (trains to staging, auto-promotes if all pass validation)
  python main.py

  # Train specific service
  python main.py --service booking

  # Train with specific date range (Aug-Oct for baseline without holiday traffic)
  python main.py --start-date 2025-08-01 --end-date 2025-10-31

  # Train directly to production (bypasses staging and validation gate)
  python main.py --direct

  # Manually promote staging models to production (if auto-promotion was blocked)
  python main.py --promote

  # Rollback to previous production models (if new models are bad)
  python main.py --rollback

  # Keep backup after successful promotion (for extra safety)
  python main.py --keep-backup

Default Behavior:
  1. Check disk space (need room for staging + backup)
  2. Backup current production models to ./smartbox_models_backup/
  3. Clean any existing staging directory
  4. Train models to staging directory (./smartbox_models_staging/)
  5. If ALL models pass validation:
     - Promote staging to production
     - Clean up staging directory
     - Clean up backup (unless --keep-backup)
  6. If ANY model fails:
     - Keep backup for rollback
     - Keep staging for debugging
     - Exit with error code 1

Data is automatically cached to parquet files in ./metrics_cache/. On first run,
data is fetched from VictoriaMetrics. Subsequent runs load from cache instantly.
        """
    )
    parser.add_argument(
        "--service", "-s",
        type=str,
        help="Train only a specific service (default: train all configured services)"
    )
    parser.add_argument(
        "--start-date",
        type=parse_date,
        help="Start date for training data in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--end-date",
        type=parse_date,
        help="End date for training data in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=30,
        help="Number of days to look back for training data (default: 30, ignored if --start-date/--end-date provided)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./metrics_cache",
        help="Directory for metrics cache. Data is automatically cached to parquet files. (default: ./metrics_cache)"
    )
    parser.add_argument(
        "--parallel", "-p",
        type=int,
        default=1,
        choices=range(1, 9),
        metavar="{1-8}",
        help="Number of parallel training threads (1=sequential, 2-8=parallel). Default: 1"
    )
    parser.add_argument(
        "--warm-cache",
        action="store_true",
        help="Pre-populate cache for all services before training. Recommended when using --parallel"
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Train directly to production directory, bypassing staging and validation gate"
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Promote staging models to production (copy from staging to production directory)"
    )
    parser.add_argument(
        "--staging-dir",
        type=_validate_directory_path,
        default=_validate_directory_path(os.environ.get("STAGING_DIR", "./smartbox_models_staging")),
        help="Staging directory for models (env: STAGING_DIR, default: ./smartbox_models_staging)"
    )
    parser.add_argument(
        "--models-dir",
        type=_validate_directory_path,
        default=_validate_directory_path(os.environ.get("MODELS_DIR", "./smartbox_models")),
        help="Production directory for models (env: MODELS_DIR, default: ./smartbox_models)"
    )
    parser.add_argument(
        "--backup-dir",
        type=_validate_directory_path,
        default=_validate_directory_path(os.environ.get("BACKUP_DIR", "./smartbox_models_backup")),
        help="Backup directory for production models (env: BACKUP_DIR, default: ./smartbox_models_backup)"
    )
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="Rollback production models from backup (restores previous version)"
    )
    parser.add_argument(
        "--keep-backup",
        action="store_true",
        help="Keep backup after successful promotion (default: cleanup backup on success)"
    )
    parser.add_argument(
        "--skip-disk-check",
        action="store_true",
        help="Skip disk space pre-flight check (use with caution)"
    )

    args = parser.parse_args()

    # Validate date arguments
    if (args.start_date is None) != (args.end_date is None):
        parser.error("--start-date and --end-date must be used together")

    if args.start_date and args.end_date and args.start_date >= args.end_date:
        parser.error("--start-date must be before --end-date")

    # Configure logging for the main module
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Handle --rollback mode: restore production from backup and exit
    if args.rollback:
        return handle_rollback(args)

    # Handle --promote mode: manually promote staging to production
    if args.promote:
        return handle_promote(args)

    # Run training
    return handle_training(args)


def handle_rollback(args: argparse.Namespace) -> int:
    """Handle --rollback mode: restore production from backup.

    Args:
        args: Parsed command line arguments containing backup_dir and models_dir.

    Returns:
        EXIT_SUCCESS on successful rollback, EXIT_ROLLBACK_ERROR on failure.
    """
    logger.info("=" * 60)
    logger.info("ROLLBACK: Restoring production models from backup")
    logger.info("=" * 60)
    logger.info("  Backup: %s", args.backup_dir)
    logger.info("  Production: %s", args.models_dir)

    result = rollback_from_backup(
        backup_dir=str(args.backup_dir),
        production_dir=str(args.models_dir),
    )

    if not result["success"]:
        logger.error("Rollback failed: %s", result.get("error", "Unknown error"))
        return EXIT_ROLLBACK_ERROR

    logger.info("")
    logger.info("Rollback successful!")
    logger.info("  Services restored: %d", len(result["services_restored"]))
    for service in result["services_restored"]:
        logger.info("    ✓ %s", service)
    logger.info("")
    logger.info("Note: Backup directory preserved at: %s", args.backup_dir)
    return EXIT_SUCCESS


def handle_promote(args: argparse.Namespace) -> int:
    """Handle --promote mode: manually promote staging to production.

    Args:
        args: Parsed command line arguments.

    Returns:
        EXIT_SUCCESS on successful promotion, EXIT_BACKUP_ERROR or EXIT_PROMOTION_ERROR on failure.
    """
    logger.info("=" * 60)
    logger.info("MANUAL PROMOTION: Staging → Production")
    logger.info("=" * 60)
    logger.info("  Staging: %s", args.staging_dir)
    logger.info("  Production: %s", args.models_dir)
    logger.info("  Backup: %s", args.backup_dir)

    # Create backup before promotion (hard stop on failure)
    logger.info("")
    logger.info("Creating backup of current production models...")
    backup_result = create_backup(
        production_dir=str(args.models_dir),
        backup_dir=str(args.backup_dir),
    )

    if not backup_result["success"]:
        logger.error("ABORTED: Backup failed - %s", backup_result.get("error"))
        logger.error("Production models are untouched.")
        return EXIT_BACKUP_ERROR

    if backup_result.get("backup_path"):
        logger.info("  Backup created: %s", backup_result["backup_path"])
    else:
        logger.info("  No existing production to backup (first deployment)")

    # Promote staging to production
    logger.info("")
    logger.info("Promoting models...")
    result = promote_models(
        staging_dir=str(args.staging_dir),
        production_dir=str(args.models_dir),
    )

    if not result["success"]:
        logger.error("")
        logger.error("Promotion failed: %s", result.get("error", "Unknown error"))
        if result.get("services_failed"):
            for failed in result["services_failed"]:
                logger.error("  ✗ %s: %s", failed["service"], failed["error"])
        logger.error("")
        logger.error("Backup available for rollback: python main.py --rollback")
        return EXIT_PROMOTION_ERROR

    logger.info("")
    logger.info("Promotion successful!")
    logger.info("  Services promoted: %d", len(result["services_promoted"]))
    for service in result["services_promoted"]:
        logger.info("    ✓ %s", service)

    # Cleanup staging and backup unless --keep-backup
    if not args.keep_backup:
        logger.info("")
        logger.info("Cleaning up...")
        cleanup_directory(str(args.staging_dir), "staging directory")
        cleanup_directory(str(args.backup_dir), "backup directory")
    else:
        logger.info("")
        logger.info("Backup preserved at: %s (--keep-backup)", args.backup_dir)

    return EXIT_SUCCESS


def handle_training(args: argparse.Namespace) -> int:
    """Handle the main training workflow.

    Args:
        args: Parsed command line arguments.

    Returns:
        Exit code indicating success or failure type.
    """
    # Initialize training pipeline
    training_pipeline = SmartboxTrainingPipeline(cache_dir=args.cache_dir)

    # Determine services to train
    if args.service:
        services = [args.service]
        logger.info("Training single service: %s", args.service)
    else:
        # Load services from config.json (falls back to discovery if not found)
        services = training_pipeline.load_services_from_config()
        if not services:
            logger.info("Discovering services from VictoriaMetrics...")
            services = training_pipeline.discover_services()

    if not services:
        logger.error("No services found to train. Please check config.json or VictoriaMetrics connection.")
        return EXIT_CONFIG_ERROR

    # Determine models directory (staging by default, or direct to production)
    if args.direct:
        models_dir = str(args.models_dir)
        logger.info("Training DIRECTLY to production directory: %s", models_dir)
        logger.info("(validation gate bypassed)")
    else:
        models_dir = str(args.staging_dir)
        logger.info("Training to staging directory: %s", models_dir)
        logger.info("(will auto-promote to production if all models pass validation)")

    # Log date range information
    if args.start_date and args.end_date:
        logger.info("Training date range: %s to %s (%d days)",
                    args.start_date.strftime("%Y-%m-%d"),
                    args.end_date.strftime("%Y-%m-%d"),
                    (args.end_date - args.start_date).days)
    else:
        logger.info("Training with last %d days of data", args.lookback_days)

    logger.info("Cache directory: %s", args.cache_dir)
    if args.parallel > 1:
        logger.info("Parallel training: %d workers", args.parallel)
    if args.warm_cache:
        logger.info("Cache warming: enabled")

    # === PRE-FLIGHT CHECKS (only for non-direct mode) ===
    if not args.direct:
        preflight_result = run_preflight_checks(args)
        if preflight_result != EXIT_SUCCESS:
            return preflight_result

    # Use enhanced time-aware training with optional parallelization
    logger.info("Using enhanced time-aware anomaly detection with explainability")

    # Train all services (with optional cache warming and parallel training)
    results = training_pipeline.train_all_services_time_aware(
        service_list=services,
        parallel_workers=args.parallel,
        warm_cache_first=args.warm_cache,
        lookback_days=args.lookback_days,
        start_date=args.start_date,
        end_date=args.end_date,
        cache_dir=args.cache_dir,
        models_dir=models_dir,
    )

    # Print results summary and handle auto-promotion
    return handle_training_results(args, results)


def run_preflight_checks(args: argparse.Namespace) -> int:
    """Run pre-flight checks before training.

    Performs disk space verification, backup creation, and staging cleanup.

    Args:
        args: Parsed command line arguments.

    Returns:
        EXIT_SUCCESS if all checks pass, EXIT_DISK_SPACE_ERROR or EXIT_BACKUP_ERROR on failure.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("PRE-FLIGHT CHECKS")
    logger.info("=" * 60)

    # 1. Disk space check
    if not args.skip_disk_check:
        logger.info("Checking disk space...")
        disk_check = check_disk_space(
            staging_dir=str(args.staging_dir),
            production_dir=str(args.models_dir),
            buffer_percent=0.2,  # 20% buffer
        )

        if disk_check.get('error'):
            logger.warning("  ⚠ Disk check warning: %s", disk_check['error'])
            logger.warning("  Continuing anyway (use --skip-disk-check to suppress)")
        elif not disk_check['sufficient']:
            logger.error("  ✗ Insufficient disk space!")
            logger.error("    Available: %s", format_bytes(disk_check['available']))
            logger.error("    Required:  %s (estimated)", format_bytes(disk_check['required']))
            logger.error("")
            logger.error("Free up disk space or use --skip-disk-check to bypass this check.")
            return EXIT_DISK_SPACE_ERROR
        else:
            logger.info("  ✓ Disk space OK: %s available, ~%s required",
                        format_bytes(disk_check['available']),
                        format_bytes(disk_check['required']))
    else:
        logger.info("Disk space check: skipped (--skip-disk-check)")

    # 2. Create backup of production models (if they exist)
    production_path = Path(args.models_dir)
    if production_path.exists() and any(production_path.iterdir()):
        logger.info("")
        logger.info("Creating backup of production models...")
        backup_result = create_backup(
            production_dir=str(args.models_dir),
            backup_dir=str(args.backup_dir),
        )

        if not backup_result['success']:
            logger.error("  ✗ Backup failed: %s", backup_result.get('error', 'Unknown error'))
            logger.error("")
            logger.error("Training aborted. Fix backup issues before proceeding.")
            return EXIT_BACKUP_ERROR

        logger.info("  ✓ Backup created: %s (%s)",
                    backup_result['backup_path'],
                    format_bytes(backup_result['size']))
    else:
        logger.info("")
        logger.info("No existing production models to backup (first run or empty directory)")

    # 3. Clean staging directory before training
    staging_path = Path(args.staging_dir)
    if staging_path.exists() and any(staging_path.iterdir()):
        logger.info("")
        logger.info("Cleaning staging directory from previous run...")
        cleanup_result = cleanup_directory(str(args.staging_dir), "staging directory")

        if not cleanup_result['success']:
            logger.warning("  ⚠ Staging cleanup failed: %s", cleanup_result.get('error', 'Unknown'))
            logger.warning("  Continuing anyway (may overwrite existing files)")
        else:
            logger.info("  ✓ Staging directory cleaned")

    logger.info("")
    logger.info("=" * 60)
    logger.info("")

    return EXIT_SUCCESS


def handle_training_results(
    args: argparse.Namespace,
    results: dict[str, Any],
) -> int:
    """Handle training results and auto-promotion.

    Args:
        args: Parsed command line arguments.
        results: Training results dictionary from the pipeline, keyed by service name.

    Returns:
        EXIT_SUCCESS on success, EXIT_TRAINING_FAILURE on failure.
    """
    # Print enhanced results summary
    logger.info("")
    logger.info("Final Results Summary:")

    # Track which services passed/failed
    passed_services: list[str] = []
    failed_services: list[dict[str, str]] = []

    for service, result in results.items():
        training_success = result["status"] == RESULT_STATUS_SUCCESS
        validation_results = result.get("validation_results", {})
        validation_summary = validation_results.get("_summary", {})
        validation_passed = validation_summary.get("overall_passed", False)

        explainability_count = result.get("total_explainability_metrics", 0)
        explainability_status = "with explainability" if explainability_count > 0 else "no explainability"

        if training_success and validation_passed:
            passed_services.append(service)
            logger.info("  ✓ %s: %s (%d metrics)", service, explainability_status, explainability_count)
        elif training_success and not validation_passed:
            reason = validation_summary.get("overall_message", "validation failed")
            failed_services.append({"service": service, "reason": reason})
            logger.info("  ⚠ %s: training OK but %s", service, reason)
        else:
            reason = result.get("message", "training failed")
            failed_services.append({"service": service, "reason": reason})
            logger.info("  ✗ %s: %s", service, reason)

    # Print overall summary
    logger.info("")
    logger.info("Overall: %d passed, %d failed", len(passed_services), len(failed_services))

    # Handle direct mode - no promotion needed
    if args.direct:
        if failed_services:
            return EXIT_TRAINING_FAILURE
        return EXIT_SUCCESS

    # Handle auto-promote (default behavior)
    logger.info("")
    logger.info("=" * 60)

    all_passed = len(failed_services) == 0 and len(passed_services) > 0

    if all_passed:
        return handle_auto_promotion(args, passed_services)
    else:
        return handle_promotion_blocked(args, failed_services)


def handle_auto_promotion(args: argparse.Namespace, passed_services: list[str]) -> int:
    """Handle automatic promotion when all services pass.

    Args:
        args: Parsed command line arguments.
        passed_services: List of service names that passed validation.

    Returns:
        EXIT_SUCCESS on successful promotion, EXIT_PROMOTION_ERROR on failure.
    """
    logger.info("All %d services passed validation. Promoting to production...", len(passed_services))

    # Backup was already created before training, so don't create another
    promote_result = promote_models(
        staging_dir=str(args.staging_dir),
        production_dir=str(args.models_dir),
    )

    if not promote_result["success"]:
        logger.error("")
        logger.error("✗ PROMOTION FAILED: %s", promote_result.get("error", "Unknown error"))
        logger.error("")
        logger.error("Staging models preserved: %s", args.staging_dir)
        if Path(args.backup_dir).exists():
            logger.error("Backup available for rollback: python main.py --rollback")
        return EXIT_PROMOTION_ERROR

    logger.info("")
    logger.info("✓ PROMOTION SUCCESSFUL!")
    logger.info("  Services promoted: %d", len(promote_result["services_promoted"]))

    # Cleanup staging directory after successful promotion
    logger.info("")
    logger.info("Cleaning up after successful promotion...")

    staging_cleanup = cleanup_directory(str(args.staging_dir), "staging directory")
    if staging_cleanup["success"]:
        logger.info("  ✓ Staging directory cleaned")
    else:
        logger.warning("  ⚠ Staging cleanup failed: %s", staging_cleanup.get("error", "Unknown"))

    # Cleanup backup unless --keep-backup was specified
    backup_path = Path(args.backup_dir)
    if backup_path.exists():
        if args.keep_backup:
            logger.info("  ✓ Backup retained at: %s (--keep-backup)", args.backup_dir)
        else:
            backup_cleanup = cleanup_directory(str(args.backup_dir), "backup directory")
            if backup_cleanup["success"]:
                logger.info("  ✓ Backup cleaned (no longer needed)")
            else:
                logger.warning("  ⚠ Backup cleanup failed: %s", backup_cleanup.get("error", "Unknown"))

    logger.info("")
    logger.info("Training and promotion completed successfully!")
    logger.info("=" * 60)
    return EXIT_SUCCESS


def handle_promotion_blocked(
    args: argparse.Namespace,
    failed_services: list[dict[str, str]],
) -> int:
    """Handle case when promotion is blocked due to failures.

    Args:
        args: Parsed command line arguments.
        failed_services: List of dicts with 'service' and 'reason' keys for each failed service.

    Returns:
        EXIT_TRAINING_FAILURE since this indicates validation failures.
    """
    logger.error("✗ PROMOTION BLOCKED: %d service(s) failed validation", len(failed_services))
    logger.error("")
    logger.error("Failed services:")
    for failed in failed_services:
        logger.error("  - %s: %s", failed["service"], failed["reason"])
    logger.error("")
    logger.error("Staging models preserved: %s", args.staging_dir)
    if Path(args.backup_dir).exists():
        logger.error("Backup available for rollback: python main.py --rollback")
    logger.error("")
    logger.error("Options:")
    logger.error("  1. Fix issues and re-train: python main.py")
    logger.error("  2. Manually promote anyway: python main.py --promote")
    logger.info("=" * 60)
    return EXIT_TRAINING_FAILURE


if __name__ == "__main__":
    sys.exit(main())
