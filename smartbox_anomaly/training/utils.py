"""Training utility functions for disk and directory management.

This module provides utilities for:
- Disk space checking before training
- Backup creation and rollback
- Model promotion from staging to production
- Directory cleanup and management
"""

from __future__ import annotations

import argparse
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from smartbox_anomaly.core.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Date Parsing
# =============================================================================


def parse_date(date_str: str) -> datetime:
    """Parse date string in YYYY-MM-DD format.

    Args:
        date_str: Date string in YYYY-MM-DD format.

    Returns:
        Parsed datetime object.

    Raises:
        argparse.ArgumentTypeError: If date format is invalid.
    """
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: {date_str}. Use YYYY-MM-DD.")


# =============================================================================
# Directory Size Utilities
# =============================================================================


def get_directory_size(path: Path) -> int:
    """Calculate total size of a directory in bytes.

    Args:
        path: Path to directory.

    Returns:
        Total size in bytes, or 0 if directory doesn't exist.
    """
    if not path.exists():
        return 0

    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = Path(dirpath) / filename
            try:
                total_size += filepath.stat().st_size
            except (OSError, FileNotFoundError):
                pass  # Skip files we can't access
    return total_size


def format_bytes(size_bytes: int) -> str:
    """Format bytes as human-readable string.

    Args:
        size_bytes: Size in bytes.

    Returns:
        Human-readable string (e.g., "1.5 GB").
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(size_bytes) < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


# =============================================================================
# Disk Space Management
# =============================================================================


def check_disk_space(
    staging_dir: str,
    production_dir: str,
    buffer_percent: float = 0.2,
) -> Dict[str, Any]:
    """Check if there's enough disk space for training and promotion.

    Pre-flight check before training starts. Estimates space needed for:
    - Staging directory (new models)
    - Backup of production (copy of current models)
    - Safety buffer (20% by default)

    Args:
        staging_dir: Path where staging models will be written.
        production_dir: Path to current production models (for backup size estimate).
        buffer_percent: Additional buffer as fraction (0.2 = 20%).

    Returns:
        Dictionary with space check results including:
        - sufficient: Whether there's enough space
        - available: Available bytes
        - required: Required bytes (with buffer)
        - error: Error message if insufficient
    """
    staging_path = Path(staging_dir)
    production_path = Path(production_dir)

    # Get current production size (this is what we'll need for backup)
    production_size = get_directory_size(production_path)

    # Estimate staging size as ~same as production (or use a minimum)
    # If no production exists yet, assume 500MB minimum for staging
    min_staging_estimate = 500 * 1024 * 1024  # 500 MB
    staging_estimate = max(production_size, min_staging_estimate)

    # Total space needed: staging + backup + buffer
    space_needed = staging_estimate + production_size
    space_with_buffer = int(space_needed * (1 + buffer_percent))

    # Check available disk space on the target filesystem
    # Use parent directory in case staging_dir doesn't exist yet
    check_path = staging_path if staging_path.exists() else staging_path.parent
    if not check_path.exists():
        check_path = Path.cwd()

    try:
        disk_usage = shutil.disk_usage(check_path)
        available_space = disk_usage.free
    except OSError as e:
        return {
            'sufficient': False,
            'error': f"Failed to check disk space: {e}",
            'available': 0,
            'required': space_with_buffer,
        }

    sufficient = available_space >= space_with_buffer

    return {
        'sufficient': sufficient,
        'available': available_space,
        'available_human': format_bytes(available_space),
        'required': space_with_buffer,
        'required_human': format_bytes(space_with_buffer),
        'production_size': production_size,
        'production_size_human': format_bytes(production_size),
        'staging_estimate': staging_estimate,
        'staging_estimate_human': format_bytes(staging_estimate),
        'buffer_percent': buffer_percent,
        'error': None if sufficient else (
            f"Insufficient disk space: need {format_bytes(space_with_buffer)}, "
            f"have {format_bytes(available_space)}"
        ),
    }


# =============================================================================
# Backup and Restore
# =============================================================================


def create_backup(production_dir: str, backup_dir: str) -> Dict[str, Any]:
    """Create a backup of production models.

    Uses a single backup directory (not timestamped) that gets replaced
    each training run. This prevents backup accumulation.

    Args:
        production_dir: Path to production models.
        backup_dir: Path where backup should be created.

    Returns:
        Dictionary with backup results including:
        - success: Whether backup succeeded
        - backup_path: Path to backup (or None if no backup needed)
        - size: Size of backup in bytes
    """
    production_path = Path(production_dir)
    backup_path = Path(backup_dir)

    if not production_path.exists():
        return {
            'success': True,
            'backup_path': None,
            'message': 'No production models to backup (first run)',
            'size': 0,
        }

    # Remove existing backup if it exists
    if backup_path.exists():
        try:
            shutil.rmtree(backup_path)
            logger.debug("Removed existing backup: %s", backup_path)
        except Exception as e:
            return {
                'success': False,
                'backup_path': None,
                'error': f"Failed to remove existing backup: {e}",
                'size': 0,
            }

    # Create new backup
    try:
        shutil.copytree(production_path, backup_path)
        backup_size = get_directory_size(backup_path)
        logger.info("Created backup: %s (%s)", backup_path, format_bytes(backup_size))
        return {
            'success': True,
            'backup_path': str(backup_path),
            'message': f"Backup created successfully",
            'size': backup_size,
            'size_human': format_bytes(backup_size),
        }
    except Exception as e:
        return {
            'success': False,
            'backup_path': None,
            'error': f"Failed to create backup: {e}",
            'size': 0,
        }


def rollback_from_backup(backup_dir: str, production_dir: str) -> Dict[str, Any]:
    """Restore production models from backup.

    Args:
        backup_dir: Path to backup directory.
        production_dir: Path to production directory to restore.

    Returns:
        Dictionary with rollback results including:
        - success: Whether rollback succeeded
        - services_restored: List of restored service names
    """
    backup_path = Path(backup_dir)
    production_path = Path(production_dir)

    if not backup_path.exists():
        return {
            'success': False,
            'error': f"Backup directory does not exist: {backup_dir}",
        }

    # Get list of services in backup
    backup_services = [d.name for d in backup_path.iterdir() if d.is_dir()]

    if not backup_services:
        return {
            'success': False,
            'error': f"No services found in backup directory: {backup_dir}",
        }

    # Remove current production if it exists
    if production_path.exists():
        try:
            shutil.rmtree(production_path)
            logger.info("Removed current production directory for rollback")
        except Exception as e:
            return {
                'success': False,
                'error': f"Failed to remove production directory: {e}",
            }

    # Restore from backup
    try:
        shutil.copytree(backup_path, production_path)
        logger.info("Restored production from backup: %s", backup_path)
        return {
            'success': True,
            'services_restored': backup_services,
            'backup_path': str(backup_path),
            'production_path': str(production_path),
        }
    except Exception as e:
        return {
            'success': False,
            'error': f"Failed to restore from backup: {e}",
        }


# =============================================================================
# Directory Cleanup
# =============================================================================


def cleanup_directory(dir_path: str, description: str = "directory") -> Dict[str, Any]:
    """Remove a directory and all its contents.

    Args:
        dir_path: Path to directory to remove.
        description: Human-readable description for logging.

    Returns:
        Dictionary with cleanup results including:
        - success: Whether cleanup succeeded
        - removed: Whether anything was removed
        - size_freed: Bytes freed by cleanup
    """
    path = Path(dir_path)

    if not path.exists():
        return {
            'success': True,
            'removed': False,
            'message': f"{description} does not exist, nothing to clean",
            'path': str(path),
        }

    try:
        size_before = get_directory_size(path)
        shutil.rmtree(path)
        logger.info("Cleaned up %s: %s (freed %s)", description, path, format_bytes(size_before))
        return {
            'success': True,
            'removed': True,
            'message': f"Successfully removed {description}",
            'path': str(path),
            'size_freed': size_before,
            'size_freed_human': format_bytes(size_before),
        }
    except Exception as e:
        logger.error("Failed to cleanup %s at %s: %s", description, path, e)
        return {
            'success': False,
            'removed': False,
            'error': f"Failed to remove {description}: {e}",
            'path': str(path),
        }


# =============================================================================
# Model Promotion
# =============================================================================


def promote_models(staging_dir: str, production_dir: str) -> Dict[str, Any]:
    """Promote models from staging to production directory.

    Note: Backup should be created BEFORE calling this function using create_backup().
    This function does not create backups - it only handles the promotion.

    Args:
        staging_dir: Path to staging models directory.
        production_dir: Path to production models directory.

    Returns:
        Dictionary with promotion results including:
        - success: Whether all promotions succeeded
        - services_promoted: List of successfully promoted services
        - services_failed: List of failed services with errors
    """
    staging_path = Path(staging_dir)
    production_path = Path(production_dir)

    if not staging_path.exists():
        return {
            'success': False,
            'error': f"Staging directory does not exist: {staging_dir}",
            'services_promoted': [],
            'services_failed': [],
        }

    # Get list of services in staging
    staging_services = [d.name for d in staging_path.iterdir() if d.is_dir()]

    if not staging_services:
        return {
            'success': False,
            'error': f"No services found in staging directory: {staging_dir}",
            'services_promoted': [],
            'services_failed': [],
        }

    # Ensure production directory exists
    production_path.mkdir(parents=True, exist_ok=True)

    # Promote each service
    promoted_services = []
    failed_services = []

    for service in staging_services:
        staging_service_path = staging_path / service
        production_service_path = production_path / service

        try:
            # Remove existing production service directory if it exists
            if production_service_path.exists():
                shutil.rmtree(production_service_path)

            # Copy staging to production
            shutil.copytree(staging_service_path, production_service_path)
            promoted_services.append(service)
            logger.info("Promoted: %s", service)

        except Exception as e:
            failed_services.append({'service': service, 'error': str(e)})
            logger.error("Failed to promote %s: %s", service, e)

    return {
        'success': len(failed_services) == 0,
        'services_promoted': promoted_services,
        'services_failed': failed_services,
        'staging_dir': staging_dir,
        'production_dir': production_dir,
    }
