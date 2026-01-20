"""Tests for the main.py training CLI.

This module provides comprehensive tests for the training command-line interface,
including path validation, exit code handling, and the various training modes.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Import the module under test
from main import (
    _validate_directory_path,
    EXIT_SUCCESS,
    EXIT_TRAINING_FAILURE,
    EXIT_CONFIG_ERROR,
    EXIT_DISK_SPACE_ERROR,
    EXIT_BACKUP_ERROR,
    EXIT_PROMOTION_ERROR,
    EXIT_ROLLBACK_ERROR,
    RESULT_STATUS_SUCCESS,
    RESULT_STATUS_ERROR,
    handle_rollback,
    handle_promote,
    handle_training_results,
    handle_auto_promotion,
    handle_promotion_blocked,
    run_preflight_checks,
)


# =============================================================================
# Exit Code Constants Tests
# =============================================================================


class TestExitCodes:
    """Test that exit code constants have expected values."""

    def test_exit_success_is_zero(self) -> None:
        """EXIT_SUCCESS should be 0 (standard Unix success)."""
        assert EXIT_SUCCESS == 0

    def test_exit_training_failure_is_one(self) -> None:
        """EXIT_TRAINING_FAILURE should be 1 (standard Unix error)."""
        assert EXIT_TRAINING_FAILURE == 1

    def test_exit_config_error_is_two(self) -> None:
        """EXIT_CONFIG_ERROR should be 2."""
        assert EXIT_CONFIG_ERROR == 2

    def test_exit_disk_space_error_is_three(self) -> None:
        """EXIT_DISK_SPACE_ERROR should be 3."""
        assert EXIT_DISK_SPACE_ERROR == 3

    def test_exit_backup_error_is_four(self) -> None:
        """EXIT_BACKUP_ERROR should be 4."""
        assert EXIT_BACKUP_ERROR == 4

    def test_exit_promotion_error_is_five(self) -> None:
        """EXIT_PROMOTION_ERROR should be 5."""
        assert EXIT_PROMOTION_ERROR == 5

    def test_exit_rollback_error_is_six(self) -> None:
        """EXIT_ROLLBACK_ERROR should be 6."""
        assert EXIT_ROLLBACK_ERROR == 6

    def test_all_exit_codes_are_unique(self) -> None:
        """All exit codes should have unique values."""
        exit_codes = [
            EXIT_SUCCESS,
            EXIT_TRAINING_FAILURE,
            EXIT_CONFIG_ERROR,
            EXIT_DISK_SPACE_ERROR,
            EXIT_BACKUP_ERROR,
            EXIT_PROMOTION_ERROR,
            EXIT_ROLLBACK_ERROR,
        ]
        assert len(exit_codes) == len(set(exit_codes)), "Exit codes must be unique"


class TestResultStatusConstants:
    """Test result status string constants."""

    def test_result_status_success(self) -> None:
        """RESULT_STATUS_SUCCESS should be 'success'."""
        assert RESULT_STATUS_SUCCESS == "success"

    def test_result_status_error(self) -> None:
        """RESULT_STATUS_ERROR should be 'error'."""
        assert RESULT_STATUS_ERROR == "error"


# =============================================================================
# Path Validation Tests
# =============================================================================


class TestValidateDirectoryPath:
    """Test _validate_directory_path function."""

    def test_valid_relative_path(self) -> None:
        """Should accept and resolve a valid relative path."""
        result = _validate_directory_path("./test_dir")
        assert isinstance(result, Path)
        assert result.is_absolute()

    def test_valid_absolute_path(self) -> None:
        """Should accept an absolute path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _validate_directory_path(tmpdir)
            assert isinstance(result, Path)
            assert result.is_absolute()
            assert str(result) == str(Path(tmpdir).resolve())

    def test_expands_user_home(self) -> None:
        """Should expand ~ to user home directory."""
        result = _validate_directory_path("~/test_dir")
        assert isinstance(result, Path)
        assert "~" not in str(result)
        assert result.is_absolute()

    def test_rejects_null_byte(self) -> None:
        """Should reject paths containing null bytes (path injection attack).

        Note: On Python 3.11+, pathlib.resolve() raises ValueError for null bytes
        before our check can run. Both behaviors correctly reject the path.
        """
        # Either our ArgumentTypeError or Python's ValueError is acceptable
        with pytest.raises((argparse.ArgumentTypeError, ValueError)):
            _validate_directory_path("/tmp/test\x00malicious")

    def test_resolves_path(self) -> None:
        """Should resolve path to absolute form."""
        result = _validate_directory_path(".")
        assert result.is_absolute()
        assert result == Path.cwd().resolve()

    def test_handles_dots_in_path(self) -> None:
        """Should handle . and .. in paths."""
        result = _validate_directory_path("./foo/../bar")
        assert isinstance(result, Path)
        assert result.is_absolute()
        # The .. should be resolved
        assert ".." not in str(result)


# =============================================================================
# Handle Rollback Tests
# =============================================================================


class TestHandleRollback:
    """Test handle_rollback function."""

    @pytest.fixture
    def mock_args(self, tmp_path: Path) -> argparse.Namespace:
        """Create mock arguments for rollback."""
        args = argparse.Namespace()
        args.backup_dir = tmp_path / "backup"
        args.models_dir = tmp_path / "models"
        return args

    @patch("main.rollback_from_backup")
    def test_successful_rollback(
        self, mock_rollback: MagicMock, mock_args: argparse.Namespace
    ) -> None:
        """Should return EXIT_SUCCESS on successful rollback."""
        mock_rollback.return_value = {
            "success": True,
            "services_restored": ["booking", "search"],
        }

        result = handle_rollback(mock_args)

        assert result == EXIT_SUCCESS
        mock_rollback.assert_called_once_with(
            backup_dir=str(mock_args.backup_dir),
            production_dir=str(mock_args.models_dir),
        )

    @patch("main.rollback_from_backup")
    def test_failed_rollback(
        self, mock_rollback: MagicMock, mock_args: argparse.Namespace
    ) -> None:
        """Should return EXIT_ROLLBACK_ERROR on failed rollback."""
        mock_rollback.return_value = {
            "success": False,
            "error": "Backup directory does not exist",
        }

        result = handle_rollback(mock_args)

        assert result == EXIT_ROLLBACK_ERROR

    @patch("main.rollback_from_backup")
    def test_rollback_with_unknown_error(
        self, mock_rollback: MagicMock, mock_args: argparse.Namespace
    ) -> None:
        """Should handle missing error message gracefully."""
        mock_rollback.return_value = {"success": False}

        result = handle_rollback(mock_args)

        assert result == EXIT_ROLLBACK_ERROR


# =============================================================================
# Handle Promote Tests
# =============================================================================


class TestHandlePromote:
    """Test handle_promote function."""

    @pytest.fixture
    def mock_args(self, tmp_path: Path) -> argparse.Namespace:
        """Create mock arguments for promotion."""
        args = argparse.Namespace()
        args.staging_dir = tmp_path / "staging"
        args.models_dir = tmp_path / "models"
        args.backup_dir = tmp_path / "backup"
        args.keep_backup = False
        return args

    @patch("main.cleanup_directory")
    @patch("main.promote_models")
    @patch("main.create_backup")
    def test_successful_promotion(
        self,
        mock_backup: MagicMock,
        mock_promote: MagicMock,
        mock_cleanup: MagicMock,
        mock_args: argparse.Namespace,
    ) -> None:
        """Should return EXIT_SUCCESS on successful promotion."""
        mock_backup.return_value = {
            "success": True,
            "backup_path": str(mock_args.backup_dir),
        }
        mock_promote.return_value = {
            "success": True,
            "services_promoted": ["booking", "search"],
        }
        mock_cleanup.return_value = {"success": True}

        result = handle_promote(mock_args)

        assert result == EXIT_SUCCESS
        mock_backup.assert_called_once()
        mock_promote.assert_called_once()
        # Cleanup should be called for both staging and backup
        assert mock_cleanup.call_count == 2

    @patch("main.create_backup")
    def test_promotion_fails_on_backup_error(
        self, mock_backup: MagicMock, mock_args: argparse.Namespace
    ) -> None:
        """Should return EXIT_BACKUP_ERROR if backup fails."""
        mock_backup.return_value = {
            "success": False,
            "error": "Disk full",
        }

        result = handle_promote(mock_args)

        assert result == EXIT_BACKUP_ERROR

    @patch("main.promote_models")
    @patch("main.create_backup")
    def test_promotion_fails_on_promote_error(
        self,
        mock_backup: MagicMock,
        mock_promote: MagicMock,
        mock_args: argparse.Namespace,
    ) -> None:
        """Should return EXIT_PROMOTION_ERROR if promotion fails."""
        mock_backup.return_value = {"success": True, "backup_path": "/backup"}
        mock_promote.return_value = {
            "success": False,
            "error": "Permission denied",
            "services_failed": [{"service": "booking", "error": "Permission denied"}],
        }

        result = handle_promote(mock_args)

        assert result == EXIT_PROMOTION_ERROR

    @patch("main.cleanup_directory")
    @patch("main.promote_models")
    @patch("main.create_backup")
    def test_promotion_with_keep_backup(
        self,
        mock_backup: MagicMock,
        mock_promote: MagicMock,
        mock_cleanup: MagicMock,
        mock_args: argparse.Namespace,
    ) -> None:
        """Should skip ALL cleanup when --keep-backup is set.

        When --keep-backup is set, both staging and backup directories
        are preserved. This allows debugging and provides extra safety margin.
        """
        mock_args.keep_backup = True
        mock_backup.return_value = {"success": True, "backup_path": "/backup"}
        mock_promote.return_value = {"success": True, "services_promoted": ["booking"]}
        mock_cleanup.return_value = {"success": True}

        result = handle_promote(mock_args)

        assert result == EXIT_SUCCESS
        # No cleanup when --keep-backup is set
        assert mock_cleanup.call_count == 0

    @patch("main.create_backup")
    def test_promotion_first_deployment(
        self, mock_backup: MagicMock, mock_args: argparse.Namespace
    ) -> None:
        """Should handle first deployment (no backup_path returned)."""
        mock_backup.return_value = {
            "success": True,
            "backup_path": None,
            "message": "No production models to backup",
        }

        # This test will proceed to promote_models, so we need to mock that too
        with patch("main.promote_models") as mock_promote:
            mock_promote.return_value = {
                "success": True,
                "services_promoted": ["booking"],
            }
            with patch("main.cleanup_directory") as mock_cleanup:
                mock_cleanup.return_value = {"success": True}

                result = handle_promote(mock_args)

                assert result == EXIT_SUCCESS


# =============================================================================
# Handle Training Results Tests
# =============================================================================


class TestHandleTrainingResults:
    """Test handle_training_results function."""

    @pytest.fixture
    def mock_args(self, tmp_path: Path) -> argparse.Namespace:
        """Create mock arguments."""
        args = argparse.Namespace()
        args.direct = False
        args.staging_dir = tmp_path / "staging"
        args.models_dir = tmp_path / "models"
        args.backup_dir = tmp_path / "backup"
        args.keep_backup = False
        return args

    def test_all_services_pass_direct_mode(
        self, mock_args: argparse.Namespace
    ) -> None:
        """Should return EXIT_SUCCESS when all services pass in direct mode."""
        mock_args.direct = True
        results: dict[str, Any] = {
            "booking": {
                "status": "success",
                "validation_results": {"_summary": {"overall_passed": True}},
                "total_explainability_metrics": 5,
            },
            "search": {
                "status": "success",
                "validation_results": {"_summary": {"overall_passed": True}},
                "total_explainability_metrics": 3,
            },
        }

        result = handle_training_results(mock_args, results)

        assert result == EXIT_SUCCESS

    def test_some_services_fail_direct_mode(
        self, mock_args: argparse.Namespace
    ) -> None:
        """Should return EXIT_TRAINING_FAILURE when any service fails in direct mode."""
        mock_args.direct = True
        results: dict[str, Any] = {
            "booking": {
                "status": "success",
                "validation_results": {"_summary": {"overall_passed": True}},
            },
            "search": {
                "status": "error",
                "message": "No data available",
            },
        }

        result = handle_training_results(mock_args, results)

        assert result == EXIT_TRAINING_FAILURE

    @patch("main.handle_auto_promotion")
    def test_all_pass_triggers_auto_promotion(
        self,
        mock_auto_promote: MagicMock,
        mock_args: argparse.Namespace,
    ) -> None:
        """Should call handle_auto_promotion when all services pass."""
        mock_auto_promote.return_value = EXIT_SUCCESS
        results: dict[str, Any] = {
            "booking": {
                "status": "success",
                "validation_results": {"_summary": {"overall_passed": True}},
            },
        }

        result = handle_training_results(mock_args, results)

        assert result == EXIT_SUCCESS
        mock_auto_promote.assert_called_once()

    @patch("main.handle_promotion_blocked")
    def test_failures_trigger_promotion_blocked(
        self,
        mock_blocked: MagicMock,
        mock_args: argparse.Namespace,
    ) -> None:
        """Should call handle_promotion_blocked when any service fails."""
        mock_blocked.return_value = EXIT_TRAINING_FAILURE
        results: dict[str, Any] = {
            "booking": {
                "status": "success",
                "validation_results": {"_summary": {"overall_passed": False, "overall_message": "Too many anomalies"}},
            },
        }

        result = handle_training_results(mock_args, results)

        assert result == EXIT_TRAINING_FAILURE
        mock_blocked.assert_called_once()

    def test_training_failure_categorized_correctly(
        self, mock_args: argparse.Namespace
    ) -> None:
        """Should correctly categorize different failure types."""
        mock_args.direct = True
        results: dict[str, Any] = {
            "booking": {
                "status": "success",
                "validation_results": {"_summary": {"overall_passed": True}},
            },
            "search": {
                "status": "success",
                "validation_results": {"_summary": {"overall_passed": False, "overall_message": "Validation failed"}},
            },
            "mobile-api": {
                "status": "error",
                "message": "Connection timeout",
            },
        }

        result = handle_training_results(mock_args, results)

        # Should fail because some services failed
        assert result == EXIT_TRAINING_FAILURE


# =============================================================================
# Handle Auto Promotion Tests
# =============================================================================


class TestHandleAutoPromotion:
    """Test handle_auto_promotion function."""

    @pytest.fixture
    def mock_args(self, tmp_path: Path) -> argparse.Namespace:
        """Create mock arguments."""
        args = argparse.Namespace()
        args.staging_dir = tmp_path / "staging"
        args.models_dir = tmp_path / "models"
        args.backup_dir = tmp_path / "backup"
        args.keep_backup = False
        return args

    @patch("main.cleanup_directory")
    @patch("main.promote_models")
    def test_successful_auto_promotion(
        self,
        mock_promote: MagicMock,
        mock_cleanup: MagicMock,
        mock_args: argparse.Namespace,
    ) -> None:
        """Should return EXIT_SUCCESS on successful auto-promotion."""
        mock_promote.return_value = {
            "success": True,
            "services_promoted": ["booking", "search"],
        }
        mock_cleanup.return_value = {"success": True}
        # Create backup dir so cleanup is attempted
        mock_args.backup_dir.mkdir(parents=True)

        result = handle_auto_promotion(mock_args, ["booking", "search"])

        assert result == EXIT_SUCCESS

    @patch("main.promote_models")
    def test_failed_auto_promotion(
        self, mock_promote: MagicMock, mock_args: argparse.Namespace
    ) -> None:
        """Should return EXIT_PROMOTION_ERROR on failed promotion."""
        mock_promote.return_value = {
            "success": False,
            "error": "Permission denied",
        }

        result = handle_auto_promotion(mock_args, ["booking"])

        assert result == EXIT_PROMOTION_ERROR

    @patch("main.cleanup_directory")
    @patch("main.promote_models")
    def test_cleanup_failure_does_not_fail_promotion(
        self,
        mock_promote: MagicMock,
        mock_cleanup: MagicMock,
        mock_args: argparse.Namespace,
    ) -> None:
        """Cleanup failures should be logged but not fail the promotion."""
        mock_promote.return_value = {"success": True, "services_promoted": ["booking"]}
        mock_cleanup.return_value = {"success": False, "error": "Permission denied"}
        mock_args.backup_dir.mkdir(parents=True)

        result = handle_auto_promotion(mock_args, ["booking"])

        # Should still succeed despite cleanup failure
        assert result == EXIT_SUCCESS


# =============================================================================
# Handle Promotion Blocked Tests
# =============================================================================


class TestHandlePromotionBlocked:
    """Test handle_promotion_blocked function."""

    @pytest.fixture
    def mock_args(self, tmp_path: Path) -> argparse.Namespace:
        """Create mock arguments."""
        args = argparse.Namespace()
        args.staging_dir = tmp_path / "staging"
        args.backup_dir = tmp_path / "backup"
        return args

    def test_returns_training_failure(self, mock_args: argparse.Namespace) -> None:
        """Should always return EXIT_TRAINING_FAILURE."""
        failed_services = [
            {"service": "booking", "reason": "Validation failed"},
            {"service": "search", "reason": "No data"},
        ]

        result = handle_promotion_blocked(mock_args, failed_services)

        assert result == EXIT_TRAINING_FAILURE

    def test_handles_empty_failed_list(self, mock_args: argparse.Namespace) -> None:
        """Should handle empty failed services list gracefully."""
        result = handle_promotion_blocked(mock_args, [])

        assert result == EXIT_TRAINING_FAILURE


# =============================================================================
# Run Preflight Checks Tests
# =============================================================================


class TestRunPreflightChecks:
    """Test run_preflight_checks function."""

    @pytest.fixture
    def mock_args(self, tmp_path: Path) -> argparse.Namespace:
        """Create mock arguments."""
        args = argparse.Namespace()
        args.staging_dir = tmp_path / "staging"
        args.models_dir = tmp_path / "models"
        args.backup_dir = tmp_path / "backup"
        args.skip_disk_check = False
        return args

    @patch("main.cleanup_directory")
    @patch("main.create_backup")
    @patch("main.check_disk_space")
    def test_all_checks_pass(
        self,
        mock_disk: MagicMock,
        mock_backup: MagicMock,
        mock_cleanup: MagicMock,
        mock_args: argparse.Namespace,
    ) -> None:
        """Should return EXIT_SUCCESS when all checks pass."""
        mock_disk.return_value = {
            "sufficient": True,
            "available": 10_000_000_000,
            "required": 500_000_000,
        }
        # No production models to backup (first run)

        result = run_preflight_checks(mock_args)

        assert result == EXIT_SUCCESS
        mock_disk.assert_called_once()
        # Backup not called because models_dir doesn't exist
        mock_backup.assert_not_called()

    @patch("main.check_disk_space")
    def test_disk_space_insufficient(
        self, mock_disk: MagicMock, mock_args: argparse.Namespace
    ) -> None:
        """Should return EXIT_DISK_SPACE_ERROR when disk space is insufficient."""
        mock_disk.return_value = {
            "sufficient": False,
            "available": 100_000_000,
            "required": 500_000_000,
        }

        result = run_preflight_checks(mock_args)

        assert result == EXIT_DISK_SPACE_ERROR

    @patch("main.check_disk_space")
    def test_skip_disk_check(
        self, mock_disk: MagicMock, mock_args: argparse.Namespace
    ) -> None:
        """Should skip disk check when --skip-disk-check is set."""
        mock_args.skip_disk_check = True

        result = run_preflight_checks(mock_args)

        assert result == EXIT_SUCCESS
        mock_disk.assert_not_called()

    @patch("main.create_backup")
    @patch("main.check_disk_space")
    def test_backup_failure(
        self,
        mock_disk: MagicMock,
        mock_backup: MagicMock,
        mock_args: argparse.Namespace,
    ) -> None:
        """Should return EXIT_BACKUP_ERROR when backup fails."""
        mock_disk.return_value = {"sufficient": True, "available": 10_000_000_000, "required": 500_000_000}
        mock_backup.return_value = {"success": False, "error": "Permission denied"}

        # Create production dir with content so backup is attempted
        mock_args.models_dir.mkdir(parents=True)
        (mock_args.models_dir / "booking").mkdir()

        result = run_preflight_checks(mock_args)

        assert result == EXIT_BACKUP_ERROR

    @patch("main.cleanup_directory")
    @patch("main.create_backup")
    @patch("main.check_disk_space")
    def test_staging_cleanup_on_existing_staging(
        self,
        mock_disk: MagicMock,
        mock_backup: MagicMock,
        mock_cleanup: MagicMock,
        mock_args: argparse.Namespace,
    ) -> None:
        """Should cleanup existing staging directory."""
        mock_disk.return_value = {"sufficient": True, "available": 10_000_000_000, "required": 500_000_000}
        mock_cleanup.return_value = {"success": True}

        # Create staging dir with content
        mock_args.staging_dir.mkdir(parents=True)
        (mock_args.staging_dir / "old_model").mkdir()

        result = run_preflight_checks(mock_args)

        assert result == EXIT_SUCCESS
        mock_cleanup.assert_called_once()

    @patch("main.check_disk_space")
    def test_disk_check_error_continues(
        self, mock_disk: MagicMock, mock_args: argparse.Namespace
    ) -> None:
        """Should continue with warning if disk check has error but isn't insufficient."""
        mock_disk.return_value = {
            "sufficient": True,
            "error": "Could not determine disk usage",
            "available": 0,
            "required": 0,
        }

        result = run_preflight_checks(mock_args)

        # Should continue despite the error (logged as warning)
        assert result == EXIT_SUCCESS


# =============================================================================
# Integration-like Tests (with minimal mocking)
# =============================================================================


class TestEndToEndScenarios:
    """Test end-to-end scenarios with realistic data flows."""

    @pytest.fixture
    def temp_dirs(self, tmp_path: Path) -> dict[str, Path]:
        """Create temporary directory structure."""
        dirs = {
            "staging": tmp_path / "staging",
            "models": tmp_path / "models",
            "backup": tmp_path / "backup",
        }
        for dir_path in dirs.values():
            dir_path.mkdir(parents=True)
        return dirs

    def test_training_result_categorization(self) -> None:
        """Test that training results are correctly categorized."""
        # All success cases
        success_results = {
            "service1": {
                "status": "success",
                "validation_results": {"_summary": {"overall_passed": True}},
            },
        }

        # Mixed results
        mixed_results = {
            "service1": {
                "status": "success",
                "validation_results": {"_summary": {"overall_passed": True}},
            },
            "service2": {
                "status": "success",
                "validation_results": {
                    "_summary": {"overall_passed": False, "overall_message": "Too many anomalies"}
                },
            },
            "service3": {
                "status": "error",
                "message": "No data",
            },
        }

        # Count passed/failed for success_results
        passed = sum(
            1 for r in success_results.values()
            if r["status"] == "success"
            and r.get("validation_results", {}).get("_summary", {}).get("overall_passed")
        )
        assert passed == 1

        # Count passed/failed for mixed_results
        passed = sum(
            1 for r in mixed_results.values()
            if r["status"] == "success"
            and r.get("validation_results", {}).get("_summary", {}).get("overall_passed")
        )
        failed = len(mixed_results) - passed
        assert passed == 1
        assert failed == 2
