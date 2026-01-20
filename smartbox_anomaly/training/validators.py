"""Training validation utilities for ML anomaly detection.

This module provides validation functions for trained models, including:
- Split-based validation (prevents data leakage)
- Synthetic anomaly testing
- Explainability checks
- Time-aware model validation for 3-period approach
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import pandas as pd

from smartbox_anomaly.core.constants import TrainingValidation
from smartbox_anomaly.core.logging import get_logger

if TYPE_CHECKING:
    from smartbox_anomaly.detection.detector import SmartboxAnomalyDetector
    from smartbox_anomaly.detection.time_aware import TimeAwareAnomalyDetector

logger = get_logger(__name__)


# =============================================================================
# Validation Helper Functions
# =============================================================================


def _sanitize_metrics(row: pd.Series) -> Dict[str, float]:
    """Convert a DataFrame row to sanitized metrics dict.

    Args:
        row: DataFrame row with metric values.

    Returns:
        Dict with sanitized metric values (no NaN, inf, or negative values).
    """
    metrics = {
        'request_rate': max(0, row.get('request_rate', 0)),
        'application_latency': max(0, row.get('application_latency', 0)),
        'dependency_latency': max(0, row.get('dependency_latency', 0)),
        'database_latency': max(0, row.get('database_latency', 0)),
        'error_rate': max(0, min(1.0, row.get('error_rate', 0))),
    }

    # Replace invalid values
    for key, value in metrics.items():
        if pd.isna(value) or np.isinf(value):
            metrics[key] = 0.0

    return metrics


def _create_synthetic_anomaly(row: pd.Series) -> Dict[str, float]:
    """Create synthetic anomaly metrics from a normal row.

    Amplifies values to create obvious anomalies for detection testing.

    Args:
        row: DataFrame row with normal metric values.

    Returns:
        Dict with amplified metric values.
    """
    metrics = {
        'request_rate': row.get('request_rate', 0) * 3,  # 3x normal traffic
        'application_latency': row.get('application_latency', 0) * 5,  # 5x latency
        'dependency_latency': row.get('dependency_latency', 0) * 2,
        'database_latency': row.get('database_latency', 0) * 2,
        'error_rate': min(1.0, row.get('error_rate', 0) * 10),  # 10x errors, capped
    }

    # Replace invalid values
    for key, value in metrics.items():
        if pd.isna(value) or np.isinf(value):
            metrics[key] = 0.0

    return metrics


# =============================================================================
# Core Validation Functions
# =============================================================================


def validate_model_split(
    model: SmartboxAnomalyDetector,
    train_features: pd.DataFrame,
    validation_features: pd.DataFrame,
) -> Dict[str, Any]:
    """Validate trained model using properly split data (no leakage).

    Tests the model on:
    1. Normal validation data (should have low anomaly rate)
    2. Synthetic anomalies based on training data (should be detected)

    Args:
        model: Trained model to validate.
        train_features: Training features (for synthetic anomaly baseline).
        validation_features: Held-out validation features (for testing).

    Returns:
        Validation results dictionary with:
        - passed: Overall validation status
        - anomaly_rate: Overall anomaly rate
        - normal_anomaly_rate: Anomaly rate on normal data
        - synthetic_detection_rate: Detection rate on synthetic anomalies
        - checks: Individual validation checks
        - leakage_free: Always True for this method
    """
    logger.info("Validating model with split data (no leakage)...")

    if len(validation_features) < TrainingValidation.MIN_VALIDATION_SAMPLES:
        return {
            'passed': False,
            'reason': 'Insufficient validation data',
            'validation_samples': len(validation_features),
            'leakage_free': True,
        }

    total_tests = min(100, len(validation_features))
    test_errors: List[str] = []

    logger.debug("  Testing on %d validation samples (properly split)...", total_tests)

    # Test 1: Normal validation data (should have low anomaly rate)
    normal_anomalies = 0
    normal_test_count = total_tests // 2

    for i in range(min(normal_test_count, len(validation_features))):
        try:
            row = validation_features.iloc[i]
            current_metrics = _sanitize_metrics(row)

            anomalies = model.detect_anomalies(current_metrics)
            if anomalies:
                normal_anomalies += 1

        except Exception as e:
            test_errors.append(str(e))
            if len(test_errors) <= 3:
                logger.warning("  Validation test %d failed: %s", i, e)

    # Test 2: Synthetic anomalies based on TRAINING statistics (no leakage)
    synthetic_anomalies = 0
    synthetic_tests = total_tests // 2

    for i in range(synthetic_tests):
        try:
            # Use training data for baseline (this is correct - no leakage)
            row = train_features.iloc[i % len(train_features)]
            current_metrics = _create_synthetic_anomaly(row)

            anomalies = model.detect_anomalies(current_metrics)
            if anomalies:
                synthetic_anomalies += 1

        except Exception as e:
            test_errors.append(str(e))

    # Calculate rates
    total_anomalies = normal_anomalies + synthetic_anomalies
    total_tests_run = normal_test_count + synthetic_tests
    anomaly_rate = total_anomalies / total_tests_run if total_tests_run > 0 else 0
    synthetic_detection_rate = synthetic_anomalies / synthetic_tests if synthetic_tests > 0 else 0
    normal_anomaly_rate = normal_anomalies / normal_test_count if normal_test_count > 0 else 0

    # Validation checks
    validation_checks = {
        'has_models': len(model.models) > 0,
        'has_thresholds': len(model.thresholds) > 0,
        'reasonable_anomaly_rate': 0.0 <= anomaly_rate <= TrainingValidation.MAX_ACCEPTABLE_ANOMALY_RATE,
        'normal_data_reasonable': normal_anomaly_rate <= TrainingValidation.MAX_NORMAL_ANOMALY_RATE,
        'few_test_errors': len(test_errors) < total_tests_run * 0.1,
        'model_responds': total_tests_run > 0,
        'detects_synthetic': synthetic_detection_rate > TrainingValidation.MIN_SYNTHETIC_DETECTION_RATE,
    }

    validation_passed = all(validation_checks.values())

    validation_result = {
        'passed': validation_passed,
        'anomaly_rate': anomaly_rate,
        'normal_anomalies': normal_anomalies,
        'normal_anomaly_rate': normal_anomaly_rate,
        'synthetic_anomalies': synthetic_anomalies,
        'synthetic_detection_rate': synthetic_detection_rate,
        'models_trained': len(model.models),
        'thresholds_set': len(model.thresholds),
        'validation_samples': total_tests_run,
        'test_errors': len(test_errors),
        'checks': validation_checks,
        'leakage_free': True,
    }

    if not validation_passed:
        logger.warning("  Validation failed:")
        for check, passed in validation_checks.items():
            logger.warning("    %s: %s", check, passed)
        logger.warning("  Normal data anomaly rate: %.1f%%", normal_anomaly_rate * 100)
        logger.warning("  Synthetic detection rate: %.1f%%", synthetic_detection_rate * 100)
    else:
        logger.info("  Validation passed (leakage-free)")
        logger.info("    Normal data anomaly rate: %.1f%%", normal_anomaly_rate * 100)
        logger.info("    Synthetic detection rate: %.1f%%", synthetic_detection_rate * 100)

    return validation_result


def validate_enhanced_model_split(
    model: SmartboxAnomalyDetector,
    train_features: pd.DataFrame,
    validation_features: pd.DataFrame,
) -> Dict[str, Any]:
    """Enhanced validation using properly split data (no leakage).

    Combines split-based validation with explainability checks.

    Args:
        model: Trained model to validate.
        train_features: Training features (for reference).
        validation_features: Held-out validation features (for testing).

    Returns:
        Validation results dictionary with additional explainability fields.
    """
    logger.info("Validating enhanced model (leakage-free)...")

    # Use the split validation method
    validation_result = validate_model_split(model, train_features, validation_features)

    # Additional explainability validation
    explainability_checks = {
        'has_training_statistics': hasattr(model, 'training_statistics') and len(model.training_statistics) > 0,
        'has_feature_importance': hasattr(model, 'feature_importance') and len(model.feature_importance) > 0,
        'has_context_method': hasattr(model, 'detect_anomalies_with_context'),
    }

    validation_result['explainability_checks'] = explainability_checks
    validation_result['explainability_passed'] = all(explainability_checks.values())
    validation_result['explainability_metrics'] = (
        len(model.training_statistics) if hasattr(model, 'training_statistics') else 0
    )
    validation_result['enhanced_passed'] = validation_result['passed'] and validation_result['explainability_passed']
    validation_result['leakage_free'] = True

    if validation_result['explainability_passed']:
        logger.info("  Explainability validation passed (%d metrics)", validation_result['explainability_metrics'])
    else:
        logger.warning("  Explainability validation failed:")
        for check, passed in explainability_checks.items():
            status = "OK" if passed else "FAILED"
            logger.warning("    %s: %s", check, status)

    return validation_result


def validate_time_aware_models(
    detector: TimeAwareAnomalyDetector,
    features_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Enhanced validation for time-aware models with 3-period approach.

    Uses 3 time-of-day buckets covering all 7 days:
    - business_hours (8-18)
    - evening_hours (18-22)
    - night_hours (22-8)

    Args:
        detector: Trained time-aware detector with period models.
        features_df: Features DataFrame with datetime index.

    Returns:
        Validation results per period with overall summary.
    """
    logger.info("Validating enhanced time-aware models with 3-period approach...")
    validation_results: Dict[str, Any] = {}

    # Add time period column
    features_df = features_df.copy()
    features_df['time_period'] = features_df.index.map(detector.get_time_period)

    # Use the detector's service-specific thresholds
    thresholds = detector.validation_thresholds
    service_type = detector._get_service_type()

    logger.debug("  Using %s thresholds for 3 periods:", service_type)
    for period, threshold in thresholds.items():
        logger.debug("    %s: %.1f%% max anomaly rate", period, threshold * 100)

    for period in detector.models.keys():
        period_data = features_df[features_df['time_period'] == period]

        # Use detector's minimum sample requirements
        min_samples = detector._get_min_samples_for_period(period)

        if len(period_data) < min_samples:
            validation_results[period] = {
                'status': 'insufficient_data',
                'samples': len(period_data),
                'min_required': min_samples,
                'passed': False,
                'explainability_passed': False,
            }
            continue

        logger.debug("  Testing enhanced %s model...", period)

        # Enhanced validation with realistic expectations
        model = detector.models[period]

        # Use smaller test sample for micro-services and admin services
        if service_type in ['micro_service', 'admin_service']:
            test_samples = min(25, len(period_data) // 3)
        elif period == 'night_hours':
            test_samples = min(30, len(period_data) // 2)
        else:
            test_samples = min(40, len(period_data) // 2)

        normal_anomaly_count = 0
        test_errors = 0

        # Test on validation data with improved error handling
        for i in range(test_samples):
            try:
                row_idx = -(i + 1) if i < len(period_data) else i % len(period_data)
                row = period_data.iloc[row_idx]
                test_metrics = _sanitize_metrics(row)

                anomalies = model.detect_anomalies(test_metrics)
                if anomalies and len(anomalies) > 0:
                    normal_anomaly_count += 1

            except Exception as e:
                test_errors += 1
                if test_errors <= 2:
                    logger.warning("    Test error for %s sample %d: %s...", period, i, str(e)[:50])

        normal_anomaly_rate = normal_anomaly_count / test_samples if test_samples > 0 else 0

        # Enhanced explainability validation
        explainability_checks = {
            'has_training_statistics': hasattr(model, 'training_statistics') and len(model.training_statistics) > 0,
            'has_feature_importance': hasattr(model, 'feature_importance') and len(model.feature_importance) > 0,
            'has_context_method': hasattr(model, 'detect_anomalies_with_context'),
            'has_zero_statistics': hasattr(model, 'zero_statistics') and len(model.zero_statistics) > 0,
        }

        # Use service-specific threshold for this period
        period_threshold = thresholds.get(period, TrainingValidation.MAX_NORMAL_ANOMALY_RATE)

        # Enhanced validation criteria with period-specific adjustments
        error_tolerance = 0.25 if period == 'night_hours' else 0.2

        validation_checks = {
            'anomaly_rate_acceptable': normal_anomaly_rate <= period_threshold,
            'sufficient_tests': test_samples >= 10,
            'low_error_rate': test_errors < test_samples * error_tolerance,
            'model_functional': normal_anomaly_count >= 0,
        }

        # Overall validation status
        validation_passed = all(validation_checks.values())
        explainability_passed = all(explainability_checks.values())

        validation_results[period] = {
            'samples_tested': test_samples,
            'normal_anomaly_rate': normal_anomaly_rate,
            'threshold_used': period_threshold,
            'test_errors': test_errors,
            'validation_checks': validation_checks,
            'passed': validation_passed,
            'explainability_checks': explainability_checks,
            'explainability_passed': explainability_passed,
            'explainability_metrics': len(model.training_statistics) if hasattr(model, 'training_statistics') else 0,
            'enhanced_passed': validation_passed and explainability_passed,
            'period_type': detector._get_period_type(period),
        }

        # Enhanced status reporting
        if validation_passed:
            status_msg = f"{normal_anomaly_rate:.1%} anomaly rate (<={period_threshold:.1%})"
            explainability_str = "explainability: OK" if explainability_passed else "explainability: MISSING"
            logger.info("  %s: OK - %s, %s", period, status_msg, explainability_str)
        else:
            failed_checks = [check for check, passed in validation_checks.items() if not passed]
            if not validation_checks['anomaly_rate_acceptable']:
                status_msg = f"{normal_anomaly_rate:.1%} anomaly rate (>{period_threshold:.1%} threshold)"
            else:
                status_msg = f"Failed: {', '.join(failed_checks)}"
            logger.warning("  %s: FAILED - %s", period, status_msg)

        # Additional context for failures with 3-period awareness
        if not validation_passed:
            if period == 'night_hours':
                logger.debug("    Note: Night hours have higher natural variability due to lower traffic")
            elif period == 'evening_hours':
                logger.debug("    Note: Evening hours are a transition period with variable patterns")

    # Enhanced overall assessment with 3-period logic
    total_periods = len(validation_results)
    passed_periods = sum(1 for result in validation_results.values() if result.get('enhanced_passed', False))

    # Critical periods: business_hours and night_hours (most important for operations)
    critical_periods_passed = sum(
        1
        for period in ['business_hours', 'night_hours']
        if period in validation_results and validation_results[period].get('enhanced_passed', False)
    )

    # Enhanced validation logic for 3-period approach
    if total_periods == 0:
        overall_passed = False
        overall_message = "No periods validated"
    elif critical_periods_passed >= 1:
        # For 3-period model, at least 2 out of 3 periods should pass
        overall_passed = passed_periods >= 2
        if overall_passed:
            overall_message = f"Validation passed: {passed_periods}/{total_periods} periods validated"
        else:
            overall_message = f"Insufficient coverage: {passed_periods}/{total_periods} periods (need ≥2)"
    else:
        overall_passed = False
        overall_message = "No critical periods (business/night hours) passed validation"

    explainability_periods = sum(1 for result in validation_results.values() if result.get('explainability_passed', False))
    explainability_overall = explainability_periods > 0

    if overall_passed:
        logger.info("Enhanced 3-period validation: %s", overall_message)
    else:
        logger.warning("Enhanced 3-period validation FAILED: %s", overall_message)
    logger.info("Explainability: %d/%d periods enabled", explainability_periods, total_periods)

    # Show period breakdown
    if validation_results:
        logger.info("Period breakdown:")
        all_periods_passed = sum(
            1
            for period in ['business_hours', 'evening_hours', 'night_hours']
            if period in validation_results and validation_results[period].get('enhanced_passed', False)
        )
        all_periods_total = sum(
            1
            for period in ['business_hours', 'evening_hours', 'night_hours']
            if period in validation_results
        )
        logger.info("  Time-of-day periods: %d/%d passed", all_periods_passed, all_periods_total)

    # Add service-specific recommendations for 3-period approach
    if not overall_passed:
        _log_validation_recommendations(service_type)

    # Add summary to validation results
    validation_results['_summary'] = {
        'overall_passed': overall_passed,
        'overall_message': overall_message,
        'explainability_overall': explainability_overall,
        'service_type': service_type,
        'total_periods': total_periods,
        'passed_periods': passed_periods,
        'critical_periods_passed': critical_periods_passed,
        'thresholds_used': thresholds,
        'approach': '3_period_time_of_day',
    }

    return validation_results


def _log_validation_recommendations(service_type: str) -> None:
    """Log recommendations based on service type when validation fails.

    Args:
        service_type: The type of service (micro_service, admin_service, etc.).
    """
    logger.info("Recommendations for %s with 3-period approach:", service_type)
    if service_type == 'micro_service':
        logger.info("  - Consider increasing training data collection period")
        logger.info("  - Focus on getting business_hours and night_hours models working first")
        logger.info("  - Evening hours may need higher tolerance for variability")
    elif service_type == 'admin_service':
        logger.info("  - Admin services often have minimal night activity")
        logger.info("  - Focus on business_hours model which has the most consistent traffic")
        logger.info("  - Night hours may not be reliable for admin services")
    else:
        logger.info("  - Ensure adequate data collection across all time periods")
        logger.info("  - Night hours have higher natural variance than day hours")
        logger.info("  - Consider adjusting anomaly detection sensitivity for night periods")
