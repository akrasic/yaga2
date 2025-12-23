"""
Centralized logging configuration for the ML anomaly detection pipeline.

This module provides consistent logging setup across all modules,
with support for different output formats and log levels.
"""

from __future__ import annotations

import logging
import sys
from types import TracebackType
from typing import Any

# Module-level logger cache
_loggers_configured: bool = False


def configure_logging(
    level: int = logging.INFO,
    verbose: bool = False,
    json_format: bool = False,
) -> None:
    """Configure logging for the entire application.

    This should be called once at application startup.

    Args:
        level: Base logging level (e.g., logging.INFO, logging.WARNING).
        verbose: If True, sets level to DEBUG.
        json_format: If True, uses JSON-formatted output.
    """
    global _loggers_configured

    if verbose:
        level = logging.DEBUG

    # Standard format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    if json_format:
        log_format = (
            '{"timestamp": "%(asctime)s", "logger": "%(name)s", '
            '"level": "%(levelname)s", "message": "%(message)s"}'
        )

    # Configure root logger
    logging.basicConfig(
        level=level,
        format=log_format,
        datefmt=date_format,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Suppress noisy third-party loggers
    _configure_third_party_loggers()

    _loggers_configured = True


def _configure_third_party_loggers() -> None:
    """Configure logging levels for third-party libraries."""
    # Suppress urllib3 connection pool warnings
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)

    # Suppress requests library debug logging
    logging.getLogger("requests").setLevel(logging.WARNING)

    # Suppress sklearn warnings (handled via warnings module in model code)
    logging.getLogger("sklearn").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name.

    Ensures logging is configured before returning the logger.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        Configured Logger instance.
    """
    if not _loggers_configured:
        configure_logging()

    return logging.getLogger(name)


def set_log_level(level: int, logger_name: str | None = None) -> None:
    """Set log level for a specific logger or root logger.

    Args:
        level: The logging level to set.
        logger_name: Optional logger name. If None, sets root logger.
    """
    if logger_name:
        logging.getLogger(logger_name).setLevel(level)
    else:
        logging.getLogger().setLevel(level)


def quiet_mode() -> None:
    """Set logging to quiet mode (WARNING and above only)."""
    set_log_level(logging.WARNING)


def verbose_mode() -> None:
    """Set logging to verbose mode (DEBUG and above)."""
    set_log_level(logging.DEBUG)


class LogContext:
    """Context manager for temporarily changing log level.

    Usage:
        with LogContext(logging.DEBUG, "mymodule"):
            # Debug logging enabled for mymodule
            do_something()
        # Original log level restored
    """

    def __init__(self, level: int, logger_name: str | None = None) -> None:
        self.level = level
        self.logger_name = logger_name
        self._original_level: int | None = None

    def __enter__(self) -> LogContext:
        logger = logging.getLogger(self.logger_name) if self.logger_name else logging.getLogger()
        self._original_level = logger.level
        logger.setLevel(self.level)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        logger = logging.getLogger(self.logger_name) if self.logger_name else logging.getLogger()
        if self._original_level is not None:
            logger.setLevel(self._original_level)


# Convenience function for structured logging
def log_event(
    logger: logging.Logger,
    level: int,
    event_type: str,
    service: str,
    message: str,
    **kwargs: Any,
) -> None:
    """Log a structured event with consistent formatting.

    Args:
        logger: Logger instance to use.
        level: Logging level.
        event_type: Type of event (e.g., ANOMALY_DETECTED, MODEL_LOADED).
        service: Service name associated with the event.
        message: Human-readable message.
        **kwargs: Additional key-value pairs to include.
    """
    extra_parts = " ".join(f"{k}={v}" for k, v in kwargs.items())
    full_message = f"{event_type} - {service}: {message}"
    if extra_parts:
        full_message = f"{full_message} [{extra_parts}]"
    logger.log(level, full_message)


# Pre-defined event types for consistent logging
class EventType:
    """Standard event types for structured logging."""

    # Anomaly events
    ANOMALY_DETECTED = "ANOMALY_DETECTED"
    NO_ANOMALIES = "NO_ANOMALIES"
    ENHANCED_ANOMALY = "ENHANCED_ANOMALY"

    # Model events
    MODEL_LOADED = "MODEL_LOADED"
    MODEL_LOAD_FAILED = "MODEL_LOAD_FAILED"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"

    # Metrics events
    METRICS_COLLECTED = "METRICS_COLLECTED"
    METRICS_FAILED = "METRICS_FAILED"

    # Inference events
    INFERENCE_START = "INFERENCE_START"
    INFERENCE_COMPLETE = "INFERENCE_COMPLETE"
    INFERENCE_ERROR = "INFERENCE_ERROR"

    # Fingerprinting events
    INCIDENT_CREATED = "INCIDENT_CREATED"
    INCIDENT_CONTINUED = "INCIDENT_CONTINUED"
    INCIDENT_RESOLVED = "INCIDENT_RESOLVED"

    # System events
    SYSTEM_STATUS = "SYSTEM_STATUS"
    CIRCUIT_BREAKER = "CIRCUIT_BREAKER"
