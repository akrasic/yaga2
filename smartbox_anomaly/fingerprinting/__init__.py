"""
Fingerprinting module - incident lifecycle tracking.

This module contains:
    - fingerprinter: AnomalyFingerprinter with incident lifecycle management
"""

from smartbox_anomaly.fingerprinting.fingerprinter import (
    SCHEMA_VERSION,
    AnomalyFingerprinter,
    create_fingerprinter,
)

__all__ = [
    "SCHEMA_VERSION",
    "AnomalyFingerprinter",
    "create_fingerprinter",
]
