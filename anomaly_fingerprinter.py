"""
Backward compatibility stub for anomaly_fingerprinter.

This module redirects imports to the new package structure.
New code should import from smartbox_anomaly.fingerprinting instead.
"""

from smartbox_anomaly.fingerprinting import (
    SCHEMA_VERSION,
    AnomalyFingerprinter,
    create_fingerprinter,
)

__all__ = [
    "AnomalyFingerprinter",
    "create_fingerprinter",
    "SCHEMA_VERSION",
]
