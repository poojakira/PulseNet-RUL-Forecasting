"""
PulseNet core exceptions.
"""

from __future__ import annotations


class PulseNetError(Exception):
    """Base category for all PulseNet exceptions."""

    def __init__(self, message: str = "", error_code: str = "PULSENET_ERROR"):
        self.message = message
        self.error_code = error_code
        super().__init__(message)


class DataError(PulseNetError):
    """Raised when data ingestion or preprocessing fails."""

    def __init__(self, message: str = "", error_code: str = "DATA_ERROR"):
        super().__init__(message, error_code)


class ModelError(PulseNetError):
    """Raised when model training, prediction, or loading fails."""

    def __init__(self, message: str = "", error_code: str = "MODEL_ERROR"):
        super().__init__(message, error_code)


class SecurityError(PulseNetError):
    """Raised when encryption or blockchain integrity checks fail."""

    def __init__(self, message: str = "", error_code: str = "SECURITY_ERROR"):
        super().__init__(message, error_code)


class ConfigurationError(PulseNetError):
    """Raised when configuration values are missing or invalid."""

    def __init__(self, message: str = "", error_code: str = "CONFIG_ERROR"):
        super().__init__(message, error_code)
