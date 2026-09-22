"""Custom exception hierarchy for the crop engine."""


class CropEngineError(Exception):
    """Base exception for all crop engine errors."""
    pass


class ConfigurationError(CropEngineError):
    """Raised when configuration is invalid or cannot be parsed."""
    pass


class VideoSourceError(CropEngineError):
    """Raised when a video file cannot be opened, read, or decoded."""
    pass


class DetectionError(CropEngineError):
    """Raised when an error occurs during frame boundary detection."""
    pass


class CropCalculationError(CropEngineError):
    """Raised when crop derivation fails."""
    pass


class ReportingError(CropEngineError):
    """Raised when network communication with the reporting service fails."""
    pass
