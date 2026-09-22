"""Crop Engine package initialization."""

__version__ = "0.1.0"

from crop_engine.config import (
    CropSearchConfig,
    FieldDetectorConfig,
    PipelineConfig,
)
from crop_engine.detector import (
    ColorThresholdDetector,
    FieldDetector,
    get_detector,
)
from crop_engine.exceptions import (
    ConfigurationError,
    CropCalculationError,
    CropEngineError,
    DetectionError,
    ReportingError,
    VideoSourceError,
)
from crop_engine.geometry import (
    CropBox,
    GeometryCalculator,
)

__all__ = [
    "CropSearchConfig",
    "FieldDetectorConfig",
    "PipelineConfig",
    "FieldDetector",
    "ColorThresholdDetector",
    "get_detector",
    "CropEngineError",
    "ConfigurationError",
    "VideoSourceError",
    "DetectionError",
    "CropCalculationError",
    "ReportingError",
    "CropBox",
    "GeometryCalculator",
]
