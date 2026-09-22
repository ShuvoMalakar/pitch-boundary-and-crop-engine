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

__all__ = [
    "CropSearchConfig",
    "FieldDetectorConfig",
    "PipelineConfig",
    "FieldDetector",
    "ColorThresholdDetector",
    "get_detector",
]
