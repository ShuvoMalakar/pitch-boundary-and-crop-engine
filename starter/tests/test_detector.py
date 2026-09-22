"""Unit tests for the FieldDetector interface and ColorThresholdDetector implementation."""

import numpy as np
import pytest
from shapely.geometry import Polygon

from crop_engine.config import FieldDetectorConfig
from crop_engine.detector import (
    ColorThresholdDetector,
    FieldDetector,
    get_detector,
)


def test_detect_valid_pitch_boundary(green_field_with_boundary_frame):
    """A frame with a visible pitch boundary must produce a valid Shapely Polygon."""
    detector = ColorThresholdDetector(min_area=1000.0)
    polygon = detector.detect(green_field_with_boundary_frame)

    assert polygon is not None
    assert isinstance(polygon, Polygon)
    assert polygon.is_valid
    assert polygon.area >= 1000.0


def test_detect_black_frame_returns_none(black_frame):
    """Black frames (camera cuts / blackouts) must return None without crashing."""
    detector = ColorThresholdDetector(min_area=1000.0)
    polygon = detector.detect(black_frame)

    assert polygon is None


def test_detect_solid_green_close_up_returns_none(green_field_frame):
    """Solid green frames (close-ups without pitch lines) must return None."""
    detector = ColorThresholdDetector(min_area=1000.0)
    polygon = detector.detect(green_field_frame)

    assert polygon is None


def test_detect_noise_frame_filtered_by_min_area(noise_frame):
    """Small sensor noise or watermarks below min_area must be filtered out."""
    detector = ColorThresholdDetector(min_area=1000.0)
    polygon = detector.detect(noise_frame)

    # Noise box in fixture is ~800 px area, which is < 1000.0 min_area
    assert polygon is None


def test_detect_empty_or_none_frame():
    """Empty or None input frames must safely return None."""
    detector = ColorThresholdDetector(min_area=1000.0)

    assert detector.detect(None) is None
    assert detector.detect(np.zeros((0, 0, 3), dtype=np.uint8)) is None


def test_detector_pluggable_seam():
    """Verify that alternative detectors can be cleanly substituted via FieldDetector."""

    class MockSAMDetector(FieldDetector):
        """Simulates a future deep-learning/SAM-based detector."""

        def detect(self, frame: np.ndarray) -> Polygon:
            return Polygon([(100, 100), (1100, 100), (1100, 600), (100, 600)])

    custom_detector: FieldDetector = MockSAMDetector()
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    result = custom_detector.detect(dummy_frame)

    assert isinstance(result, Polygon)
    assert result.is_valid
    assert result.area == 500000.0


def test_get_detector_factory_success():
    """Factory correctly instantiates ColorThresholdDetector for supported types."""
    config = FieldDetectorConfig(type="sam_mask_v1", min_area=500.0)
    detector = get_detector(config)

    assert isinstance(detector, ColorThresholdDetector)
    assert detector.min_area == 500.0


def test_get_detector_factory_unsupported_type():
    """Factory raises ValueError when given an unsupported detector type."""
    config = FieldDetectorConfig(type="unsupported_model")
    with pytest.raises(ValueError, match="Unsupported field detector type"):
        get_detector(config)
