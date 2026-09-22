"""Unit tests for the validated configuration model."""

import pytest
from pydantic import ValidationError

from crop_engine.config import (
    CropSearchConfig,
    FieldDetectorConfig,
    PipelineConfig,
)


def test_default_pipeline_config():
    """Verify that a minimal valid configuration loads cleanly with sensible defaults."""
    config = PipelineConfig(video_path="test_video.mp4")
    assert config.video_path == "test_video.mp4"
    assert config.target_fps == 30
    assert config.sample_rate_fps == 3.0
    assert config.confidence_threshold == 0.5
    assert config.max_workers == 4
    assert config.field_detector.type == "sam_mask_v1"
    assert config.field_detector.sport == "football"
    assert config.field_detector.min_area == 1000.0
    assert config.crop_search.aspect_ratio == "16:9"
    assert config.crop_search.aspect_ratio_float == pytest.approx(16 / 9, rel=0.01)
    assert config.crop_search.padding_px == 20
    assert config.debug_mode is False


def test_custom_pipeline_config_from_dict():
    """Verify parsing from a dictionary matching the original prototype layout."""
    raw = {
        "video_path": "synthetic_pitch_feed.mp4",
        "target_fps": 30,
        "confidence_threshold": 0.5,
        "field_detector": {
            "type": "sam_mask_v1",
            "sport": "football",
            "min_area": 1000,
        },
        "crop_search": {
            "aspect_ratio": "16:9",
            "padding_px": 20,
        },
        "debug_mode": True,
    }
    config = PipelineConfig.from_dict(raw)
    assert config.video_path == "synthetic_pitch_feed.mp4"
    assert config.target_fps == 30
    assert config.confidence_threshold == 0.5
    assert config.field_detector.min_area == 1000.0
    assert config.crop_search.aspect_ratio == "16:9"
    assert config.debug_mode is True


def test_fail_fast_on_extra_unexpected_keys():
    """Unexpected keys must cause immediate load-time failure."""
    raw = {
        "video_path": "test.mp4",
        "unknown_flag": 123,
    }
    with pytest.raises(ValidationError) as exc_info:
        PipelineConfig.from_dict(raw)
    assert "extra_forbidden" in str(exc_info.value)


def test_fail_fast_on_invalid_aspect_ratio():
    """Verify aspect ratio format validation."""
    invalid_ratios = ["16x9", "16-9", "invalid", "-16:9", "0:9", "16:0"]
    for ratio in invalid_ratios:
        with pytest.raises(ValidationError):
            CropSearchConfig(aspect_ratio=ratio)


def test_fail_fast_on_negative_padding():
    """Padding cannot be negative."""
    with pytest.raises(ValidationError):
        CropSearchConfig(padding_px=-5)


def test_fail_fast_on_invalid_confidence():
    """Confidence threshold must be between 0.0 and 1.0."""
    with pytest.raises(ValidationError):
        PipelineConfig(video_path="test.mp4", confidence_threshold=-0.1)
    with pytest.raises(ValidationError):
        PipelineConfig(video_path="test.mp4", confidence_threshold=1.5)


def test_fail_fast_on_invalid_min_area():
    """Detector min_area must be strictly positive."""
    with pytest.raises(ValidationError):
        FieldDetectorConfig(min_area=0)
    with pytest.raises(ValidationError):
        FieldDetectorConfig(min_area=-100)


def test_fail_fast_on_invalid_sample_rate():
    """Sample rate must be strictly positive."""
    with pytest.raises(ValidationError):
        PipelineConfig(video_path="test.mp4", sample_rate_fps=0)
    with pytest.raises(ValidationError):
        PipelineConfig(video_path="test.mp4", sample_rate_fps=-1.0)


def test_fail_fast_on_invalid_max_workers():
    """Max workers must be at least 1."""
    with pytest.raises(ValidationError):
        PipelineConfig(video_path="test.mp4", max_workers=0)
