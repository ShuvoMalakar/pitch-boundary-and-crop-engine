"""Unit tests for the exception hierarchy."""

import pytest

from src.crop_engine.exceptions import (
    ConfigurationError,
    CropCalculationError,
    CropEngineError,
    DetectionError,
    ReportingError,
    VideoSourceError,
)


def test_exception_inheritance_hierarchy():
    """All domain exceptions must inherit from the base CropEngineError."""
    exceptions = [
        ConfigurationError,
        VideoSourceError,
        DetectionError,
        CropCalculationError,
        ReportingError,
    ]
    for exc_cls in exceptions:
        assert issubclass(exc_cls, CropEngineError)
        assert issubclass(exc_cls, Exception)


def test_catch_specific_and_base_exception():
    """Specific exceptions can be caught by their type or by CropEngineError."""
    with pytest.raises(CropEngineError):
        raise VideoSourceError("Failed to open video file")

    with pytest.raises(VideoSourceError):
        raise VideoSourceError("Failed to open video file")


def test_reporting_error_distinct_from_pipeline_errors():
    """Network/reporting failures must be distinguishable from video pipeline failures."""
    reporting_err = ReportingError("Connection refused: mock_api:5000")
    video_err = VideoSourceError("Cannot decode frame")

    assert isinstance(reporting_err, ReportingError)
    assert not isinstance(reporting_err, VideoSourceError)
    assert not isinstance(video_err, ReportingError)


def test_exception_preserves_message():
    """Exception message is preserved accurately."""
    msg = "Invalid aspect ratio: 16-9"
    err = ConfigurationError(msg)
    assert str(err) == msg
