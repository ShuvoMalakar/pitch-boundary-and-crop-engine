"""Unit tests for the reporting client and wire models."""

from unittest.mock import MagicMock, patch
from pydantic import ValidationError
import pytest
import requests

from crop_engine.geometry import CropBox
from crop_engine.reporter import (
    JobEventPayload,
    JobProgressPayload,
    PlatformReporter,
)
from crop_engine.telemetry import RunMetrics


def test_job_progress_payload_validation():
    """Verify validation rules on progress update schema."""
    valid = JobProgressPayload(
        job_id="job_123",
        progress_percent=50.0,
        frames_processed=90,
        total_frames=180,
        current_fps=30.0,
    )
    assert valid.progress_percent == 50.0

    # Negative or >100% progress must fail
    with pytest.raises(ValidationError):
        JobProgressPayload(
            job_id="job_123",
            progress_percent=105.0,
            frames_processed=90,
            total_frames=180,
        )

    with pytest.raises(ValidationError):
        JobProgressPayload(
            job_id="job_123",
            progress_percent=-1.0,
            frames_processed=90,
            total_frames=180,
        )

    # Extra unexpected fields must fail fast
    with pytest.raises(ValidationError):
        JobProgressPayload(
            job_id="job_123",
            progress_percent=50.0,
            frames_processed=90,
            total_frames=180,
            unexpected_field="bad",
        )


def test_job_event_payload_validation():
    """Verify validation rules on event payload schema."""
    event = JobEventPayload(
        job_id="job_456",
        event_type="job_started",
        timestamp_iso="2026-09-22T08:00:00Z",
        details={"total_frames": 1800},
    )
    assert event.event_type == "job_started"

    # Extra unexpected fields must fail fast
    with pytest.raises(ValidationError):
        JobEventPayload(
            job_id="job_456",
            event_type="job_started",
            timestamp_iso="2026-09-22T08:00:00Z",
            bad_field=123,
        )


def test_reporter_successful_reporting():
    """Reporter returns True when endpoints return HTTP 200."""
    reporter = PlatformReporter(base_url="http://mock_api:5000", job_id="test_job")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status.return_value = None

    with patch.object(reporter.session, "post", return_value=mock_resp) as mock_post:
        # Progress
        progress = JobProgressPayload(
            job_id="test_job",
            progress_percent=10.0,
            frames_processed=18,
            total_frames=180,
        )
        assert reporter.report_progress(progress) is True
        assert mock_post.call_count == 1

        # Job lifecycle events
        assert reporter.report_job_started("video.mp4", 1800) is True
        assert (
            reporter.report_job_completed(
                RunMetrics(total_frames_in_video=1800, sampled_frames=180),
                CropBox(x=10, y=10, w=1200, h=675),
            )
            is True
        )
        assert reporter.report_job_failed("Video corrupted") is True


def test_reporter_resilient_to_network_failure():
    """Network connection errors do not raise or crash the video pipeline."""
    reporter = PlatformReporter(base_url="http://unreachable_host:5000", job_id="test_job")

    with patch.object(
        reporter.session,
        "post",
        side_effect=requests.ConnectionError("Connection refused"),
    ):
        progress = JobProgressPayload(
            job_id="test_job",
            progress_percent=10.0,
            frames_processed=18,
            total_frames=180,
        )
        # Must return False gracefully without raising
        assert reporter.report_progress(progress) is False
        assert reporter.report_job_started("video.mp4", 1800) is False
        assert reporter.report_job_failed("Pipeline error") is False
