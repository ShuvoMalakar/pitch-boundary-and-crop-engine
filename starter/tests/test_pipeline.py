"""Unit tests for the PitchCropPipeline execution, frame sampling, and concurrency."""

from pathlib import Path
import cv2
import numpy as np
import pytest
from shapely.geometry import Polygon

from crop_engine.config import PipelineConfig
from crop_engine.detector import FieldDetector
from crop_engine.exceptions import VideoSourceError
from crop_engine.pipeline import PitchCropPipeline


@pytest.fixture
def temp_synthetic_video(tmp_path) -> str:
    """Creates a temporary 60-frame (2 second at 30fps) synthetic video."""
    video_path = str(tmp_path / "test_feed.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, 30.0, (1280, 720))

    pts = np.array([[100, 100], [1180, 100], [1230, 620], [50, 620]], np.int32)

    for _ in range(60):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[:] = (34, 139, 34)
        cv2.polylines(frame, [pts], True, (255, 255, 255), 5)
        writer.write(frame)

    writer.release()
    return video_path


@pytest.fixture
def temp_black_video(tmp_path) -> str:
    """Creates a temporary 30-frame all-black video (camera cut / blackout)."""
    video_path = str(tmp_path / "black_feed.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, 30.0, (1280, 720))

    for _ in range(30):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        writer.write(frame)

    writer.release()
    return video_path


def test_pipeline_unopenable_video_raises_video_source_error():
    """Non-existent video file must immediately raise VideoSourceError."""
    config = PipelineConfig(video_path="non_existent_video_file.mp4")
    pipeline = PitchCropPipeline(config)

    with pytest.raises(VideoSourceError, match="Could not open video"):
        pipeline.run()


def test_pipeline_frame_sampling_and_concurrency(temp_synthetic_video):
    """Pipeline samples at sample_rate_fps (e.g. 3 FPS on 30 FPS video = 10x reduction)."""
    config = PipelineConfig(
        video_path=temp_synthetic_video,
        target_fps=30,
        sample_rate_fps=3.0,
        max_workers=2,
    )
    pipeline = PitchCropPipeline(config)
    result = pipeline.run()

    # 60 total frames sampled at 3 FPS from 30 FPS = step of 10 -> exactly 6 frames sampled
    assert result.metrics.total_frames_in_video == 60
    assert result.metrics.sampled_frames == 6
    assert len(result.frame_results) == 6

    # Verify chronological ordering
    frame_indices = [r.frame_idx for r in result.frame_results]
    assert frame_indices == sorted(frame_indices)
    assert frame_indices == [0, 10, 20, 30, 40, 50]

    # Detections should all be valid
    assert result.metrics.valid_detections == 6
    assert result.metrics.detection_rate == 1.0
    assert result.stable_boundary is not None
    assert result.stable_crop is not None


def test_pipeline_handles_all_black_video(temp_black_video):
    """Pipeline processes blackout feed without crashes and reports 0 valid detections."""
    config = PipelineConfig(
        video_path=temp_black_video,
        target_fps=30,
        sample_rate_fps=3.0,
        max_workers=2,
    )
    pipeline = PitchCropPipeline(config)
    result = pipeline.run()

    assert result.metrics.total_frames_in_video == 30
    assert result.metrics.sampled_frames == 3
    assert result.metrics.valid_detections == 0
    assert result.metrics.missing_detections == 3
    assert result.metrics.detection_rate == 0.0
    assert result.stable_boundary is None
    assert result.stable_crop is None


def test_pipeline_custom_detector_injection(temp_synthetic_video):
    """Verify clean dependency injection seam for custom detectors."""

    class FixedBoundaryDetector(FieldDetector):
        def detect(self, frame: np.ndarray) -> Polygon:
            return Polygon([(200, 200), (1000, 200), (1000, 500), (200, 500)])

    config = PipelineConfig(
        video_path=temp_synthetic_video,
        sample_rate_fps=3.0,
    )
    pipeline = PitchCropPipeline(config, detector=FixedBoundaryDetector())
    result = pipeline.run()

    assert result.metrics.valid_detections == 6
    assert result.stable_boundary is not None
    assert result.stable_boundary.area == (1000 - 200) * (500 - 200)
