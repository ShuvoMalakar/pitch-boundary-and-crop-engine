from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import time
from typing import Any, List, Optional

import cv2
import numpy as np
from shapely.geometry import Polygon

from crop_engine.config import PipelineConfig
from crop_engine.detector import FieldDetector, get_detector
from crop_engine.exceptions import VideoSourceError
from crop_engine.geometry import CropBox, GeometryCalculator
from crop_engine.telemetry import MetricsCollector, RunMetrics, setup_logging


@dataclass
class FrameResult:
    """Detection and crop output for a single processed frame."""

    frame_idx: int
    timestamp_s: float
    polygon: Optional[Polygon]
    crop_box: Optional[CropBox]
    intersection_area: float
    status: str  # "valid", "missing", "invalid"


@dataclass
class PipelineResult:
    """Aggregated output of a pipeline run."""

    metrics: RunMetrics
    frame_results: List[FrameResult]
    stable_boundary: Optional[Polygon]
    stable_crop: Optional[CropBox]


class PitchCropPipeline:
    """Orchestrates video frame extraction, field detection, and crop layout computation."""

    def __init__(
        self,
        config: PipelineConfig,
        detector: Optional[FieldDetector] = None,
        reporter: Optional[Any] = None,
    ):
        self.config = config
        self.detector = detector or get_detector(config.field_detector)
        self.reporter = reporter
        self.logger = setup_logging(config.debug_mode)
        self.metrics_collector = MetricsCollector()
        self.geometry = GeometryCalculator(crop_config=config.crop_search)

    def _process_frame(
        self, frame_idx: int, frame: np.ndarray, timestamp_s: float
    ) -> FrameResult:
        """Processes a single video frame: detects boundary and derives crop."""
        self.metrics_collector.record_sampled_frame()

        if frame is None or frame.size == 0:
            self.metrics_collector.record_missing_detection()
            return FrameResult(
                frame_idx=frame_idx,
                timestamp_s=timestamp_s,
                polygon=None,
                crop_box=None,
                intersection_area=0.0,
                status="missing",
            )

        try:
            poly = self.detector.detect(frame)
        except Exception as e:
            self.logger.warning(f"Detection failed on frame {frame_idx}: {e}")
            self.metrics_collector.record_invalid_detection()
            return FrameResult(
                frame_idx=frame_idx,
                timestamp_s=timestamp_s,
                polygon=None,
                crop_box=None,
                intersection_area=0.0,
                status="invalid",
            )

        if poly is None:
            self.metrics_collector.record_missing_detection()
            return FrameResult(
                frame_idx=frame_idx,
                timestamp_s=timestamp_s,
                polygon=None,
                crop_box=None,
                intersection_area=0.0,
                status="missing",
            )

        intersection_area = self.geometry.compute_intersection_area(poly)
        crop_box = self.geometry.calculate_crop(poly)

        self.metrics_collector.record_valid_detection()
        return FrameResult(
            frame_idx=frame_idx,
            timestamp_s=timestamp_s,
            polygon=poly,
            crop_box=crop_box,
            intersection_area=intersection_area,
            status="valid",
        )

    def run(self) -> PipelineResult:
        """Executes the pipeline on the configured video path."""
        start_time = time.time()
        self.logger.info(f"Opening video: {self.config.video_path}")

        cap = cv2.VideoCapture(self.config.video_path)
        if not cap.isOpened():
            raise VideoSourceError(f"Could not open video: {self.config.video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

        # Update geometry calculator dimensions if different from defaults
        if frame_w != self.geometry.frame_width or frame_h != self.geometry.frame_height:
            self.geometry = GeometryCalculator(
                frame_width=frame_w,
                frame_height=frame_h,
                crop_config=self.config.crop_search,
            )

        # Calculate sample step for sub-linear frame sampling
        step = max(1, int(round(video_fps / self.config.sample_rate_fps)))
        sample_indices = (
            list(range(0, total_frames, step))
            if total_frames > 0
            else []
        )

        self.logger.info(
            f"Video metadata: {total_frames} frames, {video_fps:.1f} FPS, {frame_w}x{frame_h}. "
            f"Sampling at {self.config.sample_rate_fps} FPS (every {step} frames, {len(sample_indices)} total)."
        )

        # Extract sampled frames and submit to concurrent thread pool
        frame_results: List[FrameResult] = []
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = []
            for idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                timestamp_s = idx / video_fps
                futures.append(
                    executor.submit(self._process_frame, idx, frame, timestamp_s)
                )

            for future in as_completed(futures):
                frame_results.append(future.result())

        cap.release()

        # Sort results chronologically
        frame_results.sort(key=lambda r: r.frame_idx)

        # Aggregate valid detections into a representative stable boundary & crop
        stable_boundary, stable_crop = self._aggregate_detections(frame_results)

        elapsed_time_s = time.time() - start_time
        metrics = self.metrics_collector.finish(
            total_frames=total_frames, elapsed_time_s=elapsed_time_s
        )

        self.logger.info(
            f"Pipeline finished: {metrics.sampled_frames} sampled frames, "
            f"{metrics.valid_detections} valid ({metrics.detection_rate:.1%}), "
            f"elapsed: {metrics.total_processing_time_s:.2f}s ({metrics.processing_fps:.1f} FPS)."
        )

        return PipelineResult(
            metrics=metrics,
            frame_results=frame_results,
            stable_boundary=stable_boundary,
            stable_crop=stable_crop,
        )

    def _aggregate_detections(
        self, results: List[FrameResult]
    ) -> tuple[Optional[Polygon], Optional[CropBox]]:
        """Aggregates valid frame detections to find the most representative boundary and crop."""
        valid_results = [r for r in results if r.status == "valid" and r.polygon is not None]
        if not valid_results:
            return None, None

        # Select the detection with median area to filter out transient outliers
        valid_results.sort(key=lambda r: r.polygon.area)
        median_result = valid_results[len(valid_results) // 2]

        return median_result.polygon, median_result.crop_box
