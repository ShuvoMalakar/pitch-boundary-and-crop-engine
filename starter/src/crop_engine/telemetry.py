from dataclasses import asdict, dataclass
import logging
import sys
import threading
from typing import Any, Dict


@dataclass
class RunMetrics:
    """Aggregated execution metrics for a video processing run."""

    total_frames_in_video: int = 0
    sampled_frames: int = 0
    valid_detections: int = 0
    missing_detections: int = 0
    invalid_detections: int = 0
    total_processing_time_s: float = 0.0

    @property
    def detection_rate(self) -> float:
        if self.sampled_frames <= 0:
            return 0.0
        return self.valid_detections / self.sampled_frames

    @property
    def processing_fps(self) -> float:
        if self.total_processing_time_s <= 0:
            return 0.0
        return self.sampled_frames / self.total_processing_time_s

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["detection_rate"] = round(self.detection_rate, 4)
        data["processing_fps"] = round(self.processing_fps, 2)
        return data


class MetricsCollector:
    """Thread-safe collector for pipeline execution metrics."""

    def __init__(self):
        self._lock = threading.RLock()
        self._metrics = RunMetrics()

    def record_sampled_frame(self) -> None:
        with self._lock:
            self._metrics.sampled_frames += 1

    def record_valid_detection(self) -> None:
        with self._lock:
            self._metrics.valid_detections += 1

    def record_missing_detection(self) -> None:
        with self._lock:
            self._metrics.missing_detections += 1

    def record_invalid_detection(self) -> None:
        with self._lock:
            self._metrics.invalid_detections += 1

    def finish(self, total_frames: int, elapsed_time_s: float) -> RunMetrics:
        with self._lock:
            self._metrics.total_frames_in_video = total_frames
            self._metrics.total_processing_time_s = max(0.0, elapsed_time_s)
            return self.get_snapshot()

    def get_snapshot(self) -> RunMetrics:
        with self._lock:
            return RunMetrics(
                total_frames_in_video=self._metrics.total_frames_in_video,
                sampled_frames=self._metrics.sampled_frames,
                valid_detections=self._metrics.valid_detections,
                missing_detections=self._metrics.missing_detections,
                invalid_detections=self._metrics.invalid_detections,
                total_processing_time_s=self._metrics.total_processing_time_s,
            )


def setup_logging(debug: bool = False) -> logging.Logger:
    """Configures structured stream logging for unattended batch execution."""
    logger = logging.getLogger("crop_engine")
    level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
