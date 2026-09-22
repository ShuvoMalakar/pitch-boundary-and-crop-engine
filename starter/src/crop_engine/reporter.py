from datetime import datetime, timezone
import logging
from typing import Any, Dict, Optional
import uuid

from pydantic import BaseModel, ConfigDict, Field
import requests

from crop_engine.geometry import CropBox
from crop_engine.telemetry import RunMetrics


class JobProgressPayload(BaseModel):
    """Validated schema for progress updates sent to the platform API."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    progress_percent: float = Field(ge=0.0, le=100.0)
    frames_processed: int = Field(ge=0)
    total_frames: int = Field(ge=0)
    current_fps: float = Field(default=0.0, ge=0.0)


class JobEventPayload(BaseModel):
    """Validated schema for milestone and outcome events sent to the platform API."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    event_type: str
    timestamp_iso: str
    details: Dict[str, Any] = Field(default_factory=dict)


class PlatformReporter:
    """HTTP reporting client that communicates job progress and events to the platform service."""

    def __init__(
        self,
        base_url: str,
        job_id: Optional[str] = None,
        timeout_s: float = 3.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.job_id = job_id or uuid.uuid4().hex[:12]
        self.timeout_s = timeout_s
        self.session = requests.Session()
        self.logger = logging.getLogger("crop_engine.reporter")

    def report_progress(self, progress: JobProgressPayload) -> bool:
        """Send a progress update to the platform."""
        url = f"{self.base_url}/api/v1/jobs/progress"
        try:
            resp = self.session.post(
                url, json=progress.model_dump(), timeout=self.timeout_s
            )
            resp.raise_for_status()
            return True
        except requests.RequestException as exc:
            self.logger.warning(
                f"Failed to report progress to {url}: {exc}. Video processing will continue."
            )
            return False

    def report_event(self, event: JobEventPayload) -> bool:
        """Send a milestone or outcome event to the platform."""
        url = f"{self.base_url}/api/v1/jobs/events"
        try:
            resp = self.session.post(
                url, json=event.model_dump(), timeout=self.timeout_s
            )
            resp.raise_for_status()
            return True
        except requests.RequestException as exc:
            self.logger.warning(
                f"Failed to report event '{event.event_type}' to {url}: {exc}. Video processing will continue."
            )
            return False

    def report_job_started(self, video_path: str, total_frames: int) -> bool:
        event = JobEventPayload(
            job_id=self.job_id,
            event_type="job_started",
            timestamp_iso=datetime.now(timezone.utc).isoformat(),
            details={"video_path": video_path, "total_frames": total_frames},
        )
        return self.report_event(event)

    def report_job_completed(
        self, metrics: RunMetrics, crop_box: Optional[CropBox]
    ) -> bool:
        details = metrics.to_dict()
        if crop_box:
            details["recommended_crop"] = crop_box.to_dict()

        event = JobEventPayload(
            job_id=self.job_id,
            event_type="job_completed",
            timestamp_iso=datetime.now(timezone.utc).isoformat(),
            details=details,
        )
        return self.report_event(event)

    def report_job_failed(self, error_message: str) -> bool:
        event = JobEventPayload(
            job_id=self.job_id,
            event_type="job_failed",
            timestamp_iso=datetime.now(timezone.utc).isoformat(),
            details={"error": error_message},
        )
        return self.report_event(event)
