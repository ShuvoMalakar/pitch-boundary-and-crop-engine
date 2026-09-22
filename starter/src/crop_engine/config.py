"""Validated configuration model for the pitch boundary and crop engine.

Replaces the raw prototype dictionary with strict Pydantic models that fail fast
at load time on unexpected keys, invalid ranges, or malformed formats.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class FieldDetectorConfig(BaseModel):
    """Configuration for the field boundary detector."""

    model_config = ConfigDict(extra="forbid")

    type: str = Field(default="sam_mask_v1")
    sport: str = Field(default="football")
    min_area: float = Field(default=1000.0, gt=0)


class CropSearchConfig(BaseModel):
    """Configuration for deriving camera crops from detected boundaries."""

    model_config = ConfigDict(extra="forbid")

    aspect_ratio: str = Field(default="16:9")
    padding_px: int = Field(default=20, ge=0)

    @field_validator("aspect_ratio")
    @classmethod
    def validate_aspect_ratio(cls, v: str) -> str:
        parts = v.strip().split(":")
        if len(parts) != 2:
            raise ValueError(f"aspect_ratio must be formatted as 'W:H' (e.g. '16:9'), got '{v}'")
        try:
            w, h = float(parts[0]), float(parts[1])
            if w <= 0 or h <= 0:
                raise ValueError
        except ValueError:
            raise ValueError(f"aspect_ratio dimensions must be positive numbers, got '{v}'")
        return v.strip()

    @property
    def aspect_ratio_float(self) -> float:
        w, h = map(float, self.aspect_ratio.split(":"))
        return w / h


class PipelineConfig(BaseModel):
    """Validated pipeline configuration replacing the raw prototype dictionary."""

    model_config = ConfigDict(extra="forbid")

    video_path: str
    target_fps: int = Field(default=30, gt=0)
    sample_rate_fps: float = Field(default=3.0, gt=0)
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_workers: int = Field(default=4, ge=1)
    field_detector: FieldDetectorConfig = Field(default_factory=FieldDetectorConfig)
    crop_search: CropSearchConfig = Field(default_factory=CropSearchConfig)
    debug_mode: bool = False
    mock_api_url: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PipelineConfig:
        return cls.model_validate(data)

