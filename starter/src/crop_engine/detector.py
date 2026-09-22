from abc import ABC, abstractmethod
from typing import Optional

import cv2
import numpy as np
from shapely.geometry import Polygon

from crop_engine.config import FieldDetectorConfig


class FieldDetector(ABC):
    """Abstract interface for pitch boundary detection."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> Optional[Polygon]:
        """Detect the field boundary polygon in a BGR frame."""
        pass


class ColorThresholdDetector(FieldDetector):
    """Detects pitch boundaries using HSV color thresholding."""

    def __init__(
        self,
        min_area: float = 1000.0,
        lower_green: Optional[np.ndarray] = None,
        upper_green: Optional[np.ndarray] = None,
    ):
        self.min_area = min_area
        self.lower_green = (
            lower_green if lower_green is not None else np.array([35, 40, 40], dtype=np.uint8)
        )
        self.upper_green = (
            upper_green if upper_green is not None else np.array([85, 255, 255], dtype=np.uint8)
        )

    def detect(self, frame: np.ndarray) -> Optional[Polygon]:
        if frame is None or frame.size == 0:
            return None

        h, w = frame.shape[:2]
        total_frame_area = float(h * w)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_green, self.upper_green)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        valid_contours = [
            c
            for c in contours
            if self.min_area <= cv2.contourArea(c) < 0.90 * total_frame_area
        ]
        if not valid_contours:
            return None

        largest = max(valid_contours, key=cv2.contourArea)
        pts = largest.reshape(-1, 2)
        if len(pts) < 3:
            return None

        try:
            poly = Polygon(pts)
            if poly.is_valid and poly.area >= self.min_area:
                return poly

            fixed_poly = poly.buffer(0)
            if fixed_poly.is_valid and not fixed_poly.is_empty and fixed_poly.area >= self.min_area:
                if fixed_poly.geom_type == "Polygon":
                    return fixed_poly
                elif fixed_poly.geom_type == "MultiPolygon":
                    return max(fixed_poly.geoms, key=lambda p: p.area)
        except Exception:
            return None

        return None


def get_detector(config: FieldDetectorConfig) -> FieldDetector:
    if config.type in ("sam_mask_v1", "color_threshold"):
        return ColorThresholdDetector(min_area=config.min_area)

    raise ValueError(f"Unsupported field detector type: '{config.type}'")
