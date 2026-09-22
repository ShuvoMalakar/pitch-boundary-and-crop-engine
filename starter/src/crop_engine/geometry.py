from dataclasses import dataclass
from typing import Optional, Tuple

from shapely.geometry import Polygon

from crop_engine.config import CropSearchConfig


@dataclass(frozen=True)
class CropBox:
    """Represents a rectangular crop box within a video frame."""

    x: int
    y: int
    w: int
    h: int

    @property
    def aspect_ratio(self) -> float:
        return self.w / self.h if self.h > 0 else 0.0

    @property
    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}


class GeometryCalculator:
    """Calculates spatial intersections and camera crops with cached invariant geometry."""

    def __init__(
        self,
        frame_width: int = 1280,
        frame_height: int = 720,
        crop_config: Optional[CropSearchConfig] = None,
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.crop_config = crop_config or CropSearchConfig()

        # Cache the invariant outer boundary polygon once at initialization
        self.outer_boundary = Polygon(
            [
                (0, 0),
                (frame_width, 0),
                (frame_width, frame_height),
                (0, frame_height),
            ]
        )

    def compute_intersection_area(self, poly: Optional[Polygon]) -> float:
        """Compute the area of intersection between a polygon and the frame boundary."""
        if poly is None or not poly.is_valid or poly.is_empty:
            return 0.0
        return float(poly.intersection(self.outer_boundary).area)

    def calculate_crop(self, poly: Optional[Polygon]) -> Optional[CropBox]:
        """Derive an aspect-ratio-constrained crop box from a pitch boundary polygon."""
        if poly is None or not poly.is_valid or poly.is_empty:
            return None

        minx, miny, maxx, maxy = poly.bounds
        pad = self.crop_config.padding_px

        # Apply padding
        minx = max(0.0, minx - pad)
        miny = max(0.0, miny - pad)
        maxx = min(float(self.frame_width), maxx + pad)
        maxy = min(float(self.frame_height), maxy + pad)

        bw = maxx - minx
        bh = maxy - miny
        if bw <= 0 or bh <= 0:
            return None

        cx = (minx + maxx) / 2.0
        cy = (miny + maxy) / 2.0
        target_ar = self.crop_config.aspect_ratio_float

        # Enforce target aspect ratio
        if bw / bh < target_ar:
            crop_w = bh * target_ar
            crop_h = bh
        else:
            crop_w = bw
            crop_h = bw / target_ar

        # Clamp crop size to frame dimensions if necessary
        if crop_w > self.frame_width:
            crop_w = float(self.frame_width)
            crop_h = crop_w / target_ar
        if crop_h > self.frame_height:
            crop_h = float(self.frame_height)
            crop_w = crop_h * target_ar

        x = int(round(max(0.0, min(float(self.frame_width) - crop_w, cx - crop_w / 2.0))))
        y = int(round(max(0.0, min(float(self.frame_height) - crop_h, cy - crop_h / 2.0))))
        w = int(round(crop_w))
        h = int(round(crop_h))

        return CropBox(x=x, y=y, w=w, h=h)

    def compute_iou(self, poly1: Optional[Polygon], poly2: Optional[Polygon]) -> float:
        """Compute the Intersection-over-Union (IoU) between two polygons."""
        if poly1 is None or poly2 is None or not poly1.is_valid or not poly2.is_valid:
            return 0.0

        intersection_area = poly1.intersection(poly2).area
        union_area = poly1.union(poly2).area

        if union_area <= 0.0:
            return 0.0

        return float(intersection_area / union_area)
