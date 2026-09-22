"""Unit tests for geometry caching, crop calculation, and IoU metrics."""

import pytest
from shapely.geometry import Polygon

from crop_engine.config import CropSearchConfig
from crop_engine.geometry import CropBox, GeometryCalculator


def test_outer_boundary_cached_once():
    """Invariant outer boundary polygon is created once at initialization."""
    calc = GeometryCalculator(frame_width=1280, frame_height=720)

    assert isinstance(calc.outer_boundary, Polygon)
    assert calc.outer_boundary.is_valid
    assert calc.outer_boundary.area == 1280 * 720

    # Ensure the cached instance is reused
    ref1 = calc.outer_boundary
    ref2 = calc.outer_boundary
    assert ref1 is ref2


def test_compute_intersection_area_inside_frame():
    """Polygon fully inside frame yields its full area."""
    calc = GeometryCalculator(frame_width=1280, frame_height=720)
    poly = Polygon([(100, 100), (500, 100), (500, 400), (100, 400)])

    area = calc.compute_intersection_area(poly)
    assert area == poly.area == 120000.0


def test_compute_intersection_area_partially_outside():
    """Polygon partially outside frame is clipped to the frame boundary."""
    calc = GeometryCalculator(frame_width=1280, frame_height=720)
    # Extends 200px beyond right edge (x=1480)
    poly = Polygon([(1000, 100), (1480, 100), (1480, 500), (1000, 500)])

    area = calc.compute_intersection_area(poly)
    # Only 1000 to 1280 (width 280, height 400) is within frame
    assert area == 280.0 * 400.0 == 112000.0


def test_compute_intersection_area_none_or_empty():
    """None or empty polygon returns 0.0."""
    calc = GeometryCalculator()
    assert calc.compute_intersection_area(None) == 0.0
    assert calc.compute_intersection_area(Polygon()) == 0.0


def test_calculate_crop_aspect_ratio():
    """Derived crop box strictly satisfies the target aspect ratio."""
    config = CropSearchConfig(aspect_ratio="16:9", padding_px=10)
    calc = GeometryCalculator(frame_width=1280, frame_height=720, crop_config=config)

    # Typical pitch trapezoid
    poly = Polygon([(150, 150), (1130, 150), (1200, 600), (80, 600)])
    crop = calc.calculate_crop(poly)

    assert crop is not None
    assert isinstance(crop, CropBox)
    assert crop.aspect_ratio == pytest.approx(16 / 9, rel=0.02)


def test_calculate_crop_stays_within_frame_bounds():
    """Crop box coordinates must remain entirely within the frame boundary."""
    config = CropSearchConfig(aspect_ratio="16:9", padding_px=50)
    calc = GeometryCalculator(frame_width=1280, frame_height=720, crop_config=config)

    # Large polygon close to borders
    poly = Polygon([(20, 20), (1260, 20), (1260, 700), (20, 700)])
    crop = calc.calculate_crop(poly)

    assert crop is not None
    assert crop.x >= 0
    assert crop.y >= 0
    assert crop.x + crop.w <= 1280
    assert crop.y + crop.h <= 720


def test_calculate_crop_none_or_empty():
    """None or empty polygon safely returns None."""
    calc = GeometryCalculator()
    assert calc.calculate_crop(None) is None
    assert calc.calculate_crop(Polygon()) is None


def test_compute_iou():
    """IoU computes intersection over union accurately."""
    calc = GeometryCalculator()

    p1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    p2 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    p3 = Polygon([(5, 0), (15, 0), (15, 10), (5, 10)])
    p4 = Polygon([(100, 100), (200, 100), (200, 200), (100, 200)])

    # Identical polygons -> IoU = 1.0
    assert calc.compute_iou(p1, p2) == pytest.approx(1.0)

    # 50% overlap -> intersection=50, union=150, IoU = 50/150 = 1/3
    assert calc.compute_iou(p1, p3) == pytest.approx(1 / 3)

    # Disjoint polygons -> IoU = 0.0
    assert calc.compute_iou(p1, p4) == 0.0

    # None handling -> IoU = 0.0
    assert calc.compute_iou(p1, None) == 0.0
