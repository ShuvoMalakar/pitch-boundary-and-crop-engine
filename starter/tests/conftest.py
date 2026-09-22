import sys
from pathlib import Path

# Ensure src/ is in sys.path so tests can import crop_engine directly
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pytest


@pytest.fixture
def black_frame():
    """A fully black 1280x720 frame (simulates camera cut / blackout)."""
    return np.zeros((720, 1280, 3), dtype=np.uint8)


@pytest.fixture
def green_field_frame():
    """A solid green 1280x720 frame with no boundary lines (simulates close-up)."""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:] = (34, 139, 34)  # BGR for forest green
    return frame


@pytest.fixture
def green_field_with_boundary_frame():
    """A green frame with a white trapezoidal pitch boundary drawn on it."""
    import cv2

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:] = (34, 139, 34)
    pts = np.array([[100, 100], [1180, 100], [1230, 620], [50, 620]], np.int32)
    cv2.polylines(frame, [pts], True, (255, 255, 255), 5)
    return frame


@pytest.fixture
def noise_frame():
    """A green frame with a tiny noise box (simulates sensor glitch / watermark)."""
    import cv2

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:] = (34, 139, 34)
    noise_pts = np.array([[10, 10], [40, 10], [40, 30], [10, 30]], np.int32)
    cv2.polylines(frame, [noise_pts], True, (255, 255, 255), 2)
    return frame
