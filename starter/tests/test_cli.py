"""Unit and integration tests for the main.py CLI entry point."""

import subprocess
import sys
from pathlib import Path
import pytest


def test_cli_help():
    """CLI prints help text and exits with code 0."""
    result = subprocess.run(
        [sys.executable, "main.py", "--help"],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent.parent),
    )
    assert result.returncode == 0
    assert "Automated Pitch Boundary & Camera Crop Engine" in result.stdout
    assert "--video" in result.stdout
    assert "--sample-rate" in result.stdout


def test_cli_fail_fast_on_invalid_arguments():
    """CLI fails fast with code 1 when given invalid configuration values."""
    result = subprocess.run(
        [sys.executable, "main.py", "--sample-rate", "-5.0"],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent.parent),
    )
    assert result.returncode == 1
    assert "Configuration Validation Error" in result.stderr
    assert "sample_rate_fps" in result.stderr


def test_cli_fail_fast_on_missing_config_file():
    """CLI fails fast with code 1 when specified config file does not exist."""
    result = subprocess.run(
        [sys.executable, "main.py", "--config", "non_existent_config.json"],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent.parent),
    )
    assert result.returncode == 1
    assert "Configuration file not found" in result.stderr


def test_cli_runs_successfully(tmp_path):
    """CLI runs end-to-end on a synthetic video and prints the execution summary."""
    import cv2
    import numpy as np

    video_path = str(tmp_path / "cli_feed.mp4")
    writer = cv2.VideoWriter(
        video_path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (1280, 720)
    )
    pts = np.array([[100, 100], [1180, 100], [1230, 620], [50, 620]], np.int32)
    for _ in range(30):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[:] = (34, 139, 34)
        cv2.polylines(frame, [pts], True, (255, 255, 255), 5)
        writer.write(frame)
    writer.release()

    result = subprocess.run(
        [
            sys.executable,
            "main.py",
            "--video",
            video_path,
            "--sample-rate",
            "3.0",
            "--workers",
            "2",
        ],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent.parent),
    )
    assert result.returncode == 0
    assert "Pipeline Execution Summary" in result.stdout
    assert "valid_detections" in result.stdout
    assert "recommended_crop" in result.stdout
