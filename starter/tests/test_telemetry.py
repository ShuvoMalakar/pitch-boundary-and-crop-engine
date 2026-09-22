"""Unit tests for telemetry, metrics collection, and structured logging."""

from concurrent.futures import ThreadPoolExecutor
import logging

from crop_engine.telemetry import MetricsCollector, RunMetrics, setup_logging


def test_run_metrics_properties():
    """Verify calculated properties for detection rate and processing FPS."""
    # Zero frames edge case
    empty_metrics = RunMetrics()
    assert empty_metrics.detection_rate == 0.0
    assert empty_metrics.processing_fps == 0.0

    # Normal run
    metrics = RunMetrics(
        total_frames_in_video=1800,
        sampled_frames=180,
        valid_detections=150,
        missing_detections=20,
        invalid_detections=10,
        total_processing_time_s=6.0,
    )
    assert round(metrics.detection_rate, 4) == round(150 / 180, 4)
    assert metrics.processing_fps == 30.0


def test_run_metrics_to_dict():
    """Verify dictionary serialization preserves all fields."""
    metrics = RunMetrics(
        total_frames_in_video=300,
        sampled_frames=30,
        valid_detections=25,
        total_processing_time_s=1.0,
    )
    d = metrics.to_dict()
    assert d["total_frames_in_video"] == 300
    assert d["sampled_frames"] == 30
    assert d["valid_detections"] == 25
    assert d["detection_rate"] == 0.8333
    assert d["processing_fps"] == 30.0


def test_metrics_collector_recording():
    """Verify recording individual frame outcomes."""
    collector = MetricsCollector()

    for _ in range(10):
        collector.record_sampled_frame()

    for _ in range(7):
        collector.record_valid_detection()

    for _ in range(2):
        collector.record_missing_detection()

    collector.record_invalid_detection()

    result = collector.finish(total_frames=100, elapsed_time_s=2.0)
    assert result.total_frames_in_video == 100
    assert result.sampled_frames == 10
    assert result.valid_detections == 7
    assert result.missing_detections == 2
    assert result.invalid_detections == 1
    assert result.total_processing_time_s == 2.0


def test_metrics_collector_thread_safety():
    """Verify thread-safe updates under heavy concurrent load."""
    collector = MetricsCollector()
    num_threads = 8
    ops_per_thread = 500

    def worker():
        for _ in range(ops_per_thread):
            collector.record_sampled_frame()
            collector.record_valid_detection()

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker) for _ in range(num_threads)]
        for f in futures:
            f.result()

    snapshot = collector.get_snapshot()
    expected_total = num_threads * ops_per_thread
    assert snapshot.sampled_frames == expected_total
    assert snapshot.valid_detections == expected_total


def test_setup_logging_levels():
    """Verify logging level is set according to debug flag."""
    logger_info = setup_logging(debug=False)
    assert logger_info.level == logging.INFO

    logger_debug = setup_logging(debug=True)
    assert logger_debug.level == logging.DEBUG
