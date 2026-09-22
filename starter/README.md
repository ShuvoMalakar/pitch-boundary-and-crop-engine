# Automated Pitch Boundary & Camera Crop Engine

A modular, observable, and efficient computer-vision pipeline for automated pitch boundary detection and camera crop layout derivation from multi-camera sports video feeds.

---

## Architecture Overview

The system is decoupled into a reusable core library (`crop_engine`) and a thin CLI runner (`main.py`):

```
starter/
├── src/crop_engine/              # Core reusable library
│   ├── __init__.py               # Package exports
│   ├── config.py                 # Pydantic v2 fail-fast configuration models
│   ├── detector.py               # FieldDetector interface & ColorThresholdDetector
│   ├── exceptions.py             # Domain exception hierarchy (CropEngineError)
│   ├── geometry.py               # GeometryCalculator (cached boundary, crop derivation, IoU)
│   ├── pipeline.py               # PitchCropPipeline (frame seeking, concurrency, aggregation)
│   ├── reporter.py               # PlatformReporter with validated wire models
│   └── telemetry.py              # RunMetrics, thread-safe MetricsCollector, structured logging
├── tests/                        # 46 unit & integration tests
│   ├── conftest.py               # Shared synthetic frame fixtures and sys.path setup
│   ├── test_cli.py               # CLI runner and fail-fast tests
│   ├── test_config.py            # Configuration validation tests
│   ├── test_detector.py          # FieldDetector and thresholding tests
│   ├── test_exceptions.py        # Exception hierarchy tests
│   ├── test_geometry.py          # Geometry caching and crop derivation tests
│   ├── test_pipeline.py          # Pipeline sampling and concurrency tests
│   ├── test_reporter.py          # Reporting client and wire model tests
│   └── test_telemetry.py         # Thread-safe metrics and logging tests
├── mock_api/                     # Platform reporting service (Flask)
├── Dockerfile                    # Container definition for pipeline runner
├── docker-compose.yml            # Multi-container orchestration (runner + mock_api)
├── main.py                       # Thin CLI execution entry point
├── requirements.txt              # Core dependencies
└── DECISIONS.md                  # Architectural decisions, trade-offs, and design rationale
```

---

## Key Features

1. **Pluggable Detection Seam (`FieldDetector`):**
   The pipeline depends strictly on the abstract `FieldDetector` interface, allowing sport-specific or ML-based detectors (e.g., SAM, YOLO) to be swapped in without modifying pipeline orchestration or downstream geometry logic.
2. **Sub-Linear Processing Efficiency:**
   Uses frame seeking (`cap.set(cv2.CAP_PROP_POS_FRAMES)`) to sample at a configurable rate (default: 3 FPS). Processing time scales with the necessary frames rather than total video length (a 10x compute reduction over 30 FPS feeds).
3. **Multi-Core Concurrency:**
   Leverages `ThreadPoolExecutor` to parallelize heavy OpenCV and Shapely operations across available CPU cores while maintaining constant memory overhead.
4. **Invariant Geometry Caching:**
   Pre-computes and caches frame boundary polygons at startup to eliminate thousands of redundant per-frame allocations.
5. **Fail-Fast Configuration:**
   Strict Pydantic v2 models (`extra="forbid"`) validate all settings at load time, aborting immediately on invalid parameters or unexpected keys before processing begins.
6. **Platform Reporting & Fault Isolation:**
   Reports lifecycle events (`job_started`, `job_completed`, `job_failed`) and progress updates to `mock_api` using validated Pydantic wire models. Network outages are logged as warnings and do not crash or abort video processing.

---

## Installation

Ensure you have Python 3.12+ installed:

```bash
cd starter

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. Running Locally via CLI (`main.py`)

```bash
# Run with automatic synthetic feed generation
python main.py --generate-synthetic

# Run with custom video, sample rate, and parallel workers
python main.py --video my_match.mp4 --sample-rate 3.0 --workers 4

# Run with platform reporting enabled
python main.py --video my_match.mp4 --api-url http://localhost:5000

# View all options
python main.py --help
```

### 2. Running via Docker Compose

```bash
cd starter
docker compose up --build
```

This starts `mock_api` on port 5001 and executes the `runner` container, which processes the feed, reports progress over the Docker network to `mock_api:5000`, and prints the execution summary.

To inspect events recorded by `mock_api`:
```bash
curl http://localhost:5001/api/v1/jobs/events
```

---

## Running the Test Suite

The test suite contains **46 unit and integration tests** covering all modules:

```bash
cd starter
python -m pytest tests -v
```

---

## Architectural Decisions

For detailed explanations of engineering trade-offs, assumptions, validation strictness, and AI disclosure, see [DECISIONS.md](DECISIONS.md).
