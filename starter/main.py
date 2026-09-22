import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict

from pydantic import ValidationError

from crop_engine.config import PipelineConfig
from crop_engine.exceptions import CropEngineError
from crop_engine.pipeline import PitchCropPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automated Pitch Boundary & Camera Crop Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--video",
        type=str,
        default="synthetic_pitch_feed.mp4",
        help="Path to the input video file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to optional JSON configuration file",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=3.0,
        help="Frame sampling rate in frames per second",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker threads for parallel frame analysis",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=os.environ.get("MOCK_API_URL", None),
        help="Base URL for the platform reporting service (e.g. http://mock_api:5000)",
    )
    parser.add_argument(
        "--aspect-ratio",
        type=str,
        default="16:9",
        help="Target aspect ratio for camera crop (e.g. 16:9, 4:3)",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=20,
        help="Padding in pixels around detected boundary",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug-level logging",
    )
    parser.add_argument(
        "--generate-synthetic",
        action="store_true",
        help="Generate synthetic test video if it does not exist locally",
    )
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> PipelineConfig:
    raw_config: Dict[str, Any] = {}

    # Load JSON file if provided
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            sys.stderr.write(f"Error: Configuration file not found: {args.config}\n")
            sys.exit(1)
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                raw_config = json.load(f)
        except json.JSONDecodeError as e:
            sys.stderr.write(f"Error: Malformed JSON in configuration file: {e}\n")
            sys.exit(1)

    # CLI arguments override or supply defaults
    if args.video:
        raw_config["video_path"] = args.video
    if args.sample_rate:
        raw_config["sample_rate_fps"] = args.sample_rate
    if args.workers:
        raw_config["max_workers"] = args.workers
    if args.api_url:
        raw_config["mock_api_url"] = args.api_url
    if args.debug:
        raw_config["debug_mode"] = True

    if "crop_search" not in raw_config:
        raw_config["crop_search"] = {}
    if args.aspect_ratio:
        raw_config["crop_search"]["aspect_ratio"] = args.aspect_ratio
    if args.padding is not None:
        raw_config["crop_search"]["padding_px"] = args.padding

    try:
        return PipelineConfig.from_dict(raw_config)
    except ValidationError as e:
        sys.stderr.write("Configuration Validation Error (fail-fast):\n")
        for err in e.errors():
            loc = " -> ".join(str(p) for p in err["loc"])
            sys.stderr.write(f"  - [{loc}]: {err['msg']}\n")
        sys.exit(1)


def main() -> None:
    args = parse_args()

    # Helper: generate synthetic video if requested or needed for default run
    if args.generate_synthetic or (
        args.video == "synthetic_pitch_feed.mp4" and not Path(args.video).exists()
    ):
        from synthetic_generator import generate_synthetic_video

        print(f"Generating synthetic feed: {args.video} ...")
        generate_synthetic_video(outputPath=args.video, numFrames=300)

    config = load_config(args)

    try:
        pipeline = PitchCropPipeline(config)
        result = pipeline.run()

        print("\n" + "=" * 50)
        print("Pipeline Execution Summary")
        print("=" * 50)
        metrics = result.metrics.to_dict()
        for k, v in metrics.items():
            print(f"  {k:<26}: {v}")

        if result.stable_crop:
            print(f"  recommended_crop          : {result.stable_crop.as_tuple}")
        print("=" * 50)

    except CropEngineError as e:
        sys.stderr.write(f"\nPipeline Error: {e}\n")
        sys.exit(1)
    except Exception as e:
        sys.stderr.write(f"\nUnexpected System Error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
