#!/usr/bin/env python3
"""CLI entrypoint: analyze a Hacky Sack video and write an annotated copy.

Examples:
    python run.py test_videos/sample.mp4
    python run.py test_videos/sample.mp4 --output output/annotated.mp4 --display
    python run.py test_videos/sample.mp4 --mock-sack   # no sack model needed at all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from hackysack_cv.config import AppConfig  # noqa: E402
from hackysack_cv.pipeline import HackySackAnalyzer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hacky Sack computer vision analyzer")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--output", default="output/annotated.mp4", help="Path to write annotated video")
    parser.add_argument("--pose-model", default="yolov8n-pose.pt", help="Ultralytics pose model path/name")
    parser.add_argument(
        "--sack-model",
        default="models/sack_detector.pt",
        help="Path to a custom-trained sack detector; falls back to color thresholding if missing",
    )
    parser.add_argument(
        "--mock-sack",
        action="store_true",
        help="Ignore the video entirely for sack detection and use a synthetic bouncing "
        "trajectory instead, so you can validate the pipeline before training/color-tuning a real detector",
    )
    parser.add_argument("--reference-height-m", type=float, default=1.75, help="Assumed player height in meters")
    parser.add_argument("--touch-radius-px", type=int, default=60, help="Proximity threshold for a touch, in pixels")
    parser.add_argument("--display", action="store_true", help="Show a live preview window while processing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = AppConfig(
        pose_model_path=args.pose_model,
        sack_model_path=args.sack_model,
        reference_player_height_m=args.reference_height_m,
        touch_radius_px=args.touch_radius_px,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    analyzer = HackySackAnalyzer(config)
    stats = analyzer.run(
        video_path=args.video,
        output_path=args.output,
        display=args.display,
        force_mock_sack=args.mock_sack,
    )

    print(f"\nAnnotated video written to: {args.output}")
    print(f"Total touches: {stats.total_touches}")
    print(f"Current round touches: {stats.current_round_touches}")
    print("Per-player touches:", dict(stats.player_touch_counts))
    print("Per-player errors:", dict(stats.player_errors))


if __name__ == "__main__":
    main()
