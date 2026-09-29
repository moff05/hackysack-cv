#!/usr/bin/env python3
"""Analyze gameplay and save an annotated MP4 plus a JSON session report."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="Input video file")
    parser.add_argument("--output", default="output/annotated.mp4", help="Annotated .mp4 output")
    parser.add_argument("--report", help="JSON report (default: output video name with .json extension)")
    parser.add_argument("--pose-model", default=str(ROOT / "yolov8n-pose.pt"))
    parser.add_argument("--sack-model", help="Custom sack weights; an explicit missing path is an error")
    parser.add_argument("--color-sack", action="store_true", help="Use green/black color detector")
    parser.add_argument("--mock-sack", action="store_true", help="Use synthetic sack motion for testing")
    parser.add_argument("--skip-players", action="store_true", help="Disable pose inference (no touch attribution)")
    parser.add_argument("--reference-height-m", type=float, default=1.75)
    parser.add_argument("--touch-radius-px", type=int, default=60)
    parser.add_argument("--ground-line-ratio", type=float, default=0.92, help="Ground height as fraction of image height, 0 < value <= 1")
    parser.add_argument("--sack-confidence", type=float, default=0.25, help="Trained detector confidence, 0 to 1")
    parser.add_argument("--max-frames", type=int, help="Analyze only the first N frames")
    parser.add_argument("--display", action="store_true", help="Live preview; Q finishes the session")
    parser.add_argument("--quiet", action="store_true", help="Hide progress messages")
    args = parser.parse_args(argv)
    if sum((args.color_sack, args.mock_sack, bool(args.sack_model))) > 1:
        parser.error("Choose only one of --color-sack, --mock-sack, and --sack-model")
    if args.report and Path(args.report).suffix != ".json":
        parser.error("--report must use a .json extension")
    if args.sack_model and not Path(args.sack_model).is_file():
        parser.error(f"Sack model does not exist: {args.sack_model}")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be greater than zero")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        from hackysack_cv.config import AppConfig
        from hackysack_cv.pipeline import HackySackAnalyzer
        model = None if args.color_sack else args.sack_model or str(ROOT / "models/sack_detector.pt")
        config = AppConfig(pose_model_path=args.pose_model, sack_model_path=model,
                           player_tracker_config=str(ROOT / "configs/player_tracker.yaml"),
                           reference_player_height_m=args.reference_height_m,
                           touch_radius_px=args.touch_radius_px, ground_line_ratio=args.ground_line_ratio,
                           sack_model_confidence=args.sack_confidence)
        report = args.report or str(Path(args.output).with_suffix(".json"))
        if not args.mock_sack and (not model or not Path(model).is_file()):
            print("Using color detection tuned for a green/black footbag. Trained weights improve detection.", file=sys.stderr)
        if args.mock_sack:
            print("Synthetic demo: sack positions are generated, not detected from footage.", file=sys.stderr)
        def progress(done, total):
            suffix = f" / {total}" if total else ""
            print(f"\rAnalyzed {done}{suffix} frames", end="", file=sys.stderr, flush=True)
        analyzer = HackySackAnalyzer(config)
        stats = analyzer.run(args.video, args.output, display=args.display, force_mock_sack=args.mock_sack,
                             skip_players=args.skip_players, report_path=report, max_frames=args.max_frames, browser_video=True,
                             progress=None if args.quiet else progress)
        if not args.quiet:
            print(file=sys.stderr)
        from hackysack_cv.report import write_review
        review = write_review(analyzer.last_report, str(Path(report).with_suffix(".html")))
        print(f"Video: {args.output}\nReport: {report}\nReview: {review}")
        if analyzer.last_report['tracking']['observed_fraction'] < 0.5:
            print("Low sack observation coverage: inspect the review before relying on scores.", file=sys.stderr)
        print(f"Touches: {stats.total_touches} | Best round: {stats.best_round_touches} | Drops: {stats.total_drops}")
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nCancelled. Incomplete video removed; previous output preserved.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
