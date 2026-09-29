#!/usr/bin/env python3
"""Generates a synthetic bouncing-ball clip so the pipeline can be smoke
tested without any real footage or trained sack weights.

Note: the drawn ball exercises sack detection/tracking/game logic end-to-end
(with --mock-sack or the default color-threshold fallback), but the plain
background has no real person in it, so YOLO pose won't find any players.
For a full demo (touch detection near actual keypoints), point run.py at real
footage of people playing.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np


def generate(
    output_path: str,
    width: int = 1280,
    height: int = 720,
    fps: int = 60,
    duration_s: float = 6.0,
    period_frames: int = 45,
    ball_color_bgr: tuple[int, int, int] = (120, 165, 90),  # green, matches
    # ColorThresholdSackDetector's default HSV range in sack_tracker.py
    ball_radius: int = 14,
) -> None:
    if min(width, height, fps, duration_s, period_frames, ball_radius) <= 0:
        raise ValueError("Video dimensions, timing, and radius must be positive")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    total_frames = int(duration_s * fps)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        raise OSError(f"Cannot write video: {output_path}")

    ground_y = int(height * 0.85)
    apex_y = int(height * 0.35)

    for frame_idx in range(total_frames):
        frame = np.full((height, width, 3), (120, 125, 140), dtype=np.uint8)  # neutral ground
        cv2.line(frame, (0, ground_y), (width, ground_y), (70, 70, 70), 2)

        phase = (frame_idx % period_frames) / period_frames
        x = int(width * (0.5 - 0.2 * math.cos(math.pi * frame_idx / period_frames)))
        arc = 4.0 * phase * (1.0 - phase)
        y = int(ground_y - arc * (ground_y - apex_y))

        cv2.circle(frame, (x, y), ball_radius + 8, (10, 10, 10), -1)
        cv2.circle(frame, (x - 5, y - 3), ball_radius, ball_color_bgr, -1)
        cv2.circle(frame, (x + 4, y + 4), ball_radius // 2, (10, 10, 10), -1)
        writer.write(frame)

    writer.release()
    print(f"Wrote {total_frames} frames ({total_frames / fps:.1f}s @ {fps}fps) to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a synthetic bouncing-ball test video")
    parser.add_argument("--output", default="test_videos/synthetic_bounce.mp4")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--duration-s", type=float, default=6.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate(
        output_path=args.output,
        width=args.width,
        height=args.height,
        fps=args.fps,
        duration_s=args.duration_s,
    )
