"""Video analysis with validated output and a reproducible session report."""
from __future__ import annotations

from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import tempfile
import shutil
import subprocess
from typing import Callable, Optional

import cv2

from hackysack_cv import hud
from hackysack_cv.config import AppConfig
from hackysack_cv.game_logic import GameStateManager
from hackysack_cv.kinematics import compute_kinematics, estimate_meters_per_pixel
from hackysack_cv.player_tracker import PlayerTracker
from hackysack_cv.sack_tracker import MockMotionSackDetector, SackTracker, build_sack_detector, compute_search_roi
from hackysack_cv.types import GameStats


class HackySackAnalyzer:
    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or AppConfig()
        self.last_report: Optional[dict] = None

    def run(
        self,
        video_path: str,
        output_path: str,
        display: bool = False,
        force_mock_sack: bool = False,
        skip_players: bool = False,
        report_path: Optional[str] = None,
        max_frames: Optional[int] = None,
        browser_video: bool = False,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> GameStats:
        self.last_report = None
        source, output = Path(video_path).resolve(), Path(output_path).resolve()
        report = Path(report_path).resolve() if report_path else None
        if source == output or report in (source, output):
            raise ValueError("Input video, output video, and report must use different paths")
        if not source.is_file():
            raise FileNotFoundError(f"Input video does not exist: {source}")
        if output.suffix.lower() != ".mp4":
            raise ValueError("Output video must have an .mp4 extension")
        if max_frames is not None and (not isinstance(max_frames, int) or max_frames <= 0):
            raise ValueError("max_frames must be a positive integer")
        cap = cv2.VideoCapture(str(source))
        writer = None
        temporary = None
        try:
            if not cap.isOpened():
                raise ValueError(f"Could not decode video: {source}")
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if width <= 0 or height <= 0:
                raise ValueError("Video has invalid dimensions")
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not math.isfinite(fps) or fps <= 0:
                fps = self.config.fps
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            total = int(frame_count) if math.isfinite(frame_count) and frame_count > 0 else 0
            output.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=f".{output.stem}-", suffix=".mp4", dir=output.parent)
            os.close(fd)
            writer = cv2.VideoWriter(temporary, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            if not writer.isOpened():
                raise OSError(f"Cannot write MP4 video to {output.parent}")
            player_tracker = None if skip_players else PlayerTracker(self.config)
            detector = (MockMotionSackDetector(width, height) if force_mock_sack
                        else build_sack_detector(self.config, width, height))
            sack_tracker = SackTracker(detector, self.config)
            game = GameStateManager(self.config, frame_height=height)
            meters_per_pixel = None
            frame_idx = 0
            events = []
            observed = coasted = 0
            stopped_early = False
            while max_frames is None or frame_idx < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                players = player_tracker.update(frame) if player_tracker else []
                roi = compute_search_roi(players, width, height, self.config)
                sack = sack_tracker.update(frame, frame_idx, roi)
                scale = estimate_meters_per_pixel(players, self.config.reference_player_height_m)
                if scale is not None:
                    meters_per_pixel = scale
                frame_events = game.process(frame_idx, sack, players)
                for kind in ("touch", "drop"):
                    event = getattr(frame_events, kind)
                    if event:
                        events.append({"type": kind, "time_s": frame_idx / fps, **asdict(event)})
                if sack:
                    coasted += int(sack.is_coasting)
                    observed += int(not sack.is_coasting)
                kinematics = None
                if sack is not None and meters_per_pixel is not None:
                    kinematics = compute_kinematics(sack.velocity, sack.position[1], game.ground_y, meters_per_pixel, fps)
                hud.draw_ground_line(frame, game.ground_y)
                hud.draw_players(frame, players)
                hud.draw_sack(frame, sack)
                hud.draw_dashboard(frame, self.config, game.stats, kinematics)
                hud.draw_tracking_status(frame, sack, force_mock_sack)
                writer.write(frame)
                frame_idx += 1
                if progress and (frame_idx == 1 or frame_idx % 30 == 0):
                    progress(frame_idx, min(total, max_frames) if total and max_frames else total)
                if display:
                    cv2.imshow("Hacky Sack CV - Q to finish", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        stopped_early = True
                        break
            if frame_idx == 0:
                raise ValueError("Input video contains no decodable frames")
            if total and frame_idx < total and not stopped_early and (max_frames is None or frame_idx < max_frames):
                raise OSError(f"Video ended unexpectedly after {frame_idx} of {total} frames")
            writer.release()
            writer = None
            codec = "mp4v"
            if browser_video and shutil.which("ffmpeg"):
                fd, encoded = tempfile.mkstemp(prefix=f".{output.stem}-web-", suffix=".mp4", dir=output.parent)
                os.close(fd)
                try:
                    result = subprocess.run(
                        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", temporary,
                         "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
                         "-movflags", "+faststart", encoded], capture_output=True, text=True,
                    )
                    if result.returncode:
                        raise RuntimeError(f"Browser video encoding failed: {result.stderr[-1000:]}")
                    os.replace(encoded, temporary)
                    codec = "h264"
                finally:
                    Path(encoded).unlink(missing_ok=True)
            os.replace(temporary, output)
            temporary = None
            self.last_report = {
                "schema_version": 1, "video_codec": codec,
                "input": str(source), "output": str(output),
                "fps": fps, "width": width, "height": height,
                "processed_frames": frame_idx, "duration_s": frame_idx / fps,
                "stopped_early": stopped_early or bool(max_frames and frame_idx == max_frames and (not total or frame_idx < total)),
                "detector": type(detector).__name__, "synthetic": force_mock_sack,
                "player_tracking": not skip_players,
                "tracking": {"observed_frames": observed, "coasted_frames": coasted,
                             "missing_frames": frame_idx - observed - coasted,
                             "observed_fraction": observed / frame_idx},
                "stats": asdict(game.stats), "events": events, "config": asdict(self.config),
                "limitations": ["Touches and drops are heuristic estimates, not verified scores.",
                                "Ground is a manually configured horizontal line.",
                                "Speed and height use assumed player height, not camera calibration."],
            }
            if report:
                report.parent.mkdir(parents=True, exist_ok=True)
                report_temp = None
                try:
                    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=report.parent,
                                                     prefix=f".{report.name}-", delete=False) as handle:
                        report_temp = Path(handle.name)
                        json.dump(self.last_report, handle, indent=2, allow_nan=False)
                        handle.write("\n")
                    os.replace(report_temp, report)
                finally:
                    if report_temp:
                        report_temp.unlink(missing_ok=True)
            if progress:
                progress(frame_idx, frame_idx)
            return game.stats
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if temporary:
                Path(temporary).unlink(missing_ok=True)
            if display:
                cv2.destroyAllWindows()
