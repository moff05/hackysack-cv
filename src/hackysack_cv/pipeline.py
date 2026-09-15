"""Top-level orchestrator: wires player tracking, sack tracking, kinematics,
game logic, and HUD rendering into a single video-in/video-out pass."""

from __future__ import annotations

from typing import Optional

import cv2

from hackysack_cv import hud
from hackysack_cv.config import AppConfig
from hackysack_cv.game_logic import GameStateManager
from hackysack_cv.kinematics import compute_kinematics, estimate_meters_per_pixel
from hackysack_cv.player_tracker import PlayerTracker
from hackysack_cv.sack_tracker import (
    MockMotionSackDetector,
    SackTracker,
    build_sack_detector,
    compute_search_roi,
)
from hackysack_cv.types import GameStats


class HackySackAnalyzer:
    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or AppConfig()

    def run(
        self,
        video_path: str,
        output_path: str,
        display: bool = False,
        force_mock_sack: bool = False,
    ) -> GameStats:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or self.config.fps

        player_tracker = PlayerTracker(self.config)
        detector = (
            MockMotionSackDetector(width, height)
            if force_mock_sack
            else build_sack_detector(self.config, width, height)
        )
        sack_tracker = SackTracker(detector, self.config)
        game_state = GameStateManager(self.config, frame_height=height)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        meters_per_pixel: Optional[float] = None
        frame_idx = 0

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                players = player_tracker.update(frame)
                roi = compute_search_roi(players, width, height, self.config)
                sack_state = sack_tracker.update(frame, frame_idx, roi)

                scale = estimate_meters_per_pixel(players, self.config.reference_player_height_m)
                if scale is not None:
                    meters_per_pixel = scale

                if sack_state is not None:
                    game_state.process(frame_idx, sack_state, players)

                kinematics = None
                if sack_state is not None and meters_per_pixel is not None:
                    kinematics = compute_kinematics(
                        velocity_px_per_frame=sack_state.velocity,
                        position_y_px=sack_state.position[1],
                        ground_y_px=game_state.ground_y,
                        meters_per_pixel=meters_per_pixel,
                        fps=fps,
                    )

                hud.draw_ground_line(frame, game_state.ground_y)
                hud.draw_players(frame, players)
                hud.draw_sack(frame, sack_state)
                hud.draw_dashboard(frame, self.config, game_state.stats, kinematics)

                writer.write(frame)
                if display:
                    cv2.imshow("Hacky Sack CV", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                frame_idx += 1
        finally:
            cap.release()
            writer.release()
            if display:
                cv2.destroyAllWindows()

        return game_state.stats
