"""Video annotation: player boxes/skeletons, sack trail, and the stats HUD.

Drawing is done with plain OpenCV primitives rather than `supervision`'s
annotator classes - their pose/keypoint API has shifted across versions and
plain `cv2.circle`/`cv2.line` calls are trivial to keep stable here. The
`supervision` package is still listed as a dependency (per the original
spec) and is trivial to swap in if you prefer its annotators.
"""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from hackysack_cv.config import AppConfig
from hackysack_cv.kinematics import KinematicsResult
from hackysack_cv.types import GameStats, PlayerState, SackState, SKELETON_EDGES

PLAYER_COLORS = [
    (66, 135, 245),   # blue
    (66, 245, 129),   # green
    (245, 66, 197),   # pink
    (245, 173, 66),   # orange
    (170, 66, 245),   # purple
]
SACK_COLOR = (0, 215, 255)
GROUND_COLOR = (60, 60, 220)


def _player_color(player_id: int) -> tuple[int, int, int]:
    return PLAYER_COLORS[player_id % len(PLAYER_COLORS)]


def draw_players(frame: np.ndarray, players: List[PlayerState]) -> None:
    for player in players:
        color = _player_color(player.player_id)
        x1, y1, x2, y2 = (int(v) for v in player.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"P{player.player_id}",
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

        kp = player.raw_keypoints
        for a, b in SKELETON_EDGES:
            if a in kp and b in kp:
                pa = tuple(int(v) for v in kp[a])
                pb = tuple(int(v) for v in kp[b])
                cv2.line(frame, pa, pb, color, 2)
        for point in kp.values():
            cv2.circle(frame, (int(point[0]), int(point[1])), 3, color, -1)


def draw_ground_line(frame: np.ndarray, ground_y: float) -> None:
    h, w = frame.shape[:2]
    y = int(ground_y)
    cv2.line(frame, (0, y), (w, y), GROUND_COLOR, 1, cv2.LINE_AA)


def draw_sack(frame: np.ndarray, sack: Optional[SackState]) -> None:
    if sack is None:
        return
    trail = list(sack.trail)
    n = len(trail)
    for i in range(1, n):
        alpha = i / n
        thickness = max(1, int(4 * alpha))
        p1 = tuple(int(v) for v in trail[i - 1])
        p2 = tuple(int(v) for v in trail[i])
        cv2.line(frame, p1, p2, SACK_COLOR, thickness, cv2.LINE_AA)

    x, y = int(sack.position[0]), int(sack.position[1])
    radius = 10 if not sack.is_coasting else 6
    cv2.circle(frame, (x, y), radius, SACK_COLOR, -1 if not sack.is_coasting else 2)


def draw_dashboard(
    frame: np.ndarray,
    config: AppConfig,
    stats: GameStats,
    kinematics: Optional[KinematicsResult],
) -> None:
    h, w = frame.shape[:2]
    panel_w, panel_h = 260, 150
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, config.hud_panel_alpha, frame, 1 - config.hud_panel_alpha, 0, frame)

    lines = [
        f"Round touches: {stats.current_round_touches}",
        f"Total touches: {stats.total_touches}",
    ]
    if kinematics is not None:
        lines.append(f"Speed: {kinematics.speed_mph:.1f} mph")
        lines.append(f"Height: {kinematics.apex_height_ft:.2f} ft")
    else:
        lines.append("Speed: --")
        lines.append("Height: --")

    y = 35
    for line in lines:
        cv2.putText(
            frame, line, (22, y), cv2.FONT_HERSHEY_SIMPLEX, config.hud_font_scale, (255, 255, 255), 1, cv2.LINE_AA
        )
        y += 28

    _draw_leaderboard(frame, stats, x=w - 270, y=10)


def _draw_leaderboard(frame: np.ndarray, stats: GameStats, x: int, y: int) -> None:
    player_ids = sorted(set(stats.player_touch_counts) | set(stats.player_errors))
    if not player_ids:
        return

    row_h = 24
    panel_w, panel_h = 260, row_h * (len(player_ids) + 1) + 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + panel_w, y + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    header_y = y + 20
    cv2.putText(
        frame, "Player  Touches  Errors", (x + 10, header_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
    )
    for i, pid in enumerate(player_ids):
        row_y = header_y + row_h * (i + 1)
        color = _player_color(pid)
        touches = stats.player_touch_counts.get(pid, 0)
        errors = stats.player_errors.get(pid, 0)
        cv2.putText(
            frame, f"P{pid}      {touches:<8}{errors}", (x + 10, row_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )
