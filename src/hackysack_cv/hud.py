"""Video annotation: player boxes/skeletons, sack trail, and the stats HUD.

Drawing uses OpenCV primitives with no additional annotation dependencies.
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
    """Render on a reference canvas, then fit to small and portrait videos."""
    h, w = frame.shape[:2]
    panel = np.full((232, 300, 3), (25, 23, 20), dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (4, 231), SACK_COLOR, -1)
    def label(text, y, size=0.5, color=(230, 230, 230)):
        cv2.putText(panel, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX,
                    size * config.hud_font_scale / 0.6, color, 1, cv2.LINE_AA)
    label("FOOTBAG / SESSION", 28, 0.55, SACK_COLOR)
    label(str(stats.current_round_touches), 80, 1.5)
    label("TOUCHES THIS ROUND", 105, 0.42, (165, 165, 165))
    label(f"Total {stats.total_touches}   Best {stats.best_round_touches}   Drops {stats.total_drops}", 139)
    cv2.line(panel, (18, 153), (280, 153), (65, 65, 65), 1)
    label(f"Est. speed   {kinematics.speed_mph:.1f} mph" if kinematics else "Est. speed   --", 180)
    label(f"Est. height  {kinematics.apex_height_ft:.2f} ft" if kinematics else "Est. height  --", 208)
    scale = min(1.0, max(1, w - 20) / 300, max(1, h - 55) / 232)
    pw, ph = max(1, int(300 * scale)), max(1, int(232 * scale))
    panel = cv2.resize(panel, (pw, ph))
    x, y = min(10, w - pw), min(10, h - ph)
    target = frame[y:y + ph, x:x + pw]
    cv2.addWeighted(panel, config.hud_panel_alpha, target, 1 - config.hud_panel_alpha, 0, target)
    if w >= 600:
        _draw_leaderboard(frame, stats, x=w - 270, y=10)


def draw_tracking_status(frame: np.ndarray, sack: Optional[SackState], synthetic: bool = False) -> None:
    h, w = frame.shape[:2]
    status = "SYNTHETIC DEMO" if synthetic else "SACK MISSING" if sack is None else "SACK PREDICTED" if sack.is_coasting else "SACK DETECTED"
    size = min(0.5, max(0.15, (w - 20) / 350))
    cv2.rectangle(frame, (0, max(0, h - 30)), (w, h), (25, 23, 20), -1)
    cv2.putText(frame, status, (8, max(10, h - 10)), cv2.FONT_HERSHEY_SIMPLEX, size, SACK_COLOR, 1, cv2.LINE_AA)


def _draw_leaderboard(frame: np.ndarray, stats: GameStats, x: int, y: int) -> None:
    player_ids = sorted(set(stats.player_touch_counts) | set(stats.player_errors),
                        key=lambda pid: (-stats.player_touch_counts.get(pid, 0), pid))
    if not player_ids:
        return
    row_h = 24
    max_rows = max(0, (frame.shape[0] - y - 70) // row_h)
    player_ids = player_ids[:max_rows]
    panel_h = row_h * (len(player_ids) + 1) + 10
    region = frame[y:y + panel_h, x:x + 260]
    region[:] = (region.astype(np.float32) * 0.4 + np.array([25, 23, 20]) * 0.6).astype(np.uint8)
    cv2.putText(frame, "PLAYER  TOUCHES  ERRORS", (x + 10, y + 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (220, 220, 220), 1, cv2.LINE_AA)
    for i, pid in enumerate(player_ids):
        cv2.putText(frame, f"P{pid:<5} {stats.player_touch_counts.get(pid, 0):<8} {stats.player_errors.get(pid, 0)}",
                    (x + 10, y + 20 + row_h * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, _player_color(pid), 1, cv2.LINE_AA)
