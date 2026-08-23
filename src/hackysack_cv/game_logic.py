"""Touch detection and round/session game-state tracking.

Two events drive everything:

- Touch: the sack is near a player's ankle/knee/chest AND its vertical
  velocity flips from downward (+y, since image y grows downward) to upward
  (-y) sharply enough to be a real kick rather than sensor noise.
- Drop: the sack reaches the ground plane without that upward deflection.
  This ends the round and charges an error to whichever player was closest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from hackysack_cv.config import AppConfig
from hackysack_cv.types import DropEvent, GameStats, PlayerState, Point, SackState, TouchEvent


def _distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _nearest_player(
    players: List[PlayerState], sack_pos: Point
) -> Optional[Tuple[int, str, float]]:
    """Closest (player_id, keypoint_name, distance) across all tracked players'
    ankle/knee/chest points."""
    best: Optional[Tuple[int, str, float]] = None
    for player in players:
        for name, point in player.named_keypoints.items():
            dist = _distance(point, sack_pos)
            if best is None or dist < best[2]:
                best = (player.player_id, name, dist)
    return best


@dataclass
class FrameEvents:
    touch: Optional[TouchEvent] = None
    drop: Optional[DropEvent] = None


class GameStateManager:
    def __init__(self, config: AppConfig, frame_height: int) -> None:
        self._config = config
        self._ground_y = frame_height * config.ground_line_ratio
        self.stats = GameStats()
        self._prev_vy: Optional[float] = None
        self._last_event_frame: int = -config.event_cooldown_frames

    @property
    def ground_y(self) -> float:
        return self._ground_y

    def _cooldown_elapsed(self, frame_idx: int) -> bool:
        return (frame_idx - self._last_event_frame) >= self._config.event_cooldown_frames

    def process(
        self, frame_idx: int, sack: SackState, players: List[PlayerState]
    ) -> FrameEvents:
        events = FrameEvents()
        vy = sack.velocity[1]
        prev_vy = self._prev_vy
        self._prev_vy = vy

        if prev_vy is None or not self._cooldown_elapsed(frame_idx):
            return events

        nearest = _nearest_player(players, sack.position)
        is_inflection = (
            prev_vy > 0
            and vy < 0
            and (prev_vy - vy) >= self._config.inflection_min_delta_vy
        )
        near_a_player = nearest is not None and nearest[2] <= self._config.touch_radius_px

        if is_inflection and near_a_player:
            player_id = nearest[0]  # type: ignore[index]
            self.stats.record_touch(player_id)
            events.touch = TouchEvent(frame_idx=frame_idx, player_id=player_id, position=sack.position)
            self._last_event_frame = frame_idx
            return events

        hit_ground = sack.position[1] >= self._ground_y
        if hit_ground and not is_inflection:
            attributed_player = nearest[0] if nearest is not None else None
            self.stats.record_drop(attributed_player)
            events.drop = DropEvent(
                frame_idx=frame_idx, player_id=attributed_player, position=sack.position
            )
            self._last_event_frame = frame_idx

        return events
