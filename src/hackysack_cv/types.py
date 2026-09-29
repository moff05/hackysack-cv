"""Shared data types for the Hacky Sack CV pipeline."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Deque, Dict, Optional, Tuple

Point = Tuple[float, float]


class CocoKeypoint(IntEnum):
    """Index layout produced by Ultralytics YOLO pose models (COCO-17)."""

    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


SKELETON_EDGES: Tuple[Tuple[CocoKeypoint, CocoKeypoint], ...] = (
    (CocoKeypoint.LEFT_SHOULDER, CocoKeypoint.RIGHT_SHOULDER),
    (CocoKeypoint.LEFT_SHOULDER, CocoKeypoint.LEFT_HIP),
    (CocoKeypoint.RIGHT_SHOULDER, CocoKeypoint.RIGHT_HIP),
    (CocoKeypoint.LEFT_HIP, CocoKeypoint.RIGHT_HIP),
    (CocoKeypoint.LEFT_HIP, CocoKeypoint.LEFT_KNEE),
    (CocoKeypoint.LEFT_KNEE, CocoKeypoint.LEFT_ANKLE),
    (CocoKeypoint.RIGHT_HIP, CocoKeypoint.RIGHT_KNEE),
    (CocoKeypoint.RIGHT_KNEE, CocoKeypoint.RIGHT_ANKLE),
    (CocoKeypoint.LEFT_SHOULDER, CocoKeypoint.LEFT_ELBOW),
    (CocoKeypoint.LEFT_ELBOW, CocoKeypoint.LEFT_WRIST),
    (CocoKeypoint.RIGHT_SHOULDER, CocoKeypoint.RIGHT_ELBOW),
    (CocoKeypoint.RIGHT_ELBOW, CocoKeypoint.RIGHT_WRIST),
    (CocoKeypoint.NOSE, CocoKeypoint.LEFT_SHOULDER),
    (CocoKeypoint.NOSE, CocoKeypoint.RIGHT_SHOULDER),
)


@dataclass
class PlayerState:
    """One tracked player in a single frame."""

    player_id: int
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    raw_keypoints: Dict[CocoKeypoint, Point]
    keypoint_conf: Dict[CocoKeypoint, float]

    @property
    def named_keypoints(self) -> Dict[str, Point]:
        """Ankles/knees plus a synthetic chest point (shoulder midpoint), the
        subset touch detection actually cares about."""
        kp = self.raw_keypoints
        out: Dict[str, Point] = {}
        for name, idx in (
            ("left_ankle", CocoKeypoint.LEFT_ANKLE),
            ("right_ankle", CocoKeypoint.RIGHT_ANKLE),
            ("left_knee", CocoKeypoint.LEFT_KNEE),
            ("right_knee", CocoKeypoint.RIGHT_KNEE),
        ):
            if idx in kp:
                out[name] = kp[idx]
        if CocoKeypoint.LEFT_SHOULDER in kp and CocoKeypoint.RIGHT_SHOULDER in kp:
            lx, ly = kp[CocoKeypoint.LEFT_SHOULDER]
            rx, ry = kp[CocoKeypoint.RIGHT_SHOULDER]
            out["chest"] = ((lx + rx) / 2.0, (ly + ry) / 2.0)
        return out

    @property
    def pixel_height(self) -> Optional[float]:
        """Bbox height as a stand-in for the player's real-world height in pixels."""
        x1, y1, x2, y2 = self.bbox
        h = y2 - y1
        return h if h > 0 else None


@dataclass
class SackObservation:
    """A single raw detection of the sack, or None if nothing was found this frame."""

    frame_idx: int
    position: Optional[Point]
    confidence: float = 0.0


@dataclass
class SackState:
    """Kalman-smoothed sack state plus a short trail for drawing/inflection checks."""

    position: Point
    velocity: Point  # px/frame
    trail: Deque[Point] = field(default_factory=lambda: deque(maxlen=15))
    is_coasting: bool = False  # true when the filter is predicting through a missed detection
    frames_since_detection: int = 0


@dataclass
class TouchEvent:
    frame_idx: int
    player_id: int
    position: Point


@dataclass
class DropEvent:
    frame_idx: int
    player_id: Optional[int]
    position: Point


@dataclass
class GameStats:
    current_round_touches: int = 0
    total_touches: int = 0
    total_drops: int = 0
    best_round_touches: int = 0
    player_touch_counts: Dict[int, int] = field(default_factory=dict)
    player_errors: Dict[int, int] = field(default_factory=dict)

    def record_touch(self, player_id: int) -> None:
        self.current_round_touches += 1
        self.total_touches += 1
        self.best_round_touches = max(self.best_round_touches, self.current_round_touches)
        self.player_touch_counts[player_id] = self.player_touch_counts.get(player_id, 0) + 1

    def record_drop(self, player_id: Optional[int]) -> None:
        self.total_drops += 1
        self.current_round_touches = 0
        if player_id is not None:
            self.player_errors[player_id] = self.player_errors.get(player_id, 0) + 1
