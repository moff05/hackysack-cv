"""Multi-player pose tracking via Ultralytics YOLO pose + BoT-SORT."""

from __future__ import annotations

from typing import List

import numpy as np

from hackysack_cv.config import AppConfig
from hackysack_cv.types import CocoKeypoint, PlayerState


class PlayerTracker:
    """Thin wrapper around `model.track(..., persist=True)` that reshapes
    Ultralytics' output into our own `PlayerState` dataclass so the rest of
    the pipeline never touches Ultralytics types directly."""

    def __init__(self, config: AppConfig) -> None:
        from ultralytics import YOLO  # deferred import: heavy + optional

        self._model = YOLO(config.pose_model_path)
        self._confidence = config.pose_confidence
        self._imgsz = config.pose_model_imgsz
        self._config = config

    def update(self, frame: np.ndarray) -> List[PlayerState]:
        # iou=0.5 (down from Ultralytics' 0.7 default): raising imgsz to fix
        # a real detection-recall gap (see pose_model_imgsz in config.py)
        # introduced a rare but real side effect -- ~0.3% of frames got a
        # spurious 4th, heavily-overlapping duplicate box for one of the 3
        # real players, and each one cascaded into a burst of new track IDs
        # afterward. A tighter NMS threshold suppresses those duplicates
        # before they ever reach the tracker, without giving up the
        # resolution increase.
        results = self._model.track(
            frame,
            persist=True,
            tracker=self._config.player_tracker_config,
            conf=self._confidence,
            imgsz=self._imgsz,
            iou=0.5,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        if result.boxes is None or result.boxes.id is None or result.keypoints is None:
            return []  # nothing detected, or tracker hasn't assigned IDs yet

        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        track_ids = result.boxes.id.int().cpu().numpy()
        kp_xy = result.keypoints.xy.cpu().numpy()  # (n, 17, 2)
        kp_conf = (
            result.keypoints.conf.cpu().numpy()
            if result.keypoints.conf is not None
            else np.ones(kp_xy.shape[:2])
        )

        players: List[PlayerState] = []
        for i in range(len(track_ids)):
            raw_keypoints = {}
            keypoint_conf = {}
            for kp_idx in CocoKeypoint:
                x, y = kp_xy[i, kp_idx]
                conf = float(kp_conf[i, kp_idx])
                if conf >= self._confidence and (x > 0 or y > 0):
                    raw_keypoints[kp_idx] = (float(x), float(y))
                    keypoint_conf[kp_idx] = conf

            players.append(
                PlayerState(
                    player_id=int(track_ids[i]),
                    bbox=tuple(boxes_xyxy[i].tolist()),
                    raw_keypoints=raw_keypoints,
                    keypoint_conf=keypoint_conf,
                )
            )
        return players
