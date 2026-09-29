"""Sack (footbag) localization.

Three interchangeable detector strategies are provided so the pipeline runs
end-to-end before a custom-trained `sack_detector.pt` exists:

  1. YoloSackDetector    - real detector, once you've trained sack_detector.pt
  2. ColorThresholdSackDetector - HSV blob tracking, works on any high-contrast
     ball (e.g. a bright orange/yellow footbag) with zero training
  3. MockMotionSackDetector - synthetic bouncing trajectory, ignores the frame
     entirely; lets you validate the rest of the pipeline (Kalman smoothing,
     touch/drop logic, HUD) on any video, even blank footage

All three implement the same `detect(frame, frame_idx) -> Optional[(x, y, conf)]`
interface so `SackTracker` doesn't care which one it's holding.
"""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import Optional, Protocol, Tuple

import cv2
import numpy as np

from hackysack_cv.config import AppConfig
from hackysack_cv.kalman import KalmanFilter2D
from hackysack_cv.types import PlayerState, Point, SackState

Roi = Tuple[int, int, int, int]  # x0, y0, x1, y1 in full-frame pixel coords


def compute_search_roi(
    players: list[PlayerState],
    frame_width: int,
    frame_height: int,
    config: AppConfig,
) -> Optional[Roi]:
    """Bounding region the sack detector should search, built from where the
    players currently are rather than the whole frame.

    Real footage is rarely a clean studio background — trees, fences, and
    other green/bright clutter sitting elsewhere in the frame can outscore
    the actual sack for a naive color-threshold detector (observed directly:
    an early real-footage test locked onto a tree canopy well above and
    behind the players instead of the ball). Restricting the search to a
    margin around the player cluster — generous to the sides, deliberately
    modest above their heads — cuts most of that out. Returns None if no
    players are tracked this frame, in which case the caller should fall
    back to searching the whole frame.
    """
    if not players:
        return None

    x1s = [p.bbox[0] for p in players]
    y1s = [p.bbox[1] for p in players]
    x2s = [p.bbox[2] for p in players]
    y2s = [p.bbox[3] for p in players]
    widths = [p.bbox[2] - p.bbox[0] for p in players]
    heights = [p.bbox[3] - p.bbox[1] for p in players]
    widths.sort()
    heights.sort()
    median_w = widths[len(widths) // 2]
    median_h = heights[len(heights) // 2]

    x0 = min(x1s) - config.sack_roi_horizontal_margin_scale * median_w
    x1 = max(x2s) + config.sack_roi_horizontal_margin_scale * median_w
    y0 = min(y1s) - config.sack_roi_upward_margin_scale * median_h
    y1 = frame_height  # down to the ground — a dropped sack can be below any player's feet

    return (
        int(max(0, x0)),
        int(max(0, y0)),
        int(min(frame_width, x1)),
        int(min(frame_height, y1)),
    )


class SackDetector(Protocol):
    acceptance_threshold: float
    """Minimum `detect()` confidence SackTracker should actually act on. Not
    a shared/comparable scale across implementations — a trained model's
    class probability and a color-blob's circularity score mean different
    things at the same numeric value, so each detector sets its own rather
    than SackTracker applying one fixed cutoff to whichever is active."""

    def detect(
        self, frame: np.ndarray, frame_idx: int, roi: Optional[Roi] = None
    ) -> Optional[Tuple[float, float, float]]:
        """Return (x, y, confidence) in full-frame pixel coordinates, or None
        if not found. `roi`, when given, restricts the search region."""
        ...


class YoloSackDetector:
    """Wraps a custom-trained Ultralytics model whose only class is the sack."""

    def __init__(self, model_path: str, confidence: float = 0.35, imgsz: int = 1280) -> None:
        from ultralytics import YOLO  # deferred import: heavy + optional

        self._model = YOLO(model_path)
        self._confidence = confidence
        # SackTracker gates on this instead of a fixed constant, since a
        # trained model's confidence scale and a color-detector's circularity
        # score are not the same thing at all (see ColorThresholdSackDetector).
        self.acceptance_threshold = confidence
        # Must match (or be a deliberate multiple of) the imgsz the model was
        # trained at. predict() silently defaults to 640 if this isn't passed
        # explicitly, which does NOT come from the checkpoint's training
        # config — training the sack_detector.pt shipped here used imgsz=1280
        # specifically because the ball is only ~15-25px across in the source
        # 1920x1440 footage; leaving this unset was caught costing real
        # detections on real footage, not just a theoretical concern.
        self._imgsz = imgsz

    def detect(
        self, frame: np.ndarray, frame_idx: int, roi: Optional[Roi] = None
    ) -> Optional[Tuple[float, float, float]]:
        search_frame = frame
        ox, oy = 0, 0
        if roi is not None:
            ox, oy, rx1, ry1 = roi
            search_frame = frame[oy:ry1, ox:rx1]
            if search_frame.size == 0:
                return None

        results = self._model.predict(search_frame, imgsz=self._imgsz, conf=self._confidence, verbose=False)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return None
        boxes = results[0].boxes
        best_idx = int(boxes.conf.argmax())
        x1, y1, x2, y2 = boxes.xyxy[best_idx].tolist()
        conf = float(boxes.conf[best_idx])
        return ox + (x1 + x2) / 2.0, oy + (y1 + y2) / 2.0, conf


class ColorThresholdSackDetector:
    """Fallback detector: finds the largest blob in an HSV color range.

    Defaults tuned 2026-09-14 from a real close-up photo of Nicholas's actual
    footbag (seafoam-green/black suede panels): sampled HSV ~(74, 69, 129) on
    the green panels (OpenCV's 0-179 hue scale), black panels reliably near
    V<15 but too noisy in H/S to use as a primary signal outdoors (shadows
    and dark clothing share that same low-V range). Green is the
    discriminative channel here specifically because the test footage's
    playing surface is a pink/tan paver driveway, not grass — on a grass
    surface this same hue range would need a much tighter mask or a
    trained-model detector instead, since the ground itself would false-positive.
    """

    def __init__(
        self,
        hsv_lower: Tuple[int, int, int] = (60, 35, 55),
        hsv_upper: Tuple[int, int, int] = (90, 255, 220),
        min_area_px: float = 25.0,
        max_area_px: float = 3000.0,
        black_value_max: int = 55,
        min_black_fraction: float = 0.08,
        acceptance_threshold: float = 0.35,
    ) -> None:
        self._lower = np.array(hsv_lower, dtype=np.uint8)
        self._upper = np.array(hsv_upper, dtype=np.uint8)
        self._min_area = min_area_px
        self._max_area = max_area_px
        self._black_value_max = black_value_max
        self._min_black_fraction = min_black_fraction
        # Circularity-based, not a model's class probability — 0.35 here
        # means something completely different than 0.35 does for
        # YoloSackDetector. SackTracker gates on whichever detector reports.
        self.acceptance_threshold = acceptance_threshold

    def detect(
        self, frame: np.ndarray, frame_idx: int, roi: Optional[Roi] = None
    ) -> Optional[Tuple[float, float, float]]:
        search_frame = frame
        ox, oy = 0, 0
        if roi is not None:
            ox, oy, rx1, ry1 = roi
            search_frame = frame[oy:ry1, ox:rx1]
            if search_frame.size == 0:
                return None

        hsv = cv2.cvtColor(search_frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._lower, self._upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        candidates = [c for c in contours if self._min_area <= cv2.contourArea(c) <= self._max_area]
        if not candidates:
            return None

        # Green alone isn't discriminative enough against real backgrounds —
        # tree canopy/foliage sits in the same hue range and, restricted to
        # just this ROI, was still winning over the actual sack in testing
        # (see the class docstring). The sack is a two-tone panelled ball
        # (~40% near-black fabric butted right up against the green panels),
        # which ordinary foliage doesn't have, so require a real fraction of
        # near-black pixels in a padded window around each candidate blob
        # before accepting it — this is the check that actually separates
        # "a leaf" from "the ball" at this resolution.
        frame_h, frame_w = search_frame.shape[:2]
        v_channel = hsv[:, :, 2]
        scored = []
        for c in candidates:
            x, y, w, h = cv2.boundingRect(c)
            pad = max(w, h)  # generous — the ball's black panels can be a similar size to its green ones
            wx0, wy0 = max(0, x - pad), max(0, y - pad)
            wx1, wy1 = min(frame_w, x + w + pad), min(frame_h, y + h + pad)
            window = v_channel[wy0:wy1, wx0:wx1]
            if window.size == 0:
                continue
            black_fraction = float(np.mean(window < self._black_value_max))
            if black_fraction < self._min_black_fraction:
                continue
            area = cv2.contourArea(c)
            scored.append((area, black_fraction, c))

        if not scored:
            return None

        # Among candidates that pass the black-adjacency check, prefer the
        # largest — same rationale as before (max_area filter already caps
        # implausibly large blobs, so bigger among survivors is more likely
        # a real, closer/clearer view of the ball than sensor noise).
        area, black_fraction, best = max(scored, key=lambda s: s[0])
        (x, y), radius = cv2.minEnclosingCircle(best)
        circularity = area / (math.pi * radius**2 + 1e-6)
        confidence = float(np.clip(circularity, 0.0, 1.0))
        return ox + x, oy + y, confidence


class MockMotionSackDetector:
    """Generates a repeating parabolic bounce path, independent of the video
    frame. Useful for smoke-testing the pipeline before any real detector
    (trained or color-based) is wired up."""

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        period_frames: int = 45,
        apex_ratio: float = 0.35,
        ground_ratio: float = 0.85,
    ) -> None:
        self._w = frame_width
        self._h = frame_height
        self._period = period_frames
        self._apex_y = frame_height * apex_ratio
        self._ground_y = frame_height * ground_ratio
        self.acceptance_threshold = 0.0  # always returns confidence=1.0 below; any threshold passes

    def detect(
        self, frame: np.ndarray, frame_idx: int, roi: Optional[Roi] = None
    ) -> Optional[Tuple[float, float, float]]:
        phase = (frame_idx % self._period) / self._period
        x = self._w * (0.5 - 0.2 * math.cos(math.pi * frame_idx / self._period))
        # Parabola: 0 at phase 0/1 (ground), 1 at phase 0.5 (apex)
        arc = 4.0 * phase * (1.0 - phase)
        y = self._ground_y - arc * (self._ground_y - self._apex_y)
        return x, y, 1.0


class SackTracker:
    """Runs a detector each frame, smooths through a Kalman filter, and keeps
    a short trail for drawing and for the touch/drop logic in game_logic.py."""

    def __init__(self, detector: SackDetector, config: AppConfig) -> None:
        self._detector = detector
        self._config = config
        self._kf = KalmanFilter2D(
            process_noise=config.kalman_process_noise,
            measurement_noise=config.kalman_measurement_noise,
        )
        self._state = SackState(position=(0.0, 0.0), velocity=(0.0, 0.0))
        self._trail_maxlen = config.trail_length
        self._frames_since_detection = 0

    def update(
        self, frame: np.ndarray, frame_idx: int, roi: Optional[Roi] = None
    ) -> Optional[SackState]:
        detection = self._detector.detect(frame, frame_idx, roi)
        measurement: Optional[Point] = None

        if (detection is not None and all(math.isfinite(v) for v in detection)
                and detection[2] >= self._detector.acceptance_threshold):
            candidate = (detection[0], detection[1])
            # Reject an implausible frame-to-frame teleport rather than
            # yanking an established track onto it — this is what actually
            # rejects a color-threshold false positive (a backlit palm frond
            # in testing scored high enough to pass every per-frame check,
            # but was ~800px from where the ball actually was). Only gated
            # while the track is recently confirmed; after a real occlusion
            # long enough to exceed the relax threshold, let it re-lock
            # anywhere rather than staying pinned to a stale prediction.
            if (
                self._kf.initialized
                and self._frames_since_detection <= self._config.sack_gate_relax_after_missed_frames
                and self._frames_since_detection <= self._config.kalman_max_coast_frames
            ):
                predicted = self._kf.peek_predicted_position()
                jump = math.hypot(candidate[0] - predicted[0], candidate[1] - predicted[1])
                if jump > self._config.sack_max_jump_px:
                    candidate = None
            measurement = candidate

        if measurement is not None:
            if self._frames_since_detection > self._config.kalman_max_coast_frames:
                self._kf.reset(measurement)
                self._state = SackState(position=measurement, velocity=(0.0, 0.0))
            self._frames_since_detection = 0
        else:
            if not self._kf.initialized:
                return None
            self._frames_since_detection += 1
            if self._frames_since_detection > self._config.kalman_max_coast_frames:
                return None  # track considered lost; caller should treat sack as absent

        position = self._kf.update(measurement)
        velocity = self._kf.velocity

        trail = deque(self._state.trail, maxlen=self._trail_maxlen)
        trail.append(position)
        while len(trail) > self._trail_maxlen:
            trail.popleft()

        self._state = SackState(
            position=position,
            velocity=velocity,
            trail=trail,
            is_coasting=measurement is None,
            frames_since_detection=self._frames_since_detection,
        )
        return self._state


def build_sack_detector(config: AppConfig, frame_width: int, frame_height: int) -> SackDetector:
    """Picks the best available detector: trained model > color threshold.

    Trained-model weights are optional (see README); this keeps `run.py`
    usable on day one with a placeholder detector, and upgradeable to a real
    one just by dropping weights into `models/sack_detector.pt`.
    """
    if config.sack_model_path and Path(config.sack_model_path).exists():
        return YoloSackDetector(config.sack_model_path, confidence=config.sack_model_confidence)
    return ColorThresholdSackDetector(acceptance_threshold=config.sack_confidence)
