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
from pathlib import Path
from typing import Optional, Protocol, Tuple

import cv2
import numpy as np

from hackysack_cv.config import AppConfig
from hackysack_cv.kalman import KalmanFilter2D
from hackysack_cv.types import Point, SackState


class SackDetector(Protocol):
    def detect(self, frame: np.ndarray, frame_idx: int) -> Optional[Tuple[float, float, float]]:
        """Return (x, y, confidence) in pixel coordinates, or None if not found."""
        ...


class YoloSackDetector:
    """Wraps a custom-trained Ultralytics model whose only class is the sack."""

    def __init__(self, model_path: str, confidence: float = 0.35) -> None:
        from ultralytics import YOLO  # deferred import: heavy + optional

        self._model = YOLO(model_path)
        self._confidence = confidence

    def detect(self, frame: np.ndarray, frame_idx: int) -> Optional[Tuple[float, float, float]]:
        results = self._model.predict(frame, conf=self._confidence, verbose=False)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return None
        boxes = results[0].boxes
        best_idx = int(boxes.conf.argmax())
        x1, y1, x2, y2 = boxes.xyxy[best_idx].tolist()
        conf = float(boxes.conf[best_idx])
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0, conf


class ColorThresholdSackDetector:
    """Fallback detector: finds the largest blob in an HSV color range.

    Defaults target a bright orange/yellow footbag under typical outdoor
    lighting. Tune `hsv_lower`/`hsv_upper` to match your actual sack's color.
    """

    def __init__(
        self,
        hsv_lower: Tuple[int, int, int] = (5, 120, 150),
        hsv_upper: Tuple[int, int, int] = (30, 255, 255),
        min_area_px: float = 25.0,
        max_area_px: float = 3000.0,
    ) -> None:
        self._lower = np.array(hsv_lower, dtype=np.uint8)
        self._upper = np.array(hsv_upper, dtype=np.uint8)
        self._min_area = min_area_px
        self._max_area = max_area_px

    def detect(self, frame: np.ndarray, frame_idx: int) -> Optional[Tuple[float, float, float]]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._lower, self._upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        candidates = [c for c in contours if self._min_area <= cv2.contourArea(c) <= self._max_area]
        if not candidates:
            return None

        best = max(candidates, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(best)
        area = cv2.contourArea(best)
        circularity = area / (math.pi * radius**2 + 1e-6)
        confidence = float(np.clip(circularity, 0.0, 1.0))
        return x, y, confidence


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

    def detect(self, frame: np.ndarray, frame_idx: int) -> Optional[Tuple[float, float, float]]:
        phase = (frame_idx % self._period) / self._period
        x = self._w * (0.3 + 0.4 * phase)
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

    def update(self, frame: np.ndarray, frame_idx: int) -> Optional[SackState]:
        detection = self._detector.detect(frame, frame_idx)
        measurement: Optional[Point] = None

        if detection is not None and detection[2] >= self._config.sack_confidence:
            measurement = (detection[0], detection[1])
            self._frames_since_detection = 0
        else:
            self._frames_since_detection += 1
            if self._frames_since_detection > self._config.kalman_max_coast_frames:
                return None  # track considered lost; caller should treat sack as absent

        position = self._kf.update(measurement)
        velocity = self._kf.velocity

        trail = self._state.trail
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
        return YoloSackDetector(config.sack_model_path, confidence=config.sack_confidence)
    return ColorThresholdSackDetector()
