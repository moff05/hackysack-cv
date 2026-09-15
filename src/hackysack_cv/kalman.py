"""Minimal constant-velocity 2D Kalman filter, used to smooth sack position/
velocity and to coast through frames where detection drops out (occlusion,
motion blur, player's foot covering the ball, etc.)."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

Point = Tuple[float, float]


class KalmanFilter2D:
    """State vector: [x, y, vx, vy]. Measurement: [x, y]."""

    def __init__(
        self,
        process_noise: float = 1e-2,
        measurement_noise: float = 1e-1,
    ) -> None:
        self._state = np.zeros((4, 1), dtype=np.float64)
        self._P = np.eye(4, dtype=np.float64) * 500.0  # initial uncertainty
        self._F = np.array(
            [
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float64,
        )
        self._H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        self._Q = np.eye(4, dtype=np.float64) * process_noise
        self._R = np.eye(2, dtype=np.float64) * measurement_noise
        self._initialized = False

    @property
    def position(self) -> Point:
        return float(self._state[0, 0]), float(self._state[1, 0])

    @property
    def initialized(self) -> bool:
        return self._initialized

    def peek_predicted_position(self) -> Point:
        """What `predict()` would return, without mutating filter state —
        lets a caller gate an incoming measurement against the current
        trajectory before deciding whether to feed it into `update()`."""
        predicted_state = self._F @ self._state
        return float(predicted_state[0, 0]), float(predicted_state[1, 0])

    @property
    def velocity(self) -> Point:
        return float(self._state[2, 0]), float(self._state[3, 0])

    def reset(self, position: Point) -> None:
        self._state = np.array([[position[0]], [position[1]], [0.0], [0.0]])
        self._P = np.eye(4, dtype=np.float64) * 500.0
        self._initialized = True

    def predict(self) -> Point:
        self._state = self._F @ self._state
        self._P = self._F @ self._P @ self._F.T + self._Q
        return self.position

    def update(self, measurement: Optional[Point]) -> Point:
        """Feed in a new detection (or None if this frame had no detection)."""
        if measurement is None:
            return self.predict()

        if not self._initialized:
            self.reset(measurement)
            return self.position

        self.predict()
        z = np.array([[measurement[0]], [measurement[1]]], dtype=np.float64)
        y = z - self._H @ self._state
        S = self._H @ self._P @ self._H.T + self._R
        K = self._P @ self._H.T @ np.linalg.inv(S)
        self._state = self._state + K @ y
        self._P = (np.eye(4) - K @ self._H) @ self._P
        return self.position
