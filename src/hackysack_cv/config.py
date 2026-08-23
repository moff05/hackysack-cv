"""Central configuration for the Hacky Sack CV pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AppConfig:
    """All tunable thresholds live here so behavior can be adjusted without touching pipeline code."""

    # --- Models ---
    pose_model_path: str = "yolov8n-pose.pt"
    sack_model_path: Optional[str] = "models/sack_detector.pt"
    pose_confidence: float = 0.4
    sack_confidence: float = 0.35

    # --- Video ---
    fps: float = 60.0
    ground_line_ratio: float = 0.92
    """Fraction of frame height treated as ground contact plane, since we don't
    have real-world floor geometry without a calibrated camera."""

    # --- Sack trajectory / touch detection ---
    touch_radius_px: int = 60
    trail_length: int = 15
    velocity_smoothing_window: int = 5
    inflection_min_delta_vy: float = 1.5
    """Minimum |Δvy| (px/frame) required to call a direction change a real
    inflection rather than measurement noise."""
    event_cooldown_frames: int = 10
    """Frames to wait after a touch/drop before another can fire, so one
    physical contact isn't double-counted across a few noisy frames."""

    # --- Kalman filter (constant velocity model) ---
    kalman_process_noise: float = 1e-2
    kalman_measurement_noise: float = 1e-1
    kalman_max_coast_frames: int = 15
    """How many consecutive missed detections (occlusion) the filter will
    keep predicting through before it gives up on the current track."""

    # --- Kinematics ---
    reference_player_height_m: float = 1.75

    # --- HUD ---
    hud_font_scale: float = 0.6
    hud_panel_alpha: float = 0.6

    # --- Keypoints of interest (COCO-17 names, see types.py for indices) ---
    tracked_keypoints: tuple[str, ...] = field(
        default_factory=lambda: (
            "left_ankle",
            "right_ankle",
            "left_knee",
            "right_knee",
            "chest",
        )
    )
