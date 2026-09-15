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

    # --- Sack detection search region ---
    sack_roi_horizontal_margin_scale: float = 0.75
    """How far the sack search region extends past the leftmost/rightmost
    tracked player, as a multiple of median player bbox width."""
    sack_roi_upward_margin_scale: float = 1.2
    """How far above the topmost player's head the sack search region
    extends, as a multiple of median player bbox height. Kept modest on
    purpose: a real kick rarely clears much more than a body-height above
    the head, and the whole point of this ROI is to exclude tree canopy /
    background clutter sitting further up the frame — see
    ColorThresholdSackDetector's false-positive history in sack_tracker.py."""

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
    sack_max_jump_px: float = 250.0
    """A detection more than this far from the filter's predicted position
    is rejected as a measurement (treated like a miss) rather than yanking
    the track onto it. This is what actually rejects a color-threshold
    false-positive jumping to a random spot elsewhere in frame (a backlit
    palm frond in testing) — the real sack can't teleport between frames.
    Rough starting point relative to this project's ~1920px-wide test
    footage at 30fps; if a genuinely fast kick keeps getting rejected as an
    "impossible jump", raise this. Only enforced once the track is
    established and recently confirmed — see sack_gate_relax_after_missed_frames."""
    sack_gate_relax_after_missed_frames: int = 5
    """Once a track has coasted through more than this many consecutive
    missed detections, the jump gate is skipped so the track can re-lock
    onto the ball anywhere in frame after a real occlusion, instead of
    staying gated to an increasingly stale predicted position."""

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
