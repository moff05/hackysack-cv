"""Pixel-space to real-world unit conversions.

Without a calibrated camera we can't get true metric scale, so we estimate a
meters-per-pixel factor from an assumed average player height. This is a
rough approximation - fine for relative speed/height comparisons and a fun
HUD number, not for anything that needs to be accurate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from hackysack_cv.types import PlayerState

MPH_PER_MPS = 2.23694
FEET_PER_METER = 3.28084


@dataclass
class KinematicsResult:
    speed_mps: float
    speed_mph: float
    apex_height_m: float
    apex_height_ft: float


def estimate_meters_per_pixel(
    players: List[PlayerState], reference_height_m: float
) -> Optional[float]:
    """Averages bbox pixel-height across all currently tracked players and
    scales against an assumed real-world height. Returns None if no players
    are visible (caller should hold onto the last known scale)."""
    heights = [p.pixel_height for p in players if p.pixel_height]
    if not heights:
        return None
    avg_pixel_height = sum(heights) / len(heights)
    return reference_height_m / avg_pixel_height


def compute_kinematics(
    velocity_px_per_frame: tuple[float, float],
    position_y_px: float,
    ground_y_px: float,
    meters_per_pixel: float,
    fps: float,
) -> KinematicsResult:
    vx, vy = velocity_px_per_frame
    speed_px_per_s = (vx**2 + vy**2) ** 0.5 * fps
    speed_mps = speed_px_per_s * meters_per_pixel
    speed_mph = speed_mps * MPH_PER_MPS

    height_px = max(0.0, ground_y_px - position_y_px)
    apex_height_m = height_px * meters_per_pixel
    apex_height_ft = apex_height_m * FEET_PER_METER

    return KinematicsResult(
        speed_mps=speed_mps,
        speed_mph=speed_mph,
        apex_height_m=apex_height_m,
        apex_height_ft=apex_height_ft,
    )
