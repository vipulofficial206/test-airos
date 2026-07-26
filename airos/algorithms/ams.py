"""
AirOS++ Algorithm 1: Adaptive Motion Smoothing (AMS)
Calculates dynamic exponential smoothing factor alpha based on velocity and acceleration.
High velocity -> High alpha (Low smoothing, low latency)
Low velocity  -> Low alpha (High smoothing, jitter cancellation)
"""

import math
from typing import Optional, Tuple

from airos.config.settings import AMSConfig
from airos.logger.airos_logger import get_logger

logger = get_logger()


class AdaptiveMotionSmoothing:
    """Adaptive Motion Smoothing (AMS) Engine."""

    def __init__(self, config: AMSConfig):
        self.config = config
        self.prev_raw_pos: Optional[Tuple[float, float]] = None
        self.prev_smoothed_pos: Optional[Tuple[float, float]] = None
        self.prev_velocity: Tuple[float, float] = (0.0, 0.0)
        self.prev_timestamp: Optional[float] = None
        self.current_alpha: float = config.alpha_min

    def smooth(
        self, pos: Tuple[float, float], timestamp: float
    ) -> Tuple[Tuple[float, float], float]:
        """Applies AMS to raw 2D coordinate input.

        Args:
            pos: Raw (x, y) coordinates.
            timestamp: Current frame timestamp.

        Returns:
            Tuple of ((smoothed_x, smoothed_y), alpha_t)
        """
        if not self.config.enabled:
            return pos, 1.0

        if self.prev_smoothed_pos is None or self.prev_timestamp is None:
            self.prev_raw_pos = pos
            self.prev_smoothed_pos = pos
            self.prev_timestamp = timestamp
            return pos, self.config.alpha_max

        dt = max(0.001, timestamp - self.prev_timestamp)

        # 1. Compute Raw Velocity Vector & Magnitude
        vx = (pos[0] - self.prev_raw_pos[0]) / dt
        vy = (pos[1] - self.prev_raw_pos[1]) / dt
        v_mag = math.hypot(vx, vy)

        # 2. Compute Acceleration Vector & Magnitude
        ax = (vx - self.prev_velocity[0]) / dt
        ay = (vy - self.prev_velocity[1]) / dt
        a_mag = math.hypot(ax, ay)

        # 3. Apply Deadzone Filtering
        if v_mag < self.config.velocity_deadzone:
            v_mag = 0.0

        # 4. Compute Dynamic Alpha Scaling via Exponential Response Formula
        speed_factor = 1.0 - math.exp(-self.config.lambda_speed * v_mag)
        alpha = self.config.alpha_min + speed_factor * (
            self.config.alpha_max - self.config.alpha_min
        )
        alpha = max(self.config.alpha_min, min(self.config.alpha_max, alpha))

        # 5. Apply Exponential Smoothing
        sx = alpha * pos[0] + (1.0 - alpha) * self.prev_smoothed_pos[0]
        sy = alpha * pos[1] + (1.0 - alpha) * self.prev_smoothed_pos[1]
        smoothed_pos = (sx, sy)

        # Update State Variables
        self.prev_raw_pos = pos
        self.prev_smoothed_pos = smoothed_pos
        self.prev_velocity = (vx, vy)
        self.prev_timestamp = timestamp
        self.current_alpha = alpha

        return smoothed_pos, alpha

    def reset(self) -> None:
        """Resets internal state of the smoothing filter."""
        self.prev_raw_pos = None
        self.prev_smoothed_pos = None
        self.prev_velocity = (0.0, 0.0)
        self.prev_timestamp = None
        self.current_alpha = self.config.alpha_min
