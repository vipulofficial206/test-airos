"""
AirOS++ Algorithm 3: Intent-Based Click Verification (ICV)
Verifies intentional clicks by detecting aspect ratio shift, stillness, and BSI stability over N frames.
"""

import math
import time
from typing import Dict, Optional, Tuple

from config.settings import ICVConfig
from airos.detector.hand_detector import HandDetection
from airos.logger.airos_logger import get_logger
from airos.tracking.hand_tracker import TrackedHand

logger = get_logger()


class IntentBasedClickVerification:
    """Intent-Based Click Verification (ICV) State Machine."""

    def __init__(self, config: ICVConfig):
        self.config = config
        self.consecutive_count: int = 0
        self.last_click_time: float = 0.0
        self.baseline_ar: Optional[float] = None

    def verify_click(
        self,
        current_det: HandDetection,
        tracked_hand: Optional[TrackedHand],
        current_time: float,
    ) -> bool:
        """Evaluates whether current frame satisfies all 4 intent conditions for click execution.

        Conditions:
        1. Aspect ratio shift >= shift_threshold
        2. BSI score >= bsi_threshold
        3. Velocity <= max_velocity_threshold
        4. Maintained for N consecutive frames
        """
        if not self.config.enabled:
            return False

        # Check cooldown timer
        if (current_time - self.last_click_time) < self.config.cooldown_sec:
            return False

        if tracked_hand is None:
            self.consecutive_count = 0
            return False

        # Condition 1: Aspect Ratio Shift relative to baseline
        baseline = tracked_hand.baseline_aspect_ratio
        ar_shift = abs(current_det.aspect_ratio - baseline)
        cond_ar = ar_shift >= self.config.aspect_ratio_shift_threshold

        # Condition 2: BSI Stability Score
        cond_bsi = current_det.bsi_score >= self.config.bsi_threshold

        # Condition 3: Stillness (Low hand velocity)
        vx, vy = tracked_hand.velocity
        v_mag = math.hypot(vx, vy)
        cond_vel = v_mag <= self.config.max_velocity_threshold

        # Evaluate combined condition
        if cond_ar and cond_bsi and cond_vel:
            self.consecutive_count += 1
            logger.debug(
                f"ICV match: Frame {self.consecutive_count}/{self.config.consecutive_frames_required} | AR Shift: {ar_shift:.3f} | BSI: {current_det.bsi_score:.2f} | Vel: {v_mag:.2f}"
            )
        else:
            self.consecutive_count = max(0, self.consecutive_count - 1)

        # Condition 4: Consecutive Frame Persistence
        if self.consecutive_count >= self.config.consecutive_frames_required:
            logger.info("ICV Intent Verified! Triggering click action.")
            self.last_click_time = current_time
            self.consecutive_count = 0
            return True

        return False

    def reset(self) -> None:
        """Resets ICV state machine."""
        self.consecutive_count = 0
        self.last_click_time = 0.0
        self.baseline_ar = None
