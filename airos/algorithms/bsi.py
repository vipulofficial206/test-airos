"""
AirOS++ Algorithm 2: Bounding Box Stability Index (BSI)
Evaluates multi-factor stability metric S_t in [0.0, 1.0] to reject noisy and jittery detections.
S_t = w_c*C_t + w_d*S_disp + w_a*S_area + w_ar*S_aspect + w_p*S_pers
"""

import math
from typing import List, Optional

from airos.config.settings import BSIConfig
from airos.detector.hand_detector import HandDetection
from airos.logger.airos_logger import get_logger
from airos.tracking.hand_tracker import TrackedHand

logger = get_logger()


class BoundingBoxStabilityIndex:
    """Bounding Box Stability Index (BSI) Engine."""

    def __init__(self, config: BSIConfig):
        self.config = config

    def evaluate(
        self, current_det: HandDetection, tracked_hand: Optional[TrackedHand] = None
    ) -> float:
        """Computes BSI stability score S_t in range [0.0, 1.0]."""
        if not self.config.enabled:
            current_det.bsi_score = 1.0
            current_det.is_stable = True
            return 1.0

        w = self.config.weights
        decay = self.config.decay_params

        # 1. Detection Confidence Component (C_t)
        c_score = max(0.0, min(1.0, current_det.confidence))

        if tracked_hand is None or len(tracked_hand.history) < 2:
            # First detection has default partial stability
            s_disp = 1.0
            s_area = 1.0
            s_aspect = 1.0
            s_pers = 1.0 / float(self.config.persistence_required_frames)
        else:
            prev_det = tracked_hand.history[-2]

            # 2. Centroid Displacement Stability (S_disp)
            disp = math.hypot(
                current_det.centroid[0] - prev_det.centroid[0],
                current_det.centroid[1] - prev_det.centroid[1],
            )
            s_disp = math.exp(-decay.gamma_displacement * disp)

            # 3. Area Consistency Stability (S_area)
            if prev_det.area > 0:
                area_ratio_diff = abs(current_det.area - prev_det.area) / prev_det.area
                s_area = math.exp(-decay.gamma_area * area_ratio_diff)
            else:
                s_area = 1.0

            # 4. Aspect Ratio Consistency Stability (S_aspect)
            if prev_det.aspect_ratio > 0:
                aspect_ratio_diff = (
                    abs(current_det.aspect_ratio - prev_det.aspect_ratio) / prev_det.aspect_ratio
                )
                s_aspect = math.exp(-decay.gamma_aspect * aspect_ratio_diff)
            else:
                s_aspect = 1.0

            # 5. Temporal Persistence Stability (S_pers)
            s_pers = min(
                1.0,
                float(tracked_hand.consecutive_frames)
                / float(self.config.persistence_required_frames),
            )

        # Weighted Sum
        s_t = (
            w.confidence * c_score
            + w.displacement * s_disp
            + w.area_consistency * s_area
            + w.aspect_ratio_consistency * s_aspect
            + w.temporal_persistence * s_pers
        )
        s_t = max(0.0, min(1.0, s_t))

        current_det.bsi_score = s_t
        current_det.is_stable = s_t >= self.config.threshold
        return s_t
