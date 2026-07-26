"""
AirOS++ Spatial Feature Extraction
Computes inter-hand distance vectors, relative spatial elevation, and aspect ratio features.
"""

from dataclasses import dataclass
import math
from typing import List, Optional, Tuple

from airos.detector.hand_detector import HandDetection


@dataclass
class SpatialFeatures:
    """Dataclass encapsulating multi-hand spatial metrics."""

    left_hand: Optional[HandDetection] = None
    right_hand: Optional[HandDetection] = None
    inter_hand_distance_px: Optional[float] = None
    vertical_elevation_diff_px: Optional[float] = None
    horizontal_offset_px: Optional[float] = None
    aspect_ratio_diff: Optional[float] = None
    both_hands_present: bool = False


class SpatialFeatureExtractor:
    """Computes inter-hand spatial relations for gesture interpretation."""

    @staticmethod
    def extract(detections: List[HandDetection]) -> SpatialFeatures:
        features = SpatialFeatures()

        for det in detections:
            if det.label == "left":
                features.left_hand = det
            elif det.label == "right":
                features.right_hand = det

        if features.left_hand is not None and features.right_hand is not None:
            features.both_hands_present = True
            lx, ly = features.left_hand.centroid
            rx, ry = features.right_hand.centroid

            dx = rx - lx
            dy = ry - ly
            features.horizontal_offset_px = dx
            features.vertical_elevation_diff_px = dy
            features.inter_hand_distance_px = math.hypot(dx, dy)
            features.aspect_ratio_diff = abs(
                features.left_hand.aspect_ratio - features.right_hand.aspect_ratio
            )

        return features
