"""
AirOS++ Spatial Gesture Interpreter
Translates tracked hand detections and spatial features into high-level system actions.
"""

from dataclasses import dataclass
from enum import Enum, auto
import time
from typing import List, Optional, Tuple

from airos.algorithms.icv import IntentBasedClickVerification
from airos.config.settings import GestureConfig
from airos.detector.hand_detector import HandDetection
from airos.detector.spatial_features import SpatialFeatureExtractor, SpatialFeatures
from airos.logger.airos_logger import get_logger
from airos.tracking.hand_tracker import HandTracker, TrackedHand

logger = get_logger()


class GestureType(Enum):
    NONE = auto()
    CURSOR_MOVE = auto()
    LEFT_CLICK = auto()
    RIGHT_CLICK = auto()
    SCROLL_UP = auto()
    SCROLL_DOWN = auto()
    VOLUME_CHANGE = auto()
    BRIGHTNESS_CHANGE = auto()


@dataclass
class GestureAction:
    """Dataclass encapsulating an interpreted gesture action to be executed by OSController."""

    gesture_type: GestureType
    cursor_target_norm: Optional[Tuple[float, float]] = None  # Normalized (0.0 to 1.0)
    value_delta: float = 0.0  # Used for volume, brightness, or scroll steps
    confidence: float = 1.0
    description: str = "Idle"


class GestureInterpreter:
    """State machine interpreting 2D spatial features and temporal intent into system actions."""

    def __init__(self, config: GestureConfig, icv_engine: IntentBasedClickVerification):
        self.config = config
        self.icv = icv_engine
        self.prev_inter_hand_dist: Optional[float] = None
        self.prev_elevation_diff: Optional[float] = None

    def interpret(
        self,
        detections: List[HandDetection],
        tracker: HandTracker,
        frame_dim: Tuple[int, int],
    ) -> List[GestureAction]:
        """Processes detections and returns list of executable GestureAction objects."""
        actions: List[GestureAction] = []
        if not detections:
            return [GestureAction(gesture_type=GestureType.NONE, description="No hand detected")]

        spatial_feats = SpatialFeatureExtractor.extract(detections)
        w_img, h_img = frame_dim
        current_time = time.time()

        # 1. Cursor Movement Interpretation (Primary Hand)
        cursor_hand = spatial_feats.left_hand or spatial_feats.right_hand
        if cursor_hand is not None:
            cx, cy = cursor_hand.centroid
            # Apply active screen margin ratio padding
            margin = self.config.cursor_margin_ratio
            norm_x = max(0.0, min(1.0, (cx - w_img * margin) / (w_img * (1.0 - 2 * margin))))
            norm_y = max(0.0, min(1.0, (cy - h_img * margin) / (h_img * (1.0 - 2 * margin))))

            actions.append(
                GestureAction(
                    gesture_type=GestureType.CURSOR_MOVE,
                    cursor_target_norm=(norm_x, norm_y),
                    confidence=cursor_hand.confidence,
                    description=f"Cursor Move ({norm_x:.2f}, {norm_y:.2f})",
                )
            )

        # 2. Click Verification via ICV Engine (Right / Action Hand)
        action_hand = spatial_feats.right_hand or spatial_feats.left_hand
        if action_hand is not None and action_hand.track_id is not None:
            tracked_hand = tracker.get_track(action_hand.track_id)
            if self.icv.verify_click(action_hand, tracked_hand, current_time):
                # Distinguish Left vs Right click based on aspect ratio magnitude shift
                if action_hand.aspect_ratio > 1.35:
                    actions.append(
                        GestureAction(
                            gesture_type=GestureType.RIGHT_CLICK,
                            confidence=action_hand.confidence,
                            description="Intent Verified Right Click",
                        )
                    )
                else:
                    actions.append(
                        GestureAction(
                            gesture_type=GestureType.LEFT_CLICK,
                            confidence=action_hand.confidence,
                            description="Intent Verified Left Click",
                        )
                    )

        # 3. Multi-Hand Dual Spatial Gestures (Volume, Brightness, Scroll)
        if spatial_feats.both_hands_present:
            cur_dist = spatial_feats.inter_hand_distance_px
            cur_elev = spatial_feats.vertical_elevation_diff_px

            if (
                self.prev_inter_hand_dist is not None
                and cur_dist is not None
                and self.prev_elevation_diff is not None
                and cur_elev is not None
            ):
                dist_delta = cur_dist - self.prev_inter_hand_dist
                elev_delta = cur_elev - self.prev_elevation_diff

                # Volume control via horizontal/distance expansion or contraction
                if abs(dist_delta) > 12.0:
                    vol_step = (
                        dist_delta / self.config.distance_max_px
                    ) * self.config.volume_sensitivity
                    actions.append(
                        GestureAction(
                            gesture_type=GestureType.VOLUME_CHANGE,
                            value_delta=vol_step,
                            description=f"Volume Step ({vol_step:+.2f})",
                        )
                    )

                # Brightness control via vertical elevation shift
                if abs(elev_delta) > 15.0:
                    bright_step = (
                        -elev_delta / 200.0
                    ) * self.config.brightness_sensitivity  # Inverted Y
                    actions.append(
                        GestureAction(
                            gesture_type=GestureType.BRIGHTNESS_CHANGE,
                            value_delta=bright_step,
                            description=f"Brightness Step ({bright_step:+.2f})",
                        )
                    )

            self.prev_inter_hand_dist = cur_dist
            self.prev_elevation_diff = cur_elev
        else:
            self.prev_inter_hand_dist = None
            self.prev_elevation_diff = None

        return actions
