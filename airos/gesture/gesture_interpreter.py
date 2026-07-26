"""
AirOS++ Intuitive Spatial Gesture Interpreter
Translates tracked 2D hand posture dynamics and spatial trajectories into intuitive operating system actions.
"""

from dataclasses import dataclass
from enum import Enum, auto
import time
from typing import List, Optional, Tuple

from config.settings import GestureConfig
from airos.algorithms.icv import IntentBasedClickVerification
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
    DOUBLE_CLICK = auto()
    DRAG_START = auto()
    DRAG_END = auto()
    MIDDLE_CLICK = auto()
    SCROLL_UP = auto()
    SCROLL_DOWN = auto()
    VOLUME_CHANGE = auto()
    BRIGHTNESS_CHANGE = auto()
    APP_SWITCH_NEXT = auto()
    MUTE_TOGGLE = auto()


@dataclass
class GestureAction:
    """Dataclass encapsulating an interpreted gesture action to be executed by OSController."""

    gesture_type: GestureType
    cursor_target_norm: Optional[Tuple[float, float]] = None  # Normalized (0.0 to 1.0)
    value_delta: float = 0.0  # Used for volume, brightness, or scroll steps
    confidence: float = 1.0
    description: str = "Idle"


class GestureInterpreter:
    """Intuitive State Machine mapping spatial hand posture shifts to mouse and desktop actions."""

    def __init__(self, config: GestureConfig, icv_engine: IntentBasedClickVerification):
        self.config = config
        self.icv = icv_engine

        # State tracking for advanced mouse gestures
        self.last_left_click_time: float = 0.0
        self.is_dragging: bool = False
        self.fist_hold_start_time: Optional[float] = None
        self.prev_inter_hand_dist: Optional[float] = None
        self.prev_elevation_diff: Optional[float] = None
        self.last_app_switch_time: float = 0.0

    def interpret(
        self,
        detections: List[HandDetection],
        tracker: HandTracker,
        frame_dim: Tuple[int, int],
    ) -> List[GestureAction]:
        actions: List[GestureAction] = []
        if not detections:
            # If hand released during drag, trigger Drag End
            if self.is_dragging:
                self.is_dragging = False
                self.fist_hold_start_time = None
                actions.append(
                    GestureAction(
                        gesture_type=GestureType.DRAG_END,
                        description="Hand Released -> Drag & Drop Released",
                    )
                )
            return actions if actions else [GestureAction(gesture_type=GestureType.NONE, description="No hand detected")]

        spatial_feats = SpatialFeatureExtractor.extract(detections)
        w_img, h_img = frame_dim
        current_time = time.time()

        # 1. Primary Hand Cursor Navigation
        cursor_hand = spatial_feats.left_hand if self.config.cursor_hand == "left" else spatial_feats.right_hand
        if cursor_hand is None:
            cursor_hand = detections[0]  # Fallback to single visible hand

        cx, cy = cursor_hand.centroid
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

        # 2. Action Hand Intuitive Gestures (Click, Double Click, Right Click, Drag)
        action_hand = spatial_feats.right_hand if self.config.action_hand == "right" else spatial_feats.left_hand
        if action_hand is None and len(detections) >= 1:
            action_hand = detections[0]

        if action_hand is not None and action_hand.track_id is not None:
            tracked_hand = tracker.get_track(action_hand.track_id)

            # Check ICV Intent Verification
            if self.icv.verify_click(action_hand, tracked_hand, current_time):
                # Right Click: Wide Horizontal Hand Posture (Aspect Ratio > 1.35)
                if action_hand.aspect_ratio > 1.35:
                    actions.append(
                        GestureAction(
                            gesture_type=GestureType.RIGHT_CLICK,
                            confidence=action_hand.confidence,
                            description="Wide Palm -> Right Click",
                        )
                    )
                # Middle Click: Vertical Narrow Hand Posture (Aspect Ratio < 0.65)
                elif action_hand.aspect_ratio < 0.65:
                    actions.append(
                        GestureAction(
                            gesture_type=GestureType.MIDDLE_CLICK,
                            confidence=action_hand.confidence,
                            description="Vertical Narrow Hand -> Middle Click",
                        )
                    )
                else:
                    # Check for Double Click (two clicks within 0.45 seconds)
                    if (current_time - self.last_left_click_time) < 0.45:
                        actions.append(
                            GestureAction(
                                gesture_type=GestureType.DOUBLE_CLICK,
                                confidence=action_hand.confidence,
                                description="Rapid Tap -> Double Click",
                            )
                        )
                        self.last_left_click_time = 0.0
                    else:
                        actions.append(
                            GestureAction(
                                gesture_type=GestureType.LEFT_CLICK,
                                confidence=action_hand.confidence,
                                description="Index Tap / Fist -> Left Click",
                            )
                        )
                        self.last_left_click_time = current_time

            # Drag & Drop State Machine: Sustained Fist Posture
            ar_shift = abs(action_hand.aspect_ratio - (tracked_hand.baseline_aspect_ratio if tracked_hand else 1.0))
            if ar_shift >= self.icv.config.aspect_ratio_shift_threshold and action_hand.bsi_score >= 0.60:
                if self.fist_hold_start_time is None:
                    self.fist_hold_start_time = current_time
                elif (current_time - self.fist_hold_start_time) >= 0.65 and not self.is_dragging:
                    self.is_dragging = True
                    actions.append(
                        GestureAction(
                            gesture_type=GestureType.DRAG_START,
                            description="Sustained Fist -> Drag & Drop Hold Started",
                        )
                    )
            else:
                if self.is_dragging:
                    self.is_dragging = False
                    self.fist_hold_start_time = None
                    actions.append(
                        GestureAction(
                            gesture_type=GestureType.DRAG_END,
                            description="Hand Opened -> Drag & Drop Released",
                        )
                    )
                else:
                    self.fist_hold_start_time = None

        # 3. Dual Hand Gestures (Volume, Brightness, Mute, Window Switcher)
        if spatial_feats.both_hands_present:
            cur_dist = spatial_feats.inter_hand_distance_px
            cur_elev = spatial_feats.vertical_elevation_diff_px

            # Check Cross-Hands Mute Gesture (Hands crossing horizontally)
            if spatial_feats.aspect_ratio_diff and spatial_feats.aspect_ratio_diff > 0.8:
                if (current_time - self.last_app_switch_time) > 1.0:
                    actions.append(
                        GestureAction(
                            gesture_type=GestureType.MUTE_TOGGLE,
                            description="Cross Hands -> Mute Audio Toggle",
                        )
                    )
                    self.last_app_switch_time = current_time

            if (
                self.prev_inter_hand_dist is not None
                and cur_dist is not None
                and self.prev_elevation_diff is not None
                and cur_elev is not None
            ):
                dist_delta = cur_dist - self.prev_inter_hand_dist
                elev_delta = cur_elev - self.prev_elevation_diff

                # Volume Control via Horizontal Expansion / Contraction
                if abs(dist_delta) > 12.0:
                    vol_step = (dist_delta / self.config.distance_max_px) * self.config.volume_sensitivity
                    actions.append(
                        GestureAction(
                            gesture_type=GestureType.VOLUME_CHANGE,
                            value_delta=vol_step,
                            description=f"Horizontal Hand Stretch -> Volume Step ({vol_step:+.2f})",
                        )
                    )

                # Brightness Control via Vertical Elevation Differential
                if abs(elev_delta) > 15.0:
                    bright_step = (-elev_delta / 200.0) * self.config.brightness_sensitivity
                    actions.append(
                        GestureAction(
                            gesture_type=GestureType.BRIGHTNESS_CHANGE,
                            value_delta=bright_step,
                            description=f"Vertical Hand Elevation -> Brightness Step ({bright_step:+.2f})",
                        )
                    )

            self.prev_inter_hand_dist = cur_dist
            self.prev_elevation_diff = cur_elev
        else:
            self.prev_inter_hand_dist = None
            self.prev_elevation_diff = None

        return actions
