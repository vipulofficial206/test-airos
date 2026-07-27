"""
AirOS++ Intuitive Spatial & Finger Gesture Interpreter
Translates tracked 2D hand posture dynamics, finger contour geometry, and spatial trajectories into intuitive operating system actions.
"""

from dataclasses import dataclass
from enum import Enum, auto
import time
from typing import List, Optional, Tuple

from config.settings import GestureConfig
from airos.algorithms.icv import IntentBasedClickVerification
from airos.detector.finger_detector import FingerDetector, FingerGesture
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
    DWELL_CLICK = auto()
    SCROLL_UP = auto()
    SCROLL_DOWN = auto()
    VOLUME_CHANGE = auto()
    BRIGHTNESS_CHANGE = auto()
    ZOOM_IN = auto()
    ZOOM_OUT = auto()
    SWIPE_LEFT = auto()
    SWIPE_RIGHT = auto()
    SWIPE_UP = auto()
    SWIPE_DOWN = auto()
    MUTE_TOGGLE = auto()
    SCREENSHOT = auto()


@dataclass
class GestureAction:
    """Dataclass encapsulating an interpreted gesture action to be executed by OSController."""

    gesture_type: GestureType
    cursor_target_norm: Optional[Tuple[float, float]] = None  # Normalized (0.0 to 1.0)
    value_delta: float = 0.0  # Used for volume, brightness, or scroll steps
    confidence: float = 1.0
    description: str = "Idle"


class GestureInterpreter:
    """Intuitive State Machine mapping finger postures & spatial trajectories to OS actions."""

    def __init__(self, config: GestureConfig, icv_engine: IntentBasedClickVerification):
        self.config = config
        self.icv = icv_engine

        # State tracking for advanced gestures
        self.last_left_click_time: float = 0.0
        self.is_dragging: bool = False
        self.fist_hold_start_time: Optional[float] = None
        self.prev_inter_hand_dist: Optional[float] = None
        self.prev_elevation_diff: Optional[float] = None
        self.last_app_switch_time: float = 0.0

        # Dwell Click State Tracking
        self.dwell_start_time: Optional[float] = None
        self.dwell_anchor_pos: Optional[Tuple[float, float]] = None

        # Swipe Motion History Tracking
        self.pos_history: List[Tuple[float, float, float]] = []  # (x, y, timestamp)

    def interpret(
        self,
        detections: List[HandDetection],
        tracker: HandTracker,
        frame_dim: Tuple[int, int],
        frame: Optional[object] = None,
    ) -> List[GestureAction]:
        actions: List[GestureAction] = []
        if not detections:
            if self.is_dragging:
                self.is_dragging = False
                self.fist_hold_start_time = None
                actions.append(
                    GestureAction(
                        gesture_type=GestureType.DRAG_END,
                        description="Hand Released -> Drag & Drop Released",
                    )
                )
            self.dwell_start_time = None
            self.dwell_anchor_pos = None
            return actions if actions else [GestureAction(gesture_type=GestureType.NONE, description="No hand detected")]

        spatial_feats = SpatialFeatureExtractor.extract(detections)
        w_img, h_img = frame_dim
        current_time = time.time()

        # 1. Primary Navigation Hand Cursor Positioning
        cursor_hand = detections[0]
        for det in detections:
            if self.config.cursor_hand == det.label:
                cursor_hand = det
                break

        # Analyze finger geometry if frame is available
        finger_analysis = None
        if frame is not None:
            finger_analysis = FingerDetector.analyze(frame, cursor_hand)

        if finger_analysis and finger_analysis.index_tip_pt:
            cx, cy = finger_analysis.index_tip_pt
        else:
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

        # 2. Hover Dwell Click Detection
        if self.dwell_anchor_pos is None:
            self.dwell_anchor_pos = (norm_x, norm_y)
            self.dwell_start_time = current_time
        else:
            dist = ((norm_x - self.dwell_anchor_pos[0]) ** 2 + (norm_y - self.dwell_anchor_pos[1]) ** 2) ** 0.5
            if dist < 0.025:  # Held stationary within 2.5% screen radius
                if self.dwell_start_time and (current_time - self.dwell_start_time) >= 1.0:
                    actions.append(
                        GestureAction(
                            gesture_type=GestureType.DWELL_CLICK,
                            confidence=0.95,
                            description="Hover Dwell Click Triggered (1.0s)",
                        )
                    )
                    self.dwell_start_time = current_time + 0.5  # Reset dwell window
            else:
                self.dwell_anchor_pos = (norm_x, norm_y)
                self.dwell_start_time = current_time

        # 3. Action Hand Intuitive Gestures (Click, Double Click, Right Click, Middle Click, Drag)
        action_hand = spatial_feats.right_hand if self.config.action_hand == "right" else spatial_feats.left_hand
        if action_hand is None and len(detections) >= 1:
            action_hand = detections[0]

        if action_hand is not None and action_hand.track_id is not None:
            tracked_hand = tracker.get_track(action_hand.track_id)

            # Check ICV Intent Verification for Explicit Gestures
            if self.icv.verify_click(action_hand, tracked_hand, current_time):
                is_two_v = finger_analysis and finger_analysis.gesture == FingerGesture.TWO_FINGERS_V
                is_three = (finger_analysis and finger_analysis.gesture == FingerGesture.THREE_FINGERS) or (
                    finger_analysis is None and action_hand.aspect_ratio < 0.40
                )
                
                if is_two_v:
                    actions.append(
                        GestureAction(
                            gesture_type=GestureType.RIGHT_CLICK,
                            confidence=action_hand.confidence,
                            description="2 Fingers (V-Sign) -> Right Click",
                        )
                    )
                elif is_three:
                    actions.append(
                        GestureAction(
                            gesture_type=GestureType.MIDDLE_CLICK,
                            confidence=action_hand.confidence,
                            description="3 Fingers -> Middle Click",
                        )
                    )
                else:
                    if (current_time - self.last_left_click_time) < 0.45:
                        actions.append(
                            GestureAction(
                                gesture_type=GestureType.DOUBLE_CLICK,
                                confidence=action_hand.confidence,
                                description="Double Tap -> Double Click",
                            )
                        )
                        self.last_left_click_time = 0.0
                    else:
                        actions.append(
                            GestureAction(
                                gesture_type=GestureType.LEFT_CLICK,
                                confidence=action_hand.confidence,
                                description="Index Pinch / Tap -> Left Click",
                            )
                        )
                        self.last_left_click_time = current_time

            # Drag & Drop State Machine: Sustained Closed Fist
            ar_shift = abs(action_hand.aspect_ratio - (tracked_hand.baseline_aspect_ratio if tracked_hand else 1.0))
            is_fist = (finger_analysis and finger_analysis.gesture == FingerGesture.FIST_CLOSED) or (
                ar_shift >= self.icv.config.aspect_ratio_shift_threshold and action_hand.bsi_score >= 0.60
            )

            if is_fist:
                if self.fist_hold_start_time is None:
                    self.fist_hold_start_time = current_time
                elif (current_time - self.fist_hold_start_time) >= 0.65 and not self.is_dragging:
                    self.is_dragging = True
                    actions.append(
                        GestureAction(
                            gesture_type=GestureType.DRAG_START,
                            description="Sustained Closed Fist -> Drag & Drop Hold Started",
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

        # 4. Trajectory Swipe Gestures (Left / Right / Up / Down)
        self.pos_history.append((norm_x, norm_y, current_time))
        if len(self.pos_history) > 10:
            self.pos_history.pop(0)

        if len(self.pos_history) >= 5 and (current_time - self.last_app_switch_time) > 0.8:
            dx = self.pos_history[-1][0] - self.pos_history[0][0]
            dy = self.pos_history[-1][1] - self.pos_history[0][1]
            dt = max(0.001, self.pos_history[-1][2] - self.pos_history[0][2])
            speed = (dx**2 + dy**2) ** 0.5 / dt

            if speed > 1.8:  # Fast spatial swipe motion
                if abs(dx) > abs(dy) and abs(dx) > 0.35:
                    if dx > 0:
                        actions.append(GestureAction(gesture_type=GestureType.SWIPE_RIGHT, description="Fast Swipe Right -> Next Desktop / Forward"))
                    else:
                        actions.append(GestureAction(gesture_type=GestureType.SWIPE_LEFT, description="Fast Swipe Left -> Prev Desktop / Back"))
                    self.last_app_switch_time = current_time
                    self.pos_history.clear()
                elif abs(dy) > abs(dx) and abs(dy) > 0.35:
                    if dy < 0:
                        actions.append(GestureAction(gesture_type=GestureType.SWIPE_UP, description="Fast Swipe Up -> Task View"))
                    else:
                        actions.append(GestureAction(gesture_type=GestureType.SWIPE_DOWN, description="Fast Swipe Down -> Show Desktop"))
                    self.last_app_switch_time = current_time
                    self.pos_history.clear()

        # 5. Dual Hand Spatial Gestures (Volume, Brightness, Mute)
        if spatial_feats.both_hands_present:
            cur_dist = spatial_feats.inter_hand_distance_px
            cur_elev = spatial_feats.vertical_elevation_diff_px

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

                if abs(dist_delta) > 12.0:
                    vol_step = (dist_delta / self.config.distance_max_px) * self.config.volume_sensitivity
                    actions.append(
                        GestureAction(
                            gesture_type=GestureType.VOLUME_CHANGE,
                            value_delta=vol_step,
                            description=f"Horizontal Hand Stretch -> Volume Step ({vol_step:+.2f})",
                        )
                    )

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
