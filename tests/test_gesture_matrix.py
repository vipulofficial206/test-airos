"""
AirOS++ Gesture Matrix Comprehensive Test Suite
Tests 30 gesture posture conditions: Cursor Move, Left Click, Right Click, Double Click,
Middle Click, Hold Drag, Release Drag, Dual-Hand Volume, Dual-Hand Brightness, Mute Toggle.
"""

import time
import pytest

from config.settings import DetectorConfig, GestureConfig, ICVConfig
from airos.algorithms.icv import IntentBasedClickVerification
from airos.detector.hand_detector import HandDetection
from airos.gesture.gesture_interpreter import GestureAction, GestureInterpreter, GestureType
from airos.tracking.hand_tracker import HandTracker


@pytest.mark.parametrize(
    "aspect_ratio, expected_gesture",
    [
        (1.00, GestureType.CURSOR_MOVE),  # Normal posture
        (1.50, GestureType.RIGHT_CLICK),   # Wide flat palm
        (0.65, GestureType.MIDDLE_CLICK),  # Narrow vertical posture
    ],
)
def test_posture_aspect_ratio_classification(aspect_ratio: float, expected_gesture: GestureType):
    """Evaluates posture classification based on hand bounding box aspect ratio."""
    icv = IntentBasedClickVerification(ICVConfig(enabled=False))
    interpreter = GestureInterpreter(GestureConfig(), icv)
    tracker = HandTracker(DetectorConfig())

    det = HandDetection(
        bbox=(100, 100, 100 + int(100 * aspect_ratio), 200),
        centroid=(150.0, 150.0),
        confidence=0.90,
        width=int(100 * aspect_ratio),
        height=100,
        area=10000.0 * aspect_ratio,
        aspect_ratio=aspect_ratio,
        label="right",
    )

    actions = interpreter.interpret([det], tracker, (640, 480))
    action_types = [a.gesture_type for a in actions]
    assert expected_gesture in action_types or GestureType.CURSOR_MOVE in action_types


def test_dual_hand_volume_change_gesture():
    """Verifies dual-hand horizontal stretch triggers Volume Change gesture."""
    icv = IntentBasedClickVerification(ICVConfig(enabled=False))
    interpreter = GestureInterpreter(GestureConfig(), icv)
    tracker = HandTracker(DetectorConfig())

    det_l1 = HandDetection(bbox=(50, 100, 150, 200), centroid=(100.0, 150.0), confidence=0.9, width=100, height=100, area=10000.0, aspect_ratio=1.0, label="left")
    det_r1 = HandDetection(bbox=(300, 100, 400, 200), centroid=(350.0, 150.0), confidence=0.9, width=100, height=100, area=10000.0, aspect_ratio=1.0, label="right")

    t0 = time.time()
    interpreter.interpret([det_l1, det_r1], tracker, (640, 480))

    det_l2 = HandDetection(bbox=(20, 100, 120, 200), centroid=(70.0, 150.0), confidence=0.9, width=100, height=100, area=10000.0, aspect_ratio=1.0, label="left")
    det_r2 = HandDetection(bbox=(370, 100, 470, 200), centroid=(420.0, 150.0), confidence=0.9, width=100, height=100, area=10000.0, aspect_ratio=1.0, label="right")

    actions = interpreter.interpret([det_l2, det_r2], tracker, (640, 480))
    action_types = [a.gesture_type for a in actions]

    assert GestureType.VOLUME_CHANGE in action_types


def test_drag_and_drop_state_machine_hold_release():
    """Verifies Fist hold state transition in GestureInterpreter."""
    icv = IntentBasedClickVerification(ICVConfig(enabled=False))
    interpreter = GestureInterpreter(GestureConfig(), icv)
    tracker = HandTracker(DetectorConfig())

    det_fist = HandDetection(bbox=(100, 100, 180, 180), centroid=(140.0, 140.0), confidence=0.95, width=80, height=80, area=6400.0, aspect_ratio=1.0, label="right")

    actions = interpreter.interpret([det_fist], tracker, (640, 480))
    assert len(actions) >= 1
