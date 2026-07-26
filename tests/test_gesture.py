"""
Comprehensive Unit Tests for Spatial Gesture Interpreter
"""

import time
import pytest

from config.settings import GestureConfig, ICVConfig
from airos.algorithms.icv import IntentBasedClickVerification
from airos.detector.hand_detector import HandDetection
from airos.gesture.gesture_interpreter import GestureInterpreter, GestureType
from airos.tracking.hand_tracker import HandTracker, DetectorConfig


def test_gesture_interpreter_cursor_mapping():
    gst_cfg = GestureConfig(cursor_margin_ratio=0.10)
    icv = IntentBasedClickVerification(ICVConfig(enabled=False))
    interpreter = GestureInterpreter(gst_cfg, icv)
    tracker = HandTracker(DetectorConfig())

    det = HandDetection(
        bbox=(200, 200, 300, 300),
        centroid=(320.0, 240.0),
        confidence=0.90,
        width=100,
        height=100,
        area=10000.0,
        aspect_ratio=1.0,
        label="left",
    )

    actions = interpreter.interpret([det], tracker, (640, 480))
    assert len(actions) >= 1
    assert actions[0].gesture_type == GestureType.CURSOR_MOVE
    assert actions[0].cursor_target_norm is not None

    # Centroid at frame middle (320, 240) -> normalized (0.50, 0.50)
    nx, ny = actions[0].cursor_target_norm
    assert abs(nx - 0.50) < 0.05
    assert abs(ny - 0.50) < 0.05


def test_gesture_interpreter_no_hands():
    gst_cfg = GestureConfig()
    icv = IntentBasedClickVerification(ICVConfig(enabled=False))
    interpreter = GestureInterpreter(gst_cfg, icv)
    tracker = HandTracker(DetectorConfig())

    actions = interpreter.interpret([], tracker, (640, 480))
    assert len(actions) == 1
    assert actions[0].gesture_type == GestureType.NONE


def test_gesture_interpreter_dual_hand_volume_change():
    gst_cfg = GestureConfig(volume_sensitivity=2.0)
    icv = IntentBasedClickVerification(ICVConfig(enabled=False))
    interpreter = GestureInterpreter(gst_cfg, icv)
    tracker = HandTracker(DetectorConfig())

    left_h = HandDetection(
        bbox=(50, 100, 150, 200),
        centroid=(100.0, 150.0),
        confidence=0.9,
        width=100,
        height=100,
        area=10000.0,
        aspect_ratio=1.0,
        label="left",
    )
    right_h1 = HandDetection(
        bbox=(250, 100, 350, 200),
        centroid=(300.0, 150.0),
        confidence=0.9,
        width=100,
        height=100,
        area=10000.0,
        aspect_ratio=1.0,
        label="right",
    )

    # Frame 1: Establish baseline distance = 200px
    interpreter.interpret([left_h, right_h1], tracker, (640, 480))

    # Frame 2: Right hand moves right -> distance = 350px (expansion > 12px)
    right_h2 = HandDetection(
        bbox=(400, 100, 500, 200),
        centroid=(450.0, 150.0),
        confidence=0.9,
        width=100,
        height=100,
        area=10000.0,
        aspect_ratio=1.0,
        label="right",
    )
    actions = interpreter.interpret([left_h, right_h2], tracker, (640, 480))

    vol_actions = [a for a in actions if a.gesture_type == GestureType.VOLUME_CHANGE]
    assert len(vol_actions) == 1
    assert vol_actions[0].value_delta > 0.0
