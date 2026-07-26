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

    nx, ny = actions[0].cursor_target_norm
    assert abs(nx - 0.50) < 0.05
    assert abs(ny - 0.50) < 0.05


def test_gesture_interpreter_double_click_detection():
    gst_cfg = GestureConfig()
    icv = IntentBasedClickVerification(ICVConfig(enabled=False))
    interpreter = GestureInterpreter(gst_cfg, icv)

    # First click timestamp set
    t0 = time.time()
    interpreter.last_left_click_time = t0

    det = HandDetection(
        bbox=(10, 10, 110, 110),
        centroid=(60.0, 60.0),
        confidence=0.90,
        width=100,
        height=100,
        area=10000.0,
        aspect_ratio=1.0,
        label="right",
        track_id=1,
    )
    tracker = HandTracker(DetectorConfig())
    tracker.update([det])

    # ICV verify click mock
    icv.enabled = False
    icv.verify_click = lambda det, track, t: True

    # Second click within 0.45s -> Double Click
    actions = interpreter.interpret([det], tracker, (640, 480))
    double_clicks = [a for a in actions if a.gesture_type == GestureType.DOUBLE_CLICK]
    assert len(double_clicks) == 1


def test_gesture_interpreter_drag_and_drop_state_machine():
    gst_cfg = GestureConfig()
    icv = IntentBasedClickVerification(ICVConfig(enabled=False))
    interpreter = GestureInterpreter(gst_cfg, icv)
    tracker = HandTracker(DetectorConfig())

    t0 = time.time()
    det_fist = HandDetection(
        bbox=(10, 10, 110, 110),
        centroid=(60.0, 60.0),
        confidence=0.90,
        width=100,
        height=100,
        area=10000.0,
        aspect_ratio=1.40,  # AR shift = 0.40 >= 0.15
        bsi_score=0.85,
        track_id=1,
    )
    tracker.update([det_fist])
    track = tracker.get_track(det_fist.track_id)
    if track:
        track.baseline_aspect_ratio = 1.0

    # Hold fist for > 0.65s
    interpreter.fist_hold_start_time = t0 - 0.70
    actions = interpreter.interpret([det_fist], tracker, (640, 480))

    drag_actions = [a for a in actions if a.gesture_type == GestureType.DRAG_START]
    assert len(drag_actions) == 1
    assert interpreter.is_dragging is True

    # Release hand (no hands detected) -> Drag End
    actions_release = interpreter.interpret([], tracker, (640, 480))
    drag_end_actions = [a for a in actions_release if a.gesture_type == GestureType.DRAG_END]
    assert len(drag_end_actions) == 1
    assert interpreter.is_dragging is False


def test_gesture_interpreter_middle_click():
    gst_cfg = GestureConfig()
    icv = IntentBasedClickVerification(ICVConfig(enabled=False))
    interpreter = GestureInterpreter(gst_cfg, icv)
    tracker = HandTracker(DetectorConfig())

    det_narrow = HandDetection(
        bbox=(10, 10, 40, 110),
        centroid=(25.0, 60.0),
        confidence=0.90,
        width=30,
        height=100,
        area=3000.0,
        aspect_ratio=0.30,  # AR < 0.65 -> Middle Click
        bsi_score=0.85,
        track_id=1,
    )
    tracker.update([det_narrow])
    icv.verify_click = lambda det, track, t: True

    actions = interpreter.interpret([det_narrow], tracker, (640, 480))
    middle_clicks = [a for a in actions if a.gesture_type == GestureType.MIDDLE_CLICK]
    assert len(middle_clicks) == 1
