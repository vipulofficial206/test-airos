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

    icv.enabled = False
    icv.verify_click = lambda det, track, t: True

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
        aspect_ratio=1.40,
        bsi_score=0.85,
        track_id=1,
    )
    tracker.update([det_fist])
    track = tracker.get_track(det_fist.track_id)
    if track:
        track.baseline_aspect_ratio = 1.0

    interpreter.fist_hold_start_time = t0 - 0.70
    actions = interpreter.interpret([det_fist], tracker, (640, 480))

    drag_actions = [a for a in actions if a.gesture_type == GestureType.DRAG_START]
    assert len(drag_actions) == 1
    assert interpreter.is_dragging is True

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
        aspect_ratio=0.30,
        bsi_score=0.85,
        track_id=1,
    )
    tracker.update([det_narrow])
    icv.verify_click = lambda det, track, t: True

    actions = interpreter.interpret([det_narrow], tracker, (640, 480))
    middle_clicks = [a for a in actions if a.gesture_type == GestureType.MIDDLE_CLICK]
    assert len(middle_clicks) == 1


def test_gesture_interpreter_dwell_click():
    gst_cfg = GestureConfig()
    icv = IntentBasedClickVerification(ICVConfig(enabled=False))
    interpreter = GestureInterpreter(gst_cfg, icv)
    tracker = HandTracker(DetectorConfig())

    det = HandDetection(bbox=(10, 10, 110, 110), centroid=(60.0, 60.0), confidence=0.90, width=100, height=100, area=10000.0, aspect_ratio=1.0)

    # Establish dwell anchor
    t0 = time.time()
    interpreter.interpret([det], tracker, (640, 480))

    # Hold stationary for > 1.0s
    interpreter.dwell_start_time = t0 - 1.1
    actions = interpreter.interpret([det], tracker, (640, 480))

    dwell_actions = [a for a in actions if a.gesture_type == GestureType.DWELL_CLICK]
    assert len(dwell_actions) == 1


def test_gesture_interpreter_swipe_right():
    gst_cfg = GestureConfig()
    icv = IntentBasedClickVerification(ICVConfig(enabled=False))
    interpreter = GestureInterpreter(gst_cfg, icv)
    tracker = HandTracker(DetectorConfig())

    t0 = time.time()
    # Fast horizontal trajectory shift from left (0.1) to right (0.9) in 0.1s
    interpreter.pos_history = [
        (0.1, 0.5, t0),
        (0.3, 0.5, t0 + 0.03),
        (0.6, 0.5, t0 + 0.06),
        (0.9, 0.5, t0 + 0.10),
        (0.95, 0.5, t0 + 0.11),
    ]

    det = HandDetection(bbox=(550, 200, 600, 250), centroid=(575.0, 225.0), confidence=0.90, width=50, height=50, area=2500.0, aspect_ratio=1.0)
    actions = interpreter.interpret([det], tracker, (640, 480))

    swipe_actions = [a for a in actions if a.gesture_type == GestureType.SWIPE_RIGHT]
    assert len(swipe_actions) == 1


def test_gesture_interpreter_swipe_left():
    gst_cfg = GestureConfig()
    icv = IntentBasedClickVerification(ICVConfig(enabled=False))
    interpreter = GestureInterpreter(gst_cfg, icv)
    tracker = HandTracker(DetectorConfig())

    t0 = time.time()
    interpreter.pos_history = [
        (0.9, 0.5, t0),
        (0.6, 0.5, t0 + 0.03),
        (0.3, 0.5, t0 + 0.06),
        (0.1, 0.5, t0 + 0.10),
        (0.05, 0.5, t0 + 0.11),
    ]

    det = HandDetection(bbox=(10, 200, 60, 250), centroid=(35.0, 225.0), confidence=0.90, width=50, height=50, area=2500.0, aspect_ratio=1.0)
    actions = interpreter.interpret([det], tracker, (640, 480))

    swipe_actions = [a for a in actions if a.gesture_type == GestureType.SWIPE_LEFT]
    assert len(swipe_actions) == 1


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

    left_h = HandDetection(bbox=(50, 100, 150, 200), centroid=(100.0, 150.0), confidence=0.9, width=100, height=100, area=10000.0, aspect_ratio=1.0, label="left")
    right_h1 = HandDetection(bbox=(250, 100, 350, 200), centroid=(300.0, 150.0), confidence=0.9, width=100, height=100, area=10000.0, aspect_ratio=1.0, label="right")

    interpreter.interpret([left_h, right_h1], tracker, (640, 480))

    right_h2 = HandDetection(bbox=(400, 100, 500, 200), centroid=(450.0, 150.0), confidence=0.9, width=100, height=100, area=10000.0, aspect_ratio=1.0, label="right")
    actions = interpreter.interpret([left_h, right_h2], tracker, (640, 480))

    vol_actions = [a for a in actions if a.gesture_type == GestureType.VOLUME_CHANGE]
    assert len(vol_actions) == 1
    assert vol_actions[0].value_delta > 0.0
