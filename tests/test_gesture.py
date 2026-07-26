"""
Unit Tests for Gesture Interpreter
"""

import pytest

from airos.algorithms.icv import IntentBasedClickVerification
from airos.config.settings import GestureConfig, ICVConfig
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
    # Centroid at middle of frame (320, 240) -> normalized (0.5, 0.5)
    nx, ny = actions[0].cursor_target_norm
    assert abs(nx - 0.50) < 0.05
    assert abs(ny - 0.50) < 0.05
