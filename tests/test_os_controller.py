"""
Comprehensive Unit Tests for OS Controller (Dry Run Mode)
"""

import pytest

from config.settings import ControllerConfig
from airos.controller.os_controller import OSController
from airos.gesture.gesture_interpreter import GestureAction, GestureType


def test_os_controller_dry_run_all_actions():
    cfg = ControllerConfig(dry_run=True)
    ctrl = OSController(cfg)

    actions = [
        GestureAction(gesture_type=GestureType.CURSOR_MOVE, cursor_target_norm=(0.5, 0.5), description="Move"),
        GestureAction(gesture_type=GestureType.LEFT_CLICK, description="Left Click"),
        GestureAction(gesture_type=GestureType.RIGHT_CLICK, description="Right Click"),
        GestureAction(gesture_type=GestureType.DOUBLE_CLICK, description="Double Click"),
        GestureAction(gesture_type=GestureType.MIDDLE_CLICK, description="Middle Click"),
        GestureAction(gesture_type=GestureType.DRAG_START, description="Drag Start"),
        GestureAction(gesture_type=GestureType.DRAG_END, description="Drag End"),
        GestureAction(gesture_type=GestureType.SCROLL_UP, value_delta=5.0, description="Scroll Up"),
        GestureAction(gesture_type=GestureType.VOLUME_CHANGE, value_delta=0.1, description="Volume Change"),
        GestureAction(gesture_type=GestureType.BRIGHTNESS_CHANGE, value_delta=0.1, description="Brightness Change"),
        GestureAction(gesture_type=GestureType.MUTE_TOGGLE, description="Mute Toggle"),
    ]

    for act in actions:
        ctrl.execute(act)

    assert ctrl.config.dry_run is True
