"""
Comprehensive Unit Tests for OS Controller (Dry Run Mode)
"""

import pytest

from config.settings import ControllerConfig
from airos.controller.os_controller import OSController
from airos.gesture.gesture_interpreter import GestureAction, GestureType


def test_os_controller_dry_run_mouse_movement():
    cfg = ControllerConfig(dry_run=True)
    ctrl = OSController(cfg)
    act = GestureAction(gesture_type=GestureType.CURSOR_MOVE, cursor_target_norm=(0.5, 0.5), description="Move Cursor")
    ctrl.execute(act)
    assert ctrl.config.dry_run is True


def test_os_controller_dry_run_left_click():
    cfg = ControllerConfig(dry_run=True)
    ctrl = OSController(cfg)
    act = GestureAction(gesture_type=GestureType.LEFT_CLICK, description="Left Click")
    ctrl.execute(act)
    assert ctrl.config.dry_run is True


def test_os_controller_dry_run_right_click():
    cfg = ControllerConfig(dry_run=True)
    ctrl = OSController(cfg)
    act = GestureAction(gesture_type=GestureType.RIGHT_CLICK, description="Right Click")
    ctrl.execute(act)
    assert ctrl.config.dry_run is True


def test_os_controller_dry_run_double_click():
    cfg = ControllerConfig(dry_run=True)
    ctrl = OSController(cfg)
    act = GestureAction(gesture_type=GestureType.DOUBLE_CLICK, description="Double Click")
    ctrl.execute(act)
    assert ctrl.config.dry_run is True


def test_os_controller_dry_run_middle_click():
    cfg = ControllerConfig(dry_run=True)
    ctrl = OSController(cfg)
    act = GestureAction(gesture_type=GestureType.MIDDLE_CLICK, description="Middle Click")
    ctrl.execute(act)
    assert ctrl.config.dry_run is True


def test_os_controller_dry_run_drag_start_and_end():
    cfg = ControllerConfig(dry_run=True)
    ctrl = OSController(cfg)
    act_start = GestureAction(gesture_type=GestureType.DRAG_START, description="Drag Start")
    act_end = GestureAction(gesture_type=GestureType.DRAG_END, description="Drag End")

    ctrl.execute(act_start)
    ctrl.execute(act_end)
    assert ctrl.config.dry_run is True


def test_os_controller_dry_run_swipes_and_shortcuts():
    cfg = ControllerConfig(dry_run=True)
    ctrl = OSController(cfg)

    swipes = [
        GestureAction(gesture_type=GestureType.SWIPE_LEFT, description="Swipe Left"),
        GestureAction(gesture_type=GestureType.SWIPE_RIGHT, description="Swipe Right"),
        GestureAction(gesture_type=GestureType.SWIPE_UP, description="Swipe Up"),
        GestureAction(gesture_type=GestureType.SWIPE_DOWN, description="Swipe Down"),
        GestureAction(gesture_type=GestureType.DWELL_CLICK, description="Dwell Click"),
        GestureAction(gesture_type=GestureType.MUTE_TOGGLE, description="Mute Toggle"),
    ]

    for s in swipes:
        ctrl.execute(s)
    assert ctrl.config.dry_run is True


def test_os_controller_dry_run_volume_and_brightness():
    cfg = ControllerConfig(dry_run=True)
    ctrl = OSController(cfg)

    ctrl.execute(GestureAction(gesture_type=GestureType.VOLUME_CHANGE, value_delta=0.10, description="Vol +0.10"))
    ctrl.execute(GestureAction(gesture_type=GestureType.BRIGHTNESS_CHANGE, value_delta=0.10, description="Bright +0.10"))
    assert ctrl.config.dry_run is True
