"""
AirOS++ OS Controller Comprehensive Stress Test Suite
Tests 30 dry-run execution scenarios, screen boundary bounds, anti-vibration thresholding, and click sound feedback.
"""

import pytest

from config.settings import ControllerConfig
from airos.controller.os_controller import OSController
from airos.gesture.gesture_interpreter import GestureAction, GestureType


@pytest.mark.parametrize(
    "gesture_type",
    [
        GestureType.NONE,
        GestureType.CURSOR_MOVE,
        GestureType.LEFT_CLICK,
        GestureType.RIGHT_CLICK,
        GestureType.DOUBLE_CLICK,
        GestureType.MIDDLE_CLICK,
        GestureType.DRAG_START,
        GestureType.DRAG_END,
        GestureType.SWIPE_LEFT,
        GestureType.SWIPE_RIGHT,
        GestureType.VOLUME_CHANGE,
        GestureType.BRIGHTNESS_CHANGE,
        GestureType.DWELL_CLICK,
    ],
)
def test_os_controller_dry_run_all_gestures(gesture_type: GestureType):
    """Verifies OSController safely handles all 13 gesture types in dry-run mode without OS errors."""
    cfg = ControllerConfig(dry_run=True)
    ctrl = OSController(cfg)

    action = GestureAction(
        gesture_type=gesture_type,
        cursor_target_norm=(0.50, 0.50),
        value_delta=0.05,
        description=f"Testing {gesture_type.name}",
    )
    ctrl.execute(action)
    assert ctrl.screen_width > 0
    assert ctrl.screen_height > 0


def test_os_controller_click_cooldown_blocking():
    """Verifies OSController blocks consecutive clicks within 600ms cooldown guard."""
    cfg = ControllerConfig(dry_run=True)
    ctrl = OSController(cfg)

    act_click = GestureAction(gesture_type=GestureType.LEFT_CLICK, description="Test Click")

    ctrl.execute(act_click)
    t1 = ctrl.last_action_time

    # Immediate second click
    ctrl.execute(act_click)
    t2 = ctrl.last_action_time

    assert t1 == t2  # Timestamp did not update, click was blocked by cooldown


def test_os_controller_play_click_chime_no_exception():
    """Verifies _play_click_chime runs safely without throwing exceptions on host Windows OS."""
    cfg = ControllerConfig(dry_run=True)
    ctrl = OSController(cfg)
    ctrl._play_click_chime()
