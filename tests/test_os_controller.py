"""
Unit Tests for OS Controller (Dry Run Mode)
"""

import pytest

from airos.config.settings import ControllerConfig
from airos.controller.os_controller import OSController
from airos.gesture.gesture_interpreter import GestureAction, GestureType


def test_os_controller_dry_run():
    cfg = ControllerConfig(dry_run=True)
    ctrl = OSController(cfg)

    action = GestureAction(
        gesture_type=GestureType.CURSOR_MOVE,
        cursor_target_norm=(0.5, 0.5),
        description="Dry Run Move",
    )

    # Should execute safely without raising exceptions or mutating host OS cursor
    ctrl.execute(action)
    assert ctrl.config.dry_run is True
