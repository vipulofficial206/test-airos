"""
AirOS++ AMS Algorithm Comprehensive Stress Test Suite
Tests 30 rigorous parameter variations, zero time deltas, high accelerations, stillness locks, and bounds clamping.
"""

import math
import time
import pytest

from config.settings import AMSConfig
from airos.algorithms.ams import AdaptiveMotionSmoothing


@pytest.mark.parametrize("alpha_min", [0.05, 0.10, 0.15, 0.20, 0.30])
@pytest.mark.parametrize("alpha_max", [0.70, 0.80, 0.85, 0.90, 0.95])
def test_ams_alpha_param_combinations(alpha_min: float, alpha_max: float):
    """Tests 25 combinations of alpha_min and alpha_max parameters."""
    cfg = AMSConfig(enabled=True, alpha_min=alpha_min, alpha_max=alpha_max)
    ams = AdaptiveMotionSmoothing(cfg)

    t0 = time.time()
    ams.smooth((100.0, 100.0), t0)
    pos, alpha = ams.smooth((200.0, 200.0), t0 + 0.033)

    assert alpha_min <= alpha <= alpha_max
    assert 100.0 <= pos[0] <= 200.0
    assert 100.0 <= pos[1] <= 200.0


def test_ams_zero_dt_protection_stress():
    """Verifies AMS protects against zero time delta (division by zero)."""
    cfg = AMSConfig(enabled=True)
    ams = AdaptiveMotionSmoothing(cfg)
    t0 = time.time()

    ams.smooth((100.0, 100.0), t0)
    pos, alpha = ams.smooth((150.0, 150.0), t0)  # Identical timestamp

    assert pos is not None
    assert alpha >= cfg.alpha_min


def test_ams_stillness_lock_deadzone_freeze():
    """Verifies AMS Stillness Lock completely freezes cursor when velocity is within deadzone."""
    cfg = AMSConfig(enabled=True, velocity_deadzone=5.0, alpha_min=0.15)
    ams = AdaptiveMotionSmoothing(cfg)

    t0 = time.time()
    ams.smooth((100.0, 100.0), t0)
    pos_frozen, alpha = ams.smooth((100.1, 100.1), t0 + 0.1)

    # Position must equal previous smoothed position
    assert pos_frozen == (100.0, 100.0)
    assert alpha == cfg.alpha_min


def test_ams_reset_clears_internal_state():
    """Verifies reset() resets all internal state variables."""
    cfg = AMSConfig(enabled=True)
    ams = AdaptiveMotionSmoothing(cfg)

    ams.smooth((100.0, 100.0), time.time())
    ams.reset()

    assert ams.prev_raw_pos is None
    assert ams.prev_smoothed_pos is None
    assert ams.prev_timestamp is None
    assert ams.prev_velocity == (0.0, 0.0)


def test_ams_extreme_acceleration_tracking():
    """Verifies AMS scales alpha up to alpha_max during sudden rapid hand acceleration."""
    cfg = AMSConfig(enabled=True, alpha_min=0.10, alpha_max=0.90, lambda_speed=0.10)
    ams = AdaptiveMotionSmoothing(cfg)

    t0 = time.time()
    ams.smooth((0.0, 0.0), t0)
    _, alpha = ams.smooth((1000.0, 1000.0), t0 + 0.01)

    assert alpha >= 0.85
