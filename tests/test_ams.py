"""
Comprehensive Unit Tests for Algorithm 1: Adaptive Motion Smoothing (AMS)
"""

import time
import pytest

from config.settings import AMSConfig
from airos.algorithms.ams import AdaptiveMotionSmoothing


def test_ams_disabled():
    cfg = AMSConfig(enabled=False)
    ams = AdaptiveMotionSmoothing(cfg)
    pos = (100.0, 100.0)
    smoothed_pos, alpha = ams.smooth(pos, time.time())
    assert smoothed_pos == pos
    assert alpha == 1.0


def test_ams_first_frame():
    cfg = AMSConfig(enabled=True)
    ams = AdaptiveMotionSmoothing(cfg)
    pos = (200.0, 150.0)
    smoothed_pos, alpha = ams.smooth(pos, time.time())
    assert smoothed_pos == pos
    assert alpha == cfg.alpha_max


def test_ams_low_velocity_smoothing():
    cfg = AMSConfig(enabled=True, alpha_min=0.15, alpha_max=0.85, velocity_deadzone=5.0)
    ams = AdaptiveMotionSmoothing(cfg)

    t0 = time.time()
    ams.smooth((100.0, 100.0), t0)

    smoothed_pos, alpha = ams.smooth((101.0, 101.0), t0 + 0.1)

    assert alpha <= 0.60
    assert smoothed_pos[0] < 101.0
    assert smoothed_pos[1] < 101.0


def test_ams_high_velocity_tracking():
    cfg = AMSConfig(enabled=True, alpha_min=0.15, alpha_max=0.85)
    ams = AdaptiveMotionSmoothing(cfg)

    t0 = time.time()
    ams.smooth((100.0, 100.0), t0)

    smoothed_pos, alpha = ams.smooth((300.0, 300.0), t0 + 0.03)

    assert alpha >= 0.70
    assert smoothed_pos[0] > 200.0


def test_ams_velocity_deadzone():
    cfg = AMSConfig(enabled=True, velocity_deadzone=10.0, alpha_min=0.15)
    ams = AdaptiveMotionSmoothing(cfg)

    t0 = time.time()
    ams.smooth((100.0, 100.0), t0)

    _, alpha = ams.smooth((100.5, 100.5), t0 + 0.1)
    assert alpha == cfg.alpha_min


def test_ams_alpha_clamping_bounds():
    cfg = AMSConfig(enabled=True, alpha_min=0.10, alpha_max=0.90)
    ams = AdaptiveMotionSmoothing(cfg)

    t0 = time.time()
    ams.smooth((0.0, 0.0), t0)

    _, alpha_high = ams.smooth((10000.0, 10000.0), t0 + 0.001)
    assert alpha_high <= 0.90

    _, alpha_low = ams.smooth((10000.0, 10000.0), t0 + 10.0)
    assert alpha_low >= 0.10


def test_ams_zero_time_delta_protection():
    cfg = AMSConfig(enabled=True)
    ams = AdaptiveMotionSmoothing(cfg)

    t0 = time.time()
    ams.smooth((100.0, 100.0), t0)
    smoothed_pos, alpha = ams.smooth((150.0, 150.0), t0)

    assert smoothed_pos is not None
    assert isinstance(alpha, float)


def test_ams_negative_time_delta_protection():
    cfg = AMSConfig(enabled=True)
    ams = AdaptiveMotionSmoothing(cfg)

    t0 = time.time()
    ams.smooth((100.0, 100.0), t0)
    smoothed_pos, alpha = ams.smooth((150.0, 150.0), t0 - 1.0)

    assert smoothed_pos is not None
    assert isinstance(alpha, float)


def test_ams_extreme_acceleration():
    cfg = AMSConfig(enabled=True)
    ams = AdaptiveMotionSmoothing(cfg)

    t0 = time.time()
    ams.smooth((10.0, 10.0), t0)
    ams.smooth((12.0, 12.0), t0 + 0.1)

    pos, alpha = ams.smooth((512.0, 512.0), t0 + 0.13)

    assert alpha > cfg.alpha_min
    assert pos[0] > 12.0


def test_ams_stationary_decay():
    cfg = AMSConfig(enabled=True, alpha_min=0.15)
    ams = AdaptiveMotionSmoothing(cfg)

    t0 = time.time()
    ams.smooth((100.0, 100.0), t0)
    ams.smooth((500.0, 500.0), t0 + 0.02)

    pos, alpha = ams.smooth((500.0, 500.0), t0 + 1.02)
    assert alpha == cfg.alpha_min
    assert isinstance(pos, tuple)


def test_ams_alpha_lower_bound_stability():
    cfg = AMSConfig(enabled=True, alpha_min=0.05)
    ams = AdaptiveMotionSmoothing(cfg)
    ams.smooth((100.0, 100.0), time.time())
    _, alpha = ams.smooth((100.01, 100.01), time.time() + 0.1)
    assert alpha >= 0.05


def test_ams_reset():
    cfg = AMSConfig(enabled=True)
    ams = AdaptiveMotionSmoothing(cfg)

    ams.smooth((100.0, 100.0), time.time())
    ams.reset()

    assert ams.prev_smoothed_pos is None
