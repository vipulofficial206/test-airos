"""
Unit and Edge Case Tests for Algorithm 1: Adaptive Motion Smoothing (AMS)
"""

import math
import pytest

from airos.algorithms.ams import AdaptiveMotionSmoothing
from config.settings import AMSConfig


def test_ams_disabled():
    cfg = AMSConfig(enabled=False)
    ams = AdaptiveMotionSmoothing(cfg)
    pos = (100.0, 200.0)
    smoothed, alpha = ams.smooth(pos, timestamp=1.0)
    assert smoothed == pos
    assert alpha == 1.0


def test_ams_first_frame():
    cfg = AMSConfig(enabled=True)
    ams = AdaptiveMotionSmoothing(cfg)
    pos = (150.0, 250.0)
    smoothed, alpha = ams.smooth(pos, timestamp=1.0)
    assert smoothed == pos
    assert alpha == cfg.alpha_max


def test_ams_low_velocity_smoothing():
    cfg = AMSConfig(enabled=True, alpha_min=0.10, alpha_max=0.90, lambda_speed=0.05)
    ams = AdaptiveMotionSmoothing(cfg)

    # Initial frame
    ams.smooth((100.0, 100.0), timestamp=0.0)

    # Small displacement (low velocity)
    pos2 = (101.0, 101.0)
    smoothed2, alpha2 = ams.smooth(pos2, timestamp=1.0)

    # Low velocity should yield alpha close to alpha_min
    assert alpha2 < 0.30
    # Smoothed position should be close to previous position (high smoothing)
    assert smoothed2[0] < pos2[0]


def test_ams_high_velocity_tracking():
    cfg = AMSConfig(enabled=True, alpha_min=0.10, alpha_max=0.90, lambda_speed=0.05)
    ams = AdaptiveMotionSmoothing(cfg)

    ams.smooth((100.0, 100.0), timestamp=0.0)

    # Large displacement (high velocity)
    pos2 = (500.0, 500.0)
    smoothed2, alpha2 = ams.smooth(pos2, timestamp=0.1)

    # High velocity should yield alpha close to alpha_max
    assert alpha2 > 0.70
    assert abs(smoothed2[0] - pos2[0]) < 100.0


def test_ams_reset():
    cfg = AMSConfig(enabled=True)
    ams = AdaptiveMotionSmoothing(cfg)
    ams.smooth((100.0, 100.0), timestamp=0.0)
    ams.smooth((200.0, 200.0), timestamp=1.0)
    ams.reset()

    assert ams.prev_raw_pos is None
    assert ams.prev_smoothed_pos is None
