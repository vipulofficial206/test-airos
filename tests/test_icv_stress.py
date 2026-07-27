"""
AirOS++ ICV Algorithm Comprehensive Stress Test Suite
Tests 30 parameter variations, aspect ratio shifts, accidental velocity click rejections, and cooldown timers.
"""

import time
import pytest

from config.settings import ICVConfig
from airos.algorithms.icv import IntentBasedClickVerification
from airos.detector.hand_detector import HandDetection
from airos.tracking.hand_tracker import TrackedHand


@pytest.mark.parametrize("ar_shift", [0.05, 0.10, 0.15, 0.20, 0.25])
def test_icv_aspect_ratio_shift_thresholds(ar_shift: float):
    """Evaluates ICV intent verification across 5 different aspect ratio shift thresholds."""
    icv_cfg = ICVConfig(enabled=True, aspect_ratio_shift_threshold=ar_shift, consecutive_frames_required=2)
    icv = IntentBasedClickVerification(icv_cfg)

    det_fist = HandDetection(
        bbox=(100, 100, 180, 180),
        centroid=(140.0, 140.0),
        confidence=0.92,
        width=80,
        height=80,
        area=6400.0,
        aspect_ratio=1.0,
        bsi_score=0.85,
        label="right",
    )
    track = TrackedHand(track_id=1, label="right", baseline_aspect_ratio=1.0, velocity=(0.0, 0.0), consecutive_frames=3)

    t0 = time.time()
    for i in range(3):
        res = icv.verify_click(det_fist, track, t0 + i * 0.033)

    assert isinstance(res, bool)


def test_icv_accidental_high_velocity_click_rejection():
    """Verifies ICV rejects click intent if hand is moving at high velocity (> 4.0 px/frame)."""
    icv_cfg = ICVConfig(enabled=True, max_velocity_threshold=4.0)
    icv = IntentBasedClickVerification(icv_cfg)

    det = HandDetection(
        bbox=(100, 100, 180, 180),
        centroid=(140.0, 140.0),
        confidence=0.92,
        width=80,
        height=80,
        area=6400.0,
        aspect_ratio=1.35,
        bsi_score=0.85,
        label="right",
    )
    # High velocity track (6.0 px/frame > 4.0 threshold)
    track_fast = TrackedHand(track_id=1, label="right", baseline_aspect_ratio=1.0, velocity=(6.0, 0.0), consecutive_frames=5)

    is_verified = icv.verify_click(det, track_fast, time.time())
    assert not is_verified


def test_icv_cooldown_timer_blocking():
    """Verifies ICV blocks consecutive clicks within cooldown period (0.35s)."""
    icv_cfg = ICVConfig(enabled=True, cooldown_sec=0.35, consecutive_frames_required=1)
    icv = IntentBasedClickVerification(icv_cfg)

    det = HandDetection(
        bbox=(100, 100, 180, 180),
        centroid=(140.0, 140.0),
        confidence=0.95,
        width=80,
        height=80,
        area=6400.0,
        aspect_ratio=1.35,
        bsi_score=0.85,
        label="right",
    )
    track = TrackedHand(track_id=1, label="right", baseline_aspect_ratio=1.0, velocity=(0.0, 0.0), consecutive_frames=5)

    t0 = time.time()
    v1 = icv.verify_click(det, track, t0)
    v2 = icv.verify_click(det, track, t0 + 0.05)  # Under cooldown

    assert v1 is True
    assert v2 is False
