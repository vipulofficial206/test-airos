"""
Comprehensive Unit Tests for Algorithm 3: Intent-Based Click Verification (ICV)
"""

import time
import pytest

from config.settings import ICVConfig
from airos.algorithms.icv import IntentBasedClickVerification
from airos.detector.hand_detector import HandDetection
from airos.tracking.hand_tracker import TrackedHand


def test_icv_click_verification_pipeline():
    cfg = ICVConfig(
        enabled=True,
        aspect_ratio_shift_threshold=0.15,
        bsi_threshold=0.60,
        max_velocity_threshold=5.0,
        consecutive_frames_required=3,
        cooldown_sec=0.2,
    )
    icv = IntentBasedClickVerification(cfg)

    det = HandDetection(
        bbox=(10, 10, 110, 110),
        centroid=(60.0, 60.0),
        confidence=0.90,
        width=100,
        height=100,
        area=10000.0,
        aspect_ratio=1.40,
        bsi_score=0.85,
    )
    track = TrackedHand(track_id=1, label="right", baseline_aspect_ratio=1.0, velocity=(0.0, 0.0), consecutive_frames=5)

    t0 = time.time()
    click1 = icv.verify_click(det, track, t0)
    assert click1 is False
    assert icv.consecutive_count == 1

    click2 = icv.verify_click(det, track, t0 + 0.03)
    assert click2 is False
    assert icv.consecutive_count == 2

    click3 = icv.verify_click(det, track, t0 + 0.06)
    assert click3 is True
    assert icv.consecutive_count == 0


def test_icv_accidental_click_rejection_due_to_high_velocity():
    cfg = ICVConfig(enabled=True, aspect_ratio_shift_threshold=0.15, bsi_threshold=0.60, max_velocity_threshold=5.0, consecutive_frames_required=3)
    icv = IntentBasedClickVerification(cfg)

    det = HandDetection(
        bbox=(10, 10, 110, 110),
        centroid=(60.0, 60.0),
        confidence=0.90,
        width=100,
        height=100,
        area=10000.0,
        aspect_ratio=1.40,
        bsi_score=0.85,
    )
    track = TrackedHand(track_id=1, label="right", baseline_aspect_ratio=1.0, velocity=(50.0, 50.0))

    t0 = time.time()
    click = icv.verify_click(det, track, t0)
    assert click is False
    assert icv.consecutive_count == 0


def test_icv_cooldown_blocking():
    cfg = ICVConfig(enabled=True, consecutive_frames_required=1, cooldown_sec=0.5)
    icv = IntentBasedClickVerification(cfg)

    det = HandDetection(bbox=(10, 10, 110, 110), centroid=(60.0, 60.0), confidence=0.90, width=100, height=100, area=10000.0, aspect_ratio=1.40, bsi_score=0.85)
    track = TrackedHand(track_id=1, label="right", baseline_aspect_ratio=1.0, velocity=(0.0, 0.0))

    t0 = time.time()
    click1 = icv.verify_click(det, track, t0)
    assert click1 is True

    click2 = icv.verify_click(det, track, t0 + 0.1)
    assert click2 is False


def test_icv_consecutive_decay_on_condition_loss():
    cfg = ICVConfig(enabled=True, consecutive_frames_required=3)
    icv = IntentBasedClickVerification(cfg)

    det_valid = HandDetection(bbox=(10, 10, 110, 110), centroid=(60.0, 60.0), confidence=0.90, width=100, height=100, area=10000.0, aspect_ratio=1.40, bsi_score=0.85)
    det_invalid = HandDetection(bbox=(10, 10, 110, 110), centroid=(60.0, 60.0), confidence=0.90, width=100, height=100, area=10000.0, aspect_ratio=1.0, bsi_score=0.85)
    track = TrackedHand(track_id=1, label="right", baseline_aspect_ratio=1.0, velocity=(0.0, 0.0))

    t0 = time.time()
    icv.verify_click(det_valid, track, t0)
    assert icv.consecutive_count == 1

    icv.verify_click(det_invalid, track, t0 + 0.03)
    assert icv.consecutive_count == 0


def test_icv_disabled():
    cfg = ICVConfig(enabled=False)
    icv = IntentBasedClickVerification(cfg)
    det = HandDetection(bbox=(10, 10, 110, 110), centroid=(60.0, 60.0), confidence=0.90, width=100, height=100, area=10000.0, aspect_ratio=1.40)
    track = TrackedHand(track_id=1, label="right")
    assert icv.verify_click(det, track, time.time()) is False


def test_icv_low_bsi_rejection():
    cfg = ICVConfig(enabled=True, bsi_threshold=0.60)
    icv = IntentBasedClickVerification(cfg)

    det_low_bsi = HandDetection(bbox=(10, 10, 110, 110), centroid=(60.0, 60.0), confidence=0.90, width=100, height=100, area=10000.0, aspect_ratio=1.40, bsi_score=0.40)
    track = TrackedHand(track_id=1, label="right", baseline_aspect_ratio=1.0)

    click = icv.verify_click(det_low_bsi, track, time.time())
    assert click is False


def test_icv_none_track_fallback():
    cfg = ICVConfig(enabled=True, consecutive_frames_required=1)
    icv = IntentBasedClickVerification(cfg)

    det = HandDetection(bbox=(10, 10, 110, 110), centroid=(60.0, 60.0), confidence=0.90, width=100, height=100, area=10000.0, aspect_ratio=1.40, bsi_score=0.80)
    click = icv.verify_click(det, tracked_hand=None, current_time=time.time())

    assert isinstance(click, bool)


def test_icv_consecutive_count_decay_floor():
    cfg = ICVConfig(enabled=True)
    icv = IntentBasedClickVerification(cfg)
    icv.consecutive_count = 0

    det_invalid = HandDetection(bbox=(10, 10, 110, 110), centroid=(60.0, 60.0), confidence=0.90, width=100, height=100, area=10000.0, aspect_ratio=1.0, bsi_score=0.85)
    track = TrackedHand(track_id=1, label="right", baseline_aspect_ratio=1.0)

    icv.verify_click(det_invalid, track, time.time())
    assert icv.consecutive_count == 0  # Should not drop below 0


def test_icv_high_consecutive_frames_requirement():
    cfg = ICVConfig(enabled=True, consecutive_frames_required=5)
    icv = IntentBasedClickVerification(cfg)

    det = HandDetection(bbox=(10, 10, 110, 110), centroid=(60.0, 60.0), confidence=0.90, width=100, height=100, area=10000.0, aspect_ratio=1.40, bsi_score=0.85)
    track = TrackedHand(track_id=1, label="right", baseline_aspect_ratio=1.0, velocity=(0.0, 0.0))

    t0 = time.time()
    for i in range(4):
        assert icv.verify_click(det, track, t0 + i * 0.03) is False

    assert icv.verify_click(det, track, t0 + 4 * 0.03) is True


def test_icv_reset():
    cfg = ICVConfig(enabled=True)
    icv = IntentBasedClickVerification(cfg)
    icv.consecutive_count = 3
    icv.last_click_time = 100.0
    icv.reset()

    assert icv.consecutive_count == 0
    assert icv.last_click_time == 0.0
