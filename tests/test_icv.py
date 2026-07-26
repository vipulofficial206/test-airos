"""
Unit Tests for Algorithm 3: Intent-Based Click Verification (ICV)
"""

import time
import pytest

from airos.algorithms.icv import IntentBasedClickVerification
from airos.config.settings import ICVConfig
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

    # Baseline hand aspect ratio = 1.0
    det = HandDetection(
        bbox=(10, 10, 110, 110),
        centroid=(60.0, 60.0),
        confidence=0.90,
        width=100,
        height=100,
        area=10000.0,
        aspect_ratio=1.40,  # AR shift = 0.40 > 0.15
        bsi_score=0.85,
    )

    track = TrackedHand(
        track_id=1,
        label="right",
        baseline_aspect_ratio=1.0,
        velocity=(0.0, 0.0),
        consecutive_frames=5,
    )

    t0 = time.time()

    # Frame 1: Match 1
    click1 = icv.verify_click(det, track, t0)
    assert click1 is False
    assert icv.consecutive_count == 1

    # Frame 2: Match 2
    click2 = icv.verify_click(det, track, t0 + 0.03)
    assert click2 is False
    assert icv.consecutive_count == 2

    # Frame 3: Match 3 -> Should trigger Click!
    click3 = icv.verify_click(det, track, t0 + 0.06)
    assert click3 is True
    assert icv.consecutive_count == 0


def test_icv_accidental_click_rejection_due_to_high_velocity():
    cfg = ICVConfig(
        enabled=True,
        aspect_ratio_shift_threshold=0.15,
        bsi_threshold=0.60,
        max_velocity_threshold=5.0,
        consecutive_frames_required=3,
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

    # High velocity hand (moving fast -> accidental trigger)
    track = TrackedHand(
        track_id=1,
        label="right",
        baseline_aspect_ratio=1.0,
        velocity=(50.0, 50.0),  # v_mag ~ 70.7 > 5.0
    )

    t0 = time.time()
    click = icv.verify_click(det, track, t0)
    assert click is False
    assert icv.consecutive_count == 0
