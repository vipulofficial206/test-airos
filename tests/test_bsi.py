"""
Comprehensive Unit Tests for Algorithm 2: Bounding Box Stability Index (BSI)
"""

import time
import pytest

from config.settings import BSIConfig
from airos.algorithms.bsi import BoundingBoxStabilityIndex
from airos.detector.hand_detector import HandDetection
from airos.tracking.hand_tracker import TrackedHand


def test_bsi_disabled():
    cfg = BSIConfig(enabled=False)
    bsi = BoundingBoxStabilityIndex(cfg)

    det = HandDetection(
        bbox=(10, 10, 110, 110),
        centroid=(60.0, 60.0),
        confidence=0.9,
        width=100,
        height=100,
        area=10000.0,
        aspect_ratio=1.0,
    )
    score = bsi.evaluate(det)
    assert score == 1.0
    assert det.is_stable is True


def test_bsi_high_confidence_stable():
    cfg = BSIConfig(enabled=True, threshold=0.60)
    bsi = BoundingBoxStabilityIndex(cfg)

    det1 = HandDetection(
        bbox=(10, 10, 110, 110),
        centroid=(60.0, 60.0),
        confidence=0.95,
        width=100,
        height=100,
        area=10000.0,
        aspect_ratio=1.0,
    )

    track = TrackedHand(track_id=1, label="left", history=[det1], consecutive_frames=5)
    det2 = HandDetection(
        bbox=(11, 11, 111, 111),
        centroid=(61.0, 61.0),
        confidence=0.95,
        width=100,
        height=100,
        area=10000.0,
        aspect_ratio=1.0,
    )

    score = bsi.evaluate(det2, track)
    assert score > 0.80
    assert det2.is_stable is True


def test_bsi_unstable_jitter_box():
    cfg = BSIConfig(enabled=True, threshold=0.60)
    bsi = BoundingBoxStabilityIndex(cfg)

    det1 = HandDetection(
        bbox=(10, 10, 110, 110),
        centroid=(60.0, 60.0),
        confidence=0.30,
        width=100,
        height=100,
        area=10000.0,
        aspect_ratio=1.0,
    )
    track = TrackedHand(track_id=1, label="left", history=[det1], consecutive_frames=1)

    # Massive aspect ratio and centroid displacement jump with low confidence
    det2 = HandDetection(
        bbox=(200, 200, 450, 250),
        centroid=(325.0, 225.0),
        confidence=0.35,
        width=250,
        height=50,
        area=12500.0,
        aspect_ratio=5.0,
    )

    score = bsi.evaluate(det2, track)
    assert score < 0.60
    assert det2.is_stable is False


def test_bsi_area_collapse_instability():
    cfg = BSIConfig(enabled=True, threshold=0.60)
    bsi = BoundingBoxStabilityIndex(cfg)

    det1 = HandDetection(
        bbox=(10, 10, 110, 110),
        centroid=(60.0, 60.0),
        confidence=0.90,
        width=100,
        height=100,
        area=10000.0,
        aspect_ratio=1.0,
    )
    track = TrackedHand(track_id=1, label="left", history=[det1], consecutive_frames=5)

    # Sudden area collapse (area drops from 10000 to 1000)
    det2 = HandDetection(
        bbox=(50, 50, 80, 80),
        centroid=(65.0, 65.0),
        confidence=0.70,
        width=30,
        height=30,
        area=900.0,
        aspect_ratio=1.0,
    )

    score = bsi.evaluate(det2, track)
    # Area stability component drops significantly
    assert score < 0.65


def test_bsi_temporal_persistence_buildup():
    cfg = BSIConfig(enabled=True, threshold=0.60, persistence_required_frames=5)
    bsi = BoundingBoxStabilityIndex(cfg)

    det = HandDetection(
        bbox=(10, 10, 110, 110),
        centroid=(60.0, 60.0),
        confidence=0.75,
        width=100,
        height=100,
        area=10000.0,
        aspect_ratio=1.0,
    )

    track1 = TrackedHand(track_id=1, label="left", history=[det], consecutive_frames=1)
    score1 = bsi.evaluate(det, track1)

    track5 = TrackedHand(track_id=1, label="left", history=[det], consecutive_frames=5)
    score5 = bsi.evaluate(det, track5)

    assert score5 > score1
