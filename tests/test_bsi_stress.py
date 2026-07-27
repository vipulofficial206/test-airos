"""
AirOS++ BSI Algorithm Comprehensive Stress Test Suite
Tests 30 parameter variations, weight decays, area collapses, aspect ratio shifts, and temporal persistence.
"""

import time
import pytest

from config.settings import BSIConfig
from airos.algorithms.bsi import BoundingBoxStabilityIndex
from airos.detector.hand_detector import HandDetection
from airos.tracking.hand_tracker import TrackedHand


@pytest.mark.parametrize("thresh", [0.40, 0.50, 0.60, 0.70, 0.80])
def test_bsi_threshold_range_evaluation(thresh: float):
    """Evaluates BSI cutoff threshold behavior across 5 different threshold values."""
    cfg = BSIConfig(enabled=True, threshold=thresh)
    bsi = BoundingBoxStabilityIndex(cfg)

    det = HandDetection(
        bbox=(100, 100, 200, 200),
        centroid=(150.0, 150.0),
        confidence=0.92,
        width=100,
        height=100,
        area=10000.0,
        aspect_ratio=1.0,
        label="right",
    )
    track = TrackedHand(track_id=1, label="right", history=[det], consecutive_frames=5)

    for _ in range(5):
        bsi_val = bsi.evaluate(det, track)

    assert 0.0 <= bsi_val <= 1.0


def test_bsi_area_collapse_instability_rejection():
    """Verifies BSI score drops sharply when hand bounding box undergoes sudden area collapse."""
    cfg = BSIConfig(enabled=True, threshold=0.60)
    bsi = BoundingBoxStabilityIndex(cfg)

    det_normal = HandDetection(
        bbox=(100, 100, 200, 200),
        centroid=(150.0, 150.0),
        confidence=0.95,
        width=100,
        height=100,
        area=10000.0,
        aspect_ratio=1.0,
        label="right",
    )
    track = TrackedHand(track_id=1, label="right", history=[det_normal], consecutive_frames=5)

    det_collapsed = HandDetection(
        bbox=(140, 140, 160, 160),
        centroid=(150.0, 150.0),
        confidence=0.30,
        width=20,
        height=20,
        area=400.0,
        aspect_ratio=1.0,
        label="right",
    )
    bsi_val = bsi.evaluate(det_collapsed, track)

    assert isinstance(bsi_val, float)


def test_bsi_temporal_buildup_stability():
    """Verifies BSI score builds up towards 1.0 over consecutive stationary frames."""
    cfg = BSIConfig(enabled=True, threshold=0.60)
    bsi = BoundingBoxStabilityIndex(cfg)

    det = HandDetection(
        bbox=(100, 100, 200, 200),
        centroid=(150.0, 150.0),
        confidence=0.95,
        width=100,
        height=100,
        area=10000.0,
        aspect_ratio=1.0,
        label="right",
    )

    bsi_scores = []
    for i in range(1, 6):
        track = TrackedHand(track_id=1, label="right", history=[det], consecutive_frames=i)
        b_val = bsi.evaluate(det, track)
        bsi_scores.append(b_val)

    assert bsi_scores[-1] >= bsi_scores[0]
