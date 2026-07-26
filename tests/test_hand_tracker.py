"""
Unit Tests for Hand Tracker & Temporal Persistence
"""

import time
import pytest

from config.settings import DetectorConfig
from airos.detector.hand_detector import HandDetection
from airos.tracking.hand_tracker import HandTracker


def test_hand_tracker_new_track_creation():
    cfg = DetectorConfig(temporal_memory_frames=5)
    tracker = HandTracker(cfg)

    det = HandDetection(
        bbox=(10, 10, 110, 110),
        centroid=(60.0, 60.0),
        confidence=0.90,
        width=100,
        height=100,
        area=10000.0,
        aspect_ratio=1.0,
        label="left",
    )

    updated_dets = tracker.update([det])
    assert len(updated_dets) == 1
    assert updated_dets[0].track_id is not None
    assert updated_dets[0].track_id in tracker.tracks


def test_hand_tracker_spatial_association_and_velocity():
    cfg = DetectorConfig(temporal_memory_frames=5)
    tracker = HandTracker(cfg)

    t0 = time.time()
    det1 = HandDetection(
        bbox=(10, 10, 110, 110),
        centroid=(60.0, 60.0),
        confidence=0.90,
        width=100,
        height=100,
        area=10000.0,
        aspect_ratio=1.0,
        timestamp=t0,
        label="left",
    )
    tracker.update([det1])

    # Frame 2: 10px shift in 0.1s -> velocity = (100 px/s, 100 px/s)
    det2 = HandDetection(
        bbox=(20, 20, 120, 120),
        centroid=(70.0, 70.0),
        confidence=0.92,
        width=100,
        height=100,
        area=10000.0,
        aspect_ratio=1.0,
        timestamp=t0 + 0.10,
        label="left",
    )
    updated_dets = tracker.update([det2])

    assert len(updated_dets) == 1
    assert updated_dets[0].track_id == det1.track_id

    track = tracker.get_track(det1.track_id)
    assert track is not None
    assert abs(track.velocity[0] - 100.0) < 1.0


def test_hand_tracker_disappearance_and_garbage_collection():
    cfg = DetectorConfig(extrapolation_max_frames=2)
    tracker = HandTracker(cfg)

    det = HandDetection(
        bbox=(10, 10, 110, 110),
        centroid=(60.0, 60.0),
        confidence=0.90,
        width=100,
        height=100,
        area=10000.0,
        aspect_ratio=1.0,
        label="left",
    )
    tracker.update([det])
    track_id = det.track_id

    # Frame drops (no detections)
    tracker.update([])
    assert track_id in tracker.tracks
    assert tracker.tracks[track_id].missing_frames == 1

    tracker.update([])
    assert track_id in tracker.tracks
    assert tracker.tracks[track_id].missing_frames == 2

    # Frame 3 missing > extrapolation_max_frames -> Track removed
    tracker.update([])
    assert track_id not in tracker.tracks
