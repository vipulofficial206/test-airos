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

    det = HandDetection(bbox=(10, 10, 110, 110), centroid=(60.0, 60.0), confidence=0.90, width=100, height=100, area=10000.0, aspect_ratio=1.0, label="left")

    updated_dets = tracker.update([det])
    assert len(updated_dets) == 1
    assert updated_dets[0].track_id is not None
    assert updated_dets[0].track_id in tracker.tracks


def test_hand_tracker_spatial_association_and_velocity():
    cfg = DetectorConfig(temporal_memory_frames=5)
    tracker = HandTracker(cfg)

    t0 = time.time()
    det1 = HandDetection(bbox=(10, 10, 110, 110), centroid=(60.0, 60.0), confidence=0.90, width=100, height=100, area=10000.0, aspect_ratio=1.0, timestamp=t0, label="left")
    tracker.update([det1])

    det2 = HandDetection(bbox=(20, 20, 120, 120), centroid=(70.0, 70.0), confidence=0.92, width=100, height=100, area=10000.0, aspect_ratio=1.0, timestamp=t0 + 0.10, label="left")
    updated_dets = tracker.update([det2])

    assert len(updated_dets) == 1
    assert updated_dets[0].track_id == det1.track_id

    track = tracker.get_track(det1.track_id)
    assert track is not None
    assert abs(track.velocity[0] - 100.0) < 1.0


def test_hand_tracker_disappearance_and_garbage_collection():
    cfg = DetectorConfig(extrapolation_max_frames=2)
    tracker = HandTracker(cfg)

    det = HandDetection(bbox=(10, 10, 110, 110), centroid=(60.0, 60.0), confidence=0.90, width=100, height=100, area=10000.0, aspect_ratio=1.0, label="left")
    tracker.update([det])
    track_id = det.track_id

    tracker.update([])
    assert track_id in tracker.tracks
    assert tracker.tracks[track_id].missing_frames == 1

    tracker.update([])
    assert track_id in tracker.tracks
    assert tracker.tracks[track_id].missing_frames == 2

    tracker.update([])
    assert track_id not in tracker.tracks


def test_hand_tracker_history_buffer_limit():
    cfg = DetectorConfig(temporal_memory_frames=3)
    tracker = HandTracker(cfg)

    t0 = time.time()
    for i in range(10):
        det = HandDetection(bbox=(10 + i, 10, 110 + i, 110), centroid=(60.0 + i, 60.0), confidence=0.90, width=100, height=100, area=10000.0, aspect_ratio=1.0, timestamp=t0 + i * 0.03, label="left")
        tracker.update([det])

    track = list(tracker.tracks.values())[0]
    assert len(track.history) <= 3


def test_hand_tracker_dual_hands_association():
    cfg = DetectorConfig()
    tracker = HandTracker(cfg)

    left_det = HandDetection(bbox=(50, 50, 150, 150), centroid=(100.0, 100.0), confidence=0.90, width=100, height=100, area=10000.0, aspect_ratio=1.0, label="left")
    right_det = HandDetection(bbox=(400, 50, 500, 150), centroid=(450.0, 100.0), confidence=0.90, width=100, height=100, area=10000.0, aspect_ratio=1.0, label="right")

    updated = tracker.update([left_det, right_det])
    assert len(updated) == 2
    assert len(tracker.tracks) == 2


def test_hand_tracker_get_non_existent_track():
    cfg = DetectorConfig()
    tracker = HandTracker(cfg)
    assert tracker.get_track(999) is None
