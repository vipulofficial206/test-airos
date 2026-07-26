"""
Comprehensive Unit Tests for Zero-Landmark Finger Contour Geometry Engine
"""

import cv2
import numpy as np
import pytest

from airos.detector.finger_detector import FingerAnalysis, FingerDetector, FingerGesture
from airos.detector.hand_detector import HandDetection


def test_finger_detector_none_frame():
    det = HandDetection(bbox=(10, 10, 100, 100), centroid=(55.0, 55.0), confidence=0.90, width=90, height=90, area=8100.0, aspect_ratio=1.0)
    analysis = FingerDetector.analyze(None, det)
    assert analysis.finger_count == 0
    assert analysis.gesture == FingerGesture.UNKNOWN


def test_finger_detector_none_detection():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    analysis = FingerDetector.analyze(frame, None)
    assert analysis.finger_count == 0


def test_finger_detector_empty_frame():
    frame = np.array([], dtype=np.uint8)
    det = HandDetection(bbox=(10, 10, 100, 100), centroid=(55.0, 55.0), confidence=0.90, width=90, height=90, area=8100.0, aspect_ratio=1.0)
    analysis = FingerDetector.analyze(frame, det)
    assert analysis.finger_count == 0


def test_finger_detector_solid_fist_image():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(frame, (100, 100), 40, (255, 255, 255), -1)

    det = HandDetection(bbox=(50, 50, 150, 150), centroid=(100.0, 100.0), confidence=0.95, width=100, height=100, area=10000.0, aspect_ratio=1.0)
    analysis = FingerDetector.analyze(frame, det)

    assert isinstance(analysis, FingerAnalysis)
    assert analysis.index_tip_pt is not None
    assert analysis.index_tip_norm is not None


def test_finger_detector_index_pointing_aspect():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[50:180, 90:110] = 255

    det = HandDetection(bbox=(80, 40, 120, 190), centroid=(100.0, 115.0), confidence=0.90, width=40, height=150, area=6000.0, aspect_ratio=0.26)
    analysis = FingerDetector.analyze(frame, det)

    assert analysis.finger_count in (1, 0)
    assert analysis.index_tip_pt[0] >= 80 and analysis.index_tip_pt[0] <= 120


def test_finger_detector_normalized_coordinates_bounds():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[100:200, 200:300] = 255

    det = HandDetection(bbox=(200, 100, 300, 200), centroid=(250.0, 150.0), confidence=0.90, width=100, height=100, area=10000.0, aspect_ratio=1.0)
    analysis = FingerDetector.analyze(frame, det)

    nx, ny = analysis.index_tip_norm
    assert 0.0 <= nx <= 1.0
    assert 0.0 <= ny <= 1.0


def test_finger_detector_invalid_bbox_clamping():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    det = HandDetection(bbox=(-50, -50, 800, 800), centroid=(375.0, 375.0), confidence=0.90, width=850, height=850, area=722500.0, aspect_ratio=1.0)
    analysis = FingerDetector.analyze(frame, det)

    assert isinstance(analysis, FingerAnalysis)


def test_finger_detector_tiny_roi_rejection():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    det = HandDetection(bbox=(10, 10, 15, 15), centroid=(12.5, 12.5), confidence=0.90, width=5, height=5, area=25.0, aspect_ratio=1.0)
    analysis = FingerDetector.analyze(frame, det)

    assert analysis.finger_count == 0
