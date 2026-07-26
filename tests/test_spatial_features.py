"""
Unit Tests for Spatial Feature Extraction & Hand Pair Relations
"""

import pytest

from airos.detector.hand_detector import HandDetection
from airos.detector.spatial_features import SpatialFeatureExtractor


def test_spatial_feature_extraction_dual_hands():
    left_hand = HandDetection(
        bbox=(100, 100, 200, 200),
        centroid=(150.0, 150.0),
        confidence=0.90,
        width=100,
        height=100,
        area=10000.0,
        aspect_ratio=1.0,
        label="left",
    )
    right_hand = HandDetection(
        bbox=(400, 100, 500, 200),
        centroid=(450.0, 150.0),
        confidence=0.88,
        width=100,
        height=100,
        area=10000.0,
        aspect_ratio=1.2,
        label="right",
    )

    feats = SpatialFeatureExtractor.extract([left_hand, right_hand])

    assert feats.both_hands_present is True
    assert feats.left_hand == left_hand
    assert feats.right_hand == right_hand
    assert feats.inter_hand_distance_px == 300.0
    assert feats.vertical_elevation_diff_px == 0.0
    assert abs(feats.aspect_ratio_diff - 0.2) < 0.001


def test_spatial_feature_extraction_single_hand():
    left_hand = HandDetection(
        bbox=(100, 100, 200, 200),
        centroid=(150.0, 150.0),
        confidence=0.90,
        width=100,
        height=100,
        area=10000.0,
        aspect_ratio=1.0,
        label="left",
    )

    feats = SpatialFeatureExtractor.extract([left_hand])

    assert feats.both_hands_present is False
    assert feats.left_hand == left_hand
    assert feats.right_hand is None
    assert feats.inter_hand_distance_px is None
