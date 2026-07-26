"""
Unit Tests for Hand Detector & Synthetic Engine
"""

import numpy as np
import pytest

from airos.config.settings import DetectorConfig, ModelConfig
from airos.detector.hand_detector import HandDetector


def test_synthetic_hand_detector_execution():
    mdl_cfg = ModelConfig(fallback_to_mock=True)
    det_cfg = DetectorConfig(max_hands=2)
    detector = HandDetector(mdl_cfg, det_cfg)

    synthetic_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(synthetic_frame)

    assert len(detections) > 0
    assert len(detections) <= 2
    for det in detections:
        assert det.width > 0
        assert det.height > 0
        assert det.confidence > 0.0
        assert det.label in ("left", "right")
