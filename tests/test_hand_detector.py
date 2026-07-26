"""
Unit Tests for Hand Detector & Synthetic Engine
"""

import numpy as np
import pytest

from config.settings import DetectorConfig, ModelConfig
from airos.detector.hand_detector import HandDetector
from airos.models.model_loader import SyntheticHandDetector
from airos.utilities.camera import SyntheticFrameGenerator


def test_synthetic_hand_detector_engine():
    gen = SyntheticFrameGenerator(640, 480)
    ret, frame = gen.read()
    assert ret is True

    synth_detector = SyntheticHandDetector(640, 480)
    raw_dets = synth_detector.infer(frame)

    assert len(raw_dets) == 2
    assert raw_dets[0]["label"] == "hand"
    assert raw_dets[0]["conf"] > 0.80


def test_hand_detector_pipeline():
    mdl_cfg = ModelConfig(fallback_to_mock=True)
    det_cfg = DetectorConfig(max_hands=2)
    detector = HandDetector(mdl_cfg, det_cfg)

    # Use synthetic frame generator frame
    gen = SyntheticFrameGenerator(640, 480)
    _, synthetic_frame = gen.read()

    detections = detector.detect(synthetic_frame)

    assert isinstance(detections, list)
    for det in detections:
        assert det.width > 0
        assert det.height > 0
        assert det.confidence > 0.0
        assert det.label in ("left", "right", "unassigned")
