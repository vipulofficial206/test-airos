"""
Full Integration & Pipeline Stress Tests for AirOS++
"""

import numpy as np
import pytest

from airos.algorithms.ams import AdaptiveMotionSmoothing
from airos.algorithms.bsi import BoundingBoxStabilityIndex
from airos.algorithms.icv import IntentBasedClickVerification
from config.settings import load_config
from airos.controller.os_controller import OSController
from airos.detector.hand_detector import HandDetector
from airos.gesture.gesture_interpreter import GestureInterpreter
from airos.tracking.hand_tracker import HandTracker


def test_full_pipeline_synthetic_execution():
    config = load_config("config/default_config.yaml")
    config.controller.dry_run = True

    detector = HandDetector(config.model, config.detector)
    tracker = HandTracker(config.detector)
    ams = AdaptiveMotionSmoothing(config.algorithms.ams)
    bsi = BoundingBoxStabilityIndex(config.algorithms.bsi)
    icv = IntentBasedClickVerification(config.algorithms.icv)
    interpreter = GestureInterpreter(config.gesture, icv)
    controller = OSController(config.controller)

    synthetic_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Run 50 pipeline iterations
    for frame_idx in range(50):
        raw_dets = detector.detect(synthetic_frame)
        tracked_dets = tracker.update(raw_dets)

        stable_dets = []
        for det in tracked_dets:
            track = tracker.get_track(det.track_id) if det.track_id else None
            bsi.evaluate(det, track)
            if det.is_stable:
                stable_dets.append(det)

        if stable_dets:
            smoothed_pos, alpha = ams.smooth(stable_dets[0].centroid, det.timestamp)
            stable_dets[0].centroid = smoothed_pos

        actions = interpreter.interpret(stable_dets, tracker, (640, 480))
        for act in actions:
            controller.execute(act)

    assert True  # Pipeline completed without uncaught exceptions
