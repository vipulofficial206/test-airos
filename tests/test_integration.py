"""
End-to-End System Pipeline Integration Tests
"""

import time
import pytest

from config.settings import load_config
from airos.algorithms.ams import AdaptiveMotionSmoothing
from airos.algorithms.bsi import BoundingBoxStabilityIndex
from airos.algorithms.icv import IntentBasedClickVerification
from airos.controller.os_controller import OSController
from airos.detector.hand_detector import HandDetector
from airos.gesture.gesture_interpreter import GestureInterpreter
from airos.tracking.hand_tracker import HandTracker
from airos.ui.hud import HUDOverlay
from airos.utilities.camera import ThreadedCameraReader
from airos.utilities.metrics import PerformanceMetricsCollector


def test_full_pipeline_synthetic_execution():
    config = load_config("config/default_config.yaml")
    config.controller.dry_run = True

    camera = ThreadedCameraReader(config.camera, mock_mode=True).start()
    detector = HandDetector(config.model, config.detector)
    tracker = HandTracker(config.detector)
    ams = AdaptiveMotionSmoothing(config.algorithms.ams)
    bsi = BoundingBoxStabilityIndex(config.algorithms.bsi)
    icv = IntentBasedClickVerification(config.algorithms.icv)
    interpreter = GestureInterpreter(config.gesture, icv)
    controller = OSController(config.controller)
    hud = HUDOverlay(config.hud)
    metrics_collector = PerformanceMetricsCollector()

    for i in range(50):
        t_start = metrics_collector.tick_start()
        frame = camera.read()
        assert frame is not None

        raw_dets = detector.detect(frame)
        tracked_dets = tracker.update(raw_dets)

        stable_dets = []
        for det in tracked_dets:
            track = tracker.get_track(det.track_id) if det.track_id else None
            bsi.evaluate(det, track)
            if det.is_stable:
                stable_dets.append(det)

        smoothed_pos = None
        if stable_dets:
            smoothed_pos, alpha = ams.smooth(stable_dets[0].centroid, det.timestamp)
            stable_dets[0].centroid = smoothed_pos

        actions = interpreter.interpret(stable_dets, tracker, (640, 480), frame=frame)
        for act in actions:
            controller.execute(act)

        metrics = metrics_collector.tick_end(t_start, smoothed_pos)
        rendered = hud.render(frame, stable_dets, actions, metrics)
        assert rendered is not None

    camera.stop()


def test_pipeline_reset_and_restart():
    config = load_config("config/default_config.yaml")
    config.controller.dry_run = True

    camera = ThreadedCameraReader(config.camera, mock_mode=True).start()
    time.sleep(0.1)
    camera.stop()
    assert camera.stopped is True

    camera.start()
    frame = camera.read()
    assert frame is not None
    camera.stop()


def test_synthetic_camera_frame_counter():
    config = load_config("config/default_config.yaml")
    camera = ThreadedCameraReader(config.camera, mock_mode=True).start()

    f1 = camera.read()
    time.sleep(0.04)
    f2 = camera.read()

    assert f1 is not None
    assert f2 is not None
    camera.stop()
