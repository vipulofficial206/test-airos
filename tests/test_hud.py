"""
Unit Tests for OpenCV HUD Overlay Telemetry Rendering
"""

import numpy as np
import pytest

from config.settings import HUDConfig
from airos.detector.hand_detector import HandDetection
from airos.gesture.gesture_interpreter import GestureAction, GestureType
from airos.ui.hud import HUDOverlay
from airos.utilities.metrics import PerformanceMetrics


def test_hud_overlay_rendering():
    hud_cfg = HUDConfig(enabled=True)
    hud = HUDOverlay(hud_cfg)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    det = HandDetection(bbox=(100, 100, 200, 200), centroid=(150.0, 150.0), confidence=0.95, width=100, height=100, area=10000.0, aspect_ratio=1.0, label="left", bsi_score=0.90, is_stable=True)
    action = GestureAction(gesture_type=GestureType.CURSOR_MOVE, cursor_target_norm=(0.5, 0.5), description="Cursor Move (0.50, 0.50)")
    metrics = PerformanceMetrics(fps=30.0, latency_ms=25.0, cpu_percent=15.0, ram_percent=2.0)

    rendered_frame = hud.render(frame, [det], [action], metrics, ams_alpha=0.5)

    assert rendered_frame is not None
    assert rendered_frame.shape == frame.shape
    assert isinstance(rendered_frame, np.ndarray)


def test_hud_overlay_disabled():
    hud_cfg = HUDConfig(enabled=False)
    hud = HUDOverlay(hud_cfg)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    metrics = PerformanceMetrics()
    rendered_frame = hud.render(frame, [], [], metrics)

    assert np.array_equal(rendered_frame, frame)


def test_hud_overlay_multi_hands():
    hud_cfg = HUDConfig(enabled=True)
    hud = HUDOverlay(hud_cfg)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    det1 = HandDetection(bbox=(50, 50, 150, 150), centroid=(100.0, 100.0), confidence=0.90, width=100, height=100, area=10000.0, aspect_ratio=1.0, label="left", is_stable=True)
    det2 = HandDetection(bbox=(400, 50, 500, 150), centroid=(450.0, 100.0), confidence=0.40, width=100, height=100, area=10000.0, aspect_ratio=1.0, label="right", is_stable=False)
    metrics = PerformanceMetrics()

    rendered = hud.render(frame, [det1, det2], [], metrics)
    assert rendered is not None


def test_hud_overlay_empty_actions():
    hud_cfg = HUDConfig(enabled=True)
    hud = HUDOverlay(hud_cfg)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    metrics = PerformanceMetrics()
    rendered = hud.render(frame, [], [], metrics)

    assert rendered is not None
