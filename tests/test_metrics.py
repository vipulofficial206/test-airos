"""
Unit Tests for Performance Metrics Collector
"""

import time
import pytest

from airos.utilities.metrics import PerformanceMetricsCollector, PerformanceMetrics


def test_metrics_collector_fps_and_latency():
    collector = PerformanceMetricsCollector(window_size=10)

    t1 = collector.tick_start()
    time.sleep(0.01)  # 10ms processing simulation
    m1 = collector.tick_end(t1, cursor_pos=(100.0, 100.0))

    assert m1.total_frames_processed == 1
    assert m1.latency_ms >= 10.0

    t2 = collector.tick_start()
    time.sleep(0.01)
    m2 = collector.tick_end(t2, cursor_pos=(105.0, 105.0))

    assert m2.total_frames_processed == 2
    assert m2.fps > 0.0


def test_metrics_collector_jitter_variance():
    collector = PerformanceMetricsCollector(window_size=10)

    for i in range(10):
        t = collector.tick_start()
        # Simulated position movement
        pos = (100.0 + i * 2.0, 100.0 + i * 2.0)
        m = collector.tick_end(t, cursor_pos=pos)

    assert isinstance(m.jitter_variance, float)
    assert m.jitter_variance >= 0.0
