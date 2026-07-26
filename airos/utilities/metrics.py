"""
AirOS++ Performance Metrics Collector
Tracks FPS, End-to-End Latency, CPU Usage, RAM Usage, and Cursor Jitter Variance.
"""

from collections import deque
from dataclasses import dataclass, field
import time
from typing import Deque, Optional, Tuple

import numpy as np
import psutil

from airos.logger.airos_logger import get_logger

logger = get_logger()


@dataclass
class PerformanceMetrics:
    fps: float = 0.0
    latency_ms: float = 0.0
    cpu_percent: float = 0.0
    ram_percent: float = 0.0
    jitter_variance: float = 0.0
    total_frames_processed: int = 0


class PerformanceMetricsCollector:
    """Collects real-time runtime system statistics and benchmark metrics."""

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.frame_times: Deque[float] = deque(maxlen=window_size)
        self.latencies: Deque[float] = deque(maxlen=window_size)
        self.cursor_positions: Deque[Tuple[float, float]] = deque(maxlen=window_size)
        self.process = psutil.Process()
        self.start_time = time.time()
        self.total_frames = 0

    def tick_start(self) -> float:
        return time.time()

    def tick_end(
        self, t_start: float, cursor_pos: Optional[Tuple[float, float]] = None
    ) -> PerformanceMetrics:
        t_now = time.time()
        latency = (t_now - t_start) * 1000.0  # ms
        self.latencies.append(latency)
        self.frame_times.append(t_now)
        self.total_frames += 1

        if cursor_pos is not None:
            self.cursor_positions.append(cursor_pos)

        # Calculate FPS
        if len(self.frame_times) >= 2:
            elapsed = self.frame_times[-1] - self.frame_times[0]
            fps = float(len(self.frame_times) - 1) / max(0.001, elapsed)
        else:
            fps = 0.0

        # Calculate Latency
        avg_latency = float(np.mean(self.latencies)) if self.latencies else 0.0

        # Calculate Cursor Jitter Variance
        jitter_var = 0.0
        if len(self.cursor_positions) >= 5:
            arr = np.array(self.cursor_positions)
            diffs = np.diff(arr, axis=0)
            speeds = np.linalg.norm(diffs, axis=1)
            jitter_var = float(np.var(speeds))

        # CPU & RAM Usage
        try:
            cpu_pct = psutil.cpu_percent(interval=None)
            ram_pct = self.process.memory_percent()
        except Exception:
            cpu_pct = 0.0
            ram_pct = 0.0

        return PerformanceMetrics(
            fps=fps,
            latency_ms=avg_latency,
            cpu_percent=cpu_pct,
            ram_percent=ram_pct,
            jitter_variance=jitter_var,
            total_frames_processed=self.total_frames,
        )
