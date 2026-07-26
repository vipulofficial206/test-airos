"""
Automated Benchmarking Suite for AirOS++
Evaluates FPS, End-to-End Latency, CPU Usage, RAM Footprint, Cursor Jitter Variance, and Click Verification Accuracy over N frame iterations.
"""

import argparse
import json
from pathlib import Path
import time
from typing import Dict, Any

import numpy as np

from airos.algorithms.ams import AdaptiveMotionSmoothing
from airos.algorithms.bsi import BoundingBoxStabilityIndex
from airos.algorithms.icv import IntentBasedClickVerification
from airos.config.settings import load_config
from airos.controller.os_controller import OSController
from airos.detector.hand_detector import HandDetector
from airos.gesture.gesture_interpreter import GestureInterpreter
from airos.logger.airos_logger import setup_logger
from airos.tracking.hand_tracker import HandTracker
from airos.utilities.camera import ThreadedCameraReader
from airos.utilities.metrics import PerformanceMetricsCollector
from benchmark.benchmark_report_generator import generate_markdown_report

logger = setup_logger(log_level="INFO")


def run_benchmark(num_frames: int = 300, config_path: str = "config/default_config.yaml") -> Dict[str, Any]:
    """Runs automated benchmark simulation and records quantitative metrics."""
    logger.info(f"Starting AirOS++ Benchmark Suite across {num_frames} frames...")

    config = load_config(config_path)
    config.controller.dry_run = True  # Ensure dry-run during benchmarks

    detector = HandDetector(config.model, config.detector)
    tracker = HandTracker(config.detector)
    ams = AdaptiveMotionSmoothing(config.algorithms.ams)
    bsi = BoundingBoxStabilityIndex(config.algorithms.bsi)
    icv = IntentBasedClickVerification(config.algorithms.icv)
    interpreter = GestureInterpreter(config.gesture, icv)
    os_controller = OSController(config.controller)
    metrics_collector = PerformanceMetricsCollector()

    camera = ThreadedCameraReader(config.camera, mock_mode=True).start()
    time.sleep(0.3)

    latencies = []
    fps_list = []
    cpu_list = []
    ram_list = []
    jitters = []
    bsi_scores = []
    click_count = 0

    start_bench_time = time.time()

    for idx in range(num_frames):
        t0 = metrics_collector.tick_start()
        frame = camera.read()
        if frame is None:
            continue

        raw_dets = detector.detect(frame)
        tracked_dets = tracker.update(raw_dets)

        stable_dets = []
        for det in tracked_dets:
            track = tracker.get_track(det.track_id) if det.track_id else None
            score = bsi.evaluate(det, track)
            bsi_scores.append(score)
            if det.is_stable:
                stable_dets.append(det)

        smoothed_pos = None
        if stable_dets:
            smoothed_pos, _ = ams.smooth(stable_dets[0].centroid, det.timestamp)
            stable_dets[0].centroid = smoothed_pos

        actions = interpreter.interpret(stable_dets, tracker, (640, 480))
        for act in actions:
            if act.description and "Click" in act.description:
                click_count += 1
            os_controller.execute(act)

        m = metrics_collector.tick_end(t0, smoothed_pos)
        latencies.append(m.latency_ms)
        fps_list.append(m.fps)
        cpu_list.append(m.cpu_percent)
        ram_list.append(m.ram_percent)
        jitters.append(m.jitter_variance)

    camera.stop()
    total_bench_duration = time.time() - start_bench_time

    results = {
        "num_frames_evaluated": num_frames,
        "total_duration_sec": round(total_bench_duration, 2),
        "mean_fps": round(float(np.mean(fps_list[5:])), 2) if len(fps_list) > 5 else 0.0,
        "mean_latency_ms": round(float(np.mean(latencies)), 2),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
        "mean_cpu_percent": round(float(np.mean(cpu_list)), 2),
        "mean_ram_percent": round(float(np.mean(ram_list)), 2),
        "mean_bsi_score": round(float(np.mean(bsi_scores)), 3),
        "jitter_variance_px2": round(float(np.mean(jitters)), 4),
        "clicks_verified": click_count,
    }

    logger.info("=====================================================")
    logger.info("            AirOS++ Benchmark Summary               ")
    logger.info("=====================================================")
    for k, v in results.items():
        logger.info(f"  {k:<25}: {v}")
    logger.info("=====================================================")

    # Save JSON report
    out_dir = Path("benchmark")
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    generate_markdown_report(results, out_dir / "benchmark_report.md")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AirOS++ Benchmark Suite")
    parser.add_argument("--frames", type=int, default=300, help="Number of benchmark frames")
    args = parser.parse_args()
    run_benchmark(args.frames)
