"""
AirOS++ Main Application Entry Point & Core Orchestration Pipeline
Author: Senior AI Architecture & Systems Engineering Team
"""

import argparse
import sys
import time

import cv2

from airos.algorithms.ams import AdaptiveMotionSmoothing
from airos.algorithms.bsi import BoundingBoxStabilityIndex
from airos.algorithms.icv import IntentBasedClickVerification
from config.settings import load_config
from airos.controller.os_controller import OSController
from airos.detector.hand_detector import HandDetector
from airos.gesture.gesture_interpreter import GestureInterpreter
from airos.logger.airos_logger import get_logger, setup_logger
from airos.tracking.hand_tracker import HandTracker
from airos.ui.hud import HUDOverlay
from airos.utilities.camera import ThreadedCameraReader
from airos.utilities.metrics import PerformanceMetricsCollector
from airos.utilities.system_info import SystemDiagnostics


def parse_args():
    parser = argparse.ArgumentParser(
        description="AirOS++: Intent-Aware Adaptive Spatial Control Framework for Touchless HCI"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without displaying OpenCV GUI window (suited for CI/headlessness)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate gesture interpretation without invoking native OS APIs",
    )
    parser.add_argument(
        "--mock-camera",
        action="store_true",
        help="Use synthetic frame generator instead of hardware webcam",
    )
    parser.add_argument(
        "--camera-id",
        type=int,
        default=None,
        help="Specify hardware webcam device index (e.g. 0, 1)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="Limit execution to N frames (0 for infinite loop)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG logging output",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration
    config = load_config(args.config)
    if args.dry_run:
        config.controller.dry_run = True
    if args.camera_id is not None:
        config.camera.device_id = args.camera_id

    # Setup Logging
    log_level = "DEBUG" if args.debug else config.system.log_level
    logger = setup_logger(
        log_level=log_level,
        log_to_file=config.system.log_to_file,
        log_file_path=config.system.log_file_path,
    )

    logger.info("Starting AirOS++ Touchless Control Framework...")
    SystemDiagnostics.print_diagnostics()

    # Initialize Core Subsystems
    detector = HandDetector(config.model, config.detector)
    tracker = HandTracker(config.detector)
    ams = AdaptiveMotionSmoothing(config.algorithms.ams)
    bsi = BoundingBoxStabilityIndex(config.algorithms.bsi)
    icv = IntentBasedClickVerification(config.algorithms.icv)
    gesture_interpreter = GestureInterpreter(config.gesture, icv)
    os_controller = OSController(config.controller)
    hud = HUDOverlay(config.hud)
    metrics_collector = PerformanceMetricsCollector()

    # Initialize Threaded Camera Reader
    camera = ThreadedCameraReader(config.camera, mock_mode=args.mock_camera).start()
    time.sleep(0.5)  # Warmup buffer

    frame_count = 0
    current_alpha = config.algorithms.ams.alpha_min

    logger.info("AirOS++ Real-time Pipeline Active. Press 'q' or 'ESC' to exit.")

    try:
        while True:
            t_start = metrics_collector.tick_start()
            frame = camera.read()
            if frame is None:
                time.sleep(0.01)
                continue

            frame_count += 1
            if args.frames > 0 and frame_count > args.frames:
                logger.info(f"Reached frame limit threshold ({args.frames}). Exiting loop.")
                break

            h_img, w_img = frame.shape[:2]

            # 1. Detection Phase
            raw_detections = detector.detect(frame)

            # 2. Tracking Association Phase
            tracked_detections = tracker.update(raw_detections)

            # 3. BSI Stability Evaluation Phase
            stable_detections = []
            for det in tracked_detections:
                track = tracker.get_track(det.track_id) if det.track_id else None
                bsi_score = bsi.evaluate(det, track)
                if det.is_stable:
                    stable_detections.append(det)

            # 4. AMS Motion Smoothing Phase for Primary Hand
            smoothed_cursor_pos = None
            if stable_detections:
                primary_hand = stable_detections[0]
                smoothed_pos, current_alpha = ams.smooth(primary_hand.centroid, det.timestamp)
                primary_hand.centroid = smoothed_pos
                smoothed_cursor_pos = smoothed_pos

            # 5. Gesture Interpretation Phase
            actions = gesture_interpreter.interpret(
                stable_detections, tracker, (w_img, h_img)
            )

            # 6. OS Control API Execution Phase
            for action in actions:
                os_controller.execute(action)

            # 7. Metrics & Telemetry Update
            metrics = metrics_collector.tick_end(t_start, smoothed_cursor_pos)

            # 8. HUD Rendering Phase
            if not args.headless:
                hud_frame = hud.render(
                    frame, stable_detections, actions, metrics, ams_alpha=current_alpha
                )
                cv2.imshow("AirOS++ Telemetry HUD", hud_frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    logger.info("User requested exit signals ('q'/ESC). Terminating.")
                    break

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Stopping application.")
    except Exception as e:
        logger.critical(f"Unhandled exception in AirOS++ execution loop: {e}", exc_info=True)
    finally:
        camera.stop()
        if not args.headless:
            cv2.destroyAllWindows()
        logger.info("AirOS++ Framework Shutdown Complete.")


if __name__ == "__main__":
    main()
