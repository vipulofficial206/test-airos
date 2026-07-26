"""
AirOS++ Real-time Telemetry OpenCV HUD Overlay
Renders FPS, Latency (ms), Bounding Boxes, Centroids, BSI Scores, AMS Alpha, Active Gesture, and CPU/RAM usage.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np

from config.settings import HUDConfig
from airos.detector.hand_detector import HandDetection
from airos.gesture.gesture_interpreter import GestureAction
from airos.utilities.metrics import PerformanceMetrics


class HUDOverlay:
    """Renders research telemetry overlay on OpenCV frames."""

    def __init__(self, config: HUDConfig):
        self.config = config

    def render(
        self,
        frame: np.ndarray,
        detections: List[HandDetection],
        actions: List[GestureAction],
        metrics: PerformanceMetrics,
        ams_alpha: float = 0.5,
    ) -> np.ndarray:
        """Renders HUD overlay onto frame copy."""
        if not self.config.enabled:
            return frame

        hud_frame = frame.copy()
        h, w = hud_frame.shape[:2]

        # 1. Render Bounding Boxes & Centroids
        if self.config.show_bounding_boxes or self.config.show_centroids:
            for det in detections:
                color = (
                    tuple(self.config.box_color_stable)
                    if det.is_stable
                    else tuple(self.config.box_color_unstable)
                )
                xmin, ymin, xmax, ymax = det.bbox

                if self.config.show_bounding_boxes:
                    cv2.rectangle(hud_frame, (xmin, ymin), (xmax, ymax), color, 2)

                if self.config.show_centroids:
                    cx, cy = int(det.centroid[0]), int(det.centroid[1])
                    cv2.circle(hud_frame, (cx, cy), 5, (255, 255, 0), -1)
                    cv2.drawMarker(
                        hud_frame,
                        (cx, cy),
                        (0, 255, 255),
                        cv2.MARKER_CROSS,
                        10,
                        1,
                    )

                # Label text above box
                label_text = f"{det.label.upper()} | Conf: {det.confidence:.2f}"
                if self.config.show_bsi:
                    label_text += f" | BSI: {det.bsi_score:.2f}"

                cv2.putText(
                    hud_frame,
                    label_text,
                    (xmin, max(20, ymin - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        # 2. Render Telemetry Sidebar Card (Top-Left)
        card_w, card_h = 240, 160
        overlay = hud_frame.copy()
        cv2.rectangle(overlay, (10, 10), (10 + card_w, 10 + card_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, hud_frame, 0.35, 0, hud_frame)
        cv2.rectangle(hud_frame, (10, 10), (10 + card_w, 10 + card_h), (0, 255, 255), 1)

        # Header Title
        cv2.putText(
            hud_frame,
            "AirOS++ Telemetry",
            (20, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Stats Lines
        y_offset = 48
        line_height = 18

        stats = []
        if self.config.show_fps:
            stats.append(f"FPS: {metrics.fps:.1f}")
        if self.config.show_latency:
            stats.append(f"Latency: {metrics.latency_ms:.1f} ms")
        if self.config.show_ams_alpha:
            stats.append(f"AMS Alpha: {ams_alpha:.2f}")
        if self.config.show_system_usage:
            stats.append(f"CPU: {metrics.cpu_percent:.1f}% | RAM: {metrics.ram_percent:.1f}%")

        for line in stats:
            cv2.putText(
                hud_frame,
                line,
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            y_offset += line_height

        # 3. Render Active Gesture Banner (Bottom-Center)
        active_desc = actions[0].description if actions else "Idle"
        cv2.putText(
            hud_frame,
            f"Active Gesture: {active_desc}",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        return hud_frame
