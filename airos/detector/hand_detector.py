"""
AirOS++ Hand Detector & Dataclass Definition
Extracts spatial bounding boxes, centroids, geometric aspect ratios, and confidence.
"""

from dataclasses import dataclass, field
import time
from typing import List, Optional, Tuple

import numpy as np

from config.settings import DetectorConfig, ModelConfig
from airos.logger.airos_logger import get_logger
from airos.models.model_loader import BaseDetectorEngine, YOLOv10ModelLoader

logger = get_logger()


@dataclass
class HandDetection:
    """Dataclass storing complete spatial and temporal metrics of a detected hand box."""

    bbox: Tuple[int, int, int, int]  # (xmin, ymin, xmax, ymax)
    centroid: Tuple[float, float]  # (cx, cy)
    confidence: float
    width: int
    height: int
    area: float
    aspect_ratio: float  # width / height
    label: str = "unassigned"  # 'left', 'right', or 'unassigned'
    track_id: Optional[int] = None
    timestamp: float = field(default_factory=time.time)
    bsi_score: float = 1.0
    is_stable: bool = True


class HandDetector:
    """YOLOv10 Hand Detection pipeline manager.

    Processes raw frames and constructs structured HandDetection objects.
    """

    def __init__(self, model_config: ModelConfig, detector_config: DetectorConfig):
        self.model_config = model_config
        self.detector_config = detector_config
        self.engine: BaseDetectorEngine = YOLOv10ModelLoader(model_config)

    def detect(self, frame: np.ndarray) -> List[HandDetection]:
        """Runs hand detection on the input frame and extracts spatial attributes."""
        if frame is None or frame.size == 0:
            logger.warning("Empty or null frame passed to HandDetector.")
            return []

        raw_detections = self.engine.infer(frame)
        current_time = time.time()
        hand_detections: List[HandDetection] = []

        for det in raw_detections:
            xmin, ymin, xmax, ymax = det["bbox"]
            conf = det["conf"]

            # Clamp coordinates to frame boundaries
            h_img, w_img = frame.shape[:2]
            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(w_img - 1, xmax), min(h_img - 1, ymax)

            w = max(1, xmax - xmin)
            h = max(1, ymax - ymin)
            cx = xmin + w / 2.0
            cy = ymin + h / 2.0
            area = float(w * h)
            aspect_ratio = float(w) / float(h)

            hand_det = HandDetection(
                bbox=(xmin, ymin, xmax, ymax),
                centroid=(cx, cy),
                confidence=conf,
                width=w,
                height=h,
                area=area,
                aspect_ratio=aspect_ratio,
                timestamp=current_time,
            )
            hand_detections.append(hand_det)

        # Sort by confidence descending and cap at max_hands
        hand_detections.sort(key=lambda x: x.confidence, reverse=True)
        hand_detections = hand_detections[: self.detector_config.max_hands]

        # Assign initial spatial handedness heuristic (leftmost centroid -> left hand in camera frame)
        if len(hand_detections) == 1:
            # If single hand detected, determine side based on frame center
            _, w_img = frame.shape[:2]
            if hand_detections[0].centroid[0] < w_img / 2.0:
                hand_detections[0].label = "left"
            else:
                hand_detections[0].label = "right"
        elif len(hand_detections) >= 2:
            # Sort by x coordinate ascending (left hand has smaller x)
            sorted_by_x = sorted(hand_detections[:2], key=lambda d: d.centroid[0])
            sorted_by_x[0].label = "left"
            sorted_by_x[1].label = "right"

        return hand_detections
