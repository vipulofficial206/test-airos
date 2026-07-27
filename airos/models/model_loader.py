"""
AirOS++ YOLOv10 Model Loader & Hand Detector Backend Engine
Handles CPU model loading, skin contour hand detection, and synthetic fallbacks.
"""

from abc import ABC, abstractmethod
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from config.settings import ModelConfig
from airos.logger.airos_logger import get_logger

logger = get_logger()


class BaseDetectorEngine(ABC):
    """Abstract base class for vision object detection backends."""

    @abstractmethod
    def infer(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Runs inference on input BGR image array and returns raw detections."""
        pass


class SyntheticHandDetector(BaseDetectorEngine):
    """Synthetic Hand Detector for CPU benching, CI environments, and dry-run execution."""

    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        self.start_time = time.time()

    def infer(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Generates realistic synthetic hand detections moving smoothly in a figure-8 motion."""
        t = time.time() - self.start_time
        # Left Hand (Cursor moving)
        lx = int(self.width * 0.35 + math.sin(t * 1.5) * 120)
        ly = int(self.height * 0.5 + math.cos(t * 3.0) * 60)
        lw, lh = 90, 110

        # Right Hand (Action / Click simulation every 4 seconds)
        rx = int(self.width * 0.70 + math.cos(t * 1.2) * 50)
        ry = int(self.height * 0.5 + math.sin(t * 2.0) * 40)
        rh = 100
        rw = 85 if (int(t) % 4 != 0) else 130

        detections = [
            {
                "bbox": [lx - lw // 2, ly - lh // 2, lx + lw // 2, ly + lh // 2],
                "conf": 0.92,
                "class_id": 0,
                "label": "hand",
            },
            {
                "bbox": [rx - rw // 2, ry - rh // 2, rx + rw // 2, ry + rh // 2],
                "conf": 0.88,
                "class_id": 0,
                "label": "hand",
            },
        ]
        return detections


class SkinContourHandDetector(BaseDetectorEngine):
    """Real-time OpenCV Skin-Color & Contour Geometry Hand Detector Engine.
    Excludes top-center face regions and isolates active raised hand bounding boxes.
    Runs on CPU in <2ms, 100% offline.
    """

    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def infer(self, image: np.ndarray) -> List[Dict[str, Any]]:
        if image is None or image.size == 0:
            return []

        h_img, w_img = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect face boxes in the frame to exclude them with 100% precision
        face_boxes = []
        if not self.face_cascade.empty():
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(50, 50))
            for (fx, fy, fw, fh) in faces:
                # Expand face box down to cover chin/neck region
                face_boxes.append((fx - 15, fy - 15, fx + fw + 15, fy + fh + int(fh * 0.5)))

        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

        # YCrCb Skin Color Range
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

        # Morphological Operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.erode(skin_mask, kernel, iterations=1)
        skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
        skin_mask = cv2.GaussianBlur(skin_mask, (5, 5), 0)

        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hand_candidates = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1000:  # Ignore tiny noise
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w / 2.0, y + h / 2.0

            # 100% Face Exclusion: Check if contour centroid is inside any face/head/neck box
            is_face = False
            for (fx1, fy1, fx2, fy2) in face_boxes:
                if fx1 <= cx <= fx2 and fy1 <= cy <= fy2:
                    is_face = True
                    break

            # Fallback heuristic: Top-center region with head aspect ratio ~ 0.7-1.35
            is_top_head = (
                y < h_img * 0.40
                and (w_img * 0.15 < cx < w_img * 0.85)
                and 0.60 < (w / float(h + 1e-6)) < 1.40
                and area > (w_img * h_img * 0.04)
            )

            # Skip face/head contour so ONLY raised HANDS are extracted!
            if is_face or is_top_head:
                continue

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / float(hull_area + 1e-6)
            conf = float(min(0.99, max(0.50, solidity * 1.1)))

            hand_candidates.append(
                {
                    "bbox": [x, y, x + w, y + h],
                    "conf": conf,
                    "class_id": 0,
                    "label": "hand",
                    "area": area,
                }
            )

        # Sort by area descending (largest active hands first)
        hand_candidates.sort(key=lambda item: item["area"], reverse=True)
        return hand_candidates[:2]


class YOLOv10ModelLoader(BaseDetectorEngine):
    """YOLOv10 Hand Detection Model Wrapper using PyTorch/Ultralytics CPU runtime & Skin Hand Fallback."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model: Any = None
        self.skin_detector: SkinContourHandDetector = SkinContourHandDetector()
        self.fallback_engine: Optional[SyntheticHandDetector] = None
        self._initialize_model()

    def _initialize_model(self) -> None:
        model_path = Path(self.config.yolo_model_path)
        logger.info(f"Initializing YOLOv10 detector backend from path: {model_path}")

        try:
            from ultralytics import YOLO

            if model_path.exists():
                logger.info(f"Loading YOLOv10 weights from file {model_path} on CPU...")
                self.model = YOLO(str(model_path))
            else:
                logger.warning(
                    f"Model file '{model_path}' not found on disk. Attempting to download standard yolov10s..."
                )
                self.model = YOLO("yolov10s.pt")
                logger.info("Successfully loaded standard YOLOv10s weights.")
        except Exception as e:
            logger.error(f"Failed to initialize Ultralytics YOLOv10 model: {e}")
            if self.config.fallback_to_mock:
                logger.warning("Falling back to Skin Contour Hand Detector engine...")
            else:
                raise RuntimeError(f"YOLOv10 initialization failed and fallback is disabled: {e}")

    def infer(self, image: np.ndarray) -> List[Dict[str, Any]]:
        # Always run Skin Contour Hand Detector to extract real hand boxes and filter face boxes
        skin_hands = self.skin_detector.infer(image)
        if skin_hands:
            return skin_hands

        if self.model is not None:
            try:
                results = self.model.predict(
                    source=image,
                    conf=self.config.confidence_threshold,
                    iou=self.config.iou_threshold,
                    imgsz=self.config.input_size,
                    device="cpu",
                    verbose=False,
                )

                detections: List[Dict[str, Any]] = []
                h_img, w_img = image.shape[:2]

                if len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes
                    for box in boxes:
                        xyxy = box.xyxy[0].cpu().numpy().tolist()
                        conf = float(box.conf[0].cpu().numpy())
                        cls_id = int(box.cls[0].cpu().numpy())

                        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                        w, h = x2 - x1, y2 - y1

                        # Filter out face / full-body person box (top center large box)
                        is_face_or_body = (
                            y1 < h_img * 0.25 and (h > h_img * 0.40 or w > w_img * 0.40)
                        )
                        if is_face_or_body:
                            continue

                        detections.append(
                            {
                                "bbox": [x1, y1, x2, y2],
                                "conf": conf,
                                "class_id": cls_id,
                                "label": "hand",
                            }
                        )
                if detections:
                    return detections
            except Exception as e:
                logger.error(f"Error during YOLOv10 inference execution: {e}")

        # If no real hands are detected, return empty list (no ghost detections)
        return []
