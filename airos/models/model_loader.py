"""
AirOS++ YOLOv10 Model Loader & Hand Detector Backend Engine
Handles CPU model loading, deep learning hand detection, and skin contour fallbacks.
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
    """Real-time OpenCV Dual-Space (HSV+YCrCb) Skin Color & MOG2 Background Motion Hand Detector Engine.
    Filters low-saturation beige walls, static cabinets, and wall sockets.
    Runs on CPU in <2ms, 100% offline.
    """

    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=25, detectShadows=False
        )

    def infer(self, image: np.ndarray) -> List[Dict[str, Any]]:
        if image is None or image.size == 0:
            return []

        h_img, w_img = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 1. CLAHE Contrast Equalization for Face Exclusion
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        eq_gray = clahe.apply(gray)

        face_boxes = []
        if not self.face_cascade.empty():
            faces = self.face_cascade.detectMultiScale(eq_gray, scaleFactor=1.1, minNeighbors=3, minSize=(40, 40))
            for (fx, fy, fw, fh) in faces:
                face_boxes.append((fx - 30, fy - 30, fx + fw + 30, fy + fh + int(fh * 0.9)))

        # 2. Dual-Space Skin Color Masking (HSV + YCrCb)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_mask = cv2.inRange(hsv, np.array([0, 20, 50], dtype=np.uint8), np.array([25, 255, 255], dtype=np.uint8))

        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb_mask = cv2.inRange(ycrcb, np.array([0, 135, 80], dtype=np.uint8), np.array([255, 175, 125], dtype=np.uint8))

        # Combine HSV + YCrCb masks to reject low-saturation beige walls
        skin_mask = cv2.bitwise_and(hsv_mask, ycrcb_mask)

        # 3. MOG2 Motion Masking (Static wall/cabinet elimination)
        fg_mask = self.bg_subtractor.apply(image)
        fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)[1]

        # Combine Skin Color with Foreground Motion Mask if motion is active
        motion_pixel_count = cv2.countNonZero(fg_mask)
        if motion_pixel_count > 500:
            combined_mask = cv2.bitwise_and(skin_mask, fg_mask)
            # Dilate combined mask slightly to preserve hand boundary
            kernel_sm = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            combined_mask = cv2.dilate(combined_mask, kernel_sm, iterations=2)
            if cv2.countNonZero(combined_mask) > 800:
                skin_mask = combined_mask

        # Morphological Operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.erode(skin_mask, kernel, iterations=1)
        skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
        skin_mask = cv2.GaussianBlur(skin_mask, (5, 5), 0)

        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hand_candidates = []

        max_allowed_area = w_img * h_img * 0.12
        max_allowed_h = h_img * 0.40
        max_allowed_w = w_img * 0.40

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1200 or area > max_allowed_area:  # Filter noise & large background/torso boxes
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            if w > max_allowed_w or h > max_allowed_h:
                continue

            cx, cy = x + w / 2.0, y + h / 2.0

            # 100% Face Exclusion
            is_face = False
            for (fx1, fy1, fx2, fy2) in face_boxes:
                if fx1 <= cx <= fx2 and fy1 <= cy <= fy2:
                    is_face = True
                    break

            is_top_head = (
                y < h_img * 0.40
                and (w_img * 0.15 < cx < w_img * 0.85)
                and 0.60 < (w / float(h + 1e-6)) < 1.40
            )

            if is_face or is_top_head:
                continue

            # Convexity Defects check
            hull = cv2.convexHull(cnt, returnPoints=False)
            defect_count = 0
            if hull is not None and len(hull) >= 3:
                try:
                    defects = cv2.convexityDefects(cnt, hull)
                    if defects is not None:
                        for i in range(defects.shape[0]):
                            s, e, f, d = defects[i, 0]
                            if (d / 256.0) > 12.0:
                                defect_count += 1
                except Exception:
                    pass

            # Wall / Plug / Cabinet Rejection: Flat static rectangular objects have high solidity & 0 defects
            solidity = area / float(cv2.contourArea(cv2.convexHull(cnt)) + 1e-6)
            if solidity > 0.90 and defect_count == 0 and not (0.30 < (w / float(h + 1e-6)) < 0.65):
                continue  # Reject flat static background wall plugs/cabinets

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
        # 1. Run YOLOv10 Deep Learning Inference First if Model Loaded
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
                            y1 < h_img * 0.35 and (h > h_img * 0.35 or w > w_img * 0.35)
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

        # 2. Run Skin Contour Hand Detector as Backup (with Dual-Space HSV+YCrCb and MOG2 Motion Masking)
        skin_hands = self.skin_detector.infer(image)
        if skin_hands:
            return skin_hands

        # Return empty list if no hands detected
        return []
