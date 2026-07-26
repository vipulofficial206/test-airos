"""
AirOS++ YOLOv10 Model Loader & Model Backend Abstraction
Handles CPU model loading, fallback inference engine, and synthetic detection generation.
"""

from abc import ABC, abstractmethod
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from airos.config.settings import ModelConfig
from airos.logger.airos_logger import get_logger

logger = get_logger()


class BaseDetectorEngine(ABC):
    """Abstract base class for vision object detection backends."""

    @abstractmethod
    def infer(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Runs inference on input BGR image array and returns raw detections.

        Returns list of dicts: [{'bbox': [x1, y1, x2, y2], 'conf': float, 'class_id': int, 'label': str}]
        """
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
        # Periodically shift aspect ratio to simulate click gesture posture change
        rh = 100
        rw = 85 if (int(t) % 4 != 0) else 130  # Aspect ratio pulse

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


class YOLOv10ModelLoader(BaseDetectorEngine):
    """YOLOv10 Hand Detection Model Wrapper using PyTorch/Ultralytics CPU runtime."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model: Any = None
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
                logger.warning("Falling back to Synthetic Hand Detector engine...")
                self.fallback_engine = SyntheticHandDetector()
            else:
                raise RuntimeError(f"YOLOv10 initialization failed and fallback is disabled: {e}")

    def infer(self, image: np.ndarray) -> List[Dict[str, Any]]:
        if self.fallback_engine is not None:
            return self.fallback_engine.infer(image)

        if self.model is None:
            raise RuntimeError("Model backend is uninitialized.")

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
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy().tolist()
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    detections.append(
                        {
                            "bbox": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                            "conf": conf,
                            "class_id": cls_id,
                            "label": "hand",
                        }
                    )
            return detections
        except Exception as e:
            logger.error(f"Error during YOLOv10 inference execution: {e}")
            if self.config.fallback_to_mock:
                if self.fallback_engine is None:
                    self.fallback_engine = SyntheticHandDetector(
                        width=image.shape[1], height=image.shape[0]
                    )
                return self.fallback_engine.infer(image)
            return []
