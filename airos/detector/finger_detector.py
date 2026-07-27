"""
AirOS++ Zero-Landmark Finger Contour Geometry Engine
Extracts finger count, index fingertip coordinate, and finger postures using OpenCV Convexity Defects on YOLOv10 ROI.
Completely offline, runs on CPU in <1.5ms, zero MediaPipe dependency.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Tuple

import cv2
import numpy as np

from airos.detector.hand_detector import HandDetection
from airos.logger.airos_logger import get_logger

logger = get_logger()


class FingerGesture(Enum):
    UNKNOWN = auto()
    FIST_CLOSED = auto()     # 0 fingers
    INDEX_POINTING = auto()  # 1 finger (Index)
    TWO_FINGERS_V = auto()   # 2 fingers (V-Sign)
    THREE_FINGERS = auto()   # 3 fingers
    FOUR_FINGERS = auto()    # 4 fingers
    FIVE_OPEN = auto()       # 5 fingers open palm


@dataclass
class FingerAnalysis:
    """Dataclass storing finger contour geometry metrics."""

    finger_count: int = 0
    gesture: FingerGesture = FingerGesture.UNKNOWN
    index_tip_pt: Optional[Tuple[int, int]] = None  # (x, y) pinpoint in frame space
    index_tip_norm: Optional[Tuple[float, float]] = None  # (0.0 to 1.0)
    convexity_defects_count: int = 0
    contour_area: float = 0.0


class FingerDetector:
    """Extracts extended finger counts and index fingertip coordinates from YOLOv10 hand ROI."""

    @staticmethod
    def analyze(frame: np.ndarray, det: HandDetection) -> FingerAnalysis:
        analysis = FingerAnalysis()
        if frame is None or det is None or frame.size == 0:
            return analysis

        h_img, w_img = frame.shape[:2]
        x1, y1, x2, y2 = det.bbox

        # Clamp ROI bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0 or (x2 - x1) < 10 or (y2 - y1) < 10:
            return analysis

        # Convert to grayscale and apply adaptive thresholding
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find largest contour in ROI
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return analysis

        max_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(max_contour)
        analysis.contour_area = area

        if area < 300:
            return analysis

        # Smooth contour slightly to eliminate self-intersections
        epsilon = 0.005 * cv2.arcLength(max_contour, True)
        max_contour = cv2.approxPolyDP(max_contour, epsilon, True)

        # Calculate Convex Hull & Defects
        hull = cv2.convexHull(max_contour, returnPoints=False)
        if hull is None or len(hull) < 3:
            return analysis

        defects = None
        try:
            defects = cv2.convexityDefects(max_contour, hull)
        except Exception as e:
            logger.debug(f"Convexity defects calculation skipped for noisy contour: {e}")
            defects = None
        defect_count = 0

        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = max_contour[s][0]
                end = max_contour[e][0]
                far = max_contour[f][0]

                # Calculate defect triangle side lengths
                a = np.linalg.norm(np.array(end) - np.array(start))
                b = np.linalg.norm(np.array(far) - np.array(start))
                c = np.linalg.norm(np.array(far) - np.array(end))

                # Cosine rule for angle at defect point
                angle = np.arccos(np.clip((b**2 + c**2 - a**2) / (2 * b * c + 1e-6), -1.0, 1.0))
                depth = d / 256.0  # Distance in pixels

                # Valid finger gap: acute angle (< 90 degrees) & sufficient defect depth (> 15px)
                if angle < np.pi / 2.0 and depth > 15.0:
                    defect_count += 1

        analysis.convexity_defects_count = defect_count

        # Finger Count = Defect Count + 1 (if hand extended)
        if area > 1200 and defect_count == 0:
            # Check aspect ratio for 0 fingers (fist) vs 1 finger (index extended)
            roi_ar = (x2 - x1) / float(y2 - y1 + 1e-6)
            if roi_ar < 0.60:
                finger_count = 1
            else:
                finger_count = 0
        else:
            finger_count = defect_count + 1 if defect_count > 0 else 1

        finger_count = max(0, min(5, finger_count))
        analysis.finger_count = finger_count

        # Map Finger Gesture Enum
        if finger_count == 0:
            analysis.gesture = FingerGesture.FIST_CLOSED
        elif finger_count == 1:
            analysis.gesture = FingerGesture.INDEX_POINTING
        elif finger_count == 2:
            analysis.gesture = FingerGesture.TWO_FINGERS_V
        elif finger_count == 3:
            analysis.gesture = FingerGesture.THREE_FINGERS
        elif finger_count == 4:
            analysis.gesture = FingerGesture.FOUR_FINGERS
        else:
            analysis.gesture = FingerGesture.FIVE_OPEN

        # Find Index Fingertip (Top-most point of largest contour in ROI)
        top_pt_roi = tuple(max_contour[max_contour[:, :, 1].argmin()][0])
        tip_x = x1 + top_pt_roi[0]
        tip_y = y1 + top_pt_roi[1]

        analysis.index_tip_pt = (tip_x, tip_y)
        analysis.index_tip_norm = (tip_x / float(w_img), tip_y / float(h_img))

        return analysis
