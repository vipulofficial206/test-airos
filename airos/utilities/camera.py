"""
AirOS++ Threaded Camera Reader
Asynchronous OpenCV video capture with auto-reconnect, zero-copy buffer queue, and synthetic frame generator.
"""

from queue import Empty, Queue
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from config.settings import CameraConfig
from airos.logger.airos_logger import get_logger

logger = get_logger()


class SyntheticFrameGenerator:
    """Generates synthetic RGB frames for headless and CI testing."""

    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        self.frame_count = 0

    def read(self) -> Tuple[bool, np.ndarray]:
        self.frame_count += 1
        # Create dark blue background
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:, :] = (35, 20, 15)  # BGR

        # Draw grid lines
        for x in range(0, self.width, 40):
            cv2.line(frame, (x, 0), (x, self.height), (50, 40, 30), 1)
        for y in range(0, self.height, 40):
            cv2.line(frame, (0, y), (self.width, y), (50, 40, 30), 1)

        time.sleep(1.0 / 30.0)  # Simulate 30 FPS timing
        return True, frame


class ThreadedCameraReader:
    """Threaded OpenCV VideoCapture reader maintaining latest frame buffer to minimize latency."""

    def __init__(self, config: CameraConfig, mock_mode: bool = False):
        self.config = config
        self.mock_mode = mock_mode
        self.cap: Optional[cv2.VideoCapture] = None
        self.synthetic_gen: Optional[SyntheticFrameGenerator] = None
        self.frame_queue: Queue = Queue(maxsize=self.config.buffer_size)
        self.stopped: bool = False
        self.thread: Optional[threading.Thread] = None

        if self.mock_mode:
            logger.info("Initializing ThreadedCameraReader in Synthetic Mock Mode.")
            self.synthetic_gen = SyntheticFrameGenerator(config.width, config.height)
        else:
            self._init_camera()

    def _init_camera(self) -> bool:
        logger.info(f"Opening OpenCV VideoCapture device index {self.config.device_id}...")
        self.cap = cv2.VideoCapture(self.config.device_id, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            logger.warning(
                f"Failed to open hardware webcam at index {self.config.device_id}. Falling back to synthetic frame generator."
            )
            self.mock_mode = True
            self.synthetic_gen = SyntheticFrameGenerator(self.config.width, self.config.height)
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.config.target_fps)
        return True

    def start(self) -> "ThreadedCameraReader":
        self.stopped = False
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()
        return self

    def _update_loop(self) -> None:
        while not self.stopped:
            if self.mock_mode and self.synthetic_gen is not None:
                ret, frame = self.synthetic_gen.read()
            else:
                if self.cap is None or not self.cap.isOpened():
                    if self.config.auto_reconnect:
                        logger.warning("Camera disconnected. Reconnecting...")
                        time.sleep(self.config.reconnect_delay_sec)
                        self._init_camera()
                        continue
                    else:
                        break
                ret, frame = self.cap.read()

            if not ret or frame is None:
                logger.warning("Blank frame received from capture source.")
                time.sleep(0.01)
                continue

            # Keep queue size at 1 to discard stale buffered frames (low latency)
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except Empty:
                    pass
            self.frame_queue.put(frame)

    def read(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Returns latest available frame array or None on timeout."""
        try:
            return self.frame_queue.get(timeout=timeout)
        except Empty:
            return None

    def stop(self) -> None:
        self.stopped = True
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if self.cap is not None:
            self.cap.release()
        logger.info("ThreadedCameraReader cleanly stopped.")
