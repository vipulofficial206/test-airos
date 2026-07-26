"""
AirOS++ OS Controller API Integration Layer
Simulates mouse movement, clicks, vertical scrolling, master volume, and display brightness.
Includes safety bounds clamping, dry-run testing mode, and platform fallbacks.
"""

import sys
import time
from typing import Optional, Tuple

import pyautogui

from airos.config.settings import ControllerConfig
from airos.gesture.gesture_interpreter import GestureAction, GestureType
from airos.logger.airos_logger import get_logger

logger = get_logger()

# Configure PyAutoGUI Failsafes
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.001


class OSController:
    """Operating System Automation Bridge."""

    def __init__(self, config: ControllerConfig):
        self.config = config
        self.screen_width, self.screen_height = pyautogui.size()
        self.last_action_time: float = 0.0
        self.volume_interface = None
        self._init_audio_backend()

    def _init_audio_backend(self) -> None:
        """Initializes PyCAW audio backend on Windows platform."""
        if sys.platform == "win32":
            try:
                from comtypes import CLSCTX_ALL
                from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

                devices = AudioUtilities.GetSpeakers()
                interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                self.volume_interface = interface.QueryInterface(IAudioEndpointVolume)
                logger.info("Successfully initialized PyCAW audio endpoint controller.")
            except Exception as e:
                logger.warning(f"Could not initialize PyCAW audio endpoint: {e}")

    def execute(self, action: GestureAction) -> None:
        """Executes a single GestureAction on the host OS."""
        if action.gesture_type == GestureType.NONE:
            return

        if self.config.dry_run:
            logger.debug(f"[DRY-RUN] Executed Action: {action.description}")
            return

        current_time = time.time()

        try:
            if action.gesture_type == GestureType.CURSOR_MOVE:
                if action.cursor_target_norm is not None:
                    nx, ny = action.cursor_target_norm
                    target_x = int(nx * self.screen_width)
                    target_y = int(ny * self.screen_height)
                    # Clamp to screen boundary
                    target_x = max(0, min(self.screen_width - 1, target_x))
                    target_y = max(0, min(self.screen_height - 1, target_y))
                    pyautogui.moveTo(target_x, target_y, _pause=False)

            elif action.gesture_type == GestureType.LEFT_CLICK:
                pyautogui.click(button="left")
                logger.info("OS Controller: Left Click Triggered")

            elif action.gesture_type == GestureType.RIGHT_CLICK:
                pyautogui.click(button="right")
                logger.info("OS Controller: Right Click Triggered")

            elif action.gesture_type == GestureType.SCROLL_UP:
                steps = int(action.value_delta)
                pyautogui.scroll(steps)

            elif action.gesture_type == GestureType.SCROLL_DOWN:
                steps = int(action.value_delta)
                pyautogui.scroll(-steps)

            elif action.gesture_type == GestureType.VOLUME_CHANGE:
                self.adjust_volume(action.value_delta)

            elif action.gesture_type == GestureType.BRIGHTNESS_CHANGE:
                self.adjust_brightness(action.value_delta)

        except Exception as e:
            logger.error(f"Error executing OS Controller action '{action.description}': {e}")

    def adjust_volume(self, delta: float) -> None:
        """Adjusts system master volume."""
        if self.volume_interface is not None:
            try:
                cur_vol = self.volume_interface.GetMasterScalarLevel()
                new_vol = max(0.0, min(1.0, cur_vol + delta))
                self.volume_interface.SetMasterScalarLevel(new_vol, None)
                logger.info(f"Volume adjusted: {cur_vol*100:.0f}% -> {new_vol*100:.0f}%")
            except Exception as e:
                logger.error(f"Failed to set master volume via PyCAW: {e}")

    def adjust_brightness(self, delta: float) -> None:
        """Adjusts primary display brightness."""
        try:
            import screen_brightness_control as sbc

            cur_b = sbc.get_brightness(display=0)
            if isinstance(cur_b, list):
                cur_b = cur_b[0]
            new_b = max(0, min(100, int(cur_b + delta * 100)))
            sbc.set_brightness(new_b, display=0)
            logger.info(f"Brightness adjusted: {cur_b}% -> {new_b}%")
        except Exception as e:
            logger.warning(f"Failed to adjust screen brightness: {e}")
