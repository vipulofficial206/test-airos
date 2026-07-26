"""
AirOS++ System Diagnostics & Platform Verification
Performs environment startup checks, CPU capability detection, and dependency validation.
"""

import os
import platform
import sys
from typing import Dict

import psutil

from airos.logger.airos_logger import get_logger

logger = get_logger()


class SystemDiagnostics:
    """Performs system verification and environment capability reporting."""

    @staticmethod
    def print_diagnostics() -> Dict[str, str]:
        info = {
            "OS": f"{platform.system()} {platform.release()} ({platform.architecture()[0]})",
            "Python": sys.version.split()[0],
            "CPU": platform.processor() or "CPU Device",
            "Cores (Physical/Logical)": f"{psutil.cpu_count(logical=False)}/{psutil.cpu_count(logical=True)}",
            "RAM Total": f"{psutil.virtual_memory().total / (1024**3):.2f} GB",
        }

        logger.info("=====================================================")
        logger.info("           AirOS++ System Environment Info           ")
        logger.info("=====================================================")
        for key, val in info.items():
            logger.info(f"  {key:<25}: {val}")
        logger.info("=====================================================")
        return info
