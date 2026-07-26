"""
AirOS++ Desktop Application Main Launcher
Author: Senior AI Architecture & HCI Engineering Team
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from airos.ui.gui_app import launch_gui_app

if __name__ == "__main__":
    launch_gui_app()
