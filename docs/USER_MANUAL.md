# AirOS++ User Manual & Installation Guide

## 1. System Requirements

- **Operating System**: Windows 10/11 (64-bit), Linux, or macOS.
- **Python Runtime**: Anaconda / Miniconda with Python 3.11.
- **Camera**: Standard integrated or USB RGB webcam (640x480 resolution @ 30 FPS).
- **Hardware**: Standard x86_64 Laptop/Desktop CPU (GPU strictly optional/not required).

---

## 2. Environment Setup

Open Anaconda Prompt or Terminal in the project root directory (`e:\vitap\capstone project`):

```bash
# 1. Create conda environment with Python 3.11
conda create -n airos-env python=3.11 -y

# 2. Activate environment
conda activate airos-env

# 3. Install core dependencies via pip
pip install -r requirements.txt
```

---

## 3. Running AirOS++

### A. Desktop Control Center Application (GUI Window)
```bash
python app.py
# Or double-click run_app.bat
```

### B. CLI Live Interactive Mode
```bash
python main.py
```

### C. Dry-Run Mode (Test gestures without moving OS mouse)
```bash
python main.py --dry-run
```

### D. Automated Benchmark Runner
```bash
python benchmark/run_benchmark.py --frames 300
```

### E. Automated Test Suite Execution
```bash
pytest tests/ -v
```

---

## 4. Complete Gesture Reference Overview

For detailed mathematical formulas and parameters, see [docs/GESTURE_MANUAL.md](file:///e:/vitap/capstone%20project/docs/GESTURE_MANUAL.md).

| Hand Position / Gesture | Interaction Action | Operational Description |
| :--- | :--- | :--- |
| **Navigation Hand Open Palm** | **Cursor Navigation** | Smooth 2D cursor movement with AMS exponential filtering. |
| **Action Hand Fist / Pinch** | **Left Click** | Intent-verified single Left Click via ICV engine. |
| **Action Hand Double Tap** | **Double Click** | Two quick pulses within 0.45s trigger Double Click. |
| **Action Hand Wide Palm** | **Right Click** | Wide horizontal posture ($AR > 1.35$) triggers Right Click. |
| **Action Hand Sustained Fist** | **Drag & Drop** | Closed fist (>0.65s) holds mouse down; opening hand releases. |
| **Action Hand Narrow Vertical** | **Middle Click** | Narrow vertical posture ($AR < 0.65$) triggers Middle Click. |
| **Dual-Hand Distance** | **Master Volume** | Expanding hand separation increases volume; contracting decreases. |
| **Dual-Hand Elevation** | **Display Brightness** | Raising/lowering vertical spatial offset adjusts screen brightness. |
| **Dual-Hand Cross Hands** | **Mute Toggle** | Crossing both hands toggles master system mute on/off. |

---

## 5. Troubleshooting & Parameter Tuning

Editing thresholds in `config/default_config.yaml` or via the **Algorithm Tuning** tab in `app.py`:
- **Adjusting Cursor Sensitivity**: Tweak `gesture.cursor_margin_ratio` (default `0.10`).
- **Adjusting Motion Smoothing**: Modify `algorithms.ams.alpha_min` and `alpha_max`.
- **Adjusting Accidental Click Sensitivity**: Increase `algorithms.icv.consecutive_frames_required` (default `4`).
