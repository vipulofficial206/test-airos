# AirOS++ User Manual & Installation Guide

## 1. System Requirements

- **Operating System**: Windows 10/11 (64-bit), Linux, or macOS.
- **Python Runtime**: Anaconda / Miniconda with Python 3.11.
- **Camera**: Standard integrated or USB RGB webcam (640x480 resolution @ 30 FPS).
- **Hardware**: Standard x86_64 Laptop/Desktop CPU (GPU strictly optional/not required).

---

## 2. Conda Environment Setup

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

### A. Live Interactive Mode (With Webcam & HUD)
```bash
python main.py
```

### B. Dry-Run Mode (Test gestures without moving OS mouse)
```bash
python main.py --dry-run
```

### C. Synthetic Mock Mode (Run without webcam peripheral)
```bash
python main.py --mock-camera --dry-run
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

## 4. Gesture Mapping Reference Guide

| Hand Position / Gesture | Interaction Action | Operational Description |
| :--- | :--- | :--- |
| **Left Hand Movement** | **Cursor Navigation** | Centroid position mapped to screen boundaries. |
| **Right Hand Fist/Pinch** | **Left Click** | ICV intent verified via aspect ratio shift ($\Delta AR$) + stillness. |
| **Right Hand Wide Posture**| **Right Click** | Secondary ICV posture threshold. |
| **Dual-Hand Distance** | **Master Volume** | Expanding hand separation increases volume; contracting decreases. |
| **Dual-Hand Elevation** | **Display Brightness** | Raising/lowering vertical spatial offset adjusts screen brightness. |

---

## 5. Troubleshooting & Configuration

Editing thresholds in `config/default_config.yaml`:
- **Adjusting Cursor Sensitivity**: Tweak `gesture.cursor_margin_ratio` (default `0.10`).
- **Adjusting Motion Smoothing**: Modify `algorithms.ams.alpha_min` and `alpha_max`.
- **Adjusting Accidental Click Sensitivity**: Increase `algorithms.icv.consecutive_frames_required` (default `4`).
