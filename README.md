# AirOS++: Intent-Aware Adaptive Spatial Control Framework for Touchless HCI

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyTorch CPU](https://img.shields.io/badge/PyTorch-CPU%20Only-orange.svg)](https://pytorch.org/)
[![MediaPipe & YOLOv10](https://img.shields.io/badge/Dual%20Backend-MediaPipe%20%2B%20YOLOv10-green.svg)](https://github.com/google/mediapipe)

AirOS++ is a research-grade, lightweight touchless operating system control framework designed to run completely offline on standard laptop CPUs using a single RGB webcam.

---

## Key Features & Research Contributions

- **Dual Neural Backend Detector**: Supports MediaPipe 21-Landmark 3D joint tracking and YOLOv10 Small (`yolov10s`) CPU backends with zero cloud/GPU dependencies.
- **Algorithm 1: Adaptive Motion Smoothing (AMS)**: Dynamic velocity and acceleration-based smoothing factor ($\alpha_t$) eliminating low-speed cursor jitter while ensuring zero lag during fast movements.
- **Algorithm 2: Bounding Box Stability Index (BSI)**: Multi-factor stability metric evaluating confidence, displacement, area ratio, aspect ratio, and temporal persistence to filter unstable detections.
- **Algorithm 3: Intent-Based Click Verification (ICV)**: Accidental click rejection engine verifying aspect ratio shifts ($\Delta AR$), high BSI stability ($S_t$), stillness, and temporal persistence over $N_{\text{icv}}$ frames.
- **Real-Time OpenCV Telemetry HUD**: Live overlay showing FPS, End-to-End Latency (ms), Bounding Boxes, Centroids, BSI scores, AMS Alpha, Active Gestures, and CPU/RAM usage.

---

## Project Architecture

```
airos/
├── config/             # YAML/JSON settings & schema
├── models/             # MediaPipe & YOLOv10 engine loader & mock fallback
├── detector/           # Hand detector & spatial feature extractor
├── tracking/           # Temporal hand tracker & track persistence
├── algorithms/         # AMS, BSI, and ICV research algorithms
├── gesture/            # Spatial gesture state machine
├── controller/         # OS Automation API bridge (PyAutoGUI, PyCAW, SBC)
├── ui/                 # Real-time OpenCV telemetry HUD
├── utilities/          # Threaded camera, metrics collector, diagnostics
├── logger/             # Thread-safe structured logging
tests/                  # Automated pytest suite (86 Unit & Integration Tests)
benchmark/              # Quantitative benchmark suite & report generator
docs/                   # Architecture, Mathematical Formulations, Gesture Manual
main.py                 # Application orchestrator CLI
```

---

## Quickstart Guide

### Anaconda Setup
```bash
conda create -n airos-env python=3.11 -y
conda activate airos-env
pip install -r requirements.txt
```

### Live Run
```bash
run_app.bat
```

### Dry-Run / Mock Run
```bash
python main.py --dry-run --mock-camera
```

### Run Benchmark Suite
```bash
python benchmark/run_benchmark.py --frames 300
```

### Run Test Suite
```bash
pytest tests/ -v
```

---

## Quantitative Performance & Reaction Time Benchmarks

| Metric / Parameter | Measured Benchmark Value | Target Constraint | Evaluation Status |
| :--- | :--- | :--- | :--- |
| **FPS (Frame Rate)** | **28.18 FPS** | >= 25.0 FPS | **PASS** |
| **Mean Latency (Reaction Time)** | **34.70 ms** | <= 45.0 ms | **PASS** |
| **95th Percentile Latency (P95)** | **35.58 ms** | <= 50.0 ms | **PASS** |
| **CPU Utilization** | **35.93%** | <= 40.0% CPU | **PASS** |
| **RAM Consumption** | **0.93%** | <= 5.0% RAM | **PASS** |
| **BSI Stability Metric** | **0.974** | >= 0.60 | **PASS** |
| **Cursor Jitter Variance** | **4.12 px²** | <= 10.0 px² | **PASS** |

---

## Documentation Links

- [Complete Hand Movement & Gesture Manual](docs/GESTURE_MANUAL.md)
- [System Architecture Specification](docs/ARCHITECTURE.md)
- [Mathematical Formulations (AMS, BSI, ICV)](docs/MATHEMATICS.md)
- [User Manual & Troubleshooting Guide](docs/USER_MANUAL.md)

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
