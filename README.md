# AirOS++: Intent-Aware Adaptive Spatial Control Framework for Touchless HCI Using YOLOv10

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyTorch CPU](https://img.shields.io/badge/PyTorch-CPU%20Only-orange.svg)](https://pytorch.org/)
[![YOLOv10](https://img.shields.io/badge/YOLOv10-Small-green.svg)](https://github.com/THU-MIG/yolov10)

AirOS++ is a research-grade, lightweight touchless operating system control framework designed to run completely offline on standard laptop CPUs using a single RGB webcam.

---

## Key Features & Research Contributions

- **Zero Landmark Dependency**: Completely eliminates MediaPipe, 21-joint skeletal keypoints, depth cameras, Leap Motion, cloud APIs, and GPU hardware requirements.
- **YOLOv10 Hand Detection**: Uses YOLOv10 Small (`yolov10s`) for robust 2D bounding box, centroid, area, aspect ratio, and confidence extraction.
- **Algorithm 1: Adaptive Motion Smoothing (AMS)**: Dynamic velocity and acceleration-based smoothing factor ($\alpha_t$) eliminating low-speed cursor jitter while ensuring zero lag during fast movements.
- **Algorithm 2: Bounding Box Stability Index (BSI)**: Multi-factor stability metric evaluating confidence, displacement, area ratio, aspect ratio, and temporal persistence to filter unstable detections.
- **Algorithm 3: Intent-Based Click Verification (ICV)**: Accidental click rejection engine verifying aspect ratio shifts ($\Delta AR$), high BSI stability ($S_t$), stillness, and temporal persistence over $N_{\text{icv}}$ frames.
- **Real-Time OpenCV HUD**: Telemetry overlay showing FPS, End-to-End Latency (ms), Bounding Boxes, Centroids, BSI scores, AMS Alpha, Active Gestures, and CPU/RAM usage.

---

## Project Architecture

```
airos/
├── config/             # YAML/JSON settings & schema
├── models/             # YOLOv10 engine loader & mock fallback
├── detector/           # Hand detector & spatial feature extractor
├── tracking/           # Temporal hand tracker & track persistence
├── algorithms/         # AMS, BSI, and ICV research algorithms
├── gesture/            # Spatial gesture state machine
├── controller/         # OS Automation API bridge (PyAutoGUI, PyCAW, SBC)
├── ui/                 # Real-time OpenCV telemetry HUD
├── utilities/          # Threaded camera, metrics collector, diagnostics
├── logger/             # Thread-safe structured logging
tests/                  # Automated pytest suite (Unit, Integration, Stress)
benchmark/              # Benchmark suite & report generator
docs/                   # Architecture, Mathematical Formulations, Manual
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
python main.py
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

## Quantitative Benchmarks

| Parameter | Value | Target Constraint | Status |
| :--- | :--- | :--- | :--- |
| **FPS** | **25.93 FPS** | >= 25.0 FPS | **PASS** |
| **Mean Latency** | **37.18 ms** | <= 45.0 ms | **PASS** |
| **CPU Usage** | **27.03%** | <= 40.0% CPU | **PASS** |
| **BSI Score** | **0.973** | >= 0.60 | **PASS** |

---

## Documentation Links

- [Complete Hand Movement & Gesture Manual](docs/GESTURE_MANUAL.md)
- [System Architecture Specification](docs/ARCHITECTURE.md)
- [Mathematical Formulations (AMS, BSI, ICV)](docs/MATHEMATICS.md)
- [User Manual & Troubleshooting Guide](docs/USER_MANUAL.md)

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
