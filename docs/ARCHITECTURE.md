# AirOS++ System Architecture & Component Design

## 1. System Overview

**AirOS++** is an offline, lightweight touchless human–computer interaction (HCI) software framework for desktop operating systems. The system translates live RGB camera video into precise cursor movement, click verification, scrolling, system volume, and brightness adjustments using a YOLOv10 hand detection model and software-level motion/stability algorithms.

```
+------------------+     +-------------------------------+     +-----------------------------------+
|  Standard RGB    | --> |  Threaded Camera Reader       | --> |  YOLOv10 Hand Detection Engine    |
|  Webcam Feed     |     |  (Zero-Copy Buffer Queue)     |     |  (PyTorch/Ultralytics CPU Mode)   |
+------------------+     +-------------------------------+     +-----------------------------------+
                                                                                 |
                                                                                 v
+--------------------------------------------------------------------------------------------------+
|                              Spatial & Temporal Analysis Pipeline                                |
|                                                                                                  |
|   +--------------------------+    +--------------------------+    +--------------------------+   |
|   | Bounding Box Stability   | -> | Adaptive Motion          | -> | Intent-Based Click       |   |
|   | Index (BSI)              |    | Smoothing (AMS)          |    | Verification (ICV)       |   |
|   +--------------------------+    +--------------------------+    +--------------------------+   |
+--------------------------------------------------------------------------------------------------+
                                                                                 |
                                                                                 v
+------------------+     +-------------------------------+     +-----------------------------------+
| Operating System | <-- |  OS Controller API Bridge     | <-- | Spatial Gesture Interpreter       |
| Desktop Control  |     |  (PyAutoGUI / PyCAW / SBC)    |     | (Cursor, Click, Vol, Brightness)  |
+------------------+     +-------------------------------+     +-----------------------------------+
                                         |
                                         v
                         +-------------------------------+
                         | Real-time Telemetry HUD       |
                         | (OpenCV Dashboard Overlay)    |
                         +-------------------------------+
```

---

## 2. Component Class Diagram

```
 +------------------------+             +--------------------------+
 |  HandDetector          | <---------> |  YOLOv10ModelLoader      |
 +------------------------+             +--------------------------+
 | + detect(frame): List  |             | + infer(frame): List     |
 +------------------------+             +--------------------------+
             |
             v
 +------------------------+             +--------------------------+
 |  HandTracker           | <---------> |  BoundingBoxStability    |
 +------------------------+             |  Index (BSI)             |
 | + update(dets): List   |             +--------------------------+
 +------------------------+             | + evaluate(det, track)   |
             |                          +--------------------------+
             v
 +------------------------+             +--------------------------+
 |  AdaptiveMotion        | <---------> | IntentBasedClick         |
 |  Smoothing (AMS)       |             | Verification (ICV)       |
 +------------------------+             +--------------------------+
 | + smooth(pos, t): pos  |             | + verify_click(): bool   |
 +------------------------+             +--------------------------+
             |                                       |
             +-------------------+-------------------+
                                 |
                                 v
                     +------------------------+
                     |  GestureInterpreter    |
                     +------------------------+
                     | + interpret(): List    |
                     +------------------------+
                                 |
                                 v
                     +------------------------+
                     |  OSController          |
                     +------------------------+
                     | + execute(action)      |
                     +------------------------+
```

---

## 3. Sequential Execution Flow

1. **Frame Capture**: `ThreadedCameraReader` captures BGR frames asynchronously at 30 FPS into a single-item queue to prevent latency accumulation.
2. **Hand Detection**: `HandDetector` feeds 320x320 scaled frames into `YOLOv10ModelLoader` executing on CPU.
3. **Temporal Tracking**: `HandTracker` associates new detections with existing track histories using spatial centroid Euclidean distance matching and velocity prediction.
4. **BSI Stability Evaluation**: `BoundingBoxStabilityIndex` computes multi-factor stability index $S_t$. Boxes with $S_t < T_{\text{BSI}}$ are marked unstable.
5. **AMS Smoothing**: `AdaptiveMotionSmoothing` dynamically adjusts dynamic factor $\alpha_t$ based on speed magnitude $v_{\text{mag}}$, smoothing cursor movements.
6. **ICV Click Intent**: `IntentBasedClickVerification` evaluates posture aspect ratio changes $\Delta AR$, stillness $v_{\text{mag}}$, and temporal frame persistence $N_{\text{icv}}$.
7. **Action Dispatch**: `GestureInterpreter` converts states into `GestureAction` payloads, which `OSController` executes via PyAutoGUI, PyCAW, and screen-brightness-control.
8. **HUD Render**: Telemetry dashboard overlays FPS, Latency (ms), Bounding Boxes, BSI, Alpha, and System Usage onto frame.
