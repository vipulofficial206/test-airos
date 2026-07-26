# AirOS++ Finger & Spatial Gesture Reference Manual

AirOS++ features an intent-aware finger geometry and spatial gesture state machine designed for touchless operating system interaction using a standard RGB webcam and YOLOv10 Small.

---

## 1. Zero-Landmark Finger Contour Geometry Engine

Instead of relying on heavy MediaPipe keypoint neural networks, AirOS++ extracts finger counts and pinpoint coordinates using **YOLOv10 Bounding Box ROI + OpenCV Convexity Defect Geometry**:

1. **YOLOv10 Hand ROI Extraction**: Isolates hand bounding box $(x_1, y_1, x_2, y_2)$.
2. **Adaptive Threshold Contour Masking**: Isolates hand silhouette within ROI.
3. **Convexity Defect Finger Analysis**: Calculates Convex Hull and measures deep valley defect depths and enclosed angles ($\theta < 90^\circ, d > 15\text{px}$).
4. **Index Fingertip Pinpointing**: Locates the top-most contour peak in the ROI for pixel-exact cursor positioning.

---

## 2. Finger-Level & Mouse Action Controls

### 👆 1. Index Finger Pointing (Precision Cursor Navigation)
* **Hand Used**: Navigation Hand (Left Hand by default, or single visible hand).
* **Posture**: Index finger extended upward ($AR < 0.60$ or 1 extended finger).
* **Target Pinpoint**: Top-most contour peak of the index finger $(x_{\text{tip}}, y_{\text{tip}})$.
* **Mathematical Screen Mapping**:
  $$x_{\text{norm}} = \text{clamp}\left( \frac{x_{\text{tip}} - W_{\text{img}} \cdot M}{W_{\text{img}} \cdot (1 - 2M)}, 0.0, 1.0 \right)$$
  $$y_{\text{norm}} = \text{clamp}\left( \frac{y_{\text{tip}} - H_{\text{img}} \cdot M}{H_{\text{img}} \cdot (1 - 2M)}, 0.0, 1.0 \right)$$
* **System Action**: Positions mouse cursor smoothly with Adaptive Motion Smoothing (AMS).

---

### 🤏 2. Left Click (Index Tap / Pinch)
* **Posture Transformation**: Quick index finger tap or tight fist pulse.
* **ICV Verification Rules**:
  1. Aspect Ratio Shift: $|AR_t - AR_{\text{baseline}}| \ge 0.15$
  2. BSI Stability Index: $S_t \ge 0.65$
  3. Stillness Velocity: $v_{\text{mag}} \le 4.0\text{ px/frame}$
  4. Temporal Window: Maintained for $N \ge 4$ consecutive frames.
* **System Action**: Executes a single Left Mouse Click (`pyautogui.click('left')`).

---

### ✌️ 3. Right Click (2 Fingers / V-Sign)
* **Posture**: Extend 2 fingers (V-Sign / Victory posture) or Wide Flat Palm ($AR > 1.35$).
* **Verification**: Detects 2 convexity defect valleys or wide horizontal box aspect ratio.
* **System Action**: Executes a Right Mouse Click (`pyautogui.click('right')`).

---

### 🤟 4. Middle Click (3 Fingers Extended)
* **Posture**: Extend 3 fingers upward.
* **Verification**: Detects 3 convexity defect valleys or narrow vertical box posture ($AR < 0.65$).
* **System Action**: Executes a Middle Mouse Click (`pyautogui.middleClick()`).

---

### ✊ 5. Drag & Drop (Closed Fist Click & Hold)
* **Posture**: Form a tight closed fist (0 extended fingers) and hold for $> 0.65\text{ seconds}$.
* **System Action**:
  - Hold Fist $> 0.65\text{s}$: Triggers Left Mouse Down (`pyautogui.mouseDown`).
  - Open Hand: Triggers Left Mouse Up (`pyautogui.mouseUp`).

---

### ⏱️ 6. Hover Dwell Click (1.0s Hold)
* **Posture**: Hold index fingertip stationary within $2.5\%$ screen radius for $1.0\text{ second}$.
* **System Action**: Automatically triggers a Left Click without requiring any posture shift.

---

## 3. Spatial Trajectory Swipes & System Controls

### ⬅️ ➡️ 7. Horizontal Swipes (Left & Right)
* **Fast Left Swipe**: Fast horizontal motion to the left ($\text{speed} > 1.8\text{ px/ms}$).
  - **Action**: Browser Back / Switch to Previous Desktop (`Alt + Left`).
* **Fast Right Swipe**: Fast horizontal motion to the right.
  - **Action**: Browser Forward / Switch to Next Desktop (`Alt + Right`).

---

### ⬆️ ⬇️ 8. Vertical Swipes (Up & Down)
* **Fast Upward Swipe**: Fast vertical motion upward.
  - **Action**: Windows Task View (`Win + Tab`).
* **Fast Downward Swipe**: Fast vertical motion downward.
  - **Action**: Show Desktop (`Win + D`).

---

### 🔊 ☀️ 9. Master Volume & Display Brightness
* **Master Volume**: Expand/contract horizontal distance between both hands ($\Delta d > 12\text{px}$).
* **Display Brightness**: Raise/lower vertical elevation offset between both hands ($\Delta h > 15\text{px}$).
* **Mute Toggle**: Cross both hands horizontally.

---

## 4. Master Gesture Mapping Reference Matrix

| Extended Fingers / Posture | Spatial Movement | Interaction Action | Target OS Function |
| :--- | :--- | :--- | :--- |
| **👆 1 Finger (Index Extended)** | Smooth Motion | Cursor Position | Index Tip Pinpoint $(X, Y)$ |
| **🤏 1 Finger Dip / Pinch** | Still Hold (0.15s) | Left Mouse Click | `pyautogui.click('left')` |
| **✌️ 2 Fingers (V-Sign)** | Still Hold | Right Mouse Click | `pyautogui.click('right')` |
| **🤟 3 Fingers Extended** | Still Hold | Middle Mouse Click | `pyautogui.middleClick()` |
| **✊ 0 Fingers (Closed Fist)** | Hold > 0.65s | Drag & Drop Hold | `pyautogui.mouseDown()` |
| **🖐️ 5 Fingers (Open Hand)** | Open Palm | Release Drag & Drop | `pyautogui.mouseUp()` |
| **⏱️ 1.0s Stationary Dwell** | Hover Still | Hover Dwell Click | Automatic Left Click |
| **⬅️ Fast Swipe Left** | Horizontal Left | Browser Back / Desktop | `Alt + Left` |
| **➡️ Fast Swipe Right** | Horizontal Right | Browser Forward / Desktop | `Alt + Right` |
| **⬆️ Fast Swipe Up** | Vertical Up | Task View | `Win + Tab` |
| **⬇️ Fast Swipe Down** | Vertical Down | Show Desktop | `Win + D` |
| **↔️ Dual Hand Stretch** | Horizontal Distance | Master Volume | PyCAW Endpoint Volume |
| **↕️ Dual Hand Elevation** | Vertical Offset | Display Brightness | SBC Screen Brightness |
| **❌ Cross Hands** | Horizontal Cross | Mute Toggle | `volumemute` |
