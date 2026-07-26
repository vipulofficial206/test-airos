# AirOS++ Hand Movement & Gesture Reference Guide

AirOS++ features an intent-aware spatial gesture state machine designed for touchless operating system interaction using a standard RGB webcam and YOLOv10 Small.

---

## 1. Single-Hand Gestures & Mouse Controls

### 🖐️ 1. Cursor Navigation (Mouse Movement)
* **Hand Used**: Navigation Hand (Left Hand by default, or single visible hand).
* **Posture**: Open, relaxed palm facing the camera.
* **Movement**: Move hand centroid $(c_x, c_y)$ across the camera view frame.
* **Mathematical Mapping**:
  - Normalized Coordinates:
    $$x_{\text{norm}} = \text{clamp}\left( \frac{c_x - W_{\text{img}} \cdot M}{W_{\text{img}} \cdot (1 - 2M)}, 0.0, 1.0 \right)$$
    $$y_{\text{norm}} = \text{clamp}\left( \frac{c_y - H_{\text{img}} \cdot M}{H_{\text{img}} \cdot (1 - 2M)}, 0.0, 1.0 \right)$$
  - Screen Target: $(X_{\text{screen}}, Y_{\text{screen}}) = (x_{\text{norm}} \cdot W_{\text{monitor}}, y_{\text{norm}} \cdot H_{\text{monitor}})$
* **Filtering**: Adaptive Motion Smoothing (AMS) dynamically filters high-frequency jitter during low velocity while providing instant responsiveness during fast movements.

---

### 👈 2. Left Click (Tap / Fist Pinch)
* **Hand Used**: Action Hand (Right Hand by default).
* **Posture Transformation**: Shift from Open Palm to a **Tight Fist** or **Index Pinch**.
* **Movement**: Hold hand still for ~0.15 seconds during posture change.
* **ICV Verification Rules**:
  1. Aspect Ratio Shift: $|AR_t - AR_{\text{baseline}}| \ge 0.15$
  2. BSI Stability Index: $S_t \ge 0.65$
  3. Stillness Velocity: $v_{\text{mag}} \le 4.0\text{ px/frame}$
  4. Temporal Window: Maintained for $N \ge 4$ consecutive frames.
* **Result**: Executes a single OS Left Click.

---

### ✌️ 3. Double Click (Double Quick Tap)
* **Hand Used**: Action Hand (Right Hand).
* **Posture Transformation**: Perform two quick fist pulses within $0.45\text{ seconds}$.
* **Result**: Executes an OS Double Click (opens files, folders, applications).

---

### 👉 4. Right Click (Wide Flat Palm)
* **Hand Used**: Action Hand (Right Hand).
* **Posture**: Open hand horizontally into a **Wide Flat Palm** ($AR > 1.35$).
* **Movement**: Hold wide palm still for ~0.15 seconds.
* **Result**: Executes a Right Click (opens OS Context Menu).

---

### 🖐️ 5. Drag & Drop (Click and Hold to Drag)
* **Hand Used**: Action Hand (Right Hand).
* **Posture**: Form a tight fist and hold continuously for $> 0.65\text{ seconds}$.
* **Movement**: Move hand while keeping fist closed to drag windows/icons across the screen. Open palm to release.
* **Result**:
  - Hold Fist $> 0.65\text{s}$: Triggers Left Mouse Down (`pyautogui.mouseDown`).
  - Open Hand: Triggers Left Mouse Up (`pyautogui.mouseUp`).

---

### 👆 6. Middle Click (Vertical Narrow Hand)
* **Hand Used**: Action Hand (Right Hand).
* **Posture**: Hold hand in a narrow vertical orientation ($AR < 0.65$).
* **Result**: Triggers a Middle Mouse Click.

---

## 2. Dual-Hand Spatial System Controls

### 🔊 7. Master Volume Up / Down
* **Hands Used**: Both hands visible simultaneously in camera view.
* **Movement**:
  - **Volume Increase**: Move hands apart horizontally (distance expansion $\Delta d > 12\text{px}$).
  - **Volume Decrease**: Bring hands closer together horizontally (distance contraction $\Delta d < -12\text{px}$).
* **Formula**: $\Delta V = (\Delta d / d_{\max}) \cdot \text{Sensitivity}$

---

### ☀️ 8. Display Brightness Increase / Decrease
* **Hands Used**: Both hands visible simultaneously in camera view.
* **Movement**:
  - **Brightness Increase**: Raise Right Hand significantly above Left Hand.
  - **Brightness Decrease**: Lower Right Hand below Left Hand.
* **Formula**: $\Delta B = (-\Delta h_{\text{elev}} / 200.0) \cdot \text{Sensitivity}$

---

### 🔇 9. Mute Audio Toggle (Cross Hands)
* **Hands Used**: Both hands visible simultaneously.
* **Movement**: Cross both hands horizontally in front of the camera.
* **Result**: Toggles system master mute on/off.

---

## 3. Hand Movement Quick Reference Matrix

| Gesture Name | Hand Posture / Movement | Action Triggered | Target Parameter |
| :--- | :--- | :--- | :--- |
| **Cursor Move** | Open Palm Movement | Mouse Cursor Motion | Screen Coordinates $(X, Y)$ |
| **Left Click** | Single Fist / Pinch Pulse | Left Mouse Click | `pyautogui.click('left')` |
| **Double Click** | Two Quick Fist Pulses (<0.45s) | Double Mouse Click | `pyautogui.doubleClick()` |
| **Right Click** | Wide Flat Palm ($AR > 1.35$) | Context Menu / Right Click | `pyautogui.click('right')` |
| **Drag & Drop** | Sustained Closed Fist (>0.65s) | Click & Drag Hold / Release | `mouseDown()` / `mouseUp()` |
| **Middle Click** | Narrow Vertical Hand ($AR < 0.65$) | Middle Mouse Click | `pyautogui.middleClick()` |
| **Volume Control**| Dual Hand Stretch (Expansion/Contraction)| Master Volume Adjust | PyCAW Endpoint Volume |
| **Brightness Control**| Dual Hand Vertical Elevation Shift | Screen Brightness Adjust | SBC Display Brightness |
| **Mute Toggle** | Cross Both Hands | Toggle Master Mute | System Mute Key |
