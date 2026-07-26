# Mathematical Formulations for AirOS++ Control Framework

This document outlines the formal mathematical formulations of the core software algorithms introduced in **AirOS++**:
1. **Adaptive Motion Smoothing (AMS)**
2. **Bounding Box Stability Index (BSI)**
3. **Intent-Based Click Verification (ICV)**

---

## 1. Algorithm 1: Adaptive Motion Smoothing (AMS)

Given raw hand centroid position $\vec{p}_t = (x_t, y_t)$ at time step $t$ with frame timestamp $\Delta t = t - t_{-1}$:

### Velocity & Acceleration Vectors
$$\vec{v}_t = \frac{\vec{p}_t - \vec{p}_{t-1}}{\Delta t}, \quad \vec{a}_t = \frac{\vec{v}_t - \vec{v}_{t-1}}{\Delta t}$$

### Speed Magnitude & Deadzone Filtering
$$v_{\text{mag}} = \|\vec{v}_t\|_2 = \sqrt{v_{x,t}^2 + v_{y,t}^2}$$

If $v_{\text{mag}} < v_{\text{deadzone}}$, then $v_{\text{mag}} = 0$.

### Dynamic Alpha Formulation
$$\alpha_t = \text{clamp}\left(\alpha_{\min} + \left(1 - e^{-\lambda \cdot v_{\text{mag}}}\right) \cdot (\alpha_{\max} - \alpha_{\min}), \alpha_{\min}, \alpha_{\max}\right)$$

where:
- $\alpha_{\min} = 0.15$: Minimum alpha for high smoothing during stationary hand posture.
- $\alpha_{\max} = 0.85$: Maximum alpha for direct tracking during rapid hand movement.
- $\lambda = 0.05$: Exponential response velocity sensitivity scaling coefficient.

### Exponential Position Smoothing
$$\vec{p}_{\text{smoothed}, t} = \alpha_t \cdot \vec{p}_t + (1 - \alpha_t) \cdot \vec{p}_{\text{smoothed}, t-1}$$

---

## 2. Algorithm 2: Bounding Box Stability Index (BSI)

The Bounding Box Stability Index $S_t \in [0.0, 1.0]$ evaluates multi-dimensional detection stability to reject false-positive bounding box jitter:

$$S_t = w_c C_t + w_d S_{\text{disp}, t} + w_a S_{\text{area}, t} + w_{ar} S_{\text{aspect}, t} + w_p S_{\text{pers}, t}$$

Subject to parameter constraint: $\sum w_i = 1.0$.

### Stability Components

1. **Detection Confidence ($C_t$)**: Direct model confidence score $C_t \in [0, 1]$.
2. **Centroid Displacement Stability ($S_{\text{disp}, t}$)**:
   $$S_{\text{disp}, t} = \exp\left(-\gamma_d \cdot \|\vec{p}_t - \vec{p}_{t-1}\|\right)$$
3. **Area Consistency Stability ($S_{\text{area}, t}$)**:
   $$S_{\text{area}, t} = \exp\left(-\gamma_a \cdot \left|\frac{A_t - A_{t-1}}{A_{t-1}}\right|\right)$$
4. **Aspect Ratio Consistency Stability ($S_{\text{aspect}, t}$)**:
   $$S_{\text{aspect}, t} = \exp\left(-\gamma_{ar} \cdot \left|\frac{AR_t - AR_{t-1}}{AR_{t-1}}\right|\right)$$
5. **Temporal Persistence Stability ($S_{\text{pers}, t}$)**:
   $$S_{\text{pers}, t} = \min\left(1.0, \frac{N_{\text{consecutive\_frames}}}{N_{\text{req}}}\right)$$

A detection box is deemed **stable** if $S_t \ge T_{\text{BSI}} = 0.60$.

---

## 3. Algorithm 3: Intent-Based Click Verification (ICV)

Intent-Based Click Verification (ICV) replaces 21-landmark skeletal keypoints by combining spatial aspect ratio shift, stability index, and velocity stillness conditions over a temporal frame window.

A click action is verified if and only if all four conditions hold simultaneously:

$$\mathcal{I}_t = \mathbb{I}\left[ |AR_t - AR_{\text{baseline}}| \ge T_{AR\_shift} \right] \land \mathbb{I}\left[ S_t \ge T_{\text{BSI\_click}} \right] \land \mathbb{I}\left[ v_{\text{mag}} \le T_{v\_click} \right]$$

### Consecutive Window Condition
$$N_{\text{icv}, t} = \begin{cases} N_{\text{icv}, t-1} + 1, & \text{if } \mathcal{I}_t = 1 \\ \max(0, N_{\text{icv}, t-1} - 1), & \text{otherwise} \end{cases}$$

$$\text{Trigger Click } \iff N_{\text{icv}, t} \ge N_{\text{required}} \quad \land \quad (t - t_{\text{last\_click}}) \ge T_{\text{cooldown}}$$
