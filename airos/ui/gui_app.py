"""
AirOS++ Desktop Control Center GUI Application
Built with Tkinter, PIL, and OpenCV for real-time video display, live parameter tuning, system diagnostics, and benchmark management.
"""

import json
from pathlib import Path
import queue
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Optional, Tuple, List

import cv2
import numpy as np
from PIL import Image, ImageTk

from config.settings import load_config, AirOSConfig
from airos.algorithms.ams import AdaptiveMotionSmoothing
from airos.algorithms.bsi import BoundingBoxStabilityIndex
from airos.algorithms.icv import IntentBasedClickVerification
from airos.controller.os_controller import OSController
from airos.detector.hand_detector import HandDetector
from airos.gesture.gesture_interpreter import GestureInterpreter
from airos.logger.airos_logger import get_logger, setup_logger
from airos.tracking.hand_tracker import HandTracker
from airos.ui.hud import HUDOverlay
from airos.utilities.camera import ThreadedCameraReader
from airos.utilities.metrics import PerformanceMetricsCollector
from airos.utilities.system_info import SystemDiagnostics

logger = get_logger()


class AirOSDesktopApp:
    """AirOS++ Standalone Desktop GUI Application."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("AirOS++ Control Center | Touchless HCI Framework")
        self.root.geometry("1320x840")
        self.root.minsize(1024, 720)

        # Style Configuration
        self._setup_theme()

        # Load Default Configuration
        self.config_path = "config/default_config.yaml"
        self.config: AirOSConfig = load_config(self.config_path)

        # Pipeline Subsystems
        self.is_running = False
        self.pipeline_thread: Optional[threading.Thread] = None
        self.camera_reader: Optional[ThreadedCameraReader] = None

        self.detector: Optional[HandDetector] = None
        self.tracker: Optional[HandTracker] = None
        self.ams: Optional[AdaptiveMotionSmoothing] = None
        self.bsi: Optional[BoundingBoxStabilityIndex] = None
        self.icv: Optional[IntentBasedClickVerification] = None
        self.gesture_interpreter: Optional[GestureInterpreter] = None
        self.os_controller: Optional[OSController] = None
        self.hud_overlay: Optional[HUDOverlay] = None
        self.metrics_collector: Optional[PerformanceMetricsCollector] = None

        # Thread Queue for Video Frames & Logs
        self.frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self.log_queue: queue.Queue = queue.Queue(maxsize=100)

        # Build GUI Layout
        self._build_ui()

        # Start periodic GUI updates
        self.root.after(30, self._update_gui_loop)

    def _setup_theme(self):
        style = ttk.Style()
        style.theme_use("clam")

        # Dark Theme Color Palette
        bg_dark = "#181825"
        panel_bg = "#1e1e2e"
        fg_text = "#cdd6f4"
        accent_blue = "#89b4fa"

        self.root.configure(bg=bg_dark)
        style.configure(".", background=panel_bg, foreground=fg_text, font=("Segoe UI", 10))
        style.configure("TNotebook", background=bg_dark, borderwidth=0)
        style.configure("TNotebook.Tab", background="#313244", foreground=fg_text, padding=[12, 6])
        style.map("TNotebook.Tab", background=[("selected", accent_blue)], foreground=[("selected", "#11111b")])
        style.configure("TFrame", background=panel_bg)
        style.configure("TLabelframe", background=panel_bg, foreground=accent_blue)
        style.configure("TLabelframe.Label", background=panel_bg, foreground=accent_blue, font=("Segoe UI", 10, "bold"))
        style.configure("Header.TLabel", background=bg_dark, foreground=accent_blue, font=("Segoe UI", 16, "bold"))
        style.configure("Status.TLabel", background="#11111b", foreground="#a6e3a1", font=("Consolas", 10))

    def _build_ui(self):
        # 1. Top Navigation Bar
        top_bar = tk.Frame(self.root, bg="#181825", height=50)
        top_bar.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        title_lbl = ttk.Label(top_bar, text="AirOS++ Touchless Desktop Control Center", style="Header.TLabel")
        title_lbl.pack(side=tk.LEFT, padx=10, pady=5)

        self.btn_toggle_pipeline = tk.Button(
            top_bar,
            text="▶ START PIPELINE",
            bg="#a6e3a1",
            fg="#11111b",
            font=("Segoe UI", 11, "bold"),
            relief=tk.FLAT,
            padx=15,
            pady=5,
            command=self.toggle_pipeline,
        )
        self.btn_toggle_pipeline.pack(side=tk.RIGHT, padx=10)

        # 2. Main Body Split Container
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Left Container: Live Camera View & Status Card
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=3)

        video_box = ttk.LabelFrame(left_frame, text="Live Camera & Telemetry Video Feed")
        video_box.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.video_label = tk.Label(video_box, bg="#11111b", text="Camera Idle. Click ▶ START PIPELINE to begin.")
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Live Status Telemetry Bar
        self.status_var = tk.StringVar(value="Status: IDLE | FPS: 0.0 | Latency: 0.0ms | CPU: 0.0% | RAM: 0.0%")
        status_lbl = ttk.Label(left_frame, textvariable=self.status_var, style="Status.TLabel", padding=6)
        status_lbl.pack(fill=tk.X, padx=5, pady=2)

        # Right Container: Multi-Tab Control Panel
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=2)

        notebook = ttk.Notebook(right_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Tab 1: System Controls
        tab_controls = ttk.Frame(notebook)
        notebook.add(tab_controls, text="Controls & Modes")
        self._build_tab_controls(tab_controls)

        # Tab 2: Visual Gesture Manual
        tab_manual = ttk.Frame(notebook)
        notebook.add(tab_manual, text="Gesture Manual")
        self._build_tab_manual(tab_manual)

        # Tab 3: Live Parameters Tuning
        tab_params = ttk.Frame(notebook)
        notebook.add(tab_params, text="Algorithm Tuning")
        self._build_tab_params(tab_params)

        # Tab 4: Diagnostics & Benchmark
        tab_diag = ttk.Frame(notebook)
        notebook.add(tab_diag, text="Benchmark & Tests")
        self._build_tab_diag(tab_diag)

    def _build_tab_controls(self, parent):
        box_mode = ttk.LabelFrame(parent, text="Execution Mode & Camera Selection")
        box_mode.pack(fill=tk.X, padx=10, pady=8)

        # Camera Source Selection
        ttk.Label(box_mode, text="Camera Source:").grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
        self.cam_var = tk.StringVar(value="Webcam Device 0")
        cam_combo = ttk.Combobox(
            box_mode,
            textvariable=self.cam_var,
            values=["Webcam Device 0", "Webcam Device 1", "Synthetic Mock Generator"],
            state="readonly",
        )
        cam_combo.grid(row=0, column=1, sticky=tk.EW, padx=10, pady=5)

        # User Handedness Preference
        ttk.Label(box_mode, text="Primary Hand:").grid(row=1, column=0, sticky=tk.W, padx=10, pady=5)
        self.hand_var = tk.StringVar(value="Right-Handed User")
        hand_combo = ttk.Combobox(
            box_mode,
            textvariable=self.hand_var,
            values=["Right-Handed User", "Left-Handed User"],
            state="readonly",
        )
        hand_combo.grid(row=1, column=1, sticky=tk.EW, padx=10, pady=5)
        hand_combo.bind("<<ComboboxSelected>>", self._on_hand_preference_changed)

        # Dry-Run Checkbox
        self.dry_run_var = tk.BooleanVar(value=True)
        chk_dry = ttk.Checkbutton(
            box_mode, text="Dry-Run Mode (Simulate without moving real host OS cursor)", variable=self.dry_run_var
        )
        chk_dry.grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=10, pady=5)

        # HUD Toggle
        self.hud_var = tk.BooleanVar(value=True)
        chk_hud = ttk.Checkbutton(box_mode, text="Enable OpenCV Telemetry HUD Overlay", variable=self.hud_var)
        chk_hud.grid(row=3, column=0, columnspan=2, sticky=tk.W, padx=10, pady=5)

        # Quick Actions Simulator
        box_actions = ttk.LabelFrame(parent, text="Quick Gesture Actions Simulator")
        box_actions.pack(fill=tk.X, padx=10, pady=8)

        grid_frame = ttk.Frame(box_actions)
        grid_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(grid_frame, text="Left Click", command=lambda: self._simulate_click("Left Click")).grid(row=0, column=0, padx=4, pady=4, sticky=tk.EW)
        ttk.Button(grid_frame, text="Right Click", command=lambda: self._simulate_click("Right Click")).grid(row=0, column=1, padx=4, pady=4, sticky=tk.EW)
        ttk.Button(grid_frame, text="Double Click", command=lambda: self._simulate_click("Double Click")).grid(row=1, column=0, padx=4, pady=4, sticky=tk.EW)
        ttk.Button(grid_frame, text="Middle Click", command=lambda: self._simulate_click("Middle Click")).grid(row=1, column=1, padx=4, pady=4, sticky=tk.EW)
        ttk.Button(grid_frame, text="Hold Drag", command=lambda: self._simulate_click("Drag Start")).grid(row=2, column=0, padx=4, pady=4, sticky=tk.EW)
        ttk.Button(grid_frame, text="Release Drag", command=lambda: self._simulate_click("Drag End")).grid(row=2, column=1, padx=4, pady=4, sticky=tk.EW)

    def _build_tab_manual(self, parent):
        box = ttk.LabelFrame(parent, text="Interactive Visual Gesture Guide")
        box.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        txt_manual = tk.Text(box, bg="#11111b", fg="#cdd6f4", font=("Segoe UI", 10), wrap=tk.WORD)
        txt_manual.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        manual_text = """
====================================================================
                     AIR-OS++ GESTURE GUIDE
====================================================================

1. 🖱️ CURSOR NAVIGATION (Move Mouse)
   • Action: Raise your Navigation Hand and move it across the camera view.
   • Result: The mouse cursor tracks your hand smoothly across the desktop.

2. 👈 LEFT CLICK (Tap / Pinch)
   • Action: Form a quick Tight Fist or Pinch with your Action Hand and hold still for 0.15s.
   • Result: Triggers a standard Left Mouse Click.

3. 👉 RIGHT CLICK (Wide Palm)
   • Action: Open your Action Hand horizontally into a Wide Flat Palm.
   • Result: Triggers a Right Mouse Click (Context Menu).

4. ✌️ DOUBLE CLICK (Double Quick Tap)
   • Action: Perform two quick fist pulses/taps within 0.45 seconds.
   • Result: Triggers a Double Click (Opens files/folders).

5. 🖐️ DRAG & DROP (Click & Hold)
   • Action: Hold a tight fist continuously for > 0.65 seconds to grab.
            Move your hand to drag. Open your palm to drop/release.
   • Result: Holds down left mouse button while dragging, releases on open hand.

6. 👆 MIDDLE CLICK (Vertical Hand)
   • Action: Hold your hand in a narrow vertical posture.
   • Result: Triggers a Middle Mouse Click.

7. 🔊 MASTER VOLUME (Dual-Hand Stretch)
   • Action: Raise both hands. Move them apart horizontally to increase volume;
            bring them closer together to decrease volume.

8. ☀️ DISPLAY BRIGHTNESS (Dual-Hand Elevation)
   • Action: Raise both hands. Raise your Right Hand above your Left Hand to brighten;
            lower your Right Hand to dim.

9. 🔇 MUTE AUDIO TOGGLE (Cross Hands)
   • Action: Cross both hands horizontally in front of camera.
====================================================================
"""
        txt_manual.insert(tk.END, manual_text)

    def _build_tab_params(self, parent):
        # Preset Selection
        box_preset = ttk.LabelFrame(parent, text="Preset Tuning Profile")
        box_preset.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(box_preset, text="Sensitivity Profile:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.preset_var = tk.StringVar(value="Balanced (Default)")
        preset_combo = ttk.Combobox(
            box_preset,
            textvariable=self.preset_var,
            values=["Gentle / Easy Mode", "Balanced (Default)", "Precision Expert Mode"],
            state="readonly",
        )
        preset_combo.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        preset_combo.bind("<<ComboboxSelected>>", self._on_preset_changed)

        # AMS Tuning
        box_ams = ttk.LabelFrame(parent, text="Algorithm 1: Adaptive Motion Smoothing (AMS)")
        box_ams.pack(fill=tk.X, padx=10, pady=6)

        ttk.Label(box_ams, text="Alpha Min (Stationary Smoothing):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=4)
        self.slider_alpha_min = tk.Scale(
            box_ams, from_=0.01, to=0.50, resolution=0.01, orient=tk.HORIZONTAL, bg="#1e1e2e", fg="#cdd6f4"
        )
        self.slider_alpha_min.set(self.config.algorithms.ams.alpha_min)
        self.slider_alpha_min.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=4)

        ttk.Label(box_ams, text="Alpha Max (Rapid Motion Response):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=4)
        self.slider_alpha_max = tk.Scale(
            box_ams, from_=0.50, to=1.00, resolution=0.01, orient=tk.HORIZONTAL, bg="#1e1e2e", fg="#cdd6f4"
        )
        self.slider_alpha_max.set(self.config.algorithms.ams.alpha_max)
        self.slider_alpha_max.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=4)

        # BSI Tuning
        box_bsi = ttk.LabelFrame(parent, text="Algorithm 2: Bounding Box Stability Index (BSI)")
        box_bsi.pack(fill=tk.X, padx=10, pady=6)

        ttk.Label(box_bsi, text="Stability Threshold Cutoff (T_BSI):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=4)
        self.slider_bsi_thresh = tk.Scale(
            box_bsi, from_=0.30, to=0.90, resolution=0.05, orient=tk.HORIZONTAL, bg="#1e1e2e", fg="#cdd6f4"
        )
        self.slider_bsi_thresh.set(self.config.algorithms.bsi.threshold)
        self.slider_bsi_thresh.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=4)

        # ICV Tuning
        box_icv = ttk.LabelFrame(parent, text="Algorithm 3: Intent-Based Click Verification (ICV)")
        box_icv.pack(fill=tk.X, padx=10, pady=6)

        ttk.Label(box_icv, text="Aspect Ratio Shift (Delta AR):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=4)
        self.slider_ar_shift = tk.Scale(
            box_icv, from_=0.05, to=0.40, resolution=0.01, orient=tk.HORIZONTAL, bg="#1e1e2e", fg="#cdd6f4"
        )
        self.slider_ar_shift.set(self.config.algorithms.icv.aspect_ratio_shift_threshold)
        self.slider_ar_shift.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=4)

        btn_apply = ttk.Button(parent, text="Apply Tuning Parameters", command=self._apply_parameters)
        btn_apply.pack(pady=8)

    def _build_tab_diag(self, parent):
        box_run = ttk.LabelFrame(parent, text="Automated Evaluation Suites")
        box_run.pack(fill=tk.X, padx=10, pady=10)

        btn_bench = ttk.Button(box_run, text="⚡ Run Quantitative Benchmark (300 Frames)", command=self._run_benchmark_gui)
        btn_bench.pack(fill=tk.X, padx=10, pady=8)

        btn_tests = ttk.Button(box_run, text="🧪 Run Pytest Test Suite", command=self._run_tests_gui)
        btn_tests.pack(fill=tk.X, padx=10, pady=8)

        box_out = ttk.LabelFrame(parent, text="Suite Results Summary")
        box_out.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.txt_diag = tk.Text(box_out, bg="#11111b", fg="#a6e3a1", font=("Consolas", 9), wrap=tk.WORD)
        self.txt_diag.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        sys_info = SystemDiagnostics.print_diagnostics()
        for k, v in sys_info.items():
            self.txt_diag.insert(tk.END, f"{k:<25}: {v}\n")

    def _on_hand_preference_changed(self, event):
        val = self.hand_var.get()
        if "Right" in val:
            self.config.gesture.cursor_hand = "right"
            self.config.gesture.action_hand = "left"
        else:
            self.config.gesture.cursor_hand = "left"
            self.config.gesture.action_hand = "right"

    def _on_preset_changed(self, event):
        preset = self.preset_var.get()
        if "Gentle" in preset:
            self.slider_alpha_min.set(0.25)
            self.slider_alpha_max.set(0.90)
            self.slider_bsi_thresh.set(0.50)
            self.slider_ar_shift.set(0.10)
        elif "Precision" in preset:
            self.slider_alpha_min.set(0.10)
            self.slider_alpha_max.set(0.75)
            self.slider_bsi_thresh.set(0.70)
            self.slider_ar_shift.set(0.20)
        else:
            self.slider_alpha_min.set(0.15)
            self.slider_alpha_max.set(0.85)
            self.slider_bsi_thresh.set(0.60)
            self.slider_ar_shift.set(0.15)

    def _apply_parameters(self):
        if self.config:
            self.config.algorithms.ams.alpha_min = self.slider_alpha_min.get()
            self.config.algorithms.ams.alpha_max = self.slider_alpha_max.get()
            self.config.algorithms.bsi.threshold = self.slider_bsi_thresh.get()
            self.config.algorithms.icv.aspect_ratio_shift_threshold = self.slider_ar_shift.get()

            if self.ams:
                self.ams.config.alpha_min = self.slider_alpha_min.get()
                self.ams.config.alpha_max = self.slider_alpha_max.get()
            if self.bsi:
                self.bsi.config.threshold = self.slider_bsi_thresh.get()
            if self.icv:
                self.icv.config.aspect_ratio_shift_threshold = self.slider_ar_shift.get()

            messagebox.showinfo("Parameters Updated", "Algorithm tuning parameters successfully applied to active pipeline.")

    def _simulate_click(self, click_type: str):
        self.status_var.set(f"Status: ACTIVE | Simulated Action: {click_type}")

    def _run_benchmark_gui(self):
        self.txt_diag.insert(tk.END, "\n[RUNNING BENCHMARK SIMULATION Across 300 Frames...]\n")
        self.txt_diag.see(tk.END)

        def bench_thread():
            from benchmark.run_benchmark import run_benchmark

            res = run_benchmark(num_frames=300)
            self.root.after(0, lambda: self._display_benchmark_results(res))

        threading.Thread(target=bench_thread, daemon=True).start()

    def _display_benchmark_results(self, res):
        self.txt_diag.insert(tk.END, "========================================\n")
        self.txt_diag.insert(tk.END, "       AirOS++ Benchmark Summary        \n")
        self.txt_diag.insert(tk.END, "========================================\n")
        for k, v in res.items():
            self.txt_diag.insert(tk.END, f"  {k:<25}: {v}\n")
        self.txt_diag.insert(tk.END, "========================================\n")
        self.txt_diag.see(tk.END)

    def _run_tests_gui(self):
        self.txt_diag.insert(tk.END, "\n[RUNNING PYTEST SUITE...]\n")
        self.txt_diag.see(tk.END)

        def test_thread():
            import pytest

            ret_code = pytest.main(["tests/", "-q"])
            self.root.after(0, lambda: self._display_test_results(ret_code))

        threading.Thread(target=test_thread, daemon=True).start()

    def _display_test_results(self, ret_code):
        status = "PASSED (100% Success)" if ret_code == 0 else f"COMPLETED (Exit Code: {ret_code})"
        self.txt_diag.insert(tk.END, f"\n[PYTEST EXECUTION RESULT]: {status}\n")
        self.txt_diag.see(tk.END)

    def toggle_pipeline(self):
        if not self.is_running:
            self.start_pipeline()
        else:
            self.stop_pipeline()

    def start_pipeline(self):
        self.config.controller.dry_run = self.dry_run_var.get()
        self.config.hud.enabled = self.hud_var.get()

        cam_selection = self.cam_var.get()
        mock_mode = cam_selection == "Synthetic Mock Generator"
        if cam_selection == "Webcam Device 1":
            self.config.camera.device_id = 1
        else:
            self.config.camera.device_id = 0

        self.detector = HandDetector(self.config.model, self.config.detector)
        self.tracker = HandTracker(self.config.detector)
        self.ams = AdaptiveMotionSmoothing(self.config.algorithms.ams)
        self.bsi = BoundingBoxStabilityIndex(self.config.algorithms.bsi)
        self.icv = IntentBasedClickVerification(self.config.algorithms.icv)
        self.gesture_interpreter = GestureInterpreter(self.config.gesture, self.icv)
        self.os_controller = OSController(self.config.controller)
        self.hud_overlay = HUDOverlay(self.config.hud)
        self.metrics_collector = PerformanceMetricsCollector()

        self.camera_reader = ThreadedCameraReader(self.config.camera, mock_mode=mock_mode).start()

        self.is_running = True
        self.btn_toggle_pipeline.configure(text="⏹ STOP PIPELINE", bg="#f38ba8")

        self.pipeline_thread = threading.Thread(target=self._pipeline_worker_loop, daemon=True)
        self.pipeline_thread.start()

    def stop_pipeline(self):
        self.is_running = False
        if self.camera_reader:
            self.camera_reader.stop()
            self.camera_reader = None

        self.btn_toggle_pipeline.configure(text="▶ START PIPELINE", bg="#a6e3a1")
        self.status_var.set("Status: IDLE | Pipeline Stopped")
        self.video_label.configure(image="", text="Camera Idle. Click ▶ START PIPELINE to begin.")

    def _pipeline_worker_loop(self):
        current_alpha = 0.5

        while self.is_running and self.camera_reader:
            t_start = self.metrics_collector.tick_start()
            frame = self.camera_reader.read()
            if frame is None:
                time.sleep(0.01)
                continue

            h_img, w_img = frame.shape[:2]

            raw_dets = self.detector.detect(frame)
            tracked_dets = self.tracker.update(raw_dets)

            stable_dets = []
            for det in tracked_dets:
                track = self.tracker.get_track(det.track_id) if det.track_id else None
                self.bsi.evaluate(det, track)
                if det.is_stable:
                    stable_dets.append(det)

            smoothed_cursor_pos = None
            if stable_dets:
                smoothed_pos, current_alpha = self.ams.smooth(stable_dets[0].centroid, det.timestamp)
                stable_dets[0].centroid = smoothed_pos
                smoothed_cursor_pos = smoothed_pos

            actions = self.gesture_interpreter.interpret(stable_dets, self.tracker, (w_img, h_img), frame=frame)
            for action in actions:
                self.os_controller.execute(action)

            metrics = self.metrics_collector.tick_end(t_start, smoothed_cursor_pos)

            rendered_frame = self.hud_overlay.render(
                frame, stable_dets, actions, metrics, ams_alpha=current_alpha
            )

            rgb_frame = cv2.cvtColor(rendered_frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(rgb_frame)

            img_pil = img_pil.resize((640, 480), Image.Resampling.LANCZOS)
            img_tk = ImageTk.PhotoImage(image=img_pil)

            if not self.frame_queue.full():
                self.frame_queue.put((img_tk, metrics, actions[0].description if actions else "Idle"))

    def _update_gui_loop(self):
        try:
            while not self.frame_queue.empty():
                img_tk, metrics, gesture_desc = self.frame_queue.get_nowait()
                self.video_label.configure(image=img_tk, text="")
                self.video_label.image = img_tk

                status_str = (
                    f"Status: ACTIVE | FPS: {metrics.fps:.1f} | Latency: {metrics.latency_ms:.1f}ms | "
                    f"CPU: {metrics.cpu_percent:.1f}% | RAM: {metrics.ram_percent:.1f}% | Gesture: {gesture_desc}"
                )
                self.status_var.set(status_str)
        except queue.Empty:
            pass

        self.root.after(30, self._update_gui_loop)


def launch_gui_app():
    root = tk.Tk()
    app = AirOSDesktopApp(root)
    root.mainloop()


if __name__ == "__main__":
    launch_gui_app()
