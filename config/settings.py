"""
AirOS++ Configuration Settings Loader & Schema Validator
Author: Senior AI Architecture Team
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class SystemConfig:
    name: str = "AirOS++"
    version: str = "1.0.0"
    log_level: str = "INFO"
    log_to_file: bool = True
    log_file_path: str = "airos.log"
    device: str = "cpu"
    threads: int = 4


@dataclass
class CameraConfig:
    device_id: int = 0
    width: int = 640
    height: int = 480
    target_fps: int = 30
    buffer_size: int = 1
    auto_reconnect: bool = True
    reconnect_delay_sec: float = 2.0


@dataclass
class ModelConfig:
    yolo_model_path: str = "models/yolov10s.pt"
    fallback_to_mock: bool = True
    confidence_threshold: float = 0.45
    iou_threshold: float = 0.45
    input_size: int = 320


@dataclass
class DetectorConfig:
    max_hands: int = 2
    temporal_memory_frames: int = 5
    extrapolation_max_frames: int = 3


@dataclass
class AMSConfig:
    enabled: bool = True
    alpha_min: float = 0.15
    alpha_max: float = 0.85
    lambda_speed: float = 0.05
    velocity_deadzone: float = 1.5


@dataclass
class BSIWeightsConfig:
    confidence: float = 0.25
    displacement: float = 0.25
    area_consistency: float = 0.20
    aspect_ratio_consistency: float = 0.15
    temporal_persistence: float = 0.15


@dataclass
class BSIDecayConfig:
    gamma_displacement: float = 0.08
    gamma_area: float = 2.0
    gamma_aspect: float = 2.0


@dataclass
class BSIConfig:
    enabled: bool = True
    threshold: float = 0.60
    weights: BSIWeightsConfig = field(default_factory=BSIWeightsConfig)
    decay_params: BSIDecayConfig = field(default_factory=BSIDecayConfig)
    persistence_required_frames: int = 5


@dataclass
class ICVConfig:
    enabled: bool = True
    aspect_ratio_shift_threshold: float = 0.15
    bsi_threshold: float = 0.65
    max_velocity_threshold: float = 4.0
    consecutive_frames_required: int = 4
    cooldown_sec: float = 0.35


@dataclass
class AlgorithmsConfig:
    ams: AMSConfig = field(default_factory=AMSConfig)
    bsi: BSIConfig = field(default_factory=BSIConfig)
    icv: ICVConfig = field(default_factory=ICVConfig)


@dataclass
class GestureConfig:
    cursor_hand: str = "left"
    action_hand: str = "right"
    cursor_margin_ratio: float = 0.10
    scroll_speed_multiplier: float = 12.0
    volume_sensitivity: float = 1.5
    brightness_sensitivity: float = 1.5
    distance_min_px: float = 50.0
    distance_max_px: float = 400.0


@dataclass
class ControllerConfig:
    pyautogui_failsafe: bool = True
    dry_run: bool = False
    action_cooldown_ms: int = 100


@dataclass
class HUDConfig:
    enabled: bool = True
    show_fps: bool = True
    show_latency: bool = True
    show_confidence: bool = True
    show_bsi: bool = True
    show_ams_alpha: bool = True
    show_system_usage: bool = True
    show_bounding_boxes: bool = True
    show_centroids: bool = True
    box_color_stable: list[int] = field(default_factory=lambda: [0, 255, 0])
    box_color_unstable: list[int] = field(default_factory=lambda: [0, 0, 255])


@dataclass
class AirOSConfig:
    system: SystemConfig = field(default_factory=SystemConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    algorithms: AlgorithmsConfig = field(default_factory=AlgorithmsConfig)
    gesture: GestureConfig = field(default_factory=GestureConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    hud: HUDConfig = field(default_factory=HUDConfig)


def load_config(config_path: str | Path = "config/default_config.yaml") -> AirOSConfig:
    """Loads, parses, and validates the AirOS configuration file."""
    path = Path(config_path)
    if not path.is_absolute():
        # Resolve relative to current directory or fallback
        path = Path.cwd() / path

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw_data: Dict[str, Any] = yaml.safe_load(f) or {}

    # Validate against JSON schema if present
    schema_path = path.parent / "config_schema.json"
    if schema_path.exists():
        try:
            import jsonschema

            with open(schema_path, "r", encoding="utf-8") as sf:
                schema = json.load(sf)
            jsonschema.validate(instance=raw_data, schema=schema)
        except ImportError:
            pass  # jsonschema optional for fallback
        except Exception as e:
            raise ValueError(f"Configuration schema validation failed: {e}")

    # Build dataclass instance safely
    sys_cfg = SystemConfig(**raw_data.get("system", {}))
    cam_cfg = CameraConfig(**raw_data.get("camera", {}))
    mdl_cfg = ModelConfig(**raw_data.get("model", {}))
    det_cfg = DetectorConfig(**raw_data.get("detector", {}))

    alg_raw = raw_data.get("algorithms", {})
    ams_cfg = AMSConfig(**alg_raw.get("ams", {}))

    bsi_raw = alg_raw.get("bsi", {})
    bsi_weights = BSIWeightsConfig(**bsi_raw.get("weights", {}))
    bsi_decay = BSIDecayConfig(**bsi_raw.get("decay_params", {}))
    bsi_cfg = BSIConfig(
        enabled=bsi_raw.get("enabled", True),
        threshold=bsi_raw.get("threshold", 0.60),
        weights=bsi_weights,
        decay_params=bsi_decay,
        persistence_required_frames=bsi_raw.get("persistence_required_frames", 5),
    )

    icv_cfg = ICVConfig(**alg_raw.get("icv", {}))
    alg_cfg = AlgorithmsConfig(ams=ams_cfg, bsi=bsi_cfg, icv=icv_cfg)

    gst_cfg = GestureConfig(**raw_data.get("gesture", {}))
    ctrl_cfg = ControllerConfig(**raw_data.get("controller", {}))
    hud_cfg = HUDConfig(**raw_data.get("hud", {}))

    return AirOSConfig(
        system=sys_cfg,
        camera=cam_cfg,
        model=mdl_cfg,
        detector=det_cfg,
        algorithms=alg_cfg,
        gesture=gst_cfg,
        controller=ctrl_cfg,
        hud=hud_cfg,
    )
