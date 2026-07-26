"""
Unit Tests for Configuration Settings & Schema Validator
"""

from pathlib import Path
import pytest

from config.settings import load_config, AirOSConfig, SystemConfig, CameraConfig


def test_load_default_config():
    config = load_config("config/default_config.yaml")
    assert isinstance(config, AirOSConfig)
    assert config.system.name == "AirOS++"
    assert config.camera.width == 640
    assert config.camera.height == 480
    assert config.algorithms.ams.enabled is True
    assert config.algorithms.bsi.enabled is True
    assert config.algorithms.icv.enabled is True


def test_config_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_config("config/non_existent_config.yaml")


def test_system_config_defaults():
    sys_cfg = SystemConfig()
    assert sys_cfg.device == "cpu"
    assert sys_cfg.threads == 4
