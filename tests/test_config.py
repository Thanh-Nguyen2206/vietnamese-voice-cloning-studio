from pathlib import Path

import pytest

from voice_studio.config import StudioConfig, load_config


def test_config_round_trip():
    config = StudioConfig(output_dir=Path("out"), cache_dir=Path("cache")).validate()
    assert StudioConfig.from_dict(config.to_dict()).to_dict() == config.to_dict()


def test_environment_config_and_validation(tmp_path):
    config = load_config(tmp_path, {"VVCS_DEVICE": "cpu", "VVCS_NFE": "48", "VVCS_OFFLINE": "true"})
    assert config.device == "cpu" and config.nfe == 48 and config.offline
    with pytest.raises(ValueError, match="device"):
        StudioConfig(device="tpu").validate()
