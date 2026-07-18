import numpy as np
import pytest

from scripts.data_prep import assess_training_audio, guard_training_path


def test_invalid_demo_data_is_blocked(tmp_path):
    path = tmp_path / "data" / "_invalid_demo_data" / "metadata.csv"
    with pytest.raises(ValueError, match="sóng sin"):
        guard_training_path(path)
    guard_training_path(path, allow_invalid_demo_data=True)


def test_sine_wave_is_rejected_as_too_simple():
    sr = 24_000
    time = np.arange(sr * 3) / sr
    errors, _ = assess_training_audio(np.sin(2 * np.pi * 1_000 * time), sr)
    assert any("sóng sin" in error for error in errors)


def test_realistic_noise_is_not_marked_as_sine():
    errors, _ = assess_training_audio(np.random.default_rng(42).normal(0, 0.05, 24_000), 24_000)
    assert not any("sóng sin" in error for error in errors)
