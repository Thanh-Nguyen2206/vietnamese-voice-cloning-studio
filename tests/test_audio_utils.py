import numpy as np
import pytest

from voice_studio.audio_utils import concatenate_audio, validate_waveform


def test_nan_and_inf_are_rejected():
    with pytest.raises(ValueError, match="NaN"):
        validate_waveform(np.array([0.0, np.nan, np.inf]), 24_000)


def test_near_silence_is_detected():
    warnings = validate_waveform(np.zeros(24_000, dtype=np.float32), 24_000)
    assert any("im lặng" in warning for warning in warnings)


def test_high_clipping_is_warned():
    warnings = validate_waveform(np.ones(24_000, dtype=np.float32), 24_000)
    assert any("clipping" in warning for warning in warnings)


def test_concatenation_adds_silence_and_peak_normalizes():
    wave = concatenate_audio([np.ones(100), np.ones(100) * -2], 1_000, silence_ms=100)
    assert len(wave) == 300
    assert np.max(np.abs(wave)) <= 0.991
    assert np.all(wave[100:200] == 0)
