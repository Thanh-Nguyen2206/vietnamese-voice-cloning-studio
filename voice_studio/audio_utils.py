"""Small audio validation, metric and concatenation utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class AudioMetrics:
    duration: float
    rms: float
    peak: float
    clipping_ratio: float
    spectral_flatness: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def validate_waveform(wave: np.ndarray, sample_rate: int) -> list[str]:
    """Return quality warnings; invalid numeric or empty waveforms raise errors."""

    array = np.asarray(wave)
    if array.ndim not in {1, 2} or array.size == 0:
        raise ValueError("Audio phải là waveform không rỗng, mono hoặc stereo")
    if sample_rate <= 0:
        raise ValueError("Sample rate phải lớn hơn 0")
    if not np.isfinite(array).all():
        raise ValueError("Audio chứa NaN hoặc Inf")
    mono = array.mean(axis=1) if array.ndim == 2 else array
    rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64))))
    warnings: list[str] = []
    if rms < 1e-4:
        warnings.append("audio gần im lặng")
    clipping = float(np.mean(np.abs(mono) >= 0.999))
    if clipping > 0.01:
        warnings.append(f"clipping cao ({clipping:.1%})")
    if array.ndim == 2:
        warnings.append("audio stereo cần chuyển mono")
    return warnings


def spectral_flatness(wave: np.ndarray) -> float:
    mono = np.asarray(wave, dtype=np.float64).reshape(-1)
    if mono.size < 256:
        return 0.0
    spectrum = np.abs(np.fft.rfft(mono * np.hanning(mono.size))) ** 2 + 1e-12
    return float(np.exp(np.mean(np.log(spectrum))) / np.mean(spectrum))


def audio_metrics(wave: np.ndarray, sample_rate: int) -> AudioMetrics:
    validate_waveform(wave, sample_rate)
    mono = np.asarray(wave, dtype=np.float32)
    if mono.ndim == 2:
        mono = mono.mean(axis=1)
    return AudioMetrics(
        duration=float(mono.size / sample_rate),
        rms=float(np.sqrt(np.mean(np.square(mono, dtype=np.float64)))),
        peak=float(np.max(np.abs(mono))),
        clipping_ratio=float(np.mean(np.abs(mono) >= 0.999)),
        spectral_flatness=spectral_flatness(mono),
    )


def peak_normalize(wave: np.ndarray, target: float = 0.99) -> np.ndarray:
    array = np.nan_to_num(np.asarray(wave, dtype=np.float32))
    peak = float(np.max(np.abs(array))) if array.size else 0.0
    return array / peak * target if peak > target else array


def concatenate_audio(chunks: list[np.ndarray], sample_rate: int, silence_ms: int = 180) -> np.ndarray:
    """Concatenate mono chunks with configurable silence and final peak normalization."""

    if not chunks:
        raise ValueError("Không có audio chunk để ghép")
    if not 0 <= silence_ms <= 2_000:
        raise ValueError("silence_ms phải nằm trong [0, 2000]")
    arrays = [np.asarray(chunk, dtype=np.float32).reshape(-1) for chunk in chunks]
    silence = np.zeros(round(sample_rate * silence_ms / 1000), dtype=np.float32)
    parts: list[np.ndarray] = []
    for index, chunk in enumerate(arrays):
        parts.append(chunk)
        if index + 1 < len(arrays) and silence.size:
            parts.append(silence)
    return peak_normalize(np.concatenate(parts))
