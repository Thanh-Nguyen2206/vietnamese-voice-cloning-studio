"""Typed application configuration sourced from environment variables."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping


def _bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class StudioConfig:
    """Runtime settings shared by the UI, inference and evaluation tools."""

    model_id: str = "hynt/F5-TTS-Vietnamese-ViVoice"
    device: str = "auto"
    cache_dir: Path | None = None
    output_dir: Path = Path("outputs")
    sample_rate: int = 24_000
    nfe: int = 32
    seed: int = 42
    max_text_chars: int = 10_000
    max_chunk_chars: int = 280
    chunk_silence_ms: int = 180
    engine_timeout: float = 40.0
    whisper_model: str = "small"
    offline: bool = False
    enable_cloud_engines: bool = True

    def validate(self) -> "StudioConfig":
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device phải là auto, cpu, cuda hoặc mps")
        if self.sample_rate <= 0 or self.nfe <= 0:
            raise ValueError("sample_rate và nfe phải lớn hơn 0")
        if not 80 <= self.max_chunk_chars <= self.max_text_chars:
            raise ValueError("max_chunk_chars phải từ 80 đến max_text_chars")
        if not 0 <= self.chunk_silence_ms <= 2_000:
            raise ValueError("chunk_silence_ms phải nằm trong [0, 2000]")
        if self.engine_timeout <= 0:
            raise ValueError("engine_timeout phải lớn hơn 0")
        return self

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["cache_dir"] = str(self.cache_dir) if self.cache_dir else None
        data["output_dir"] = str(self.output_dir)
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "StudioConfig":
        values = dict(data)
        if values.get("cache_dir"):
            values["cache_dir"] = Path(str(values["cache_dir"]))
        values["output_dir"] = Path(str(values.get("output_dir", "outputs")))
        return cls(**values).validate()


def load_config(root: Path | None = None, environ: Mapping[str, str] | None = None) -> StudioConfig:
    """Load config from ``VVCS_*`` variables, applying safe defaults."""

    env = os.environ if environ is None else environ
    root = (root or Path.cwd()).resolve()
    cache = env.get("VVCS_CACHE_DIR", "").strip()
    output = Path(env.get("VVCS_OUTPUT_DIR", "outputs"))
    if not output.is_absolute():
        output = root / output
    cfg = StudioConfig(
        model_id=env.get("VVCS_MODEL_ID", StudioConfig.model_id),
        device=env.get("VVCS_DEVICE", "auto").strip().lower() or "auto",
        cache_dir=Path(cache).expanduser() if cache else None,
        output_dir=output,
        sample_rate=int(env.get("VVCS_SAMPLE_RATE", StudioConfig.sample_rate)),
        nfe=int(env.get("VVCS_NFE", StudioConfig.nfe)),
        seed=int(env.get("VVCS_SEED", StudioConfig.seed)),
        max_text_chars=int(env.get("VVCS_MAX_TEXT_CHARS", StudioConfig.max_text_chars)),
        max_chunk_chars=int(env.get("VVCS_MAX_CHUNK_CHARS", StudioConfig.max_chunk_chars)),
        chunk_silence_ms=int(env.get("VVCS_CHUNK_SILENCE_MS", StudioConfig.chunk_silence_ms)),
        engine_timeout=float(env.get("VVCS_ENGINE_TIMEOUT", StudioConfig.engine_timeout)),
        whisper_model=env.get("VVCS_WHISPER_MODEL", StudioConfig.whisper_model),
        offline=_bool(env.get("VVCS_OFFLINE", "0")),
        enable_cloud_engines=_bool(env.get("VVCS_ENABLE_CLOUD_ENGINES", "1")),
    )
    return cfg.validate()
