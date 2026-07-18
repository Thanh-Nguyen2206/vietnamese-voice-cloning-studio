"""Objective TTS evaluation primitives with lazy optional dependencies."""

from __future__ import annotations

import csv
import json
import statistics
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf

from .audio_utils import audio_metrics


def normalize_metric_text(text: str, *, ignore_case_punctuation: bool = False) -> str:
    """NFC-normalize comparison text; Vietnamese accents are always preserved."""

    import re

    value = unicodedata.normalize("NFC", text or "")
    value = re.sub(r"\s+", " ", value).strip()
    if ignore_case_punctuation:
        value = value.lower()
        value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
        value = re.sub(r"\s+", " ", value).strip()
    return value


def _distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, ref_item in enumerate(reference, start=1):
        current = [i]
        for j, hyp_item in enumerate(hypothesis, start=1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ref_item != hyp_item)))
        previous = current
    return previous[-1]


def text_error_rates(reference: str, hypothesis: str, *, relaxed: bool = False) -> tuple[float, float]:
    """Return WER and CER, preferring jiwer and using an equivalent local fallback."""

    ref = normalize_metric_text(reference, ignore_case_punctuation=relaxed)
    hyp = normalize_metric_text(hypothesis, ignore_case_punctuation=relaxed)
    if not ref:
        raise ValueError("Reference text không được rỗng")
    try:
        from jiwer import cer, wer

        return float(wer(ref, hyp)), float(cer(ref, hyp))
    except ImportError:
        ref_words, hyp_words = ref.split(), hyp.split()
        return (
            _distance(ref_words, hyp_words) / max(len(ref_words), 1),
            _distance(list(ref), list(hyp)) / max(len(ref), 1),
        )


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=np.float64).reshape(-1)
    b = np.asarray(second, dtype=np.float64).reshape(-1)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("Speaker embeddings phải cùng shape và không rỗng")
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        raise ValueError("Speaker embedding có norm bằng 0")
    return float(np.dot(a, b) / denom)


class WhisperTranscriber:
    """Lazy faster-whisper wrapper so unit tests never download a model."""

    def __init__(self, model_name: str, device: str) -> None:
        self.model_name = model_name
        self.device = "cuda" if device == "cuda" else "cpu"
        self._model: Any = None

    def __call__(self, audio_path: Path) -> str:
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError("Thiếu faster-whisper; chạy pip install faster-whisper") from exc
            compute = "float16" if self.device == "cuda" else "int8"
            self._model = WhisperModel(self.model_name, device=self.device, compute_type=compute)
        segments, _ = self._model.transcribe(str(audio_path), language="vi", beam_size=5)
        return " ".join(segment.text.strip() for segment in segments).strip()


class SpeakerEncoder:
    """Lazy Resemblyzer encoder. Similarity is relative, not biometric proof."""

    def __init__(self, device: str) -> None:
        self.device = device
        self._encoder: Any = None

    def __call__(self, audio_path: Path) -> np.ndarray:
        try:
            from resemblyzer import VoiceEncoder, preprocess_wav
        except ImportError as exc:
            raise RuntimeError("Thiếu resemblyzer; chạy pip install resemblyzer") from exc
        if self._encoder is None:
            self._encoder = VoiceEncoder(device=self.device)
        return np.asarray(self._encoder.embed_utterance(preprocess_wav(audio_path)))


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    reference_text: str
    generated_audio: Path
    engine: str
    reference_audio: Path | None = None
    inference_time: float | None = None
    checkpoint: str | None = None
    seed: int | None = None
    nfe: int | None = None
    metadata: dict[str, Any] | None = None


def load_manifest(path: Path) -> list[EvaluationCase]:
    """Load and validate the JSON evaluation manifest."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Không đọc được manifest {path}: {exc}") from exc
    rows = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("Manifest phải có mảng 'cases' không rỗng")
    base = path.parent
    seen: set[str] = set()
    cases: list[EvaluationCase] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Case {index} phải là object")
        missing = [key for key in ("id", "reference_text", "generated_audio", "engine") if not row.get(key)]
        if missing:
            raise ValueError(f"Case {index} thiếu field: {', '.join(missing)}")
        case_id = str(row["id"])
        if case_id in seen:
            raise ValueError(f"Case ID bị trùng: {case_id}")
        seen.add(case_id)
        generated = Path(str(row["generated_audio"]))
        reference = Path(str(row["reference_audio"])) if row.get("reference_audio") else None
        cases.append(
            EvaluationCase(
                case_id=case_id,
                reference_text=str(row["reference_text"]),
                generated_audio=generated if generated.is_absolute() else (base / generated).resolve(),
                reference_audio=(
                    reference
                    if reference and reference.is_absolute()
                    else (base / reference).resolve()
                    if reference
                    else None
                ),
                engine=str(row["engine"]),
                inference_time=float(row["inference_time"])
                if row.get("inference_time") is not None
                else None,
                checkpoint=row.get("checkpoint"),
                seed=row.get("seed"),
                nfe=row.get("nfe"),
                metadata=row.get("metadata") or {},
            )
        )
    return cases


def evaluate_cases(
    cases: list[EvaluationCase],
    transcribe: Callable[[Path], str],
    embed: Callable[[Path], np.ndarray] | None,
    *,
    relaxed: bool = False,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    embedding_cache: dict[Path, np.ndarray] = {}
    for case in cases:
        row: dict[str, Any] = {
            "case_id": case.case_id,
            "engine": case.engine,
            "status": "error",
            "reference_text": case.reference_text,
            "generated_audio": str(case.generated_audio),
            "reference_audio": str(case.reference_audio or ""),
            "checkpoint": case.checkpoint,
            "seed": case.seed,
            "nfe": case.nfe,
            "inference_time": case.inference_time,
        }
        try:
            if not case.generated_audio.is_file():
                raise FileNotFoundError(f"Không tìm thấy generated audio: {case.generated_audio}")
            wave, sr = sf.read(case.generated_audio, dtype="float32", always_2d=False)
            metrics = audio_metrics(wave, sr).to_dict()
            transcript = transcribe(case.generated_audio)
            wer, cer = text_error_rates(case.reference_text, transcript, relaxed=relaxed)
            row.update(metrics)
            row.update(
                {
                    "transcript": transcript,
                    "wer": wer,
                    "cer": cer,
                    "real_time_factor": (
                        case.inference_time / metrics["duration"]
                        if case.inference_time is not None and metrics["duration"]
                        else None
                    ),
                }
            )
            if case.reference_audio:
                if embed is None:
                    raise RuntimeError("Speaker encoder không khả dụng")
                if not case.reference_audio.is_file():
                    raise FileNotFoundError(f"Không tìm thấy reference audio: {case.reference_audio}")
                ref_embedding = embedding_cache.setdefault(case.reference_audio, embed(case.reference_audio))
                row["speaker_similarity"] = cosine_similarity(ref_embedding, embed(case.generated_audio))
            else:
                row["speaker_similarity"] = None
            row["status"] = "ok"
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        results.append(row)
    return results


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def summarize(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        grouped[str(row["engine"])].append(row)
    summaries: list[dict[str, Any]] = []
    for engine, rows in sorted(grouped.items()):
        good = [row for row in rows if row["status"] == "ok"]

        def values(key: str) -> list[float]:
            return [float(row[key]) for row in good if row.get(key) is not None]
        summaries.append(
            {
                "engine": engine,
                "samples": len(rows),
                "failures": len(rows) - len(good),
                "mean_wer": _mean(values("wer")),
                "median_wer": _median(values("wer")),
                "mean_cer": _mean(values("cer")),
                "median_cer": _median(values("cer")),
                "mean_speaker_similarity": _mean(values("speaker_similarity")),
                "mean_inference_time": _mean(values("inference_time")),
                "mean_real_time_factor": _mean(values("real_time_factor")),
            }
        )
    return summaries


def write_reports(results: list[dict[str, Any]], output_dir: Path, manifest: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = summarize(results)
    (output_dir / "results.json").write_text(
        json.dumps(
            {"manifest": str(manifest), "results": results, "summary": summaries},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    fields = sorted({key for row in results for key in row})
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    def value(item: Any) -> str:
        return "—" if item is None else f"{item:.4f}" if isinstance(item, float) else str(item)

    lines = [
        "# Báo cáo đánh giá TTS",
        "",
        f"Manifest: `{manifest}`",
        "",
        "> Speaker similarity là metric tương đối, không phải bằng chứng nhận dạng sinh trắc học.",
        "",
        "| Engine | Mẫu | WER mean/median | CER mean/median | Similarity | Inference (s) | RTF | Lỗi |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['engine']} | {row['samples']} | {value(row['mean_wer'])} / {value(row['median_wer'])} | "
            f"{value(row['mean_cer'])} / {value(row['median_cer'])} | "
            f"{value(row['mean_speaker_similarity'])} | {value(row['mean_inference_time'])} | "
            f"{value(row['mean_real_time_factor'])} | {row['failures']} |"
        )
    failures = [row for row in results if row["status"] != "ok"]
    if failures:
        lines += ["", "## Lỗi từng mẫu", ""] + [
            f"- `{row['case_id']}`: {row.get('error')}" for row in failures
        ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
