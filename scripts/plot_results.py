#!/usr/bin/env python3
"""Render evaluation charts from an evaluate.py results.json file.

Đọc ``outputs/evaluation/results.json`` (do ``scripts/evaluate.py`` sinh ra) và
xuất các biểu đồ PNG phục vụ báo cáo (Report 3 — Results Interpretation and
Visualization). Nhãn trên biểu đồ giữ dạng ASCII để tránh lỗi thiếu glyph
tiếng Việt trên các hệ thống không cài font Unicode đầy đủ.

Cách chạy:
    python scripts/plot_results.py \
        --results outputs/evaluation/results.json \
        --output-dir outputs/evaluation/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Thứ tự cố định + nhãn hiển thị cho 6 engine (giữ nhất quán giữa các biểu đồ).
ENGINE_ORDER = ["f5tts", "xtts", "mms", "piper", "edge", "bark"]
ENGINE_LABEL = {
    "f5tts": "F5-TTS",
    "xtts": "XTTS-v2",
    "mms": "MMS-TTS",
    "piper": "Piper",
    "edge": "Edge-TTS",
    "bark": "Bark",
}
# Bảng màu: F5-TTS (engine chính) nổi bật, engine nhân bản giọng khác cùng tông,
# TTS giọng cố định trung tính hơn.
COLORS = {
    "f5tts": "#2563eb",  # xanh dương đậm — engine chính
    "xtts": "#7c3aed",   # tím — voice cloning
    "mms": "#0d9488",    # teal — TTS cố định
    "piper": "#0891b2",  # cyan — TTS cố định
    "edge": "#64748b",   # xám xanh — cloud
    "bark": "#dc2626",   # đỏ — baseline yếu
}
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
})

# Chuỗi hiển thị theo ngôn ngữ (ASCII-safe cho vi để tránh thiếu glyph).
STRINGS = {
    "vi": {
        "wer_title": "Word Error Rate (thap = tot hon)",
        "cer_title": "Character Error Rate (thap = tot hon)",
        "acc_suptitle": "Do chinh xac phat am — ASR round-trip (Whisper small)",
        "sim_title": "Speaker similarity (SECS) — giong sinh ra vs audio mau (cao = giong hon)",
        "sim_ylabel": "Cosine similarity",
        "sim_band": "vung 'nhan ban giong' (>0.75)",
        "speed_title": "Toc do tao — RTF = thoi gian tao / thoi luong (thap = nhanh hon)",
        "speed_ylabel": "Real-time factor (log)",
        "rtf1": "RTF = 1 (real-time)",
        "tr_x": "Speaker similarity (giong giong audio mau) ->",
        "tr_y": "Intelligibility = 1 - WER (de hieu) ->",
        "tr_title": "Danh doi chat luong: de hieu vs giu dac trung giong\n(goc tren-phai = tot nhat; bong cang to = tao cang nhanh)",
    },
    "en": {
        "wer_title": "Word Error Rate (lower is better)",
        "cer_title": "Character Error Rate (lower is better)",
        "acc_suptitle": "Intelligibility via ASR round-trip (Whisper small)",
        "sim_title": "Speaker similarity (SECS): generated vs. reference (higher is better)",
        "sim_ylabel": "Cosine similarity",
        "sim_band": "voice-cloning region (>0.75)",
        "speed_title": "Generation speed — RTF = inference time / duration (lower is faster)",
        "speed_ylabel": "Real-time factor (log)",
        "rtf1": "RTF = 1 (real-time)",
        "tr_x": "Speaker similarity (closeness to reference voice) ->",
        "tr_y": "Intelligibility = 1 - WER ->",
        "tr_title": "Quality trade-off: intelligibility vs. speaker identity\n(top-right = best; larger bubble = faster generation)",
    },
}
_S: dict[str, str] = STRINGS["en"]


def load_summary(results_path: Path) -> dict[str, dict]:
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    by_engine = {row["engine"]: row for row in payload.get("summary", [])}
    detail = {row["engine"]: row for row in payload.get("results", []) if row.get("status") == "ok"}
    return {"summary": by_engine, "detail": detail}


def _engines_present(data: dict) -> list[str]:
    present = set(data["summary"])
    ordered = [e for e in ENGINE_ORDER if e in present]
    ordered += [e for e in sorted(present) if e not in ENGINE_ORDER]
    return ordered


def _labels(engines: list[str]) -> list[str]:
    return [ENGINE_LABEL.get(e, e) for e in engines]


def _bars(ax, engines, values, *, ylabel, title, pct=False, fmt="{:.3f}"):
    colors = [COLORS.get(e, "#3b82f6") for e in engines]
    bars = ax.bar(_labels(engines), values, color=colors, width=0.62)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold", loc="left")
    top = max(values) if values else 1
    ax.set_ylim(0, top * 1.18)
    for rect, val in zip(bars, values):
        label = f"{val*100:.1f}%" if pct else fmt.format(val)
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + top * 0.02,
                label, ha="center", va="bottom", fontsize=9.5)
    return bars


def plot_wer_cer(data: dict, out: Path) -> Path:
    engines = _engines_present(data)
    wer = [data["summary"][e]["mean_wer"] for e in engines]
    cer = [data["summary"][e]["mean_cer"] for e in engines]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    _bars(ax1, engines, wer, ylabel="WER", title=_S["wer_title"], pct=True)
    _bars(ax2, engines, cer, ylabel="CER", title=_S["cer_title"], pct=True)
    for ax in (ax1, ax2):
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle(_S["acc_suptitle"], fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = out / "wer_cer.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_speaker_similarity(data: dict, out: Path) -> Path:
    engines = _engines_present(data)
    sim = [data["summary"][e]["mean_speaker_similarity"] or 0.0 for e in engines]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    _bars(ax, engines, sim, ylabel=_S["sim_ylabel"], title=_S["sim_title"])
    ax.axhspan(0.75, 1.0, color="#22c55e", alpha=0.08)
    ax.text(len(engines) - 0.5, 0.76, _S["sim_band"],
            ha="right", va="bottom", fontsize=8.5, color="#16a34a")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    path = out / "speaker_similarity.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_speed(data: dict, out: Path) -> Path:
    engines = _engines_present(data)
    rtf = [data["summary"][e]["mean_real_time_factor"] for e in engines]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    colors = [COLORS.get(e, "#3b82f6") for e in engines]
    bars = ax.bar(_labels(engines), rtf, color=colors, width=0.62)
    ax.set_yscale("log")
    ax.set_ylabel(_S["speed_ylabel"])
    ax.set_title(_S["speed_title"], fontweight="bold", loc="left")
    ax.axhline(1.0, color="#111827", lw=1, ls="--")
    ax.text(len(engines) - 0.5, 1.05, _S["rtf1"], ha="right", va="bottom", fontsize=8.5)
    for rect, val in zip(bars, rtf):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() * 1.05,
                f"{val:.2f}x", ha="center", va="bottom", fontsize=9.5)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    path = out / "speed_rtf.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_tradeoff(data: dict, out: Path) -> Path:
    """Bieu do danh doi: de hieu (1-WER) vs giong giong mau (SECS)."""
    engines = _engines_present(data)
    # Lech nhan thu cong (offset points) de tranh chong nhau o cum giua.
    label_offset = {
        "f5tts": (8, 8), "xtts": (8, -14), "mms": (10, 14),
        "piper": (10, -18), "edge": (-58, 16), "bark": (8, 8),
    }
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    for e in engines:
        s = data["summary"][e]
        x = s["mean_speaker_similarity"] or 0.0
        y = 1.0 - s["mean_wer"]
        rtf = s["mean_real_time_factor"] or 0.1
        size = max(90.0, 900.0 / max(rtf, 0.2))  # engine nhanh → bong to
        ax.scatter(x, y, s=size, color=COLORS.get(e, "#3b82f6"), alpha=0.72,
                   edgecolors="white", linewidths=1.5, zorder=3)
        ax.annotate(ENGINE_LABEL.get(e, e), (x, y),
                    xytext=label_offset.get(e, (7, 7)),
                    textcoords="offset points", fontsize=10, fontweight="bold", zorder=4)
    ax.set_xlabel(_S["tr_x"])
    ax.set_ylabel(_S["tr_y"])
    ax.set_title(_S["tr_title"], fontweight="bold", loc="left")
    ax.axhline(1.0 - 0.10, color="#94a3b8", lw=0.9, ls=":")
    ax.axvline(0.75, color="#94a3b8", lw=0.9, ls=":")
    ax.margins(0.16)
    fig.tight_layout()
    path = out / "tradeoff.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Ve bieu do tu results.json cua evaluate.py")
    parser.add_argument("--results", type=Path, default=Path("outputs/evaluation/results.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation/figures"))
    parser.add_argument("--lang", choices=["vi", "en"], default="vi", help="Ngon ngu nhan bieu do")
    args = parser.parse_args()

    global _S
    _S = STRINGS[args.lang]
    data = load_summary(args.results.resolve())
    if not data["summary"]:
        print("Khong co summary trong results.json — chay evaluate.py truoc.")
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    made = [
        plot_wer_cer(data, args.output_dir),
        plot_speaker_similarity(data, args.output_dir),
        plot_speed(data, args.output_dir),
        plot_tradeoff(data, args.output_dir),
    ]
    for path in made:
        print(f"Da ve: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
