#!/usr/bin/env python3
"""Render evaluation charts from an evaluate.py results.json file.

Doc outputs/.../results.json (do scripts/evaluate.py sinh ra) va xuat cac bieu do PNG
cho bao cao. Khi manifest co NHIEU cau/engine, cac bieu do cot hien mean ± std (error
bar) va co them bieu do phan bo WER theo tung cau de truc quan hoa do bat dinh.

Cach chay:
    python scripts/plot_results.py --results outputs/benchmark/evaluation/results.json \
        --output-dir outputs/benchmark/figures_en --lang en
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ENGINE_ORDER = ["f5tts", "xtts", "mms", "piper", "edge", "bark"]
ENGINE_LABEL = {"f5tts": "F5-TTS", "xtts": "XTTS-v2", "mms": "MMS-TTS",
                "piper": "Piper", "edge": "Edge-TTS", "bark": "Bark"}
COLORS = {"f5tts": "#2563eb", "xtts": "#7c3aed", "mms": "#0d9488",
          "piper": "#0891b2", "edge": "#64748b", "bark": "#dc2626"}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "-",
})

STRINGS = {
    "vi": {
        "wer_title": "Word Error Rate (thap = tot hon)", "cer_title": "Character Error Rate (thap = tot hon)",
        "acc_suptitle": "Do chinh xac phat am — ASR round-trip (Whisper), mean ± std tren {n} cau",
        "sim_title": "Speaker similarity (SECS) — giong sinh ra vs audio mau (cao = giong hon)",
        "sim_ylabel": "Cosine similarity", "sim_band": "vung 'nhan ban giong' (>0.75)",
        "speed_title": "Toc do tao — RTF = thoi gian tao / thoi luong (thap = nhanh hon)",
        "speed_ylabel": "Real-time factor (log)", "rtf1": "RTF = 1 (real-time)",
        "tr_x": "Speaker similarity (giong giong audio mau) ->", "tr_y": "Intelligibility = 1 - WER (de hieu) ->",
        "tr_title": "Danh doi chat luong: de hieu vs giu dac trung giong\n(goc tren-phai = tot nhat; bong cang to = tao cang nhanh)",
        "dist_title": "Phan bo WER theo tung cau ({n} cau) — moi diem la mot cau",
        "dist_y": "WER (moi cau)",
    },
    "en": {
        "wer_title": "Word Error Rate (lower is better)", "cer_title": "Character Error Rate (lower is better)",
        "acc_suptitle": "Intelligibility via ASR round-trip (Whisper), mean ± std over {n} sentences",
        "sim_title": "Speaker similarity (SECS): generated vs. reference (higher is better)",
        "sim_ylabel": "Cosine similarity", "sim_band": "voice-cloning region (>0.75)",
        "speed_title": "Generation speed — RTF = inference time / duration (lower is faster)",
        "speed_ylabel": "Real-time factor (log)", "rtf1": "RTF = 1 (real-time)",
        "tr_x": "Speaker similarity (closeness to reference voice) ->", "tr_y": "Intelligibility = 1 - WER ->",
        "tr_title": "Quality trade-off: intelligibility vs. speaker identity\n(top-right = best; larger bubble = faster generation)",
        "dist_title": "Per-sentence WER distribution ({n} sentences) — each point is one sentence",
        "dist_y": "WER (per sentence)",
    },
}
_S = STRINGS["en"]


def _stat(values: list[float]) -> dict:
    values = [float(v) for v in values if v is not None]
    if not values:
        return {"mean": None, "std": 0.0, "values": [], "n": 0}
    return {"mean": statistics.mean(values),
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "values": values, "n": len(values)}


def load_stats(results_path: Path) -> dict:
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    rows = [r for r in payload.get("results", []) if r.get("status") == "ok"]
    by_engine: dict[str, list[dict]] = {}
    for r in rows:
        by_engine.setdefault(r["engine"], []).append(r)
    stats: dict[str, dict] = {}
    for engine, erows in by_engine.items():
        stats[engine] = {k: _stat([r.get(k) for r in erows])
                         for k in ("wer", "cer", "speaker_similarity",
                                   "real_time_factor", "inference_time")}
    n = max((len(v) for v in by_engine.values()), default=0)
    return {"stats": stats, "n_sentences": n}


def _engines(stats: dict) -> list[str]:
    present = set(stats)
    return [e for e in ENGINE_ORDER if e in present] + [e for e in sorted(present) if e not in ENGINE_ORDER]


def _labels(engines):
    return [ENGINE_LABEL.get(e, e) for e in engines]


def _bars(ax, engines, means, errs, *, ylabel, title, pct=False):
    colors = [COLORS.get(e, "#3b82f6") for e in engines]
    bars = ax.bar(_labels(engines), means, yerr=errs, color=colors, width=0.62,
                  error_kw={"ecolor": "#334155", "elinewidth": 1.1, "capsize": 4})
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold", loc="left")
    top = max((m + e) for m, e in zip(means, errs)) if means else 1
    ax.set_ylim(0, top * 1.2)
    for rect, m, e in zip(bars, means, errs):
        label = f"{m*100:.1f}%" if pct else f"{m:.3f}"
        ax.text(rect.get_x() + rect.get_width() / 2, m + e + top * 0.02, label,
                ha="center", va="bottom", fontsize=9)
    return bars


def plot_wer_cer(data, out):
    st, engines = data["stats"], _engines(data["stats"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    _bars(ax1, engines, [st[e]["wer"]["mean"] for e in engines],
          [st[e]["wer"]["std"] for e in engines], ylabel="WER", title=_S["wer_title"], pct=True)
    _bars(ax2, engines, [st[e]["cer"]["mean"] for e in engines],
          [st[e]["cer"]["std"] for e in engines], ylabel="CER", title=_S["cer_title"], pct=True)
    for ax in (ax1, ax2):
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle(_S["acc_suptitle"].format(n=data["n_sentences"]), fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = out / "wer_cer.png"; fig.savefig(p, bbox_inches="tight"); plt.close(fig); return p


def plot_speaker_similarity(data, out):
    st, engines = data["stats"], _engines(data["stats"])
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    _bars(ax, engines, [st[e]["speaker_similarity"]["mean"] or 0 for e in engines],
          [st[e]["speaker_similarity"]["std"] for e in engines],
          ylabel=_S["sim_ylabel"], title=_S["sim_title"])
    ax.axhspan(0.75, 1.0, color="#22c55e", alpha=0.08)
    ax.text(len(engines) - 0.5, 0.76, _S["sim_band"], ha="right", va="bottom",
            fontsize=8.5, color="#16a34a")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    p = out / "speaker_similarity.png"; fig.savefig(p, bbox_inches="tight"); plt.close(fig); return p


def plot_speed(data, out):
    st, engines = data["stats"], _engines(data["stats"])
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    means = [st[e]["real_time_factor"]["mean"] for e in engines]
    errs = [st[e]["real_time_factor"]["std"] for e in engines]
    colors = [COLORS.get(e, "#3b82f6") for e in engines]
    ax.bar(_labels(engines), means, yerr=errs, color=colors, width=0.62,
           error_kw={"ecolor": "#334155", "elinewidth": 1.1, "capsize": 4})
    ax.set_yscale("log"); ax.set_ylabel(_S["speed_ylabel"])
    ax.set_title(_S["speed_title"], fontweight="bold", loc="left")
    ax.axhline(1.0, color="#111827", lw=1, ls="--")
    ax.text(len(engines) - 0.5, 1.05, _S["rtf1"], ha="right", va="bottom", fontsize=8.5)
    for i, m in enumerate(means):
        ax.text(i, m * 1.10, f"{m:.2f}x", ha="center", va="bottom", fontsize=9)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    p = out / "speed_rtf.png"; fig.savefig(p, bbox_inches="tight"); plt.close(fig); return p


def plot_tradeoff(data, out):
    st, engines = data["stats"], _engines(data["stats"])
    label_offset = {"f5tts": (11, 9), "xtts": (11, -16), "mms": (-16, -32),
                    "piper": (12, 20), "edge": (-54, 20), "bark": (10, 9)}
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    for e in engines:
        x = st[e]["speaker_similarity"]["mean"] or 0.0
        y = 1.0 - (st[e]["wer"]["mean"] or 0.0)
        rtf = st[e]["real_time_factor"]["mean"] or 0.1
        size = min(1400.0, max(90.0, 900.0 / max(rtf, 0.2)))  # cap so labels stay readable
        ax.scatter(x, y, s=size, color=COLORS.get(e, "#3b82f6"), alpha=0.72,
                   edgecolors="white", linewidths=1.5, zorder=3)
        ax.annotate(ENGINE_LABEL.get(e, e), (x, y), xytext=label_offset.get(e, (7, 7)),
                    textcoords="offset points", fontsize=10, fontweight="bold", zorder=4)
    ax.set_xlabel(_S["tr_x"]); ax.set_ylabel(_S["tr_y"])
    ax.set_title(_S["tr_title"], fontweight="bold", loc="left")
    ax.axhline(1.0 - 0.10, color="#94a3b8", lw=0.9, ls=":")
    ax.axvline(0.75, color="#94a3b8", lw=0.9, ls=":")
    ax.margins(0.16)
    fig.tight_layout()
    p = out / "tradeoff.png"; fig.savefig(p, bbox_inches="tight"); plt.close(fig); return p


def plot_wer_distribution(data, out):
    """Strip plot: WER cua tung cau, giup thay do bat dinh (khong chi mean)."""
    st, engines = data["stats"], _engines(data["stats"])
    fig, ax = plt.subplots(figsize=(9, 4.6))
    import random
    random.seed(0)
    for i, e in enumerate(engines):
        vals = st[e]["wer"]["values"]
        xs = [i + random.uniform(-0.14, 0.14) for _ in vals]
        ax.scatter(xs, vals, s=52, color=COLORS.get(e, "#3b82f6"), alpha=0.75,
                   edgecolors="white", linewidths=1, zorder=3)
        if vals:
            m = statistics.mean(vals)
            ax.plot([i - 0.28, i + 0.28], [m, m], color="#111827", lw=2, zorder=4)
    ax.set_xticks(range(len(engines)))
    ax.set_xticklabels(_labels(engines), rotation=15)
    ax.set_ylabel(_S["dist_y"])
    ax.set_title(_S["dist_title"].format(n=data["n_sentences"]), fontweight="bold", loc="left")
    ax.set_ylim(bottom=-0.03)
    ax.text(0.01, 0.97, "— = mean", transform=ax.transAxes, fontsize=8.5,
            va="top", color="#111827")
    fig.tight_layout()
    p = out / "wer_distribution.png"; fig.savefig(p, bbox_inches="tight"); plt.close(fig); return p


def main() -> int:
    parser = argparse.ArgumentParser(description="Ve bieu do tu results.json cua evaluate.py")
    parser.add_argument("--results", type=Path, default=Path("outputs/evaluation/results.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation/figures"))
    parser.add_argument("--lang", choices=["vi", "en"], default="vi")
    args = parser.parse_args()

    global _S
    _S = STRINGS[args.lang]
    data = load_stats(args.results.resolve())
    if not data["stats"]:
        print("Khong co ket qua 'ok' trong results.json — chay evaluate.py truoc.")
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    made = [plot_wer_cer(data, args.output_dir),
            plot_speaker_similarity(data, args.output_dir),
            plot_speed(data, args.output_dir),
            plot_tradeoff(data, args.output_dir),
            plot_wer_distribution(data, args.output_dir)]
    for p in made:
        print(f"Da ve: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
