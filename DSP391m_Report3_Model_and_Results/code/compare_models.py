"""
So sánh mô hình (không cần web)
================================
Sinh audio cho cùng một văn bản bằng NHIỀU mô hình rồi xuất:
  • file .wav cho từng mô hình  → outputs/comparison/<model>__<idx>.wav
  • báo cáo Markdown với số liệu khách quan (RMS, độ phẳng phổ, độ dài)

Số liệu giúp phân biệt giọng người sạch với nhiễu tĩnh mà không cần nghe:
  • RMS quá nhỏ (< 0.02)        → gần như im lặng / nhiễu
  • độ phẳng phổ cao (> 0.30)   → nhiều nhiễu, ít cấu trúc giọng

Cách chạy:
  python scripts/compare_models.py --ref reference_audio/audio.wav
  python scripts/compare_models.py --ref ref.wav --ref-text "..." \
      --models base ft_step_0001000 --nfe 32
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Mặc định chạy CPU cho ổn định, trừ khi người dùng đặt VVCS_DEVICE.
os.environ.setdefault("VVCS_DEVICE", "cpu")
import app  # noqa: E402  (đặt sau khi set env & sys.path)

DEFAULT_TEXTS = [
    "Xin chào, đây là hệ thống nhân bản giọng nói tiếng Việt.",
    "Trí tuệ nhân tạo đang phát triển rất mạnh mẽ trong những năm gần đây.",
    "Thời tiết hôm nay tại thành phố Hồ Chí Minh khá đẹp và trong sáng.",
]


def parse_args():
    p = argparse.ArgumentParser(description="So sánh các mô hình nhân bản giọng nói")
    p.add_argument("--ref", required=True, help="Đường dẫn audio mẫu (.wav)")
    p.add_argument("--ref-text", default="",
                   help="Transcript audio mẫu (trống → Whisper tự nhận diện)")
    p.add_argument("--models", nargs="*", default=None,
                   help=f"Các key mô hình; mặc định = tất cả trừ 'bark' (rất chậm trên CPU). "
                        f"Khả dụng: {', '.join(app.MODEL_REGISTRY)}")
    p.add_argument("--texts", nargs="*", default=None,
                   help="Các câu cần đọc; mặc định dùng bộ câu mẫu")
    p.add_argument("--nfe", type=int, default=app.DEFAULT_NFE)
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=app.DEFAULT_SEED,
                   help="Seed cố định để so sánh công bằng; -1 = ngẫu nhiên")
    p.add_argument("--out-dir", default=str(app.OUTPUT_DIR / "comparison"))
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Mặc định bỏ 'bark' (≈100 giây/câu trên CPU); thêm thủ công qua --models nếu cần.
    models = args.models or [k for k in app.MODEL_REGISTRY if k != "bark"]
    for k in models:
        if k not in app.MODEL_REGISTRY:
            sys.exit(f"❌ Không có mô hình '{k}'. Khả dụng: {', '.join(app.MODEL_REGISTRY)}")
    texts = args.texts or DEFAULT_TEXTS

    # Seed: -1 → ngẫu nhiên một lần, dùng chung cho mọi model/câu.
    seed = args.seed if args.seed >= 0 else __import__("random").randint(0, 2**31 - 1)

    # Transcript audio mẫu (một lần) + chuẩn hoá giống app.
    ref_text = app._normalize_vi(
        args.ref_text.strip() or app._transcribe_whisper(args.ref), add_end_punct=False)
    print(f"Audio mẫu : {args.ref}")
    print(f"Transcript: {ref_text!r}")
    print(f"Mô hình   : {', '.join(models)}")
    print(f"Seed      : {seed}")
    print(f"Thiết bị  : {app.DEVICE}\n")

    rows = []
    for mi, key in enumerate(models):
        label = app.MODEL_REGISTRY[key]["label"]
        for ti, text in enumerate(texts):
            print(f"→ [{key}] câu {ti + 1}/{len(texts)} ...")
            try:
                wave, sr = app._infer_one(
                    key, args.ref, ref_text,
                    app._normalize_vi(text, add_end_punct=True),
                    args.speed, args.nfe, seed=seed)
                fname = f"{key}__{ti:02d}.wav"
                sf.write(str(out_dir / fname), wave, sr)
                rms = float(np.sqrt(np.mean(wave ** 2)))
                flat = app._spectral_flatness(wave)
                note = app._quality_note(wave, sr).split("·")[0].strip()
                rows.append((key, label, ti, text, rms, flat,
                             len(wave) / sr, fname, note))
            except Exception as e:
                rows.append((key, label, ti, text, 0, 0, 0, "-", f"❌ {e}"))

    # Báo cáo Markdown
    report = out_dir / "report.md"
    with open(report, "w", encoding="utf-8") as f:
        f.write("# Báo cáo so sánh mô hình nhân bản giọng nói\n\n")
        f.write(f"- Audio mẫu: `{args.ref}`\n- Transcript: {ref_text}\n")
        f.write(f"- NFE steps: {args.nfe} · Tốc độ: {args.speed} · Seed: {seed} "
                f"· Thiết bị: {app.DEVICE}\n\n")
        f.write("| Mô hình | Câu | RMS | Độ phẳng phổ | Độ dài (s) | Nhận định | File |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for key, label, ti, text, rms, flat, dur, fname, note in rows:
            f.write(f"| {key} | {ti} | {rms:.3f} | {flat:.3f} | {dur:.1f} | {note} | `{fname}` |\n")
        f.write("\n**Cách đọc:** RMS < 0.02 hoặc độ phẳng phổ > 0.30 thường là "
                "nhiễu tĩnh; mô hình tốt cho RMS vừa phải và độ phẳng phổ thấp.\n")

    print(f"\n✅ Xong. Audio + báo cáo tại: {out_dir}/")
    print(f"   Mở báo cáo: {report}")


if __name__ == "__main__":
    main()
