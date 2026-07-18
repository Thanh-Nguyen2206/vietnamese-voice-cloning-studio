"""
=============================================================================
Phase 2: Data Preprocessing Pipeline
File  : scripts/data_prep.py

Chức năng:
  1. Load toàn bộ audio cá nhân từ thư mục input (hỗ trợ .wav, .mp3, .m4a, .flac)
  2. Chuyển đổi về chuẩn: 24kHz, Mono, PCM 16-bit
  3. Lọc bỏ khoảng lặng (silence removal)
  4. Cắt tự động thành segments 3-10 giây (dựa trên khoảng nghỉ tự nhiên)
  5. Lọc segments có SNR quá thấp (< 30dB)
  6. Xuất ra thư mục output + file metadata.csv

Cách chạy:
  python scripts/data_prep.py \
      --input_dir  data/raw \
      --output_dir data/processed \
      --sample_rate 24000 \
      --min_duration 3.0 \
      --max_duration 10.0

Yêu cầu SRS 2.2:
  - Tổng thời lượng: 30-60 phút audio sạch
  - Định dạng xuất: .wav (PCM Linear 16-bit), Mono
  - Tần số lấy mẫu: 24kHz
  - Phân đoạn: 3-10 giây
  - Độ nhiễu: < 30dB, không echo
=============================================================================
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import List

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

# =============================================================================
# CẤU HÌNH MẶC ĐỊNH
# =============================================================================
DEFAULT_SAMPLE_RATE = 24000       # Hz - chuẩn của F5-TTS
DEFAULT_MIN_DURATION = 3.0        # giây - segment tối thiểu
DEFAULT_MAX_DURATION = 10.0       # giây - segment tối đa (tránh OOM khi train)
SILENCE_TOP_DB = 30               # ngưỡng dB để phát hiện silence
SILENCE_PAD_SEC = 0.15            # khoảng nghỉ chèn giữa các đoạn trong remove_silence()
SUPPORTED_FORMATS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".wma"}


def is_invalid_demo_path(path: str | Path) -> bool:
    """Return whether a path points into the quarantined synthetic demo dataset."""
    return "_invalid_demo_data" in Path(path).resolve().parts


def guard_training_path(path: str | Path, allow_invalid_demo_data: bool = False) -> None:
    if is_invalid_demo_path(path) and not allow_invalid_demo_data:
        raise ValueError(
            "Từ chối data/_invalid_demo_data: đây là dữ liệu sóng sin giả, không được dùng "
            "để fine-tune. Chỉ dùng --allow-invalid-demo-data-for-technical-tests cho unit/smoke test."
        )


def assess_training_audio(audio: np.ndarray, sr: int) -> tuple[list[str], list[str]]:
    """Return fatal errors and warnings for a prospective training waveform."""
    errors: list[str] = []
    warnings: list[str] = []
    array = np.asarray(audio)
    if array.size == 0:
        return ["audio rỗng"], warnings
    if not np.isfinite(array).all():
        return ["audio chứa NaN/Inf"], warnings
    mono = array.mean(axis=1) if array.ndim == 2 else array.reshape(-1)
    if array.ndim == 2:
        warnings.append("audio stereo sẽ được chuyển mono")
    duration = mono.size / sr
    rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64))))
    clipping = float(np.mean(np.abs(mono) >= 0.999))
    if duration < 0.25:
        errors.append("audio quá ngắn")
    if rms < 1e-4:
        errors.append("audio im lặng hoặc gần im lặng")
    if clipping > 0.05:
        errors.append(f"clipping quá cao ({clipping:.1%})")
    elif clipping > 0.01:
        warnings.append(f"clipping cao ({clipping:.1%})")
    if mono.size >= 2048 and rms >= 1e-4:
        spectrum = np.abs(np.fft.rfft(mono[: min(mono.size, sr * 10)] *
                                      np.hanning(min(mono.size, sr * 10)))) ** 2
        concentration = float(spectrum.max() / max(spectrum.sum(), 1e-12))
        if concentration > 0.45:
            errors.append("tín hiệu quá đơn giản/giống sóng sin, không phải dữ liệu giọng thật")
    return errors, warnings


def find_audio_files(input_dir: str) -> List[Path]:
    """
    Quét thư mục input và trả về danh sách các file audio hỗ trợ.
    Hỗ trợ cả thư mục con (recursive search).

    Args:
        input_dir: Đường dẫn thư mục chứa audio gốc

    Returns:
        List các Path objects trỏ tới file audio
    """
    audio_files = []
    input_path = Path(input_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Thư mục input không tồn tại: {input_dir}")

    # Duyệt đệ quy qua tất cả thư mục con
    for fpath in sorted(input_path.rglob("*")):
        if fpath.is_file() and fpath.suffix.lower() in SUPPORTED_FORMATS:
            audio_files.append(fpath)

    if not audio_files:
        raise FileNotFoundError(
            f"Không tìm thấy file audio nào trong: {input_dir}\n"
            f"Định dạng hỗ trợ: {SUPPORTED_FORMATS}"
        )

    return audio_files


def load_and_normalize(filepath: Path, target_sr: int) -> np.ndarray:
    """
    Load file audio → chuyển về Mono, resample về target sample rate.

    Dùng librosa vì tự động:
      - Decode nhiều format (mp3, m4a, flac, ...)
      - Chuyển Stereo → Mono
      - Resample về target_sr

    Args:
        filepath: Đường dẫn file audio
        target_sr: Tần số lấy mẫu đích (24000 Hz)

    Returns:
        np.ndarray: Mảng audio 1D đã chuẩn hóa, dtype float32
    """
    # mono=True: tự động mix stereo → mono
    # sr=target_sr: tự động resample
    audio, sr = librosa.load(str(filepath), sr=target_sr, mono=True)

    # Chuẩn hóa biên độ về khoảng [-1, 1] để tránh clipping
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.95  # Để headroom 5%

    return audio


def remove_silence(
    audio: np.ndarray,
    sr: int,
    top_db: int = SILENCE_TOP_DB,
) -> np.ndarray:
    """
    Loại bỏ các khoảng lặng dài ở đầu, cuối và giữa audio.

    Sử dụng librosa.effects.split() để tìm các đoạn có tiếng nói,
    sau đó nối lại với khoảng nghỉ ngắn giữa các đoạn.

    Args:
        audio: Mảng audio 1D
        sr: Sample rate
        top_db: Ngưỡng dB - âm thanh dưới ngưỡng này coi là silence.
                Giá trị 30 phù hợp cho môi trường thu âm yên tĩnh.

    Returns:
        Audio đã loại bỏ silence, giữ lại khoảng nghỉ tự nhiên ngắn
    """
    # Tìm các interval [start, end] có tiếng (non-silent)
    intervals = librosa.effects.split(audio, top_db=top_db)

    if len(intervals) == 0:
        return np.array([], dtype=np.float32)

    # Khoảng nghỉ ngắn chèn giữa các đoạn → giữ nhịp tự nhiên (xem SILENCE_PAD_SEC)
    pad_samples = int(SILENCE_PAD_SEC * sr)
    padding = np.zeros(pad_samples, dtype=np.float32)

    chunks = []
    for i, (start, end) in enumerate(intervals):
        chunks.append(audio[start:end])
        # Chèn khoảng nghỉ giữa các đoạn (không chèn sau đoạn cuối)
        if i < len(intervals) - 1:
            chunks.append(padding)

    return np.concatenate(chunks)


def segment_audio(
    audio: np.ndarray,
    sr: int,
    min_dur: float = DEFAULT_MIN_DURATION,
    max_dur: float = DEFAULT_MAX_DURATION,
    top_db: int = SILENCE_TOP_DB,
) -> List[np.ndarray]:
    """
    Cắt audio dài thành các đoạn ngắn 3-10 giây.

    Chiến lược cắt (ưu tiên điểm cắt tự nhiên):
      1. Tìm tất cả khoảng lặng trong audio → dùng làm điểm cắt tiềm năng
      2. Gom các đoạn non-silent liên tiếp cho đến khi tổng ≥ min_dur
      3. Nếu tổng vượt max_dur → cắt cứng tại max_dur
      4. Bỏ đoạn < min_dur (quá ngắn, không đủ context)

    Args:
        audio: Mảng audio đã loại silence
        sr: Sample rate
        min_dur: Độ dài tối thiểu mỗi segment (giây)
        max_dur: Độ dài tối đa mỗi segment (giây)
        top_db: Ngưỡng silence detection

    Returns:
        List các np.ndarray, mỗi phần tử là một segment 3-10 giây
    """
    total_duration = len(audio) / sr

    # Trường hợp đặc biệt: audio ngắn hơn min_dur → bỏ qua
    if total_duration < min_dur:
        return []

    # Trường hợp audio đã nằm trong khoảng [min, max] → trả về nguyên
    if total_duration <= max_dur:
        return [audio]

    # --- Chiến lược chính: Cắt theo khoảng lặng tự nhiên ---
    # LƯU Ý: hàm này luôn nhận audio ĐÃ QUA remove_silence(), nghĩa là mọi khoảng
    # lặng gốc (dù dài bao nhiêu) đã bị collapse về đúng SILENCE_PAD_SEC. Do đó,
    # BẤT KỲ ranh giới đoạn nào librosa.effects.split() phát hiện được ở đây đều là
    # một điểm nghỉ THẬT trong bản ghi gốc — không cần so ngưỡng độ dài khoảng lặng
    # nữa. (Trước đây so `gap >= ngưỡng mẫu cố định` nhưng biên interval của librosa
    # bị lượng tử hoá theo hop_length nên gap đo được LUÔN nhỏ hơn pad thực đã chèn
    # → điều kiện không bao giờ đúng → mọi đoạn dài đều bị cắt cứng tại max_duration.)
    intervals = librosa.effects.split(audio, top_db=top_db,
                                      frame_length=2048, hop_length=512)

    segments = []
    current_segment_start = 0  # vị trí bắt đầu segment hiện tại (samples)

    for i, (start, end) in enumerate(intervals):
        current_pos = end  # vị trí kết thúc đoạn non-silent hiện tại
        current_dur = (current_pos - current_segment_start) / sr

        # Còn interval tiếp theo → ranh giới hiện tại là điểm nghỉ hợp lệ (xem trên).
        has_silence_gap = i < len(intervals) - 1

        # Điều kiện cắt: đủ dài VÀ có điểm nghỉ tự nhiên
        if current_dur >= min_dur and has_silence_gap:
            segment = audio[current_segment_start:current_pos]
            segments.append(segment)
            current_segment_start = intervals[i + 1][0] if i < len(intervals) - 1 else current_pos

        # Nếu vượt quá max_dur → cắt cứng để tránh OOM
        elif current_dur >= max_dur:
            hard_cut_end = current_segment_start + int(max_dur * sr)
            segment = audio[current_segment_start:hard_cut_end]
            segments.append(segment)
            current_segment_start = hard_cut_end

    # Xử lý đoạn audio còn lại cuối cùng
    remaining = audio[current_segment_start:]
    remaining_dur = len(remaining) / sr
    if remaining_dur >= min_dur:
        # Nếu đoạn cuối vẫn quá dài, cắt cứng
        while remaining_dur > max_dur:
            cut = remaining[:int(max_dur * sr)]
            segments.append(cut)
            remaining = remaining[int(max_dur * sr):]
            remaining_dur = len(remaining) / sr
        if remaining_dur >= min_dur:
            segments.append(remaining)

    return segments


def estimate_snr(audio: np.ndarray, sr: int) -> float:
    """
    Ước tính Signal-to-Noise Ratio (SNR) đơn giản.

    Phương pháp: So sánh năng lượng tín hiệu (non-silent) với
    năng lượng nền (silent regions).

    Args:
        audio: Mảng audio 1D
        sr: Sample rate

    Returns:
        SNR ước tính (dB). Giá trị cao = ít nhiễu.
    """
    # Tìm các vùng có tiếng và vùng im
    intervals = librosa.effects.split(audio, top_db=20)

    if len(intervals) == 0:
        return 0.0

    # Năng lượng vùng có tiếng (signal)
    signal_parts = np.concatenate([audio[s:e] for s, e in intervals])
    signal_power = np.mean(signal_parts ** 2)

    # Năng lượng vùng im (noise) - lấy từ các khoảng giữa intervals
    noise_parts = []
    prev_end = 0
    for start, end in intervals:
        if start > prev_end:
            noise_parts.append(audio[prev_end:start])
        prev_end = end
    # Phần cuối audio sau interval cuối cùng
    if prev_end < len(audio):
        noise_parts.append(audio[prev_end:])

    if not noise_parts or all(len(p) == 0 for p in noise_parts):
        # Không có vùng im → coi như SNR rất cao (toàn tiếng nói)
        return 60.0

    noise_concat = np.concatenate(noise_parts)
    if len(noise_concat) == 0:
        return 60.0

    noise_power = np.mean(noise_concat ** 2)

    # Tránh chia cho 0
    if noise_power < 1e-10:
        return 60.0

    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def process_pipeline(args: argparse.Namespace) -> None:
    """
    Pipeline chính: Load → Normalize → Remove Silence → Segment → Export.

    Workflow:
      1. Quét thư mục input tìm audio files
      2. Với mỗi file:
         a. Load + resample về 24kHz Mono
         b. Loại bỏ silence
         c. Cắt thành segments 3-10s
         d. Kiểm tra SNR từng segment
         e. Lưu WAV + ghi metadata
      3. In báo cáo tổng kết
    """
    guard_training_path(args.input_dir, args.allow_invalid_demo_data_for_technical_tests)
    guard_training_path(args.output_dir, args.allow_invalid_demo_data_for_technical_tests)
    # --- Khởi tạo thư mục output ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_dir = Path(args.output_dir).parent / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = metadata_dir / "metadata.csv"

    # --- Tìm audio files ---
    audio_files = find_audio_files(args.input_dir)
    print(f"\n📂 Tìm thấy {len(audio_files)} file audio trong: {args.input_dir}")
    for f in audio_files:
        print(f"   └── {f.name}")

    # --- Thống kê ---
    total_input_duration = 0.0
    total_output_duration = 0.0
    total_segments = 0
    skipped_short = 0
    skipped_snr = 0
    segment_counter = 0  # Đếm toàn cục để đặt tên file

    # --- Mở file CSV để ghi metadata ---
    # Metadata này sẽ dùng cho Phase 3 (training) để mapping audio ↔ text
    csv_file = open(metadata_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file, delimiter="|")
    writer.writerow(["audio_file", "duration_sec", "snr_db", "source_file", "text"])

    # --- Xử lý từng file ---
    for audio_path in tqdm(audio_files, desc="🔄 Processing files", unit="file"):
        print(f"\n{'='*60}")
        print(f"📄 Đang xử lý: {audio_path.name}")

        # Bước 2a: Load + Normalize (resample về 24kHz, Mono)
        try:
            audio = load_and_normalize(audio_path, target_sr=args.sample_rate)
        except Exception as e:
            print(f"   ❌ Lỗi load file: {e}")
            continue

        fatal, quality_warnings = assess_training_audio(audio, args.sample_rate)
        for warning in quality_warnings:
            print(f"   ⚠️  {warning}")
        if fatal:
            print(f"   ❌ Audio không hợp lệ: {'; '.join(fatal)}")
            continue

        input_dur = len(audio) / args.sample_rate
        total_input_duration += input_dur
        print(f"   ⏱  Thời lượng gốc: {input_dur:.1f}s "
              f"({input_dur/60:.1f} phút)")

        # Bước 2b: Loại bỏ silence
        audio_clean = remove_silence(audio, sr=args.sample_rate)
        clean_dur = len(audio_clean) / args.sample_rate
        removed_dur = input_dur - clean_dur
        print(f"   🔇 Sau khi loại silence: {clean_dur:.1f}s "
              f"(đã bỏ {removed_dur:.1f}s silence)")

        if len(audio_clean) == 0:
            print("   ⚠️  File toàn silence, bỏ qua.")
            continue

        # Bước 2c: Cắt thành segments
        segments = segment_audio(
            audio_clean,
            sr=args.sample_rate,
            min_dur=args.min_duration,
            max_dur=args.max_duration,
        )
        print(f"   ✂️  Số segments: {len(segments)}")

        # Bước 2d+2e: Kiểm tra SNR + Lưu từng segment
        for seg in segments:
            seg_dur = len(seg) / args.sample_rate

            # Bỏ segment quá ngắn (safety check)
            if seg_dur < args.min_duration:
                skipped_short += 1
                continue

            # Kiểm tra chất lượng audio (SNR)
            snr = estimate_snr(seg, args.sample_rate)
            if snr < args.min_snr:
                skipped_snr += 1
                print(f"   ⚠️  Bỏ segment (SNR={snr:.1f}dB < {args.min_snr}dB)")
                continue

            # Đặt tên file output: segment_00001.wav, segment_00002.wav, ...
            segment_counter += 1
            out_filename = f"segment_{segment_counter:05d}.wav"
            out_path = output_dir / out_filename

            # Lưu WAV: 24kHz, Mono, PCM 16-bit (subtype='PCM_16')
            sf.write(str(out_path), seg, args.sample_rate, subtype="PCM_16")

            # Ghi metadata (cột "text" để trống → sẽ điền ở Phase annotation)
            writer.writerow([
                out_filename,
                f"{seg_dur:.2f}",
                f"{snr:.1f}",
                audio_path.name,
                ""  # text placeholder - cần gán transcript sau
            ])

            total_output_duration += seg_dur
            total_segments += 1

    csv_file.close()

    # ==========================================================================
    # BÁO CÁO TỔNG KẾT
    # ==========================================================================
    print(f"\n{'='*60}")
    print("📊 BÁO CÁO TỔNG KẾT")
    print(f"{'='*60}")
    print(f"  📂 Input:  {len(audio_files)} files, "
          f"{total_input_duration:.1f}s ({total_input_duration/60:.1f} phút)")
    print(f"  📂 Output: {total_segments} segments, "
          f"{total_output_duration:.1f}s ({total_output_duration/60:.1f} phút)")
    print(f"  ❌ Bỏ qua (quá ngắn < {args.min_duration}s): {skipped_short}")
    print(f"  ❌ Bỏ qua (SNR < {args.min_snr}dB):          {skipped_snr}")
    print(f"  📋 Metadata: {metadata_path}")
    print(f"  📁 Output dir: {output_dir}")

    # Cảnh báo nếu tổng thời lượng không đạt yêu cầu SRS (30-60 phút)
    output_minutes = total_output_duration / 60
    if output_minutes < 30:
        print(f"\n  ⚠️  CẢNH BÁO: Tổng thời lượng output ({output_minutes:.1f} phút) "
              f"< 30 phút!")
        print("     Theo SRS 2.2, cần tối thiểu 30 phút audio sạch để fine-tune.")
        print("     Hãy thu thêm dữ liệu hoặc giảm min_duration.")
    elif output_minutes > 60:
        print(f"\n  ℹ️  Tổng thời lượng ({output_minutes:.1f} phút) > 60 phút.")
        print("     Dữ liệu đủ dùng. Có thể chọn lọc segments chất lượng cao nhất.")
    else:
        print(f"\n  ✅ Tổng thời lượng ({output_minutes:.1f} phút) nằm trong "
              f"khoảng lý tưởng 30-60 phút!")

    print(f"\n{'='*60}")
    print("✅ Phase 2 hoàn tất! Tiếp theo:")
    print(f"   1. Kiểm tra audio trong: {output_dir}")
    print(f"   2. Gán transcript vào cột 'text' trong: {metadata_path}")
    print("   3. Chuyển sang Phase 3: Fine-tuning")
    print(f"{'='*60}")


# =============================================================================
# CLI - Giao diện dòng lệnh
# =============================================================================
def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Ví dụ chạy:
      python scripts/data_prep.py --input_dir data/raw --output_dir data/processed
      python scripts/data_prep.py --input_dir data/raw --min_duration 4 --max_duration 8
    """
    parser = argparse.ArgumentParser(
        description="Vietnamese Voice Cloning Studio - Data Preprocessing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  # Chạy với cấu hình mặc định (3-10s, 24kHz)
  python scripts/data_prep.py --input_dir data/raw --output_dir data/processed

  # Tùy chỉnh segment length
  python scripts/data_prep.py --input_dir data/raw --min_duration 4 --max_duration 8

  # Lọc SNR chặt hơn
  python scripts/data_prep.py --input_dir data/raw --min_snr 35
        """
    )

    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Thư mục chứa file audio gốc (hỗ trợ .wav, .mp3, .m4a, .flac)"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Thư mục xuất các segment WAV đã xử lý"
    )
    parser.add_argument(
        "--sample_rate", type=int, default=DEFAULT_SAMPLE_RATE,
        help=f"Tần số lấy mẫu đích (default: {DEFAULT_SAMPLE_RATE} Hz)"
    )
    parser.add_argument(
        "--min_duration", type=float, default=DEFAULT_MIN_DURATION,
        help=f"Độ dài tối thiểu mỗi segment (default: {DEFAULT_MIN_DURATION}s)"
    )
    parser.add_argument(
        "--max_duration", type=float, default=DEFAULT_MAX_DURATION,
        help=f"Độ dài tối đa mỗi segment (default: {DEFAULT_MAX_DURATION}s)"
    )
    parser.add_argument(
        "--min_snr", type=float, default=20.0,
        help="SNR tối thiểu (dB) để giữ segment (default: 20.0 dB)"
    )
    parser.add_argument(
        "--allow-invalid-demo-data-for-technical-tests", action="store_true",
        help="CHỈ smoke test code path; không được dùng để tuyên bố chất lượng fine-tune",
    )

    return parser.parse_args()


def validate_metadata(metadata_path: Path, processed_dir: Path | None = None,
                      allow_invalid_demo_data: bool = False) -> dict[str, object]:
    """Validate official pipe-delimited metadata and return a dataset summary."""
    guard_training_path(metadata_path, allow_invalid_demo_data)
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata không tồn tại: {metadata_path}")
    base = processed_dir or metadata_path.parent.parent / "processed"
    guard_training_path(base, allow_invalid_demo_data)
    rows: list[dict[str, str]] = []
    with metadata_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="|")
        required = {"audio_file", "text"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("Metadata cần header pipe-delimited có audio_file và text")
        rows = list(reader)
    durations: list[float] = []
    sample_rates: Counter[int] = Counter()
    transcript_lengths: list[int] = []
    transcripts: Counter[str] = Counter()
    missing = invalid = 0
    for row in rows:
        audio_ref = (row.get("audio_file") or "").strip()
        text = (row.get("text") or "").strip()
        audio_path = Path(audio_ref)
        if not audio_path.is_absolute():
            audio_path = base / audio_path
        if not audio_ref or not text:
            invalid += 1
        if not audio_path.is_file():
            missing += 1
            continue
        try:
            info = sf.info(audio_path)
            wave, sr = sf.read(audio_path, dtype="float32", always_2d=False)
            fatal, _ = assess_training_audio(wave, sr)
            if fatal:
                invalid += 1
            durations.append(float(info.duration))
            sample_rates[int(info.samplerate)] += 1
        except (RuntimeError, ValueError, OSError):
            invalid += 1
        if text:
            transcript_lengths.append(len(text))
            transcripts[text.casefold()] += 1
    duplicates = sum(count - 1 for count in transcripts.values() if count > 1)
    valid_count = max(len(rows) - invalid - missing, 0)
    validation_count = max(1, round(valid_count * 0.1)) if valid_count >= 2 else 0
    return {
        "file_count": len(rows), "total_duration_sec": sum(durations),
        "mean_duration_sec": float(np.mean(durations)) if durations else 0.0,
        "min_duration_sec": min(durations, default=0.0),
        "max_duration_sec": max(durations, default=0.0),
        "sample_rate_distribution": dict(sorted(sample_rates.items())),
        "transcript_length_min": min(transcript_lengths, default=0),
        "transcript_length_max": max(transcript_lengths, default=0),
        "duplicate_transcript_count": duplicates, "missing_file_count": missing,
        "invalid_row_count": invalid, "train_count": valid_count - validation_count,
        "validation_count": validation_count,
    }


def parse_validate_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate metadata.csv và audio cho fine-tuning")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path)
    parser.add_argument("--report", type=Path, help="Ghi summary JSON")
    parser.add_argument("--allow-invalid-demo-data-for-technical-tests", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "validate":
        validation_args = parse_validate_args(sys.argv[2:])
        try:
            summary = validate_metadata(
                validation_args.metadata, validation_args.processed_dir,
                validation_args.allow_invalid_demo_data_for_technical_tests,
            )
        except (ValueError, FileNotFoundError) as exc:
            print(f"❌ Metadata không hợp lệ: {exc}", file=sys.stderr)
            raise SystemExit(2)
        rendered = json.dumps(summary, ensure_ascii=False, indent=2)
        print(rendered)
        if validation_args.report:
            validation_args.report.parent.mkdir(parents=True, exist_ok=True)
            validation_args.report.write_text(rendered + "\n", encoding="utf-8")
        raise SystemExit(1 if summary["invalid_row_count"] or summary["missing_file_count"] else 0)

    args = parse_args()

    # Validate arguments
    assert args.min_duration > 0, "min_duration phải > 0"
    assert args.max_duration > args.min_duration, "max_duration phải > min_duration"
    assert args.sample_rate > 0, "sample_rate phải > 0"

    print("=" * 60)
    print(" Vietnamese Voice Cloning Studio - Data Preprocessing Pipeline")
    print("=" * 60)
    print(f"  Input dir    : {args.input_dir}")
    print(f"  Output dir   : {args.output_dir}")
    print(f"  Sample rate  : {args.sample_rate} Hz")
    print(f"  Segment range: {args.min_duration}-{args.max_duration}s")
    print(f"  Min SNR      : {args.min_snr} dB")
    print("=" * 60)

    process_pipeline(args)
