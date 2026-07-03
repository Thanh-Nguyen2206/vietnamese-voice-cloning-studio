"""
Vietnamese Voice Cloning Studio
================================
Giao diện web để nhân bản giọng nói tiếng Việt: tải một đoạn audio mẫu, nhập
văn bản, và mô hình sẽ đọc văn bản đó bằng giọng của audio mẫu.

Điểm chính của bản này:
  • Dùng đúng pipeline suy luận chính thức của thư viện nền (convert_char_to_pinyin,
    ước lượng độ dài, cross-fade) — đây là pipeline mà mô hình ĐÃ được huấn luyện cùng.
  • Dựng DiT từ NGUYÊN cấu hình arch F5TTS_Base (đủ text_mask_padding & pe_attn_head)
    để khớp 100% mô hình gốc — nếu thiếu sẽ cho giọng méo, không nghe ra nội dung.
  • Cho phép chọn & SO SÁNH nhiều mô hình trong cùng một lần chạy.
  • Ổn định & nhất quán: chạy tuần tự (khoá + queue), cố định seed để tái lập kết quả,
    chuẩn hoá văn bản tiếng Việt, và giới hạn đỉnh để tránh méo tiếng.

Chạy:  python app.py            # mở http://localhost:7860
       python app.py --share    # tạo link public (Colab/Kaggle)
"""
import os
import re
import sys
import time
import random
import argparse
import threading
import traceback
from pathlib import Path

import numpy as np
import torch
import soundfile as sf
import gradio as gr
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

APP_TITLE = "Vietnamese Voice Cloning Studio"

# ─── Thiết bị tính toán ─────────────────────────────────────────────────────────
# Mô hình mặc định chạy trên CPU vì cho kết quả ổn định & đã được kiểm chứng cho ra
# giọng sạch. Có thể ép sang GPU/MPS qua biến môi trường VVCS_DEVICE nếu muốn nhanh
# hơn (cuda thường an toàn; mps có thể sai số học với một số phép toán của DiT).
def _auto_device() -> str:
    env = os.environ.get("VVCS_DEVICE", "").strip().lower()
    if env:
        return env
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = _auto_device()

# ─── Thông tin mô hình nền ──────────────────────────────────────────────────────
REPO_ID        = "hynt/F5-TTS-Vietnamese-ViVoice"   # mô hình tiếng Việt 1000h
VOCAB_FILENAME = "config.json"                       # repo lưu vocab dưới tên này
CKPT_FILENAME  = "model_last.pt"
CHECKPOINT_DIR = ROOT / "checkpoints"
OUTPUT_DIR     = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Kiến trúc DiT của F5TTS_Base (PHẢI khớp mô hình gốc — không sửa).
# QUAN TRỌNG: phải đủ CẢ `text_mask_padding=False` và `pe_attn_head=1`. Thiếu hai
# trường này thì DiT lấy default của bản v1 → trọng số nạp đúng shape nhưng forward
# pass chạy SAI → giọng méo, không nghe ra nội dung ("garbled speech").
def _load_arch():
    """Lấy nguyên cấu hình arch F5TTS_Base từ thư viện để khớp 100% mô hình gốc."""
    try:
        from importlib.resources import files
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(str(files("f5_tts").joinpath("configs/F5TTS_Base.yaml")))
        return OmegaConf.to_container(cfg.model.arch, resolve=True)
    except Exception as e:
        print(f"[VVCS] Không đọc được F5TTS_Base.yaml ({e}); dùng cấu hình tường minh.")
        return dict(
            dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512,
            text_mask_padding=False, conv_layers=4, pe_attn_head=1,
        )

DIT_CFG = _load_arch()
TARGET_SR = 24000

# Tham số suy luận mặc định
DEFAULT_NFE  = 32
CFG_STRENGTH = 2.0
SWAY_COEF    = -1.0
DEFAULT_SEED = 42          # cố định để kết quả tái lập & so sánh công bằng
MAX_GEN_CHARS = 2000       # chặn văn bản quá dài (tránh treo trên CPU)

# ─── Đăng ký mô hình để so sánh ─────────────────────────────────────────────────
def _discover_finetuned():
    found = []
    for ck in sorted(CHECKPOINT_DIR.glob("step_*/model.pt")):
        step = ck.parent.name
        found.append((f"ft_{step}", f"Fine-tune · {step}", ck))
    return found

def build_registry():
    # Mỗi mục: engine ("f5tts" | "xtts" | "mms" | "piper" | "edge" | "bark")
    # + nhãn song ngữ (label = tiếng Việt, label_en = English).
    reg = {
        "base": {
            "label": "F5-TTS — nhân bản giọng, tiếng Việt 1000 giờ (khuyên dùng)",
            "label_en": "F5-TTS — voice cloning, Vietnamese 1000h (recommended)",
            "engine": "f5tts", "ckpt": None,
        },
        "xtts": {
            "label": "XTTS-v2 (viXTTS) — nhân bản giọng, cần audio mẫu sạch",
            "label_en": "XTTS-v2 (viXTTS) — voice cloning, needs a clean sample",
            "engine": "xtts",
        },
        "mms": {
            "label": "MMS-TTS (Meta) — một giọng cố định, chạy local, nhanh",
            "label_en": "MMS-TTS (Meta) — single fixed voice, local, fast",
            "engine": "mms",
        },
        "piper": {
            "label": "Piper (Rhasspy) — một giọng cố định, nhanh nhất trên CPU",
            "label_en": "Piper (Rhasspy) — single fixed voice, fastest on CPU",
            "engine": "piper",
        },
        "edge": {
            "label": "Edge-TTS (Microsoft) — giọng neural, chạy cloud, cần mạng",
            "label_en": "Edge-TTS (Microsoft) — neural voice, cloud-based, needs internet",
            "engine": "edge",
        },
        "bark": {
            "label": "Bark (Suno) — giọng preset, chậm, chỉ để đối chiếu",
            "label_en": "Bark (Suno) — preset voice, slow, for reference only",
            "engine": "bark",
        },
    }
    # Các checkpoint fine-tune demo (huấn luyện trên dữ liệu giả) KHÔNG xuất hiện
    # trên site chính — chỉ bật lại khi cần nghiên cứu: VVCS_SHOW_DEMO_CKPTS=1.
    if os.environ.get("VVCS_SHOW_DEMO_CKPTS") == "1":
        for key, label, ck in _discover_finetuned():
            reg[key] = {"label": f"F5-TTS {label} (dữ liệu demo)",
                        "label_en": f"F5-TTS {label} (demo data)",
                        "engine": "f5tts", "ckpt": ck}
    return reg

# Engine nào CẦN audio mẫu (các engine còn lại dùng giọng cố định/preset).
REF_REQUIRED_ENGINES = {"f5tts", "xtts"}

MODEL_REGISTRY = build_registry()

# ─── Bộ nhớ đệm & khoá đồng bộ ──────────────────────────────────────────────────
# RLock (reentrant) để suy luận chạy TUẦN TỰ: tránh hai request cùng nạp model
# (gấp đôi RAM → OOM) hoặc chạy song song trên cùng một model (không thread-safe).
_LOCK = threading.RLock()
_vocab_file    = None
_base_ckpt     = None
_vocoder       = None
_model_cache   = {}        # key -> CFM model đã nạp
_whisper_model = None


def _ensure_assets():
    """Tải vocab + checkpoint gốc + vocoder (một lần). Có dự phòng CPU cho vocoder."""
    global _vocab_file, _base_ckpt, _vocoder, DEVICE
    if _vocab_file is None:
        _vocab_file = hf_hub_download(repo_id=REPO_ID, filename=VOCAB_FILENAME)
    if _base_ckpt is None:
        _base_ckpt = hf_hub_download(repo_id=REPO_ID, filename=CKPT_FILENAME)
    if _vocoder is None:
        from f5_tts.infer.utils_infer import load_vocoder
        try:
            _vocoder = load_vocoder(vocoder_name="vocos", device=DEVICE)
        except Exception as e:
            if DEVICE != "cpu":
                print(f"[VVCS] Vocoder lỗi trên '{DEVICE}' ({e}); chuyển sang CPU.")
                DEVICE = "cpu"
                _vocoder = load_vocoder(vocoder_name="vocos", device=DEVICE)
            else:
                raise


def get_model(key: str):
    """Nạp (và cache) mô hình theo key. An toàn luồng nhờ _LOCK."""
    with _LOCK:
        if key in _model_cache:
            return _model_cache[key]

        _ensure_assets()
        from f5_tts.model import DiT
        from f5_tts.infer.utils_infer import load_model

        # Luôn dựng từ checkpoint gốc trước (đúng định dạng EMA) để có mô hình hợp lệ.
        model = load_model(
            DiT, DIT_CFG, _base_ckpt,
            mel_spec_type="vocos", vocab_file=_vocab_file,
            use_ema=True, device=DEVICE,
        )

        ckpt = MODEL_REGISTRY[key]["ckpt"]
        if ckpt is not None:
            # Đè trọng số fine-tune cục bộ (state_dict thuần của CFM).
            sd = torch.load(str(ckpt), map_location=DEVICE, weights_only=True)
            cur = {k: v.shape for k, v in model.state_dict().items()}
            compat = {k: v for k, v in sd.items() if k in cur and v.shape == cur[k]}
            model.load_state_dict(compat, strict=False)
            print(f"[VVCS] {key}: nạp {len(compat)}/{len(sd)} khoá fine-tune "
                  f"(bỏ {len(sd) - len(compat)} khoá lệch shape)")

        model.eval()
        _model_cache[key] = model
        return model


def _set_seed(seed: int):
    """Cố định seed để output tái lập được (và so sánh model công bằng)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _transcribe_whisper(audio_path: str) -> str:
    """Tự nhận diện transcript tiếng Việt từ audio mẫu (nếu người dùng để trống)."""
    global _whisper_model
    with _LOCK:
        try:
            from faster_whisper import WhisperModel
            if _whisper_model is None:
                _whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
            segments, _ = _whisper_model.transcribe(audio_path, language="vi", beam_size=5)
            return " ".join(s.text.strip() for s in segments).strip()
        except Exception:
            pass
        try:
            import whisper
            if _whisper_model is None:
                _whisper_model = whisper.load_model("small")
            return _whisper_model.transcribe(audio_path, language="vi")["text"].strip()
        except Exception:
            return ""


# ─── Chuẩn hoá văn bản tiếng Việt ───────────────────────────────────────────────
# Mô hình hynt được huấn luyện trên văn bản CHỮ THƯỜNG, giữ dấu câu. Chuẩn hoá đầu
# vào về cùng phân phối giúp phát âm ổn định & rõ hơn.

def _normalize_vi(text: str, add_end_punct: bool = True) -> str:
    text = (text or "").strip().lower()
    text = text.replace("…", ".").replace("“", '"').replace("”", '"').replace("’", "'")
    text = re.sub(r"\s+", " ", text)                 # gộp khoảng trắng/xuống dòng
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)     # bỏ space trước dấu câu
    if add_end_punct and text and text[-1] not in ".!?,;:\"')":
        text += "."                                   # kết câu rõ ràng → đỡ trôi cuối câu
    return text


# ─── Số liệu khách quan để so sánh chất lượng ───────────────────────────────────

def _spectral_flatness(wave: np.ndarray) -> float:
    """Độ phẳng phổ: tiếng nói người ~0.02–0.20; nhiễu trắng → ~1.0."""
    if wave.size < 256:
        return 0.0
    sp = np.abs(np.fft.rfft(wave * np.hanning(len(wave)))) ** 2 + 1e-12
    return float(np.exp(np.mean(np.log(sp))) / np.mean(sp))


_VERDICTS = {
    "vi": {"silent": "Cảnh báo: gần như im lặng / nhiễu",
           "noisy": "Cảnh báo: nhiều nhiễu, ít cấu trúc giọng",
           "ok": "Đạt: có cấu trúc giọng nói", "flat": "độ phẳng phổ"},
    "en": {"silent": "Warning: near-silent / noise",
           "noisy": "Warning: noisy, little voice structure",
           "ok": "Pass: structured speech", "flat": "spectral flatness"},
}


def _quality_note(wave: np.ndarray, sr: int, lang: str = "vi") -> str:
    v = _VERDICTS.get(lang, _VERDICTS["vi"])
    rms = float(np.sqrt(np.mean(wave ** 2))) if wave.size else 0.0
    flat = _spectral_flatness(wave)
    dur = len(wave) / sr if sr else 0.0
    if rms < 0.02:
        verdict = v["silent"]
    elif flat > 0.30:
        verdict = v["noisy"]
    else:
        verdict = v["ok"]
    return f"**{verdict}**  ·  RMS={rms:.3f}  ·  {v['flat']}={flat:.3f}  ·  {dur:.1f}s"


def _postprocess(wave: np.ndarray) -> np.ndarray:
    """Giới hạn đỉnh để tránh méo do clipping (vocoder thỉnh thoảng vượt 1.0)."""
    wave = np.nan_to_num(np.asarray(wave, dtype=np.float32))
    peak = float(np.abs(wave).max()) if wave.size else 0.0
    if peak > 0.99:
        wave = wave / peak * 0.99
    return wave


# ─── Suy luận ───────────────────────────────────────────────────────────────────

def _infer_f5tts(model_key, ref_audio_path, ref_text, gen_text, speed, nfe, seed=None):
    """Đường F5-TTS (KHÔNG đổi) — pipeline chính thức, an toàn luồng."""
    from f5_tts.infer.utils_infer import infer_process, preprocess_ref_audio_text

    with _LOCK:
        # Nạp model & tiền xử lý TRƯỚC (các bước này có thể tiêu tốn RNG, nhất là lần
        # nạp model đầu tiên). Đặt seed NGAY TRƯỚC infer_process để nhiễu khởi tạo của
        # ODE giống hệt nhau giữa các lần → kết quả tái lập 100%.
        model = get_model(model_key)
        ref_audio_proc, ref_text_proc = preprocess_ref_audio_text(
            ref_audio_path, ref_text, show_info=lambda *a, **k: None
        )
        if seed is not None:
            _set_seed(int(seed))
        wave, sr, _ = infer_process(
            ref_audio_proc, ref_text_proc, gen_text, model, _vocoder,
            mel_spec_type="vocos", target_rms=0.1, cross_fade_duration=0.15,
            nfe_step=int(nfe), cfg_strength=CFG_STRENGTH, sway_sampling_coef=SWAY_COEF,
            speed=float(speed), device=DEVICE, show_info=lambda *a, **k: None,
        )
    return _postprocess(wave), sr


def _infer_one(model_key, ref_audio_path, ref_text, gen_text, speed, nfe, seed=None):
    """Điều phối theo engine của mô hình. Kết quả luôn qua _postprocess (giới hạn đỉnh).

    Lưu ý: XTTS/Bark không dùng tham số NFE (đó là của flow-matching F5-TTS).
    """
    engine = MODEL_REGISTRY[model_key].get("engine", "f5tts")
    if engine == "f5tts":
        return _infer_f5tts(model_key, ref_audio_path, ref_text, gen_text, speed, nfe, seed)

    import engines  # nạp lười: chỉ import thư viện engine khi thật sự được chọn
    fn = {"xtts": engines.xtts_infer, "bark": engines.bark_infer,
          "mms": engines.mms_infer, "piper": engines.piper_infer,
          "edge": engines.edge_infer}.get(engine)
    if fn is None:
        raise ValueError(f"Engine không hỗ trợ: {engine}")
    wave, sr = fn(ref_audio_path, ref_text, gen_text, speed, seed, device=DEVICE)
    return _postprocess(wave), sr


MAX_SLOTS = 4   # số mô hình tối đa so sánh cùng lúc


def _model_label(key: str, lang: str) -> str:
    entry = MODEL_REGISTRY[key]
    return entry["label"] if lang == "vi" else entry.get("label_en", entry["label"])


def synthesize(model_keys, ref_audio_path, ref_text, gen_text, speed, nfe, seed,
               lang="vi", progress=gr.Progress()):
    """Sinh audio cho mọi mô hình được chọn; trả về dữ liệu cho từng slot UI."""
    L = I18N.get(lang) or I18N["vi"]
    if not model_keys:
        raise gr.Error(L["err_no_model"])
    keys = list(model_keys)[:MAX_SLOTS]
    engines_sel = {MODEL_REGISTRY[k].get("engine", "f5tts") for k in keys}
    # Chỉ F5-TTS & XTTS nhân bản giọng nên mới cần audio mẫu.
    if not ref_audio_path and engines_sel & REF_REQUIRED_ENGINES:
        raise gr.Error(L["err_no_ref"])
    if not gen_text or not gen_text.strip():
        raise gr.Error(L["err_no_text"])
    if len(gen_text) > MAX_GEN_CHARS:
        raise gr.Error(L["err_too_long"].format(n=len(gen_text), m=MAX_GEN_CHARS))

    # Seed: số nguyên cố định để tái lập; < 0 = ngẫu nhiên mỗi lần.
    try:
        seed = int(seed)
    except (TypeError, ValueError):
        seed = DEFAULT_SEED
    if seed < 0:
        seed = random.randint(0, 2**31 - 1)

    # Chuẩn hoá văn bản cần đọc; transcript mẫu chỉ chuẩn hoá nhẹ (không thêm dấu cuối).
    gen_text_n = _normalize_vi(gen_text, add_end_punct=True)

    # Chỉ F5-TTS cần transcript của audio mẫu (XTTS nhân bản trực tiếp từ audio,
    # các engine còn lại dùng giọng cố định). Chỉ chạy Whisper khi có chọn F5-TTS.
    _ref_text = ref_text.strip() if ref_text and ref_text.strip() else ""
    whisper_note = ""
    if "f5tts" in engines_sel and not _ref_text and ref_audio_path:
        progress(0.05, desc=L["whisper_progress"])
        _ref_text = _transcribe_whisper(ref_audio_path)
        whisper_note = (L["whisper_note"].format(t=_ref_text)
                        if _ref_text else L["whisper_na"])
    _ref_text = _normalize_vi(_ref_text, add_end_punct=False)

    results = []
    for i, key in enumerate(keys):
        label = _model_label(key, lang)
        progress((i + 1) / (len(keys) + 1),
                 desc=f"({i + 1}/{len(keys)}) {label.split('—')[0].strip()}")
        try:
            # Cùng seed cho mọi model → so sánh công bằng (cùng nhiễu khởi tạo).
            t0 = time.time()
            wave, sr = _infer_one(key, ref_audio_path, _ref_text,
                                  gen_text_n, speed, nfe, seed=seed)
            gen_s = time.time() - t0
            sf.write(str(OUTPUT_DIR / f"{key}.wav"), wave, sr)
            note = _quality_note(wave, sr, lang)
            results.append(((sr, wave),
                            f"**{label}**\n\n{note}  ·  {L['gen_time'].format(s=gen_s)}"))
        except Exception as e:
            traceback.print_exc()
            results.append((None, f"**{label}**\n\n{L['slot_err'].format(e=e)}"))

    seed_line = f"Seed: {seed}"
    note_out = f"{whisper_note}\n{seed_line}".strip() if whisper_note else seed_line

    # Mỗi slot: (cột chứa card, markdown thông tin, audio) — ẩn cột khi không dùng.
    updates = []
    for i in range(MAX_SLOTS):
        if i < len(results):
            audio, md = results[i]
            updates += [gr.update(visible=True),
                        gr.update(value=md),
                        gr.update(value=audio)]
        else:
            updates += [gr.update(visible=False),
                        gr.update(value=""),
                        gr.update(value=None)]
    return [note_out] + updates


# ─── Song ngữ (i18n) ─────────────────────────────────────────────────────────────

I18N = {
    "vi": dict(
        hero_sub=("So sánh trực tiếp các mô hình chuyển văn bản thành giọng nói tiếng Việt — "
                  "nghe kết quả cạnh nhau, kèm chỉ số chất lượng khách quan và thời gian tạo."),
        badge_device="Thiết bị: {d}",
        badge_engines="6 engine",
        badge_slots=f"So sánh tối đa {MAX_SLOTS} mô hình mỗi lần",
        tips_label="Về các mô hình và mẹo để giọng rõ nhất",
        tips_md=(
            "| Engine | Nhân bản giọng | Chạy ở đâu | Tốc độ (CPU) | Ghi chú |\n"
            "|---|---|---|---|---|\n"
            "| F5-TTS (gốc) | Có | Local | Nhanh | Tiếng Việt 1000 giờ — chất lượng tốt nhất |\n"
            "| XTTS-v2 (viXTTS) | Có | Local | Vừa | Cần audio mẫu sạch, không nhạc nền |\n"
            "| MMS-TTS (Meta) | Không (1 giọng) | Local | Rất nhanh | VITS tiếng Việt gọn nhẹ |\n"
            "| Piper (Rhasspy) | Không (1 giọng) | Local | Nhanh nhất | ONNX, kết quả tất định |\n"
            "| Edge-TTS (Microsoft) | Không (1 giọng) | Cloud | Nhanh | Cần internet; nếu mất mạng sẽ báo lỗi sau tối đa 40 giây |\n"
            "| Bark (Suno) | Không (preset) | Local | Rất chậm | Tiếng Việt hạn chế — chỉ để đối chiếu |\n\n"
            "Lần đầu chọn engine mới, hệ thống sẽ tải mô hình về máy (XTTS khoảng 2GB, "
            "Bark khoảng 1GB, MMS/Piper vài trăm MB) nên lần chạy đầu sẽ lâu hơn bình thường.\n\n"
            "**Mẹo chất lượng:**\n"
            "- Audio mẫu sạch 5–10 giây, một giọng, không nhạc nền hay tiếng vọng "
            "(chỉ F5-TTS và XTTS dùng audio mẫu).\n"
            "- Nhập đúng transcript của audio mẫu (chỉ F5-TTS dùng) thay vì để Whisper tự đoán.\n"
            "- Tăng NFE lên 48–64 cho F5-TTS nếu muốn giọng nét hơn (engine khác bỏ qua NFE).\n"
            "- Seed cố định cho kết quả lặp lại y hệt; đặt -1 để đổi ngẫu nhiên mỗi lần.\n"
            "- Thanh tốc độ áp dụng cho mọi engine trừ Bark."
        ),
        input_hdr="Đầu vào",
        ref_audio="Audio mẫu (3–10 giây) — mặc định là mẫu sạch sẵn có",
        ref_text="Transcript của audio mẫu (tuỳ chọn — chỉ F5-TTS dùng)",
        ref_text_ph="Để trống để hệ thống tự nhận diện bằng Whisper",
        gen_text="Văn bản cần đọc",
        models=f"Chọn mô hình để so sánh (tối đa {MAX_SLOTS})",
        speed="Tốc độ đọc",
        nfe="Độ mịn NFE (chỉ F5-TTS)",
        seed="Seed (-1 = ngẫu nhiên)",
        btn="Tạo và so sánh giọng nói",
        results_hdr="Kết quả",
        runinfo="Thông tin lần chạy (transcript Whisper, seed)",
        slot="Kết quả",
        examples_label="Ví dụ văn bản",
        err_no_model="Hãy chọn ít nhất một mô hình.",
        err_no_ref="Hãy tải lên hoặc thu âm audio mẫu (3–10 giây).",
        err_no_text="Hãy nhập văn bản cần đọc.",
        err_too_long="Văn bản quá dài ({n} ký tự, tối đa {m}). Hãy chia nhỏ.",
        whisper_progress="Whisper đang nhận diện transcript mẫu...",
        whisper_note="[Whisper tự nhận diện] {t}",
        whisper_na="[Whisper không khả dụng — vẫn thử suy luận]",
        gen_time="tạo mất {s:.1f} giây",
        slot_err="Lỗi: {e}",
    ),
    "en": dict(
        hero_sub=("Compare Vietnamese text-to-speech models head-to-head — listen to results "
                  "side by side with objective quality metrics and generation time."),
        badge_device="Device: {d}",
        badge_engines="6 engines",
        badge_slots=f"Compare up to {MAX_SLOTS} models per run",
        tips_label="About the models and tips for the clearest voice",
        tips_md=(
            "| Engine | Voice cloning | Runs | Speed (CPU) | Notes |\n"
            "|---|---|---|---|---|\n"
            "| F5-TTS (base) | Yes | Local | Fast | Vietnamese 1000h — best quality |\n"
            "| XTTS-v2 (viXTTS) | Yes | Local | Medium | Needs a clean sample, no background music |\n"
            "| MMS-TTS (Meta) | No (1 voice) | Local | Very fast | Compact Vietnamese VITS |\n"
            "| Piper (Rhasspy) | No (1 voice) | Local | Fastest | ONNX, deterministic output |\n"
            "| Edge-TTS (Microsoft) | No (1 voice) | Cloud | Fast | Needs internet; fails clearly after at most 40 seconds |\n"
            "| Bark (Suno) | No (preset) | Local | Very slow | Limited Vietnamese — reference only |\n\n"
            "The first time you pick a new engine, its model is downloaded (XTTS ~2GB, "
            "Bark ~1GB, MMS/Piper a few hundred MB), so the first run takes longer.\n\n"
            "**Quality tips:**\n"
            "- Use a clean 5–10 s single-speaker sample without background music or echo "
            "(only F5-TTS and XTTS use the sample).\n"
            "- Type the exact transcript of the sample (only F5-TTS uses it) instead of relying on Whisper.\n"
            "- Raise NFE to 48–64 for a crisper F5-TTS voice (other engines ignore NFE).\n"
            "- A fixed seed reproduces identical results; set -1 for a new random take.\n"
            "- The speed slider applies to every engine except Bark."
        ),
        input_hdr="Input",
        ref_audio="Reference audio (3–10 s) — a clean built-in sample by default",
        ref_text="Reference transcript (optional — used by F5-TTS only)",
        ref_text_ph="Leave empty to auto-transcribe with Whisper",
        gen_text="Text to speak",
        models=f"Models to compare (up to {MAX_SLOTS})",
        speed="Speaking speed",
        nfe="NFE quality (F5-TTS only)",
        seed="Seed (-1 = random)",
        btn="Generate and compare",
        results_hdr="Results",
        runinfo="Run info (Whisper transcript, seed)",
        slot="Result",
        examples_label="Sample texts",
        err_no_model="Please select at least one model.",
        err_no_ref="Please upload or record a reference sample (3–10 seconds).",
        err_no_text="Please enter the text to speak.",
        err_too_long="Text too long ({n} characters, max {m}). Please split it.",
        whisper_progress="Whisper is transcribing the reference sample...",
        whisper_note="[Whisper auto-transcript] {t}",
        whisper_na="[Whisper unavailable — proceeding anyway]",
        gen_time="generated in {s:.1f} s",
        slot_err="Error: {e}",
    ),
}


# ─── Giao diện ──────────────────────────────────────────────────────────────────

CSS = """
.gradio-container {max-width: 1200px !important; margin: 0 auto !important;}
#hero {border-radius: 18px; padding: 30px 30px 24px;
  background: linear-gradient(135deg, #12365c 0%, #1f4e79 45%, #2e75b6 100%);
  color: #ffffff; margin: 6px 0 4px;}
#hero h1 {margin: 0 0 8px; font-size: 1.9rem; letter-spacing: .2px; color: #ffffff;}
#hero p {margin: 0; opacity: .9; font-size: .98rem; line-height: 1.55; max-width: 900px;}
#hero .badges {margin-top: 14px;}
#hero .badges span {display: inline-block; margin: 0 8px 6px 0; padding: 5px 14px;
  border-radius: 999px; background: rgba(255,255,255,.16); font-size: .78rem;
  letter-spacing: .3px;}
#lang-row {display: flex; justify-content: flex-end !important; margin: 2px 0 0;}
#lang-toggle {min-width: 0 !important; flex-grow: 0 !important; margin-left: auto !important;}
.panel {border: 1px solid var(--border-color-primary) !important;
  border-radius: 16px !important; padding: 16px !important;
  background: var(--background-fill-primary) !important;
  box-shadow: 0 4px 18px rgba(15, 43, 70, .05) !important;}
.panel-title {font-weight: 700 !important; font-size: 1.02rem !important;
  margin: 2px 0 4px !important;}
#generate-btn {border-radius: 12px !important; font-weight: 700 !important;
  letter-spacing: .2px !important;}
.result-card {border: 1px solid var(--border-color-primary) !important;
  border-radius: 14px !important; padding: 12px 14px 8px !important;
  margin-bottom: 10px !important;
  background: var(--background-fill-secondary) !important;}
footer {display: none !important;}
"""


def _hero_html(lang: str) -> str:
    L = I18N[lang]
    badges = "".join(
        f"<span>{b}</span>" for b in
        [L["badge_device"].format(d=DEVICE.upper()), L["badge_engines"], L["badge_slots"]]
    )
    return (f'<div id="hero"><h1>{APP_TITLE}</h1>'
            f'<p>{L["hero_sub"]}</p><div class="badges">{badges}</div></div>')


def _model_choices(lang: str):
    return [(_model_label(k, lang), k) for k in MODEL_REGISTRY]


def create_ui():
    L0 = I18N["vi"]

    with gr.Blocks(title=APP_TITLE, css=CSS) as app:
        lang_state = gr.State("vi")

        with gr.Row(elem_id="lang-row"):
            lang = gr.Radio(
                choices=[("Tiếng Việt", "vi"), ("English", "en")], value="vi",
                show_label=False, container=False, elem_id="lang-toggle",
            )
        hero = gr.HTML(_hero_html("vi"))

        with gr.Accordion(L0["tips_label"], open=False) as tips_acc:
            tips_md = gr.Markdown(L0["tips_md"])

        with gr.Row(equal_height=False):
            with gr.Column(scale=1, elem_classes=["panel"]):
                input_hdr = gr.Markdown(f"### {L0['input_hdr']}",
                                        elem_classes=["panel-title"])
                _sample_ref = ROOT / "reference_audio" / "sample_clean_vi.wav"
                ref_audio = gr.Audio(
                    label=L0["ref_audio"], type="filepath",
                    sources=["upload", "microphone"],
                    value=str(_sample_ref) if _sample_ref.exists() else None,
                )
                ref_text = gr.Textbox(label=L0["ref_text"],
                                      placeholder=L0["ref_text_ph"], lines=2)
                gen_text = gr.Textbox(
                    label=L0["gen_text"], lines=4,
                    value="Xin chào, đây là hệ thống nhân bản giọng nói tiếng Việt.",
                )
                models = gr.CheckboxGroup(choices=_model_choices("vi"),
                                          value=["base"], label=L0["models"])
                with gr.Row():
                    speed = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label=L0["speed"])
                    nfe = gr.Slider(16, 64, value=DEFAULT_NFE, step=4, label=L0["nfe"])
                seed = gr.Number(value=DEFAULT_SEED, precision=0, label=L0["seed"])
                btn = gr.Button(L0["btn"], variant="primary", size="lg",
                                elem_id="generate-btn")

            with gr.Column(scale=1, elem_classes=["panel"]):
                results_hdr = gr.Markdown(f"### {L0['results_hdr']}",
                                          elem_classes=["panel-title"])
                whisper_display = gr.Textbox(label=L0["runinfo"],
                                             interactive=False, lines=2)
                slot_cols, slot_infos, slot_audios = [], [], []
                for i in range(MAX_SLOTS):
                    with gr.Column(visible=False, elem_classes=["result-card"]) as col:
                        info = gr.Markdown("")
                        audio = gr.Audio(label=f"{L0['slot']} {i + 1}",
                                         interactive=False, autoplay=(i == 0))
                    slot_cols.append(col)
                    slot_infos.append(info)
                    slot_audios.append(audio)

        with gr.Accordion(L0["examples_label"], open=True) as examples_acc:
            gr.Examples(
                examples=[
                    ["Trí tuệ nhân tạo đang phát triển rất mạnh mẽ trong những năm gần đây."],
                    ["Chào mừng bạn đến với hệ thống tổng hợp giọng nói tiếng Việt."],
                    ["Thời tiết hôm nay tại thành phố Hồ Chí Minh khá đẹp và trong sáng."],
                ],
                inputs=[gen_text], label="",
            )

        # ── Chuyển ngôn ngữ: cập nhật mọi nhãn trên giao diện ──
        def switch_lang(lang_v, current_models):
            L = I18N[lang_v]
            return [
                lang_v,
                _hero_html(lang_v),
                gr.update(label=L["tips_label"]),
                L["tips_md"],
                f"### {L['input_hdr']}",
                gr.update(label=L["ref_audio"]),
                gr.update(label=L["ref_text"], placeholder=L["ref_text_ph"]),
                gr.update(label=L["gen_text"]),
                gr.update(label=L["models"], choices=_model_choices(lang_v),
                          value=current_models),
                gr.update(label=L["speed"]),
                gr.update(label=L["nfe"]),
                gr.update(label=L["seed"]),
                gr.update(value=L["btn"]),
                f"### {L['results_hdr']}",
                gr.update(label=L["runinfo"]),
                gr.update(label=L["examples_label"]),
            ] + [gr.update(label=f"{L['slot']} {i + 1}") for i in range(MAX_SLOTS)]

        lang.change(
            fn=switch_lang,
            inputs=[lang, models],
            outputs=[lang_state, hero, tips_acc, tips_md, input_hdr,
                     ref_audio, ref_text, gen_text, models, speed, nfe, seed,
                     btn, results_hdr, whisper_display, examples_acc] + slot_audios,
        )

        # ── Tạo giọng: whisper_display + (cột, info, audio) cho từng slot ──
        outputs = [whisper_display]
        for i in range(MAX_SLOTS):
            outputs += [slot_cols[i], slot_infos[i], slot_audios[i]]

        btn.click(
            fn=synthesize,
            inputs=[models, ref_audio, ref_text, gen_text, speed, nfe, seed, lang_state],
            outputs=outputs,
            concurrency_limit=1,          # chạy tuần tự cho ổn định
        )
    return app


# ─── Điểm vào ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=APP_TITLE)
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--share", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("=" * 60)
    print(f"  {APP_TITLE}")
    print(f"  Thiết bị: {DEVICE}")
    print(f"  Mô hình khả dụng: {', '.join(MODEL_REGISTRY)}")
    print("=" * 60)

    # Khởi động sẵn mô hình gốc; nếu GPU/MPS lỗi thì tự lùi về CPU.
    try:
        get_model("base")
        print("[VVCS] Sẵn sàng!")
    except Exception as e:
        if DEVICE != "cpu":
            print(f"[VVCS] Nạp model trên '{DEVICE}' lỗi ({e}); chuyển sang CPU.")
            DEVICE = "cpu"
            _model_cache.clear(); _vocoder = None
            try:
                get_model("base")
                print("[VVCS] Sẵn sàng (CPU)!")
            except Exception as e2:
                print(f"[VVCS] Cảnh báo: vẫn không nạp được mô hình gốc: {e2}")
        else:
            print(f"[VVCS] Cảnh báo: không nạp sẵn được mô hình gốc: {e}")

    app = create_ui()
    app.queue(default_concurrency_limit=1)   # serialize toàn bộ request
    print(f"\nServer: http://localhost:{args.port}")
    app.launch(server_name=args.host, server_port=args.port, share=args.share,
               theme=gr.themes.Soft())
