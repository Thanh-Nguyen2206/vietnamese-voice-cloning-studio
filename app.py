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
import html as html_mod
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
        step = ck.parent.name.replace("step_", "").lstrip("0") or "0"
        found.append((f"ft_{ck.parent.name}", f"step {step}", ck))
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
    # Checkpoint fine-tune trong checkpoints/step_*/model.pt LUÔN hiển thị: các
    # checkpoint huấn luyện trên dữ liệu giả trước đây đã bị xoá khỏi đĩa (03/07/2026),
    # nên bất kỳ checkpoint nào xuất hiện ở đây từ nay về sau chắc chắn là kết quả
    # fine-tune THẬT của người dùng trên giọng của chính họ — không còn lý do để ẩn
    # hay gắn nhãn "dữ liệu demo" mặc định nữa.
    for key, label, ck in _discover_finetuned():
        reg[key] = {"label": f"Giọng cá nhân (tinh chỉnh, {label})",
                    "label_en": f"Personal voice (fine-tuned, {label})",
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


def _quality_parts(wave: np.ndarray, sr: int, lang: str = "vi"):
    """Trả về (verdict, ok?, rms, flat, dur) — dùng cho cả text lẫn thẻ HTML."""
    v = _VERDICTS.get(lang, _VERDICTS["vi"])
    rms = float(np.sqrt(np.mean(wave ** 2))) if wave.size else 0.0
    flat = _spectral_flatness(wave)
    dur = len(wave) / sr if sr else 0.0
    if rms < 0.02:
        return v["silent"], False, rms, flat, dur
    if flat > 0.30:
        return v["noisy"], False, rms, flat, dur
    return v["ok"], True, rms, flat, dur


def _quality_note(wave: np.ndarray, sr: int, lang: str = "vi") -> str:
    verdict, _ok, rms, flat, dur = _quality_parts(wave, sr, lang)
    v = _VERDICTS.get(lang, _VERDICTS["vi"])
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


MAX_SLOTS = 6   # đủ chỗ so sánh CẢ 6 engine cùng lúc


def _model_label(key: str, lang: str) -> str:
    entry = MODEL_REGISTRY[key]
    return entry["label"] if lang == "vi" else entry.get("label_en", entry["label"])


def _slot_html(label, state, lang, metrics=None, gen_s=None, err=""):
    """Thẻ trạng thái HTML cho từng slot: queued / generating / done / error."""
    L = I18N.get(lang) or I18N["vi"]
    v = _VERDICTS.get(lang, _VERDICTS["vi"])
    title = f'<div class="slot-title">{html_mod.escape(label)}</div>'
    if state == "queued":
        pill = f'<span class="pill pill-queued">{L["queued_pill"]}</span>'
        body = ""
    elif state == "generating":
        pill = (f'<span class="pill pill-gen"><span class="spinner"></span>'
                f'{L["generating_pill"]}</span>')
        body = '<div class="indet"><div></div></div>'
    elif state == "done":
        verdict, ok, rms, flat, dur = metrics
        pill = (f'<span class="pill pill-done">{L["done_pill"]} · '
                f'{L["gen_time"].format(s=gen_s)}</span>')
        body = ('<div class="chips">'
                f'<span class="chip {"chip-ok" if ok else "chip-warn"}">{html_mod.escape(verdict)}</span>'
                f'<span class="chip">RMS {rms:.3f}</span>'
                f'<span class="chip">{v["flat"]} {flat:.3f}</span>'
                f'<span class="chip">{dur:.1f}s</span></div>')
    else:  # error
        pill = f'<span class="pill pill-err">{L["error_pill"]}</span>'
        body = f'<div class="slot-err">{html_mod.escape(str(err)[:300])}</div>'
    return f'<div class="slot-head">{title}{pill}</div>{body}'


def _status_html(done, total, current_short, lang, total_s=None, extra=""):
    """Thanh trạng thái tổng: tiến độ x/y + engine đang chạy (thay overlay Gradio)."""
    L = I18N.get(lang) or I18N["vi"]
    if total_s is not None:
        left = (f'<span class="pill pill-done">'
                f'{L["status_done"].format(n=total, s=total_s)}</span>')
        bar = '<div class="bar"><div style="width:100%"></div></div>'
    else:
        left = (f'<span class="pill pill-gen"><span class="spinner"></span>'
                f'{L["status_running"].format(i=done + 1, n=total, name=html_mod.escape(current_short))}</span>')
        pct = int(100 * done / max(total, 1))
        bar = f'<div class="bar"><div style="width:{pct}%"></div></div>'
    note = f'<span class="status-note">{html_mod.escape(extra)}</span>' if extra else ""
    return f'<div class="status"><div class="status-row">{left}{note}</div>{bar}</div>'


def synthesize(model_keys, ref_audio_path, ref_text, gen_text, speed, nfe, seed,
               lang="vi"):
    """Generator: sinh audio cho từng mô hình và CẬP NHẬT UI NGAY khi mỗi mô hình
    xong. Engine nhanh chạy trước. Trạng thái hiển thị bằng thẻ HTML riêng cho từng
    card + thanh tiến độ tổng (event dùng show_progress='hidden' nên KHÔNG còn
    lớp overlay của Gradio che kết quả trong lúc chạy).
    """
    L = I18N.get(lang) or I18N["vi"]
    if not model_keys:
        raise gr.Error(L["err_no_model"])
    # Giữ đúng thứ tự hiển thị trong danh sách chọn mô hình (thứ tự MODEL_REGISTRY),
    # để card kết quả xếp cùng thứ tự với các ô checkbox bên trái — không xáo theo
    # tốc độ engine nữa (người dùng phản hồi thứ tự "nhanh trước" gây khó theo dõi).
    _registry_order = {k: i for i, k in enumerate(MODEL_REGISTRY)}
    keys = sorted(set(model_keys), key=lambda k: _registry_order.get(k, 9))[:MAX_SLOTS]
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

    labels = [_model_label(k, lang) for k in keys]
    shorts = [lb.split("—")[0].strip() for lb in labels]
    t_start = time.time()

    # Trạng thái từng slot + cơ chế "dirty": mỗi lần yield CHỈ gửi giá trị cho slot
    # vừa thay đổi (gr.update() rỗng cho slot khác) → không reset audio đang phát.
    slot_html = [""] * MAX_SLOTS
    slot_audio = [None] * MAX_SLOTS
    dirty = set(range(MAX_SLOTS))

    def render(status, note):
        updates = [status, note]
        for i in range(MAX_SLOTS):
            if i in dirty:
                updates += [gr.update(visible=i < len(keys)),
                            gr.update(value=slot_html[i]),
                            gr.update(value=slot_audio[i])]
            else:
                updates += [gr.update(), gr.update(), gr.update()]
        dirty.clear()
        return updates

    # Hiện ngay toàn bộ slot ở trạng thái "trong hàng đợi".
    for i in range(len(keys)):
        slot_html[i] = _slot_html(labels[i], "queued", lang)
    yield render(_status_html(0, len(keys), shorts[0], lang), L["preparing"])

    # Chỉ F5-TTS cần transcript của audio mẫu (XTTS nhân bản trực tiếp từ audio,
    # các engine còn lại dùng giọng cố định). Chỉ chạy Whisper khi có chọn F5-TTS.
    _ref_text = ref_text.strip() if ref_text and ref_text.strip() else ""
    whisper_note = ""
    if "f5tts" in engines_sel and not _ref_text and ref_audio_path:
        yield render(_status_html(0, len(keys), shorts[0], lang,
                                  extra=L["whisper_progress"]), L["preparing"])
        _ref_text = _transcribe_whisper(ref_audio_path)
        whisper_note = (L["whisper_note"].format(t=_ref_text)
                        if _ref_text else L["whisper_na"])
    _ref_text = _normalize_vi(_ref_text, add_end_punct=False)

    seed_line = f"Seed: {seed}"
    note_out = f"{whisper_note}\n{seed_line}".strip() if whisper_note else seed_line

    for i, key in enumerate(keys):
        slot_html[i] = _slot_html(labels[i], "generating", lang)
        dirty.add(i)
        yield render(_status_html(i, len(keys), shorts[i], lang), note_out)
        try:
            # Cùng seed cho mọi model → so sánh công bằng (cùng nhiễu khởi tạo).
            t0 = time.time()
            wave, sr = _infer_one(key, ref_audio_path, _ref_text,
                                  gen_text_n, speed, nfe, seed=seed)
            gen_s = time.time() - t0
            sf.write(str(OUTPUT_DIR / f"{key}.wav"), wave, sr)
            slot_html[i] = _slot_html(labels[i], "done", lang,
                                      metrics=_quality_parts(wave, sr, lang),
                                      gen_s=gen_s)
            slot_audio[i] = (sr, wave)
        except Exception as e:
            traceback.print_exc()
            slot_html[i] = _slot_html(labels[i], "error", lang, err=e)
            slot_audio[i] = None
        dirty.add(i)
        if i + 1 < len(keys):
            status = _status_html(i + 1, len(keys), shorts[i + 1], lang)
        else:
            status = _status_html(len(keys), len(keys), "", lang,
                                  total_s=time.time() - t_start)
        yield render(status, note_out)


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
        preparing="Đang chuẩn bị...",
        queued="Trong hàng đợi — engine nhanh sẽ chạy trước...",
        generating="Đang tạo giọng nói...",
        queued_pill="Chờ đến lượt",
        generating_pill="Đang tạo...",
        done_pill="Hoàn tất",
        error_pill="Lỗi",
        status_running="Đang tạo {i}/{n}: {name}",
        status_done="Xong cả {n} mô hình · tổng {s:.0f} giây",
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
        preparing="Preparing...",
        queued="Queued — faster engines run first...",
        generating="Generating speech...",
        queued_pill="Queued",
        generating_pill="Generating...",
        done_pill="Done",
        error_pill="Error",
        status_running="Generating {i}/{n}: {name}",
        status_done="All {n} models done · {s:.0f} s total",
    ),
}


# ─── Giao diện ──────────────────────────────────────────────────────────────────

CSS = """
/* ───────────────── Design system ───────────────── */
.gradio-container {max-width: 1240px !important; margin: 0 auto !important;}
footer {display: none !important;}

/* Hero */
#hero {position: relative; overflow: hidden; border-radius: 20px; padding: 34px 34px 26px;
  background: linear-gradient(135deg, #1e1b4b 0%, #312e81 42%, #4f46e5 100%);
  color: #ffffff; margin: 6px 0 6px;
  box-shadow: 0 12px 34px rgba(49, 46, 129, .35);}
#hero::after {content: ""; position: absolute; right: -120px; top: -120px;
  width: 340px; height: 340px; border-radius: 50%;
  background: radial-gradient(circle, rgba(139,92,246,.35) 0%, transparent 70%);}
#hero h1 {margin: 0 0 10px; font-size: 2rem; letter-spacing: .2px; color: #ffffff;}
#hero p {margin: 0; opacity: .92; font-size: .98rem; line-height: 1.6; max-width: 900px;}
#hero .badges {margin-top: 16px; position: relative; z-index: 1;}
#hero .badges span {display: inline-block; margin: 0 8px 6px 0; padding: 5px 15px;
  border-radius: 999px; background: rgba(255,255,255,.13);
  border: 1px solid rgba(255,255,255,.22); font-size: .78rem; letter-spacing: .3px;}

/* Language toggle */
#lang-row {display: flex; justify-content: flex-end !important; margin: 2px 0 0;}
#lang-toggle {width: 300px !important; min-width: 300px !important;
  flex: 0 0 300px !important; margin-left: auto !important;
  border: 2px solid #6366f1 !important; border-radius: 14px !important;
  padding: 8px 16px !important; background: var(--background-fill-primary) !important;
  box-shadow: 0 3px 14px rgba(99, 102, 241, .22) !important;}
#lang-toggle .wrap {gap: 10px !important; flex-direction: row !important;}
#lang-toggle label {font-weight: 600 !important; white-space: nowrap !important;}

/* Panels & cards */
.panel {border: 1px solid var(--border-color-primary) !important;
  border-radius: 18px !important; padding: 18px !important;
  background: var(--background-fill-primary) !important;
  box-shadow: 0 6px 24px rgba(30, 27, 75, .07) !important;}
.panel-title {font-weight: 700 !important; font-size: 1.05rem !important;
  margin: 2px 0 6px !important; padding-left: 12px !important;
  border-left: 4px solid #6366f1 !important;}
.result-card {border: 1px solid var(--border-color-primary) !important;
  border-radius: 16px !important; padding: 14px 16px 10px !important;
  margin-bottom: 12px !important;
  background: var(--background-fill-secondary) !important;
  transition: border-color .25s ease, box-shadow .25s ease;}
.result-card:hover {border-color: #6366f1 !important;
  box-shadow: 0 4px 18px rgba(99, 102, 241, .14) !important;}

/* Generate button */
#generate-btn {border-radius: 14px !important; font-weight: 700 !important;
  letter-spacing: .3px !important; font-size: 1.02rem !important;
  background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%) !important;
  border: none !important; color: #fff !important;
  box-shadow: 0 6px 18px rgba(79, 70, 229, .35) !important;
  transition: transform .15s ease, box-shadow .15s ease !important;}
#generate-btn:hover {transform: translateY(-1px);
  box-shadow: 0 9px 24px rgba(79, 70, 229, .45) !important;}

/* Status pills & chips (slot cards + status bar) */
@keyframes vvcs-spin {to {transform: rotate(360deg);}}
@keyframes vvcs-slide {0% {transform: translateX(-100%);} 100% {transform: translateX(400%);}}
.spinner {display: inline-block; width: 12px; height: 12px; margin-right: 8px;
  border: 2px solid rgba(129,140,248,.35); border-top-color: #6366f1;
  border-radius: 50%; animation: vvcs-spin .8s linear infinite; vertical-align: -2px;}
.slot-head {display: flex; justify-content: space-between; align-items: center;
  gap: 10px; margin: 2px 0 8px; flex-wrap: wrap;}
.slot-title {font-weight: 700; font-size: .95rem; line-height: 1.4;}
.pill {padding: 4px 13px; border-radius: 999px; font-size: .75rem;
  font-weight: 600; white-space: nowrap;}
.pill-queued {background: rgba(148,163,184,.16); color: #94a3b8;
  border: 1px solid rgba(148,163,184,.3);}
.pill-gen {background: rgba(99,102,241,.14); color: #818cf8;
  border: 1px solid rgba(99,102,241,.35);}
.pill-done {background: rgba(34,197,94,.13); color: #22c55e;
  border: 1px solid rgba(34,197,94,.32);}
.pill-err {background: rgba(239,68,68,.13); color: #ef4444;
  border: 1px solid rgba(239,68,68,.32);}
.indet {height: 4px; border-radius: 99px; overflow: hidden; margin: 4px 0 8px;
  background: rgba(99,102,241,.12);}
.indet > div {width: 30%; height: 100%; border-radius: 99px;
  background: linear-gradient(90deg, #6366f1, #a78bfa);
  animation: vvcs-slide 1.2s ease-in-out infinite;}
.chips {display: flex; flex-wrap: wrap; gap: 6px; margin: 2px 0 10px;}
.chip {padding: 3px 11px; border-radius: 9px; font-size: .73rem;
  background: rgba(148,163,184,.1); border: 1px solid var(--border-color-primary);}
.chip-ok {background: rgba(34,197,94,.1); color: #16a34a;
  border-color: rgba(34,197,94,.3); font-weight: 600;}
.chip-warn {background: rgba(245,158,11,.12); color: #d97706;
  border-color: rgba(245,158,11,.3); font-weight: 600;}
.slot-err {font-size: .8rem; color: #ef4444; margin: 2px 0 8px; line-height: 1.5;}

/* Overall status bar */
#status-bar .status {border: 1px solid var(--border-color-primary); border-radius: 14px;
  padding: 12px 14px; margin-bottom: 12px;
  background: var(--background-fill-secondary);}
#status-bar .status-row {display: flex; justify-content: space-between;
  align-items: center; gap: 10px; margin-bottom: 9px; flex-wrap: wrap;}
#status-bar .status-note {font-size: .78rem; opacity: .75;}
#status-bar .bar {height: 6px; border-radius: 99px; overflow: hidden;
  background: rgba(99,102,241,.12);}
#status-bar .bar > div {height: 100%; border-radius: 99px;
  background: linear-gradient(90deg, #4f46e5, #a78bfa);
  transition: width .5s ease;}
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

    with gr.Blocks(title=APP_TITLE) as app:
        lang_state = gr.State("vi")

        with gr.Row(elem_id="lang-row"):
            lang = gr.Radio(
                choices=[("Tiếng Việt", "vi"), ("English", "en")], value="vi",
                label="Ngôn ngữ · Language", elem_id="lang-toggle",
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
                status_bar = gr.HTML("", elem_id="status-bar")
                whisper_display = gr.Textbox(label=L0["runinfo"],
                                             interactive=False, lines=2)
                _wave_opts = gr.WaveformOptions(
                    waveform_color="#818cf8", waveform_progress_color="#4f46e5")
                slot_cols, slot_infos, slot_audios = [], [], []
                for i in range(MAX_SLOTS):
                    with gr.Column(visible=False, elem_classes=["result-card"]) as col:
                        info = gr.HTML("")
                        audio = gr.Audio(label=f"{L0['slot']} {i + 1}",
                                         interactive=False, autoplay=(i == 0),
                                         waveform_options=_wave_opts)
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

        # ── Tạo giọng: status_bar + whisper_display + (cột, info, audio) từng slot ──
        outputs = [status_bar, whisper_display]
        for i in range(MAX_SLOTS):
            outputs += [slot_cols[i], slot_infos[i], slot_audios[i]]

        btn.click(
            fn=synthesize,
            inputs=[models, ref_audio, ref_text, gen_text, speed, nfe, seed, lang_state],
            outputs=outputs,
            concurrency_limit=1,          # chạy tuần tự cho ổn định
            # QUAN TRỌNG: tắt overlay loading của Gradio — trước đây nó phủ thanh
            # tiến trình lên TẤT CẢ card kết quả cho tới khi event kết thúc, làm
            # người dùng tưởng phải chờ đủ 6 mô hình. Trạng thái giờ hiển thị bằng
            # status_bar + huy hiệu trên từng card, cập nhật theo từng yield.
            show_progress="hidden",
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
               css=CSS,
               theme=gr.themes.Soft(
                   primary_hue="indigo", secondary_hue="violet", radius_size="lg",
                   font=[gr.themes.GoogleFont("Be Vietnam Pro"),
                         "ui-sans-serif", "system-ui", "sans-serif"],
               ))
