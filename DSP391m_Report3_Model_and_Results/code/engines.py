"""
engines.py — Các engine TTS bổ sung để SO SÁNH nhiều mô hình trên cùng site.

Thiết kế để KHÔNG ảnh hưởng tới đường F5-TTS trong app.py:
  • Không import gì nặng ở top-level (chỉ import khi engine được chọn).
  • Mỗi engine nạp lười (lazy) và cache riêng, có khoá đồng bộ riêng.
  • Nếu một engine lỗi khi nạp, app.py bắt ngoại lệ theo từng ô kết quả nên
    F5-TTS và các engine khác vẫn chạy bình thường.

Các engine:
  • XTTS-v2 (viXTTS, capleaf/viXTTS): NHÂN BẢN giọng từ audio mẫu + đọc tiếng Việt.
  • Bark (suno/bark-small): mô hình so sánh; dùng giọng preset (KHÔNG nhân bản).
  • MMS-TTS (facebook/mms-tts-vie): VITS của Meta, tiếng Việt, chạy local rất nhanh.
  • Piper (rhasspy/piper, giọng vi_VN-vais1000): ONNX local, nhanh nhất trên CPU.
  • Edge-TTS (rany2/edge-tts, giọng vi-VN-HoaiMyNeural): giọng neural thương mại của
    Microsoft — baseline chất lượng cao; CHẠY QUA CLOUD, cần internet.
"""
import os
import re
import threading

import numpy as np

from voice_studio.config import load_config

_CONFIG = load_config()

# Khoá RIÊNG cho từng engine: engine này tải/nạp mô hình lần đầu (có thể mất
# nhiều phút) sẽ KHÔNG chặn engine khác. (Trước đây dùng một khoá chung nên
# Edge-TTS bị kẹt chờ XTTS/Bark tải mô hình → tưởng như "không hoạt động".)
_XTTS_LOCK  = threading.RLock()
_BARK_LOCK  = threading.RLock()
_MMS_LOCK   = threading.RLock()
_PIPER_LOCK = threading.RLock()
TARGET_SR = 24000


def friendly_engine_error(engine: str, error: BaseException) -> RuntimeError:
    """Map dependency, timeout and memory failures to safe end-user messages."""
    if isinstance(error, ModuleNotFoundError):
        package = error.name or "optional dependency"
        return RuntimeError(
            f"{engine} chưa sẵn sàng vì thiếu '{package}'. Cài dependencies trong requirements.txt."
        )
    message = str(error)
    if "out of memory" in message.lower():
        return RuntimeError(
            f"{engine} hết bộ nhớ. Hãy bỏ bớt engine, dùng CPU hoặc khởi động lại app."
        )
    return RuntimeError(f"{engine} không thể tạo audio: {message}")

# ─────────────────────────── XTTS-v2 (viXTTS) ────────────────────────────────
_XTTS_REPO = "capleaf/viXTTS"
_xtts_model = None


def _shim_transformers():
    """coqui-tts import 'isin_mps_friendly' vốn đã bị bỏ ở transformers mới → vá lại."""
    import torch
    import transformers.pytorch_utils as pu
    if not hasattr(pu, "isin_mps_friendly"):
        def isin_mps_friendly(elements, test_elements):
            return torch.isin(elements, test_elements)
        pu.isin_mps_friendly = isin_mps_friendly


def _num_to_vi(s: str) -> str:
    try:
        from num2words import num2words
        return num2words(int(s), lang="vi")
    except Exception:
        return s


def _load_xtts(device: str):
    global _xtts_model
    if _xtts_model is not None:
        return _xtts_model

    _shim_transformers()
    from huggingface_hub import snapshot_download
    model_dir = snapshot_download(_XTTS_REPO)

    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    config = XttsConfig()
    config.load_json(os.path.join(model_dir, "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=model_dir, use_deepspeed=False)
    model.to(device)
    model.eval()

    # Stock coqui-tts không hỗ trợ tokenizer tiếng Việt → vá tối thiểu cho 'vi'.
    tok = model.tokenizer
    try:
        tok.char_limits["vi"] = 250
    except Exception:
        pass
    _orig_pre = tok.preprocess_text

    def _pre(txt, lang):
        if lang == "vi":
            txt = txt.replace('"', "").lower()
            txt = re.sub(r"\d+", lambda m: _num_to_vi(m.group()), txt)
            return re.sub(r"\s+", " ", txt).strip()
        return _orig_pre(txt, lang)

    tok.preprocess_text = _pre
    _xtts_model = model
    return model


def _clean_reference(ref_audio):
    """XTTS rất nhạy với audio mẫu bẩn (nhạc nền, im lặng, lệch mức). Dùng lại bước
    tiền xử lý đã kiểm chứng của F5-TTS (cắt im lặng + chuẩn hoá) để tăng độ ổn định."""
    try:
        from f5_tts.infer.utils_infer import preprocess_ref_audio_text
        # ref_text != "" để BỎ bước ASR (Whisper) không cần thiết ở đây.
        clean_path, _ = preprocess_ref_audio_text(
            ref_audio, "x", show_info=lambda *a, **k: None
        )
        return clean_path
    except Exception:
        return ref_audio


def xtts_infer(ref_audio, ref_text, gen_text, speed, seed, device="cpu"):
    """Nhân bản giọng từ ref_audio và đọc gen_text bằng tiếng Việt (24 kHz).

    LƯU Ý: XTTS nhạy hơn F5-TTS nhiều với chất lượng audio mẫu — mẫu có nhạc nền
    hoặc nhiễu dễ khiến XTTS đọc sai từ. Nên dùng mẫu SẠCH, một giọng.
    """
    import torch
    ref = _clean_reference(ref_audio)
    with _XTTS_LOCK:
        model = _load_xtts(device)
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
            audio_path=[ref], gpt_cond_len=10, max_ref_length=30,
        )
        if seed is not None:
            torch.manual_seed(int(seed))
        out = model.inference(
            gen_text, "vi", gpt_cond_latent, speaker_embedding,
            temperature=0.3, length_penalty=1.0, repetition_penalty=5.0,
            top_k=50, top_p=0.85, speed=float(speed), enable_text_splitting=True,
        )
    wav = np.asarray(out["wav"], dtype=np.float32)
    return wav, TARGET_SR


# ─────────────────────────────── Bark (Suno) ─────────────────────────────────
_BARK_REPO = "suno/bark-small"
_BARK_PRESET = "v2/en_speaker_6"   # Bark không có preset tiếng Việt → dùng giọng chung
_bark_model = None
_bark_proc = None


def _load_bark(device: str):
    global _bark_model, _bark_proc
    if _bark_model is not None:
        return _bark_model, _bark_proc
    from transformers import AutoProcessor, BarkModel
    _bark_proc = AutoProcessor.from_pretrained(_BARK_REPO)
    _bark_model = BarkModel.from_pretrained(_BARK_REPO).to(device)
    _bark_model.eval()
    return _bark_model, _bark_proc


def bark_infer(ref_audio, ref_text, gen_text, speed, seed, device="cpu"):
    """Sinh giọng bằng Bark (giọng preset — KHÔNG dùng ref_audio). Chậm trên CPU."""
    import torch
    with _BARK_LOCK:
        model, proc = _load_bark(device)
        if seed is not None:
            torch.manual_seed(int(seed))
        inputs = proc(gen_text, voice_preset=_BARK_PRESET)
        inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        audio = model.generate(**inputs)
    sr = int(model.generation_config.sample_rate)
    wav = audio.detach().cpu().numpy().squeeze().astype(np.float32)
    return wav, sr


# ───────────────────────── MMS-TTS (Meta, VITS, tiếng Việt) ──────────────────────
_MMS_REPO = "facebook/mms-tts-vie"
_mms_model = None
_mms_tok = None


def _load_mms(device: str):
    global _mms_model, _mms_tok
    if _mms_model is None:
        from transformers import AutoTokenizer, VitsModel
        _mms_tok = AutoTokenizer.from_pretrained(_MMS_REPO)
        _mms_model = VitsModel.from_pretrained(_MMS_REPO).to(device)
        _mms_model.eval()
    return _mms_model, _mms_tok


def mms_infer(ref_audio, ref_text, gen_text, speed, seed, device="cpu"):
    """VITS tiếng Việt của Meta — MỘT giọng cố định (không nhân bản), 16 kHz, rất nhanh."""
    import torch
    with _MMS_LOCK:
        model, tok = _load_mms(device)
        if seed is not None:
            torch.manual_seed(int(seed))          # VITS có thành phần ngẫu nhiên
        model.speaking_rate = float(speed)         # tốc độ đọc của VITS
        inputs = tok(gen_text, return_tensors="pt").to(device)
        with torch.no_grad():
            wav_t = model(**inputs).waveform
    sr = int(model.config.sampling_rate)
    return wav_t.squeeze().cpu().numpy().astype(np.float32), sr


# ───────────────────── Piper (rhasspy, ONNX, giọng vi_VN) ────────────────────────
_PIPER_VOICE = "vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium"
_piper_voice = None


def _load_piper():
    global _piper_voice
    if _piper_voice is None:
        from huggingface_hub import hf_hub_download
        from piper import PiperVoice
        onnx = hf_hub_download("rhasspy/piper-voices", f"{_PIPER_VOICE}.onnx")
        cfg = hf_hub_download("rhasspy/piper-voices", f"{_PIPER_VOICE}.onnx.json")
        _piper_voice = PiperVoice.load(onnx, config_path=cfg)
    return _piper_voice


def piper_infer(ref_audio, ref_text, gen_text, speed, seed, device="cpu"):
    """Piper ONNX — MỘT giọng vi_VN cố định, cực nhanh trên CPU, kết quả tất định."""
    import io
    import wave as wave_mod

    import soundfile as sf
    from piper import SynthesisConfig

    with _PIPER_LOCK:
        voice = _load_piper()
        syn = SynthesisConfig(length_scale=1.0 / max(float(speed), 0.1))
        buf = io.BytesIO()
        with wave_mod.open(buf, "wb") as wf:
            voice.synthesize_wav(gen_text, wf, syn_config=syn)
        buf.seek(0)
        wav, sr = sf.read(buf, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return np.asarray(wav, dtype=np.float32), int(sr)


# ─────────────── Edge-TTS (Microsoft neural, cloud — cần internet) ───────────────
_EDGE_VOICE = "vi-VN-HoaiMyNeural"


_EDGE_TIMEOUT_S = _CONFIG.engine_timeout


def edge_infer(ref_audio, ref_text, gen_text, speed, seed, device="cpu"):
    """Giọng neural vi-VN của Microsoft qua edge-tts (baseline thương mại, CLOUD).

    Không giữ khoá nào (không có mô hình local) và có timeout cứng: lỗi mạng sẽ
    báo rõ ràng ngay thay vì treo giao diện.
    """
    import asyncio
    import tempfile

    import edge_tts
    import librosa

    pct = int(round((float(speed) - 1.0) * 100))
    rate = f"{'+' if pct >= 0 else ''}{pct}%"

    async def _run(path):
        comm = edge_tts.Communicate(gen_text, _EDGE_VOICE, rate=rate)
        await asyncio.wait_for(comm.save(path), timeout=_EDGE_TIMEOUT_S)

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        tmp = f.name
    try:
        asyncio.run(_run(tmp))
        wav, sr = librosa.load(tmp, sr=TARGET_SR, mono=True)
    except (asyncio.TimeoutError, TimeoutError):
        raise RuntimeError(
            f"Edge-TTS không phản hồi sau {_EDGE_TIMEOUT_S}s — engine này chạy qua "
            f"cloud của Microsoft, hãy kiểm tra kết nối internet. "
            f"(Edge-TTS timed out after {_EDGE_TIMEOUT_S}s — it needs internet access.)"
        )
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(
            f"Edge-TTS lỗi mạng/cloud (cần internet): {e} "
            f"(Edge-TTS network/cloud error — internet required.)"
        )
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
    return np.asarray(wav, dtype=np.float32), TARGET_SR
