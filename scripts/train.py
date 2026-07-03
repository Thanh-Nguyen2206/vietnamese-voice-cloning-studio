"""
=============================================================================
Phase 3: Fine-tuning for Vietnamese Voice Cloning Studio
File  : scripts/train.py

Pipeline:
  1. Load config YAML
  2. Download & khởi tạo model pre-trained (hynt/F5-TTS-Vietnamese-ViVoice)
  3. Load dataset từ data/processed + metadata.csv
  4. Training loop với Accelerate (FP16 + Gradient Accumulation)
  5. Lưu checkpoint định kỳ + TensorBoard logging

Hardware: NVIDIA T4 (16GB VRAM)
  - batch_size = 1, grad_accum = 8 → effective batch = 8
  - Mixed Precision FP16 → tiết kiệm ~50% VRAM

Cách chạy:
  python scripts/train.py --config configs/train_config.yaml

=============================================================================
"""

import os
import sys
import math
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import yaml
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from accelerate import Accelerator
from accelerate.utils import set_seed
from safetensors.torch import load_file as safetensors_load
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm

from f5_tts.model import CFM, DiT
from f5_tts.model.utils import get_tokenizer


# =============================================================================
# LOGGING SETUP
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# 1. CONFIGURATION
# =============================================================================
def load_config(yaml_path: str) -> Dict[str, Any]:
    """
    Load toàn bộ config từ file YAML.

    Args:
        yaml_path: Đường dẫn tới file train_config.yaml

    Returns:
        Dict chứa tất cả hyperparameters
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info(f"📋 Loaded config from: {yaml_path}")
    return config


# =============================================================================
# 2. DATASET
# =============================================================================
class VoiceDataset(Dataset):
    """
    Dataset cho F5-TTS fine-tuning.

    Đọc metadata.csv để mapping: audio_file → text transcript

    Mỗi sample trả về:
      - audio: waveform tensor [T_audio] (24kHz, mono)
      - text:  chuỗi text gốc (tokenize sau trong collate)
      - audio_len: số samples thực tế

    Hỗ trợ 2 format metadata (delimiter='|'):
      Format A (2 cột): /absolute/path/to/file.wav|Text transcript
      Format B (5 cột): audio_file|duration_sec|snr_db|source_file|text
    """

    def __init__(
        self,
        processed_dir: str,
        metadata_path: str,
        sample_rate: int = 24000,
    ):
        self.processed_dir = Path(processed_dir)
        self.sample_rate = sample_rate
        self.samples = []

        # Đọc metadata.csv
        if not Path(metadata_path).exists():
            raise FileNotFoundError(f"Metadata file không tìm thấy: {metadata_path}")

        with open(metadata_path, "r", encoding="utf-8") as f:
            all_lines = f.readlines()

        skipped = 0

        for line_num, line in enumerate(all_lines, start=1):
            line = line.strip()
            if not line:
                continue

            # Bỏ qua header (dòng chứa tên cột)
            if line_num == 1 and "audio_file" in line.lower():
                logger.info(f"   Bỏ qua header: {line}")
                continue

            parts = line.split("|")

            if len(parts) == 2:
                # Format A: /absolute/path/file.wav|Text
                audio_ref = parts[0].strip()
                text = parts[1].strip()
            elif len(parts) >= 5:
                # Format B: audio_file|duration|snr|source|text
                audio_ref = parts[0].strip()
                text = parts[4].strip()
            else:
                logger.warning(f"⚠️ Dòng {line_num}: format không hợp lệ "
                               f"(cần 2 hoặc 5 cột, có {len(parts)}): {line}")
                skipped += 1
                continue

            # Bỏ qua nếu chưa có transcript
            if not text:
                logger.warning(f"⚠️ Dòng {line_num}: chưa có text cho '{audio_ref}', bỏ qua")
                skipped += 1
                continue

            # Xác định đường dẫn audio (hỗ trợ cả absolute và relative path)
            audio_ref_path = Path(audio_ref)
            if audio_ref_path.is_absolute():
                # Đường dẫn tuyệt đối → dùng trực tiếp
                audio_path = audio_ref_path
            else:
                # Đường dẫn tương đối → nối với processed_dir
                audio_path = self.processed_dir / audio_ref

            if not audio_path.exists():
                logger.warning(f"⚠️ Dòng {line_num}: file không tồn tại: {audio_path}")
                skipped += 1
                continue

            self.samples.append({
                "audio_path": str(audio_path),
                "text": text,
            })

        if not self.samples:
            raise ValueError(
                "Không tìm thấy sample nào hợp lệ!\n"
                f"Đã đọc {len(all_lines)} dòng, bỏ qua {skipped} dòng.\n"
                "Kiểm tra:\n"
                "  1. File metadata.csv có đúng format: path|text\n"
                "  2. Các file .wav có tồn tại ở đường dẫn đã ghi không?"
            )

        logger.info(f"📊 Dataset: {len(self.samples)} samples hợp lệ"
                     f" (bỏ qua {skipped} dòng)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load audio bằng torchaudio (nhanh hơn librosa cho training)
        waveform, sr = torchaudio.load(sample["audio_path"])

        # Chuyển về mono nếu cần
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample nếu sr khác target
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Squeeze: [1, T] → [T]
        waveform = waveform.squeeze(0)

        return {
            "audio": waveform,           # [T_audio]
            "text": sample["text"],      # str
            "audio_len": waveform.shape[0],
        }


def create_collate_fn(vocab_char_map: dict, hop_length: int = 256):
    """
    Tạo collate function để gom batch với variable-length sequences.

    F5-TTS CFM model cần:
      - audio: [B, T_max_audio] (padded waveforms)
      - text:  [B, T_max_text]  (padded token IDs)
      - lens:  [B]              (actual mel frame counts, dtype=long)

    ⚠️ QUAN TRỌNG: lens PHẢI là số nguyên (mel frame counts), KHÔNG phải
    float ratios. Hàm mask_from_frac_lengths() trong CFM dùng lens.amax()
    để xác định max sequence length. Nếu truyền ratio ~1.0 thì amax()=1
    → tạo mask shape [B,1] thay vì [B, T_mel] → RuntimeError broadcast.

    Args:
        vocab_char_map: Dict mapping character → token ID
        hop_length: Hop length của mel spectrogram (phải khớp config audio)
    """

    def collate_fn(batch):
        # --- Audio padding ---
        audios = [item["audio"] for item in batch]
        audio_lens = [item["audio_len"] for item in batch]
        max_audio_len = max(audio_lens)

        # Pad tất cả audio về cùng độ dài (pad zeros ở cuối)
        padded_audios = torch.zeros(len(batch), max_audio_len)
        for i, audio in enumerate(audios):
            padded_audios[i, :len(audio)] = audio

        # --- Tính mel frame lengths (SỐ NGUYÊN) ---
        # torchaudio.transforms.MelSpectrogram với center=True (default F5-TTS)
        # có output length là: T_audio // hop_length + 1
        # Fix lỗi Off-by-One kinh điển: phải cộng 1 để khớp với mel_spec.shape[-1]
        mel_lens = torch.tensor(
            [alen // hop_length + 1 for alen in audio_lens],
            dtype=torch.long,
        )

        # --- Text tokenization ---
        texts = [item["text"] for item in batch]
        text_token_lists = []

        for text in texts:
            # Character-level tokenization (khớp với F5-TTS pre-training)
            tokens = []
            for char in text.lower():
                if char in vocab_char_map:
                    tokens.append(vocab_char_map[char])
                else:
                    # Unknown char → skip (hoặc dùng UNK token)
                    pass
            text_token_lists.append(tokens)

        # Pad text tokens
        max_text_len = max(len(t) for t in text_token_lists) if text_token_lists else 1
        padded_text = torch.zeros(len(batch), max_text_len, dtype=torch.long)
        for i, tokens in enumerate(text_token_lists):
            if tokens:
                padded_text[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)

        return {
            "audio": padded_audios,     # [B, T_max_audio]
            "text": padded_text,        # [B, T_max_text]
            "lens": mel_lens,           # [B] — actual mel frame counts (long)
        }

    return collate_fn


# =============================================================================
# 3. MODEL BUILDING & LOADING
# =============================================================================
def build_model(config: Dict[str, Any], vocab_char_map: dict) -> CFM:
    """
    Khởi tạo model F5-TTS (CFM + DiT) với kiến trúc khớp pre-trained.

    CFM = Conditional Flow Matching wrapper
    DiT = Diffusion Transformer backbone (~336M params)

    Args:
        config: Dict config từ YAML
        vocab_char_map: Dict mapping char → token ID

    Returns:
        CFM model (chưa load weights)
    """
    dit_cfg = config["model"]["dit"]
    audio_cfg = config["audio"]

    # Tạo DiT backbone
    dit = DiT(
        dim=dit_cfg["dim"],
        depth=dit_cfg["depth"],
        heads=dit_cfg["heads"],
        ff_mult=dit_cfg["ff_mult"],
        text_dim=dit_cfg["text_dim"],
        conv_layers=dit_cfg["conv_layers"],
        # PHẢI khớp mô hình gốc: chính DiT tự +1 nội bộ cho filler token, nên ở đây
        # truyền đúng len(vocab). (Bản cũ +1 thừa → text_embed lệch shape, bị bỏ khi
        # nạp lại checkpoint → fine-tune coi như mất tác dụng phần text.)
        text_num_embeds=len(vocab_char_map),
    )

    # Tạo CFM wrapper (bao gồm mel spectrogram module)
    model = CFM(
        transformer=dit,
        mel_spec_kwargs=dict(
            n_fft=audio_cfg["n_fft"],
            hop_length=audio_cfg["hop_length"],
            win_length=audio_cfg["win_length"],
            n_mel_channels=audio_cfg["n_mel_channels"],
            target_sample_rate=audio_cfg["sample_rate"],
        ),
        vocab_char_map=vocab_char_map,
    )

    # Đếm parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"🏗️  Model: {total_params / 1e6:.1f}M total params, "
                f"{trainable_params / 1e6:.1f}M trainable")

    return model


def _find_checkpoint_file(repo_id: str, config_filename: str) -> str:
    """
    Tự động tìm file checkpoint trong HuggingFace repo.

    Chiến lược (theo thứ tự ưu tiên):
      1. Thử tên file từ config YAML (ckpt_filename)
      2. Nếu 404 → quét repo bằng list_repo_files()
      3. Ưu tiên: .safetensors > .pt > .pth > .bin
      4. Ưu tiên file có tên chứa "model" (loại trừ optimizer, config, etc.)

    Args:
        repo_id: HuggingFace repo ID (vd: "hynt/F5-TTS-Vietnamese-ViVoice")
        config_filename: Tên file từ config YAML (có thể sai)

    Returns:
        Tên file checkpoint chính xác trong repo

    Raises:
        FileNotFoundError nếu không tìm thấy checkpoint nào
    """
    # Bước 1: Thử tên file từ config trước
    try:
        hf_hub_download(repo_id=repo_id, filename=config_filename)
        logger.info(f"✅ Tìm thấy checkpoint từ config: {config_filename}")
        return config_filename
    except Exception:
        logger.warning(f"⚠️ Không tìm thấy '{config_filename}' trong repo.")
        logger.info("   Đang quét repo để tìm checkpoint...")

    # Bước 2: Quét toàn bộ files trong repo
    try:
        all_files = list_repo_files(repo_id)
    except Exception as e:
        raise FileNotFoundError(
            f"Không thể truy cập repo '{repo_id}': {e}\n"
            f"Kiểm tra: repo có tồn tại và public không?"
        )

    logger.info(f"📂 Files trong repo '{repo_id}':")
    for f in all_files:
        logger.info(f"   └── {f}")

    # Bước 3: Lọc file checkpoint theo extension (ưu tiên safetensors)
    CKPT_EXTENSIONS = (".safetensors", ".pt", ".pth", ".bin")
    checkpoint_files = [
        f for f in all_files
        if f.endswith(CKPT_EXTENSIONS)
    ]

    if not checkpoint_files:
        raise FileNotFoundError(
            f"Không tìm thấy file checkpoint nào trong repo '{repo_id}'.\n"
            f"Files hiện có: {all_files}\n"
            f"Extensions tìm kiếm: {CKPT_EXTENSIONS}"
        )

    # Bước 4: Sắp xếp ưu tiên
    #   - Ưu tiên file chứa "model" trong tên (loại optimizer, scheduler...)
    #   - Ưu tiên .safetensors > .pt > .pth > .bin
    ext_priority = {ext: i for i, ext in enumerate(CKPT_EXTENSIONS)}

    def sort_key(filename):
        name_lower = filename.lower()
        # Loại trừ file optimizer/scheduler/config (không phải model weights)
        is_excluded = any(kw in name_lower for kw in ["optim", "scheduler", "config"])
        has_model = "model" in name_lower
        ext = "." + filename.rsplit(".", 1)[-1] if "." in filename else ""
        return (
            is_excluded,                          # False trước (không bị loại)
            not has_model,                        # True nếu có "model" → ưu tiên
            ext_priority.get(ext, 99),            # Extension priority
        )

    checkpoint_files.sort(key=sort_key)
    selected = checkpoint_files[0]

    logger.info(f"✅ Auto-detected checkpoint: {selected}")
    return selected


def download_and_load_weights(model: CFM, config: Dict[str, Any]) -> CFM:
    """
    Download pre-trained weights từ HuggingFace và load vào model.

    Tự động detect tên file checkpoint nếu config bị sai (404).
    Hỗ trợ: .safetensors, .pt, .pth, .bin

    Args:
        model: CFM model đã khởi tạo
        config: Dict config chứa repo_id và filename

    Returns:
        Model đã load pre-trained weights
    """
    model_cfg = config["model"]
    repo_id = model_cfg["repo_id"]
    config_filename = model_cfg["ckpt_filename"]

    # Auto-detect checkpoint file (fallback nếu config filename sai)
    ckpt_filename = _find_checkpoint_file(repo_id, config_filename)

    logger.info(f"⬇️  Downloading: {repo_id}/{ckpt_filename}")

    # Download từ HuggingFace Hub
    ckpt_path = hf_hub_download(
        repo_id=repo_id,
        filename=ckpt_filename,
    )

    logger.info(f"📦 Checkpoint tải về: {ckpt_path}")

    # Load weights theo format
    if ckpt_filename.endswith(".safetensors"):
        state_dict = safetensors_load(ckpt_path)
    else:
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        # Một số checkpoint lưu dạng {"model": state_dict, ...}
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]
        elif isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

    # Load vào model (strict=False cho phép thiếu/thừa keys nhỏ)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        logger.warning(f"⚠️ Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        logger.warning(f"⚠️ Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
    if not missing and not unexpected:
        logger.info("   Tất cả keys khớp hoàn hảo!")

    logger.info("✅ Pre-trained weights loaded thành công!")
    return model


def download_vocab(config: Dict[str, Any]) -> dict:
    """
    Download vocab.txt từ HuggingFace và tạo vocab_char_map.

    vocab_char_map: dict {character: token_id}
    Dùng cho character-level tokenization tiếng Việt.

    Returns:
        vocab_char_map dict
    """
    model_cfg = config["model"]
    repo_id = model_cfg["repo_id"]
    vocab_filename = model_cfg.get("vocab_filename", "vocab.txt")

    logger.info(f"⬇️  Downloading vocab: {repo_id}/{vocab_filename}")

    try:
        vocab_path = hf_hub_download(
            repo_id=repo_id,
            filename=vocab_filename,
        )
        # vocab của repo là plain-text 1 token/dòng → phải dùng tokenizer "custom"
        vocab_char_map, _ = get_tokenizer(vocab_path, tokenizer="custom")
        logger.info(f"✅ Vocab loaded: {len(vocab_char_map)} characters")

    except Exception as e:
        logger.warning(f"⚠️ Không tải được vocab từ repo: {e}")
        logger.info("   Sử dụng default Vietnamese character set...")
        vocab_char_map = _build_default_vietnamese_vocab()

    return vocab_char_map


def _build_default_vietnamese_vocab() -> dict:
    """
    Tạo vocab mặc định cho tiếng Việt nếu không tải được từ HF.
    Bao gồm: a-z, dấu tiếng Việt, số, dấu câu thông dụng.
    """
    chars = list(" abcdefghijklmnopqrstuvwxyz")
    # Nguyên âm có dấu tiếng Việt
    vi_chars = list("àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệ"
                    "ìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữự"
                    "ỳýỷỹỵđ")
    punctuation = list(".,!?;:-()\"' ")
    digits = list("0123456789")

    all_chars = chars + vi_chars + punctuation + digits
    # Loại bỏ duplicate
    seen = set()
    unique = []
    for c in all_chars:
        if c not in seen:
            seen.add(c)
            unique.append(c)

    vocab = {c: i + 1 for i, c in enumerate(unique)}  # 0 reserved for PAD
    logger.info(f"   Built default vocab: {len(vocab)} characters")
    return vocab


# =============================================================================
# 4. LEARNING RATE SCHEDULER
# =============================================================================
def get_lr(
    step: int,
    warmup_steps: int,
    total_steps: int,
    base_lr: float,
    min_lr: float,
) -> float:
    """
    Cosine annealing LR schedule với linear warm-up.

    Schedule:
      - Bước 0 → warmup_steps: LR tăng tuyến tính từ 0 → base_lr
      - Bước warmup_steps → total_steps: LR giảm cosine từ base_lr → min_lr

    Phù hợp cho fine-tuning: warm-up tránh shock ban đầu,
    cosine decay giảm dần để hội tụ ổn định.

    Args:
        step: Bước hiện tại (global step)
        warmup_steps: Số bước warm-up
        total_steps: Tổng số bước training
        base_lr: Learning rate cao nhất
        min_lr: Learning rate tối thiểu

    Returns:
        Learning rate cho bước hiện tại
    """
    if step < warmup_steps:
        # Linear warm-up
        return base_lr * step / max(warmup_steps, 1)
    else:
        # Cosine annealing
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(progress, 1.0)
        return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


# =============================================================================
# 5. CHECKPOINT MANAGEMENT
# =============================================================================
def save_checkpoint(
    accelerator: Accelerator,
    model: CFM,
    optimizer: AdamW,
    global_step: int,
    epoch: int,
    loss: float,
    save_dir: str,
    keep_last_n: int = 3,
):
    """
    Lưu checkpoint training.

    Lưu:
      - Model state dict (unwrapped từ Accelerate)
      - Optimizer state dict
      - Training state (step, epoch, loss)

    Tự động xóa checkpoint cũ, chỉ giữ keep_last_n gần nhất.

    Args:
        accelerator: Accelerator instance
        model: Trained model
        optimizer: Optimizer
        global_step: Bước training hiện tại
        epoch: Epoch hiện tại
        loss: Loss gần nhất
        save_dir: Thư mục lưu checkpoint
        keep_last_n: Số checkpoint giữ lại
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Chỉ lưu trên main process (tránh duplicate khi multi-GPU)
    if not accelerator.is_main_process:
        return

    ckpt_dir = save_path / f"step_{global_step:07d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Unwrap model từ Accelerate wrapper
    unwrapped_model = accelerator.unwrap_model(model)

    # Lưu model weights
    torch.save(unwrapped_model.state_dict(), ckpt_dir / "model.pt")

    # Lưu optimizer state
    torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")

    # Lưu training state
    torch.save({
        "global_step": global_step,
        "epoch": epoch,
        "loss": loss,
    }, ckpt_dir / "training_state.pt")

    logger.info(f"💾 Checkpoint saved: {ckpt_dir}")

    # Xóa checkpoint cũ (giữ keep_last_n gần nhất)
    all_ckpts = sorted(save_path.glob("step_*"))
    if len(all_ckpts) > keep_last_n:
        for old_ckpt in all_ckpts[:-keep_last_n]:
            import shutil
            shutil.rmtree(old_ckpt)
            logger.info(f"🗑️  Xóa checkpoint cũ: {old_ckpt.name}")


def load_checkpoint(
    model: CFM,
    optimizer: AdamW,
    checkpoint_dir: str,
) -> int:
    """
    Load checkpoint gần nhất để resume training.

    Args:
        model: Model để load weights
        optimizer: Optimizer để load state
        checkpoint_dir: Thư mục chứa checkpoints

    Returns:
        global_step để tiếp tục training (0 nếu không có checkpoint)
    """
    ckpt_path = Path(checkpoint_dir)
    if not ckpt_path.exists():
        return 0

    all_ckpts = sorted(ckpt_path.glob("step_*"))
    if not all_ckpts:
        return 0

    latest_ckpt = all_ckpts[-1]
    logger.info(f"📂 Resume từ checkpoint: {latest_ckpt.name}")

    # Load model weights
    model_state = torch.load(latest_ckpt / "model.pt", map_location="cpu")
    model.load_state_dict(model_state)

    # Load optimizer state
    optim_state = torch.load(latest_ckpt / "optimizer.pt", map_location="cpu")
    optimizer.load_state_dict(optim_state)

    # Load training state
    train_state = torch.load(latest_ckpt / "training_state.pt", map_location="cpu")
    global_step = train_state["global_step"]
    logger.info(f"   Resumed at step {global_step}, epoch {train_state['epoch']}")

    return global_step


# =============================================================================
# 6. TRAINING LOOP (CỐT LÕI)
# =============================================================================
def train(config: Dict[str, Any]):
    """
    Main training function.

    Flow:
      1. Khởi tạo Accelerator (FP16 + Gradient Accumulation)
      2. Build model + load pre-trained weights
      3. Tạo dataset + dataloader
      4. Training loop:
         - Forward: model(audio, text, lens) → flow matching loss
         - Backward: accelerator.backward(loss)
         - Mỗi N steps: optimizer.step() (gradient accumulation)
         - Mỗi M steps: save checkpoint
      5. Lưu model cuối cùng
    """
    train_cfg = config["training"]
    ckpt_cfg = config["checkpoint"]
    log_cfg = config["logging"]

    # -----------------------------------------------------------------
    # Bước 1: Khởi tạo Accelerator
    # Accelerator quản lý:
    #   - Mixed Precision (FP16): tự động cast forward pass sang float16
    #   - Gradient Accumulation: gom gradient qua N mini-batches
    #   - Device placement: tự động đưa tensor lên GPU
    # -----------------------------------------------------------------
    accelerator = Accelerator(
        mixed_precision=train_cfg["mixed_precision"],  # "fp16"
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
    )
    set_seed(train_cfg["seed"])

    logger.info("=" * 60)
    logger.info(" Vietnamese Voice Cloning Studio — Fine-tuning")
    logger.info("=" * 60)
    logger.info(f"  Device          : {accelerator.device}")
    logger.info(f"  Mixed Precision : {train_cfg['mixed_precision']}")
    logger.info(f"  Batch Size      : {train_cfg['batch_size']}")
    logger.info(f"  Grad Accum Steps: {train_cfg['gradient_accumulation_steps']}")
    logger.info(f"  Effective Batch : {train_cfg['batch_size'] * train_cfg['gradient_accumulation_steps']}")
    logger.info(f"  Learning Rate   : {train_cfg['learning_rate']}")
    logger.info(f"  Epochs          : {train_cfg['epochs']}")
    logger.info("=" * 60)

    # -----------------------------------------------------------------
    # Bước 2: Download vocab + Build model + Load pre-trained weights
    # -----------------------------------------------------------------
    vocab_char_map = download_vocab(config)
    model = build_model(config, vocab_char_map)
    model = download_and_load_weights(model, config)

    # -----------------------------------------------------------------
    # Bước 3: Dataset & DataLoader
    # -----------------------------------------------------------------
    data_cfg = config["data"]
    dataset = VoiceDataset(
        processed_dir=data_cfg["processed_dir"],
        metadata_path=data_cfg["metadata_path"],
        sample_rate=config["audio"]["sample_rate"],
    )

    collate_fn = create_collate_fn(
        vocab_char_map,
        hop_length=config["audio"]["hop_length"],  # Phải khớp với model mel_spec
    )
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 2),
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    # -----------------------------------------------------------------
    # Bước 4: Optimizer (AdamW)
    # AdamW với weight decay để regularization
    # LR nhỏ (1e-5) cho fine-tuning → tránh catastrophic forgetting
    # -----------------------------------------------------------------
    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        betas=tuple(train_cfg["betas"]),
        eps=train_cfg["eps"],
    )

    # -----------------------------------------------------------------
    # Bước 5: Prepare với Accelerate
    # accelerator.prepare() tự động:
    #   - Wrap model với FP16 autocast
    #   - Wrap optimizer với GradScaler
    #   - Wrap dataloader với device placement
    # -----------------------------------------------------------------
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    # -----------------------------------------------------------------
    # Resume từ checkpoint (nếu có)
    # -----------------------------------------------------------------
    resume_step = 0
    raw_model = accelerator.unwrap_model(model)
    if Path(ckpt_cfg["save_dir"]).exists():
        resume_step = load_checkpoint(raw_model, optimizer, ckpt_cfg["save_dir"])
        if resume_step > 0:
            logger.info(f"🔄 Resumed training from step {resume_step}")

    # -----------------------------------------------------------------
    # Tính tổng số steps
    # -----------------------------------------------------------------
    steps_per_epoch = len(dataloader)
    total_steps = steps_per_epoch * train_cfg["epochs"]
    logger.info(f"📊 Steps per epoch: {steps_per_epoch}")
    logger.info(f"📊 Total steps: {total_steps}")

    # -----------------------------------------------------------------
    # TensorBoard writer
    # -----------------------------------------------------------------
    tb_writer = None
    if log_cfg.get("tensorboard", False) and accelerator.is_main_process:
        log_dir = Path(log_cfg["log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=str(log_dir))
        logger.info(f"📈 TensorBoard: {log_dir}")

    # =================================================================
    # TRAINING LOOP
    # =================================================================
    model.train()
    global_step = resume_step
    best_loss = float("inf")
    start_time = time.time()

    logger.info("\n🚀 Bắt đầu training...")

    for epoch in range(train_cfg["epochs"]):
        epoch_loss = 0.0
        epoch_steps = 0

        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{train_cfg['epochs']}",
            disable=not accelerator.is_main_process,
        )

        for step, batch in enumerate(progress_bar):
            # Skip đến resume_step (nếu resume training)
            if global_step < resume_step:
                global_step += 1
                continue

            # =========================================================
            # FORWARD PASS
            # accelerator.accumulate(model) tự động:
            #   - Bật/tắt gradient sync giữa các accumulation steps
            #   - Chỉ sync gradient ở step cuối cùng của accumulation
            # =========================================================
            with accelerator.accumulate(model):
                
                # --- AUTO FIX OFF-BY-ONE (Safety Check) ---
                # Đảm bảo lens.amax() tuyệt đối khớp với chiều dài của mel_spec
                # mà mô hình sẽ tự sinh ra bên trong (T_mel = inp.shape[-1] // hop_length + 1)
                expected_t_mel = batch["audio"].shape[-1] // config["audio"]["hop_length"] + 1
                current_max = batch["lens"].amax().item()
                
                if current_max != expected_t_mel:
                    # Tránh lỗi "tensor a (319) must match tensor b (320)"
                    # Ép lens của sample dài nhất khớp với expected_t_mel
                    max_idx = torch.argmax(batch["lens"])
                    batch["lens"][max_idx] = expected_t_mel

                # F5-TTS CFM forward:
                #   Input: raw audio [B, T] + text tokens [B, T_text] + lens [B]
                #   Output: flow matching loss (scalar)
                loss, _, _ = model(
                    inp=batch["audio"],   # [B, T_audio] — raw waveform
                    text=batch["text"],   # [B, T_text]  — token IDs
                    lens=batch["lens"],   # [B]          — actual mel lengths
                )

                # BACKWARD PASS
                # accelerator.backward() tự động:
                #   - Scale loss (FP16 GradScaler)
                #   - Chia loss cho grad_accum_steps
                accelerator.backward(loss)

                # Gradient clipping (ổn định training, tránh exploding gradients)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(),
                        train_cfg["max_grad_norm"],
                    )

                # OPTIMIZER STEP
                # Chỉ thực sự update weights mỗi grad_accum_steps
                # (Accelerator xử lý tự động)

                # Update learning rate (manual scheduler)
                current_lr = get_lr(
                    step=global_step,
                    warmup_steps=train_cfg["warmup_steps"],
                    total_steps=total_steps,
                    base_lr=train_cfg["learning_rate"],
                    min_lr=train_cfg["min_lr"],
                )
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr

                optimizer.step()
                optimizer.zero_grad()

            # =========================================================
            # LOGGING
            # =========================================================
            loss_val = loss.item()
            epoch_loss += loss_val
            epoch_steps += 1
            global_step += 1

            # Progress bar update
            progress_bar.set_postfix({
                "loss": f"{loss_val:.4f}",
                "lr": f"{current_lr:.2e}",
                "step": global_step,
            })

            # TensorBoard + console logging
            if global_step % log_cfg["log_every_n_steps"] == 0:
                elapsed = time.time() - start_time
                steps_per_sec = global_step / max(elapsed, 1)

                if accelerator.is_main_process:
                    logger.info(
                        f"  Step {global_step:>6d} | "
                        f"Loss: {loss_val:.4f} | "
                        f"LR: {current_lr:.2e} | "
                        f"Speed: {steps_per_sec:.2f} steps/s"
                    )

                    if tb_writer:
                        tb_writer.add_scalar("train/loss", loss_val, global_step)
                        tb_writer.add_scalar("train/lr", current_lr, global_step)
                        tb_writer.add_scalar("train/epoch", epoch + 1, global_step)

            # =========================================================
            # CHECKPOINT
            # =========================================================
            if global_step % ckpt_cfg["save_every_n_steps"] == 0:
                save_checkpoint(
                    accelerator=accelerator,
                    model=model,
                    optimizer=optimizer,
                    global_step=global_step,
                    epoch=epoch,
                    loss=loss_val,
                    save_dir=ckpt_cfg["save_dir"],
                    keep_last_n=ckpt_cfg["keep_last_n"],
                )

        # =========================================================
        # END OF EPOCH
        # =========================================================
        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        logger.info(
            f"\n📊 Epoch {epoch + 1}/{train_cfg['epochs']} | "
            f"Avg Loss: {avg_epoch_loss:.4f} | "
            f"Steps: {epoch_steps}"
        )

        if tb_writer:
            tb_writer.add_scalar("train/epoch_loss", avg_epoch_loss, epoch + 1)

        # Track best loss
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            logger.info(f"   🏆 New best loss: {best_loss:.4f}")

    # =================================================================
    # TRAINING COMPLETE
    # =================================================================
    total_time = time.time() - start_time
    hours = total_time // 3600
    minutes = (total_time % 3600) // 60

    logger.info("\n" + "=" * 60)
    logger.info(" ✅ TRAINING HOÀN TẤT!")
    logger.info("=" * 60)
    logger.info(f"  Tổng thời gian : {int(hours)}h {int(minutes)}m")
    logger.info(f"  Total steps    : {global_step}")
    logger.info(f"  Best loss      : {best_loss:.4f}")

    # Lưu model cuối cùng
    save_checkpoint(
        accelerator=accelerator,
        model=model,
        optimizer=optimizer,
        global_step=global_step,
        epoch=train_cfg["epochs"],
        loss=best_loss,
        save_dir=ckpt_cfg["save_dir"],
        keep_last_n=ckpt_cfg["keep_last_n"] + 1,  # +1 giữ thêm final checkpoint
    )

    if tb_writer:
        tb_writer.close()

    logger.info(f"\n  Checkpoints: {ckpt_cfg['save_dir']}/")
    logger.info(f"  TensorBoard: tensorboard --logdir {log_cfg['log_dir']}")
    logger.info(f"\n  Tiếp theo → Phase 4: Inference & Evaluation")
    logger.info("=" * 60)


# =============================================================================
# 7. CLI ENTRY POINT
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Vietnamese Voice Cloning Studio - Fine-tuning Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Cách chạy:
  python scripts/train.py --config configs/train_config.yaml

Resume training (tự động detect checkpoint):
  python scripts/train.py --config configs/train_config.yaml
        """
    )

    parser.add_argument(
        "--config", type=str, required=True,
        help="Đường dẫn file config YAML (configs/train_config.yaml)"
    )

    args = parser.parse_args()

    if not Path(args.config).exists():
        logger.error(f"❌ Config file không tìm thấy: {args.config}")
        sys.exit(1)

    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
