# Vietnamese Voice Cloning Studio

Đồ án DSP391m tại FPT University: web app song ngữ để nhân bản giọng nói tiếng Việt bằng
F5-TTS và so sánh cùng input với XTTS-v2, MMS-TTS, Piper, Edge-TTS và Bark.

> Model nền `hynt/F5-TTS-Vietnamese-ViVoice` dùng giấy phép
> **CC-BY-NC-SA-4.0**. Repository này chỉ dành cho nghiên cứu/giáo dục phi thương mại.
> Chỉ sử dụng giọng của chính bạn hoặc giọng mà bạn có quyền và sự đồng ý để sử dụng.

## Trạng thái có thể kiểm chứng

- Inference F5-TTS giữ utility chính thức, gồm preprocessing/tokenization của thư viện.
- DiT giữ bắt buộc `text_mask_padding=False` và `pe_attn_head=1` ở cả inference/training.
- Có chunking câu dài, evaluation WER/CER/speaker similarity, data validation và unit tests offline.
- Pipeline fine-tune đã được gia cố nhưng **chưa được xác minh end-to-end bằng dữ liệu người thật/GPU**.
- Không có kết quả WER/CER/SECS mẫu trong repo; phải sinh audio thật rồi chạy evaluation.

Tài liệu chi tiết: [kiến trúc](docs/ARCHITECTURE.md), [fine-tuning](docs/FINETUNING.md),
[evaluation](docs/EVALUATION.md), [deployment](docs/DEPLOYMENT.md),
[kịch bản demo](docs/DEMO_GUIDE.md), [dependency/model licenses](docs/LICENSES.md).

## Kiến trúc và engine

```text
Gradio UI / compare_models.py
        │
        ├── F5-TTS ViVoice ── reference audio + transcript ── voice cloning
        ├── XTTS-v2 ───────── reference audio ─────────────── voice cloning
        ├── MMS / Piper / Bark ────────────────────────────── fixed/preset local TTS
        └── Edge-TTS ──────────────────────────────────────── cloud TTS

generated WAV ── evaluate.py ── Whisper WER/CER + Resemblyzer similarity + audio metrics
```

| Engine | Loại | Voice cloning | Nơi chạy | Ghi chú |
|---|---|---:|---|---|
| F5-TTS ViVoice | local/offline sau khi cache | Có | CPU/CUDA, MPS thử nghiệm | Engine chính, 24 kHz |
| viXTTS | local/offline sau khi cache | Có | CPU/CUDA | Cần reference sạch |
| MMS-TTS | local | Không | CPU/CUDA | Một giọng tiếng Việt |
| Piper | local | Không | CPU | ONNX, nhanh |
| Edge-TTS | cloud | Không | Internet | Timeout có kiểm soát |
| Bark | local | Không | CPU/CUDA | Rất chậm trên CPU, tiếng Việt hạn chế |

Voice cloning học đặc trưng giọng từ reference. TTS thông thường dùng giọng cố định. “Cloud”
gửi text tới dịch vụ bên ngoài; “offline” không cần mạng sau khi model đã được cache.

## Quick start

Yêu cầu: Python 3.10/3.11, ffmpeg và đủ dung lượng để cache model.

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python verify_env.py
python app.py                             # http://localhost:7860
```

CPU là fallback an toàn nhưng Bark/training rất chậm. NVIDIA CUDA phù hợp cho inference/training:

```bash
VVCS_DEVICE=cuda python app.py
VVCS_DEVICE=mps python app.py             # macOS Apple Silicon: thử nghiệm
VVCS_OFFLINE=1 python app.py              # chỉ dùng cache local, Edge bị chặn
```

Linux là môi trường deploy được khuyến nghị. macOS chạy CPU/MPS nhưng không có CUDA. Windows cần
ffmpeg trong `PATH`; Piper/Coqui có thể phụ thuộc wheel theo phiên bản Python.

### Cấu hình môi trường

Các biến chính: `VVCS_DEVICE`, `VVCS_MODEL_ID`, `VVCS_CACHE_DIR`, `VVCS_OUTPUT_DIR`,
`VVCS_NFE`, `VVCS_SEED`, `VVCS_MAX_TEXT_CHARS`, `VVCS_MAX_CHUNK_CHARS`,
`VVCS_CHUNK_SILENCE_MS`, `VVCS_ENGINE_TIMEOUT`, `VVCS_WHISPER_MODEL`, `VVCS_OFFLINE`,
`VVCS_ENABLE_CLOUD_ENGINES`.

## Sử dụng app

1. Upload reference sạch 3–10 giây và nhập transcript chính xác.
2. Xác nhận bạn có quyền/sự đồng ý sử dụng giọng.
3. Nhập text, chọn engine, NFE 32/48/64 và `auto` chunking cho câu dài.
4. Chạy. Một engine lỗi không ngăn engine kế tiếp; card giữ thứ tự người dùng chọn.

Audio tham chiếu nên một người nói, không nhạc/echo, không clipping. App giới hạn tổng text mặc
định 10.000 ký tự; khi tắt chunking, giới hạn trực tiếp là 2.000 ký tự.

## Data preparation và metadata

Đặt audio người thật vào `data/raw/`. Tuyệt đối không dùng `data/_invalid_demo_data/`.

```bash
python scripts/data_prep.py \
  --input_dir data/raw --output_dir data/processed \
  --sample_rate 24000 --min_duration 3 --max_duration 10
```

Schema chính thức là pipe-delimited UTF-8:

```text
audio_file|duration_sec|snr_db|source_file|text
segment_00001.wav|5.20|34.1|recording.wav|Transcript chính xác ở đây.
```

Sau khi điền transcript:

```bash
python scripts/data_prep.py validate \
  --metadata data/metadata/metadata.csv \
  --processed-dir data/processed \
  --report outputs/dataset_summary.json
```

Validator báo duration/sample-rate/transcript distribution, duplicate, missing/invalid row và
deterministic train/validation count; nó chặn silence, NaN/Inf, clipping quá cao và tín hiệu giống sin.

## Fine-tuning

Chỉ chạy với dữ liệu người thật đã validate. Cần GPU CUDA thực tế; CPU fallback tồn tại để kiểm tra
code path nhưng không phù hợp huấn luyện đầy đủ.

```bash
python scripts/train.py --config configs/train_config.yaml
python scripts/train.py --config configs/train_config.yaml --resume latest
python scripts/train.py --config configs/train_config.yaml --resume checkpoints/step_0001000
tensorboard --logdir logs
```

Không auto-resume. Checkpoint resume phải có metadata và đúng base model. Training lưu config snapshot,
dataset summary, git commit, validation loss, periodic checkpoint và `checkpoints/best/model.pt`.
Xem [docs/FINETUNING.md](docs/FINETUNING.md) và notebook Colab.

## Evaluation và benchmark NFE

Copy manifest mẫu, thay đường dẫn bằng audio sinh thật và thêm `inference_time` nếu có:

```bash
cp evaluation/test_cases.example.json evaluation/test_cases.json
python scripts/evaluate.py \
  --manifest evaluation/test_cases.json \
  --output-dir outputs/evaluation --device auto
```

Output: `results.json`, `results.csv`, `report.md`. Batch giữ partial failure; nếu mọi mẫu lỗi, exit code
là 1. Speaker similarity chỉ là metric tương đối, không phải nhận dạng sinh trắc học.

Benchmark cùng seed/text cho từng NFE rồi evaluate từng manifest:

```bash
for nfe in 32 48 64; do
  python scripts/compare_models.py --ref reference_audio/sample_clean_vi.wav \
    --models base --nfe "$nfe" --seed 42 --out-dir "outputs/nfe_${nfe}"
done
```

Không kết luận NFE nào tốt nhất nếu chưa có report trên cùng bộ câu.

## Testing và CI

Unit tests không tải model hoặc gọi cloud:

```bash
pip install -r requirements-dev.txt
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q   # biến này tránh plugin global hỏng trong một số Conda env
ruff check voice_studio tests scripts/evaluate.py
python -m compileall -q .
```

Heavy tests phải opt-in bằng marker `integration` hoặc `gpu`. GitHub Actions chỉ compile, lint và unit test.

## Docker và Hugging Face Spaces

```bash
docker compose build
docker compose up
```

Model cache, outputs, data và checkpoints là volume; image không bake dữ liệu/weights người dùng.
Hướng dẫn Spaces, resource limit, cold start và secrets nằm tại [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## Troubleshooting và giới hạn

- `verify_env.py` báo thiếu package: cài lại `requirements.txt` trong đúng virtualenv.
- Edge lỗi/timeout: kiểm tra Internet hoặc bỏ engine này; dùng `VVCS_OFFLINE=1` khi cần riêng tư.
- CUDA OOM: chỉ chọn một engine, giảm độ dài chunk, restart app; Bark/XTTS dùng nhiều RAM/VRAM.
- Transcript reference sai hoặc audio bẩn làm giảm chất lượng cloning.
- WER phụ thuộc chất lượng Whisper; similarity phụ thuộc Resemblyzer và chỉ dùng để so sánh tương đối.
- Chưa có dữ liệu thật/checkpoint được kiểm chứng trong repo; fine-tuning/GPU/integration chưa được tuyên bố pass.
- Base model phi thương mại. Muốn thương mại phải thay model hoặc có giấy phép phù hợp và review license
  của F5-TTS, engine/checkpoint/dataset liên quan.

## Project structure

```text
app.py                    Gradio + orchestration
engines.py                optional engines, lazy loading, error mapping
voice_studio/             config, text/audio/evaluation logic thuần
scripts/data_prep.py      preprocess + metadata validation
scripts/train.py          fine-tuning
scripts/compare_models.py comparison CLI
scripts/evaluate.py       objective evaluation CLI
evaluation/               example manifest
tests/                    offline unit tests
docs/                     architecture/evaluation/fine-tuning/deploy/demo
Dockerfile                CPU demo image
```

## License và ethical use

Repository hiện chưa khai báo license riêng cho phần code dự án; model/checkpoint/dataset/dependency vẫn giữ
license riêng. Base ViVoice là CC-BY-NC-SA-4.0. Không dùng để mạo danh, lừa đảo,
né xác thực hoặc tạo nội dung không có sự đồng ý. Khi chia sẻ output, nên ghi rõ đó là audio tổng hợp.
