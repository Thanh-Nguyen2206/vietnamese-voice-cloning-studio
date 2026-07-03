# 🎙️ Vietnamese Voice Cloning Studio

Nhân bản giọng nói tiếng Việt từ một đoạn audio mẫu — kèm giao diện **so sánh
nhiều mô hình** cạnh nhau. (Đồ án DSP391m — FPTU Summer 2026.)

> Mô hình nền: bản tiếng Việt 1000 giờ trên HuggingFace · Vocoder: Vocos.

## Có gì trong bản này

- **Giao diện web** (`app.py`): tải audio mẫu → nhập văn bản → nghe kết quả.
  Chọn một hoặc nhiều mô hình để **so sánh** bằng tai và bằng **số liệu khách quan**.
- **So sánh không cần web** (`scripts/compare_models.py`): sinh audio cho nhiều
  mô hình, xuất báo cáo Markdown kèm chỉ số chất lượng.
- **Pipeline huấn luyện** (`scripts/data_prep.py`, `scripts/train.py`) để fine-tune
  trên giọng của riêng bạn.

## Sáu engine để so sánh

Site so sánh trực tiếp **6 mô hình** trên cùng đầu vào, tối đa 4 mô hình mỗi lần
(`engines.py` chứa 5 engine bổ sung, nạp lười — lần đầu chọn sẽ tải mô hình).
Giao diện **song ngữ Việt/Anh** (nút chuyển ở góc trên bên phải).

| Engine | Nguồn | Nhân bản giọng? | Tiếng Việt | Ghi chú |
|---|---|---|---|---|
| **F5-TTS** (gốc) | `hynt/...ViVoice` | Có | Rất tốt | Chất lượng tốt nhất. Không đổi. |
| **XTTS-v2 (viXTTS)** | `capleaf/viXTTS` | Có | Tốt | **Rất nhạy với audio mẫu** — cần mẫu SẠCH, không nhạc nền. |
| **MMS-TTS (Meta)** | `facebook/mms-tts-vie` | Không (1 giọng) | Khá | VITS gọn nhẹ, local, rất nhanh. |
| **Piper (Rhasspy)** | `rhasspy/piper` (vi_VN-vais1000) | Không (1 giọng) | Khá | ONNX, **nhanh nhất trên CPU**, tất định. |
| **Edge-TTS (Microsoft)** | `rany2/edge-tts` (vi-VN-HoaiMy) | Không (1 giọng) | Rất tốt | Baseline thương mại — **chạy cloud, cần internet**; timeout 40s, mỗi engine khoá riêng nên không chặn nhau. |
| **Bark (Suno)** | `suno/bark-small` | Không (preset) | Hạn chế | **Chậm trên CPU** (~100s/câu), để đối chiếu. |

> Site kèm sẵn **audio mẫu sạch** (`reference_audio/sample_clean_vi.wav`) làm mặc định
> để mọi engine chạy tốt ngay. XTTS đặc biệt cần mẫu sạch (mẫu có nhạc nền dễ khiến
> XTTS đọc sai từ — F5-TTS thì bền hơn). Mỗi kết quả hiển thị RMS, độ phẳng phổ và
> **thời gian tạo** để so sánh khách quan cả chất lượng lẫn tốc độ.
>
> Các checkpoint fine-tune demo (huấn luyện trên dữ liệu giả) **không hiển thị** trên
> site; bật lại để nghiên cứu bằng biến môi trường `VVCS_SHOW_DEMO_CKPTS=1`.

## ⚡ Lưu ý quan trọng về "nhiễu tĩnh" (đã sửa)

Bản trước cho ra **nhiễu tĩnh / gần như im lặng** thay vì giọng người vì hai lý do:

1. **Checkpoint hỏng.** Các file `checkpoints/step_*/model.pt` được fine-tune trên
   dữ liệu **giả** (`segment_*.wav` thực chất là **sóng sin**, không phải tiếng nói —
   xem `data/_invalid_demo_data/`). App cũ tự động đè checkpoint này lên mô hình gốc
   → làm hỏng đầu ra.
2. **Tự viết lại pipeline suy luận** và bỏ qua bước token hoá (`convert_char_to_pinyin`)
   mà mô hình đã được huấn luyện cùng → giọng méo ngay cả với mô hình gốc.

Bản hiện tại dùng **đúng pipeline suy luận chính thức** và **mặc định mô hình gốc**
(cho ra giọng người sạch). Các checkpoint cũ vẫn được giữ và liệt kê trong giao diện
so sánh để bạn tự nghe thấy chúng kém hơn.

## Cài đặt

```bash
bash setup.sh            # có GPU (Colab/Kaggle/RunPod)
# hoặc:
pip install -r requirements.txt
python verify_env.py     # kiểm tra môi trường
```

## Chạy giao diện web

```bash
python app.py                 # http://localhost:7860
python app.py --share         # link public (Colab/Kaggle)
python app.py --port 8080     # đổi port
```

Mặc định mô hình chạy trên CPU cho ổn định. Muốn nhanh hơn trên máy có GPU:

```bash
VVCS_DEVICE=cuda python app.py     # hoặc VVCS_DEVICE=mps (thử nghiệm)
```

### Cách dùng

1. **Tải audio mẫu** (3–10 giây) — giọng muốn nhân bản.
2. **Nhập transcript** của audio mẫu (để trống → Whisper tự nhận diện).
3. **Nhập văn bản** cần đọc.
4. **Chọn mô hình** để so sánh (mặc định: mô hình gốc).
5. Bấm **Tạo & so sánh giọng nói**. Mỗi kết quả kèm nhận định chất lượng
   (RMS, độ phẳng phổ) để đối chiếu khách quan.

## So sánh mô hình bằng dòng lệnh

```bash
python scripts/compare_models.py --ref reference_audio/audio.wav
# chỉ định mô hình & câu:
python scripts/compare_models.py --ref ref.wav \
    --models base ft_step_0001000 --nfe 32
```

Kết quả: `outputs/comparison/*.wav` + `outputs/comparison/report.md`.

---

## Fine-tune trên giọng của bạn

> ⚠️ Cần **giọng người thật** (30–60 phút, 24kHz mono). Dữ liệu sin/tổng hợp sẽ cho
> ra mô hình nhiễu (xem `data/_invalid_demo_data/`).

```bash
# 1) Tiền xử lý
python scripts/data_prep.py \
    --input_dir data/raw --output_dir data/processed \
    --sample_rate 24000 --min_duration 3.0 --max_duration 10.0
# 2) Điền transcript vào cột "text" của data/metadata/metadata.csv
# 3) Fine-tune
python scripts/train.py --config configs/train_config.yaml
```

Checkpoint lưu vào `checkpoints/step_XXXXXXX/model.pt`; tự động xuất hiện trong
danh sách so sánh của app ở lần khởi động sau.

---

## Cấu trúc thư mục

```
vietnamese-voice-cloning-studio/
├── app.py                     ← Giao diện web (CHẠY FILE NÀY)
├── requirements.txt
├── setup.sh                   ← Cài đặt môi trường
├── verify_env.py              ← Kiểm tra môi trường
├── configs/train_config.yaml  ← Config fine-tune
├── scripts/
│   ├── data_prep.py           ← Tiền xử lý audio
│   ├── train.py               ← Fine-tuning
│   └── compare_models.py      ← So sánh mô hình (CLI)
├── data/
│   ├── raw/                   ← Audio gốc của bạn
│   ├── processed/             ← Audio đã xử lý
│   ├── metadata/              ← metadata.csv
│   └── _invalid_demo_data/    ← Dữ liệu giả (KHÔNG dùng để train)
├── checkpoints/step_*/model.pt ← Checkpoint fine-tune (dữ liệu demo)
├── outputs/                   ← Audio sinh ra + báo cáo so sánh
└── logs/                      ← TensorBoard logs
```

## Chi tiết kỹ thuật

- Kiến trúc nền: DiT (dim=1024, depth=22, heads=16) ~336M tham số, mô hình tiếng
  Việt 1000h `hynt/F5-TTS-Vietnamese-ViVoice`.
- Vocoder: Vocos (24kHz).
- Giấy phép mô hình nền: CC-BY-NC-SA-4.0 (chỉ dùng cho nghiên cứu phi thương mại).
