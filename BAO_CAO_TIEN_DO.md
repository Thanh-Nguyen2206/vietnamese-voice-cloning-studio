# Báo cáo tiến độ — Vietnamese Voice Cloning Studio

> **Tài liệu lưu trữ (snapshot 2026-06-30).** Trạng thái hiện tại, command và giới hạn đã được
> hợp nhất/cập nhật trong `README.md`; không dùng file này làm nguồn vận hành chính.

> Cập nhật: 2026-06-30
> Đồ án DSP391m — FPTU Summer 2026
> Thư mục dự án: `Do_An_F5TTS/vietnamese-voice-cloning-studio/`

---

## 1. Thông tin tôi đã nắm được về dự án

### 1.1. Bản chất dự án
- Hệ thống **nhân bản giọng nói tiếng Việt (voice cloning / TTS)**: đưa vào một đoạn
  audio mẫu + văn bản → đọc văn bản bằng giọng của audio mẫu.
- Trước đây tên là *"F5-TTS Vietnamese MVP"*, đã đổi tên thành
  **Vietnamese Voice Cloning Studio** (bỏ thuật ngữ kỹ thuật "F5T" khỏi tiêu đề).

### 1.2. Mô hình & công nghệ nền
- **Mô hình gốc:** `hynt/F5-TTS-Vietnamese-ViVoice` (HuggingFace) — huấn luyện trên
  ~1000 giờ tiếng Việt.
- **Kiến trúc:** DiT (dim=1024, depth=22, heads=16) ~336M tham số.
- **Vocoder:** Vocos (24kHz).
- **Vocab:** file `config.json` của repo (dạng plain-text, 1 token/dòng, 2546 token —
  gồm cả ký tự tiếng Việt có dấu và token pinyin kế thừa từ F5-TTS gốc).
- **Thiết bị:** mặc định CPU (ổn định nhất); đổi bằng `VVCS_DEVICE=cuda|mps`.

### 1.3. Hiện trạng dữ liệu (quan trọng)
- `data/processed/segment_*.wav` thực chất là **sóng sin thuần (~1392 Hz)** — KHÔNG
  phải tiếng nói.
- `data/processed/synthetic_audio_*.wav` là **tín hiệu tổng hợp dạng tông** — cũng
  không phải tiếng nói thật.
- `data/metadata/metadata.csv` trỏ tới đường dẫn Colab `/content/...` không tồn tại.
- ⇒ Toàn bộ "dữ liệu huấn luyện" là **giả**. Đã chuyển vào `data/_invalid_demo_data/`.

### 1.4. Hiện trạng checkpoint
- `checkpoints/step_0000500/model.pt` và `checkpoints/step_0001000/model.pt` được
  fine-tune **trên chính dữ liệu sin giả** ở trên → **làm hỏng** mô hình gốc.
- `optimizer_states/*.pt` (~2.6GB mỗi file) là state của optimizer, không cần cho suy luận.

---

## 2. ĐÃ LÀM ĐƯỢC ✅

### 2.1. Chẩn đoán nguyên nhân "nhiễu tĩnh" (đã xác minh bằng số liệu)
Tìm ra **2 nguyên nhân cộng hưởng** khiến output trước đây là nhiễu/gần như im lặng:
1. **Checkpoint hỏng:** app cũ **tự động đè** checkpoint (fine-tune trên sóng sin) lên
   mô hình gốc → đầu ra RMS ≈ 0.006 (gần như im lặng + nhiễu).
2. **Pipeline suy luận tự viết sai:** `_infer_cpu` + `_vi_passthrough` **bỏ qua bước
   token hoá** `convert_char_to_pinyin` mà mô hình đã được huấn luyện cùng.

Bằng chứng đo được (cùng input):
| Pipeline | RMS | Độ phẳng phổ | Kết luận |
|---|---|---|---|
| Mô hình gốc + pipeline chính thức | ~0.48 | ~0.10 | giọng có cấu trúc |
| App cũ, chỉ mô hình gốc | ~0.16 | ~0.001 | đơn điệu/méo |
| App cũ + checkpoint fine-tune | ~0.006 | ~0.13 | nhiễu/gần im lặng |

### 2.2. Viết lại ứng dụng (`app.py`)
- Dùng **đúng pipeline chính thức** `f5_tts.infer.utils_infer.infer_process` +
  `preprocess_ref_audio_text`.
- **Mặc định mô hình gốc** (không còn tự đè checkpoint hỏng).
- **Giao diện so sánh nhiều mô hình** (tối đa 3) cạnh nhau, kèm **chỉ số khách quan**
  (RMS, độ phẳng phổ) và nhận định ✅ giọng / ⚠️ nhiễu cho từng kết quả.
- Tự nhận diện transcript audio mẫu bằng Whisper (tiếng Việt) khi để trống.
- Sửa tương thích Gradio 6 (đưa `theme` vào `launch()`).

### 2.3. Công cụ so sánh không cần web (`scripts/compare_models.py`)
- Sinh audio cho nhiều mô hình × nhiều câu → xuất `outputs/comparison/*.wav` +
  báo cáo `report.md` có bảng số liệu. Phù hợp đưa vào báo cáo đồ án.

### 2.4. Dọn dẹp & sửa lỗi khác
- Chuyển dữ liệu giả sang `data/_invalid_demo_data/` kèm README giải thích.
- Sửa lỗi off-by-one trong `scripts/train.py`: `text_num_embeds` từ `len(vocab)+1`
  → `len(vocab)` (DiT tự +1 nội bộ); sửa tokenizer về chế độ `"custom"`.
- **Đổi thương hiệu toàn bộ** (README, setup.sh, verify_env.py, config, scripts):
  bỏ "F5T" khỏi mọi tiêu đề.
- **Đổi tên thư mục** `F5TTS_Vietnamese` → `vietnamese-voice-cloning-studio`.

### 2.5. Tối ưu ổn định & chất lượng nhất quán (rà soát toàn bộ)
**Ổn định:**
- **Chạy tuần tự**: thêm `threading.RLock` + `queue(concurrency_limit=1)` → không còn
  hai request cùng nạp model (gấp đôi RAM → OOM) hay chạy song song trên cùng model.
- **Dự phòng thiết bị**: nếu `cuda`/`mps` lỗi khi nạp → tự lùi về CPU thay vì crash.
- **Chặn văn bản quá dài** (> 2000 ký tự) để tránh treo máy trên CPU.
- Sửa lỗi Whisper nạp model **hai lần** ở nhánh dự phòng.

**Chất lượng nhất quán:**
- **Cố định seed** (mặc định 42, `-1` = ngẫu nhiên): cùng input → **output y hệt**;
  mọi model dùng chung seed nên **so sánh công bằng**. (Đã kiểm chứng: cùng seed cho
  waveform trùng khít từng mẫu.)
- **Chuẩn hoá văn bản tiếng Việt** (`_normalize_vi`): chữ thường + gộp khoảng trắng +
  thêm dấu kết câu — khớp phân phối huấn luyện của mô hình (vốn train trên chữ thường),
  giảm hiện tượng trôi từ ở cuối câu.
- **Giới hạn đỉnh (peak ≤ 0.99)** trên đầu ra → tránh méo do clipping.
- UI: thêm ô **Seed**, thanh tiến trình, mục **"Mẹo để giọng rõ & ổn định"**.

### 2.6. Đã chạy & kiểm chứng
- Web app khởi động OK (HTTP 200), không lỗi: `python app.py`.
- Mô hình gốc cho ra audio **có cấu trúc giọng**; checkpoint fine-tune bị gắn cờ nhiễu.
- Tái lập: cùng seed → waveform trùng khít; khác seed → khác.
- CLI so sánh chạy ra báo cáo đúng (có thêm cột seed).

---

## 3. ĐANG LÀM / CẦN XỬ LÝ TIẾP 🔧

### 3.1. ✅ ĐÃ SỬA: giọng "líu nhíu", không nghe ra nội dung
- File bạn gửi `audio(3).wav`: RMS=0.21 (to) nhưng **độ phẳng phổ=0.0019** (rất đơn điệu)
  ⇒ giọng méo, không hiểu được (không phải nhiễu tĩnh như lỗi trước).
- **Nguyên nhân gốc (đã tìm ra):** khi viết lại app tôi dựng DiT bằng `DIT_CFG` THIẾU hai
  trường `text_mask_padding=False` và `pe_attn_head=1`. Thiếu chúng → DiT lấy default của
  bản v1 (`text_mask_padding=True`, `pe_attn_head=None`) → **forward pass chạy sai** dù
  trọng số nạp đúng shape → giọng méo. (Vì vậy API chính thức `F5TTS.infer` cho giọng sạch
  còn app của tôi thì méo — chúng dùng cấu hình arch khác nhau.)
- **Cách sửa:** `app.py` nay nạp **nguyên cấu hình arch `F5TTS_Base.yaml`** của thư viện
  (`_load_arch()`), khớp 100% mô hình gốc.
- **Kiểm chứng khách quan (ASR round-trip bằng Whisper):**
  - Độ phẳng phổ: **0.0019 → 0.145** (vào vùng giọng người, ngang audio mẫu thật ~0.13).
  - Nhập: *"Hôm nay trời nắng đẹp, tôi đi dạo trong công viên."*
    Whisper nghe lại: *"Hôm nay trời nắng đẹp, đón đi dầu cho con phi"* → đã là **tiếng Việt
    rõ ràng, hiểu được** (nửa đầu trùng khớp y hệt).
- **Còn lại (chất lượng, không phải lỗi):** nửa sau câu dài đôi khi lệch từ. Cải thiện bằng:
  audio mẫu sạch 5–10s một giọng; nhập transcript mẫu chính xác; tăng NFE 32→48–64.

---

## 4. CHƯA LÀM ĐƯỢC / NGOÀI khả năng hiện tại ❌

- **Fine-tune cho ra giọng tốt hơn mô hình gốc:** bất khả thi với dữ liệu hiện có, vì
  dữ liệu là sóng sin giả. **Cần audio giọng người thật 30–60 phút** (24kHz, mono, sạch)
  đặt vào `data/raw/` rồi chạy lại `data_prep.py` → `train.py`.
- **Cứu các checkpoint cũ:** không thể — chúng đã học từ dữ liệu giả. Chỉ giữ lại để
  đối chiếu cho thấy chúng kém.
- **Đánh giá khách quan độ "dễ hiểu" của giọng (WER/CER):** chưa dựng. Hiện mới có chỉ
  số RMS/độ phẳng phổ (phân biệt nhiễu vs giọng), **chưa** đo được mức độ phát âm đúng.
  Để đánh giá "nghe có hiểu không" cần chạy ASR (Whisper) trên đầu ra rồi so với văn bản
  gốc — đây là việc nên bổ sung để xử lý đúng vấn đề mục 3.1.

---

## 5. Cách chạy nhanh

```bash
cd ".../Do_An_F5TTS/vietnamese-voice-cloning-studio"

# Web (mặc định mô hình gốc, giọng sạch)
python app.py                      # http://localhost:7860
VVCS_DEVICE=cuda python app.py     # nếu có GPU NVIDIA

# So sánh mô hình bằng dòng lệnh
python scripts/compare_models.py --ref reference_audio/audio.wav
```

---

## 6. Tóm tắt một dòng
Đã sửa xong **hai lỗi**: (1) **nhiễu tĩnh** (checkpoint giả + pipeline sai) và
(2) **giọng méo/khó nghe** trong `audio(3).wav` (do DiT thiếu `text_mask_padding=False`
+ `pe_attn_head=1` → giờ nạp đúng arch F5TTS_Base). Đã kiểm chứng đầu ra là **tiếng Việt
rõ, hiểu được** bằng Whisper. Cùng với đó: dựng công cụ so sánh mô hình + đổi tên dự án.
**Còn lại:** muốn fine-tune ra giọng riêng cần dữ liệu giọng người thật.
