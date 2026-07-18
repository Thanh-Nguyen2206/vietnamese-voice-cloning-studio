# Tổng quan dự án — Vietnamese Voice Cloning Studio

> **Lưu ý sau audit:** nội dung dưới đây là baseline trước đợt hoàn thiện. Trạng thái code,
> evaluation, tests, Docker và command mới nhất nằm trong `README.md` và `docs/`.

> Cập nhật: 2026-07-18 · Đồ án DSP391m — FPTU Summer 2026
> File này gộp toàn bộ trạng thái dự án (thay cho việc phải đọc rải rác README + BAO_CAO_TIEN_DO) + kế hoạch hoàn thiện.

---

## 1. Dự án là gì

Web app nhân bản giọng nói tiếng Việt (voice cloning TTS): người dùng tải lên một đoạn audio mẫu (giọng muốn nhân bản) + nhập văn bản → hệ thống đọc văn bản đó bằng giọng trong audio mẫu. App còn cho so sánh cùng lúc **6 engine TTS khác nhau** trên cùng input để đối chiếu chất lượng/tốc độ.

- Model nền: `hynt/F5-TTS-Vietnamese-ViVoice` (DiT, dim=1024/depth=22/heads=16, ~336M tham số, train trên ~1000 giờ tiếng Việt), vocoder Vocos 24kHz.
- 5 engine so sánh thêm: XTTS-v2 (viXTTS), MMS-TTS (Meta), Piper (ONNX, nhanh nhất CPU), Edge-TTS (Microsoft, cloud), Bark (chậm, đối chiếu).
- Giao diện Gradio song ngữ Việt/Anh, streaming kết quả (engine nhanh hiện trước).
- Có pipeline fine-tune riêng (`scripts/data_prep.py`, `scripts/train.py`) để huấn luyện thêm trên giọng của người dùng.

## 2. Cấu trúc thư mục hiện tại

```
vietnamese-voice-cloning-studio/
├── app.py                  913 dòng — Gradio UI + inference chính (F5-TTS gốc)
├── engines.py               280 dòng — 5 engine so sánh (lazy-load)
├── verify_env.py            kiểm tra môi trường
├── setup.sh                 cài đặt (GPU/Colab)
├── requirements.txt
├── configs/train_config.yaml
├── scripts/
│   ├── data_prep.py         521 dòng — tiền xử lý audio thô → segment
│   ├── train.py             1043 dòng — fine-tuning loop
│   └── compare_models.py    120 dòng — so sánh CLI, xuất report.md
├── notebooks/finetune_colab.ipynb   — huấn luyện trên Colab (GPU)
├── data/
│   ├── raw/                 RỖNG (chỉ có README hướng dẫn)
│   ├── processed/            RỖNG
│   ├── metadata/              RỖNG
│   └── _invalid_demo_data/   dữ liệu giả cũ (sóng sin), đã cách ly
├── reference_audio/          2 file mẫu (audio.wav, sample_clean_vi.wav)
├── outputs/                  audio đã sinh + report so sánh cũ
└── logs/                     (TensorBoard, hiện trống)
```

Checkpoint fine-tune cũ (`checkpoints/step_0000500`, `step_0001000`) **không có trong repo** (đã gitignore) vì train trên dữ liệu giả — chỉ tồn tại cục bộ nếu còn, và bị coi là hỏng.

## 3. Lịch sử commit (6 commit, nhánh `main`, sạch — không có gì chưa commit)

1. `d249db2` Initial commit
2. `ee42982` Loại các file lớn khỏi repo, sửa ví dụ CLI cũ
3. `544c02f`… `544c08f` So sánh 6 engine cùng lúc, streaming, sửa toggle ngôn ngữ
4. `5aeb758` Streaming UX thật (per-card status), thiết kế lại UI
5. `89d05f5` Sắp xếp kết quả theo thứ tự người dùng chọn model, không theo tốc độ engine
6. `0efae2d` (mới nhất) Chuẩn bị pipeline fine-tune thật: sửa bug data_prep/train, thêm Colab notebook

## 4. ĐÃ LÀM ✅

**Chẩn đoán & sửa 2 lỗi nghiêm trọng ở bản đầu:**
- Nhiễu tĩnh/gần im lặng: do checkpoint hỏng (fine-tune trên sóng sin giả) tự động đè lên model gốc, và pipeline tự viết bỏ sót bước tokenize `convert_char_to_pinyin`. → Đã sửa: dùng đúng pipeline chính thức `f5_tts.infer.utils_infer`, mặc định load model gốc, không tự đè checkpoint.
- Giọng méo/líu nhíu: do thiếu 2 field cấu hình kiến trúc DiT (`text_mask_padding=False`, `pe_attn_head=1`) khi dựng model thủ công → forward pass sai dù shape trọng số đúng. → Đã sửa: nạp nguyên cấu hình `F5TTS_Base.yaml` của thư viện. Kiểm chứng bằng ASR round-trip (Whisper): độ phẳng phổ 0.0019 → 0.145, output nghe được tiếng Việt rõ ràng.

**Tính năng:**
- App so sánh 6 engine cùng lúc, streaming từng thẻ kết quả khi xong (không chờ toàn bộ).
- Chỉ số khách quan mỗi kết quả: RMS, độ phẳng phổ (spectral flatness), thời gian tạo.
- Seed cố định (mặc định 42) → tái lập kết quả, so sánh công bằng giữa các model.
- Chuẩn hoá văn bản tiếng Việt, giới hạn peak chống clipping, giới hạn độ dài văn bản (>2000 ký tự) chống treo máy CPU.
- Tự nhận transcript audio mẫu bằng Whisper khi để trống.
- Giao diện song ngữ Việt/Anh.
- Ổn định hoá: khóa luồng (RLock) + hàng đợi concurrency=1 chống OOM khi 2 request cùng nạp model; tự lùi CPU nếu cuda/mps lỗi.
- CLI so sánh không cần web (`compare_models.py`) → xuất report.md có bảng số liệu, phù hợp đưa vào báo cáo đồ án.
- Dọn dữ liệu giả, đổi tên thương hiệu (bỏ "F5T"), đổi tên thư mục dự án.
- **Mới nhất (0efae2d):** sửa bug trong `data_prep.py`/`train.py`, thêm Colab notebook để chuẩn bị fine-tune thật.

## 5. ĐANG LÀM / DỞ DANG 🔧

- Pipeline fine-tune "thật" (data_prep → train → Colab notebook) vừa được sửa bug và thêm notebook ở commit mới nhất, nhưng **chưa từng chạy thành công với dữ liệu giọng người thật** — chưa có bằng chứng end-to-end hoạt động.
- Chất lượng câu dài: nửa sau câu đôi khi lệch từ (out-of-sync) khi audio mẫu không đủ sạch hoặc transcript mẫu không chính xác. Đã có hướng khắc phục (tăng NFE 32→48–64, mẫu sạch 5–10s) nhưng chưa tự động hoá hay kiểm chứng có hệ thống.

## 6. CHƯA LÀM / BLOCK ❌

- **Không có dữ liệu huấn luyện thật.** `data/raw/`, `data/processed/`, `data/metadata/` đều rỗng — chỉ có README hướng dẫn. Đây là điểm nghẽn lớn nhất: không thể fine-tune ra giọng riêng nếu không có 30–60 phút audio giọng người thật (24kHz, mono, sạch) + transcript.
- **Chưa có đánh giá WER/CER khách quan.** Mới đo được RMS/spectral flatness (phân biệt nhiễu vs giọng có cấu trúc), chưa đo được "phát âm đúng bao nhiêu %". `jiwer` và `faster-whisper` đã có trong requirements.txt nhưng chưa có script nào dùng chúng.
- **Chưa có SECS (speaker similarity)** dù `resemblyzer` đã khai trong requirements — chưa có script tính cosine similarity giữa giọng mẫu và giọng sinh ra để đánh giá "giống giọng mẫu" khách quan.
- **Chưa có test tự động** (pytest/CI) cho app.py, engines.py, scripts/*.
- **Checkpoint fine-tune cũ không cứu được** — chỉ giữ để đối chiếu cho thấy chúng kém hơn (train trên dữ liệu giả).
- Chưa deploy/đóng gói (Docker, HF Spaces, hay tương tự) để người ngoài dùng thử mà không cần setup môi trường local.

## 7. Rủi ro / cần lưu ý

- Model nền license **CC-BY-NC-SA-4.0** — chỉ dùng nghiên cứu phi thương mại, không được dùng thương mại nếu không đổi model.
- `data/_invalid_demo_data/` là bẫy: nếu vô tình trỏ `data_prep.py`/`train.py` vào đây sẽ tái tạo lỗi cũ (train trên sóng sin).
- Edge-TTS cần internet (cloud) — không hoạt động offline, cần fallback rõ ràng trong so sánh.
- Bark rất chậm trên CPU (~100s/câu) — cân nhắc ẩn mặc định hoặc cảnh báo thời gian chờ.

---

## 8. Kế hoạch hoàn thiện dự án

### Giai đoạn A — Thu thập dữ liệu thật (điều kiện tiên quyết, ưu tiên cao nhất)
1. Thu âm hoặc chọn nguồn audio giọng người thật: 30–60 phút, 1 người nói, 24kHz mono, phòng yên tĩnh (hoặc dùng bộ dữ liệu mở như ViVoice/VietSpeech nếu chỉ cần chứng minh pipeline).
2. Đặt file thô vào `data/raw/`, chạy `python scripts/data_prep.py --input_dir data/raw --output_dir data/processed` để cắt đoạn 3–10s.
3. Điền transcript chính xác vào `data/metadata/metadata.csv` (tay hoặc dùng Whisper rồi soát lại — không tin tưởng ASR 100%).
4. Nghe lại thủ công 5–10 đoạn ngẫu nhiên để xác nhận không lặp lại lỗi "dữ liệu giả" trước đây.

### Giai đoạn B — Chạy fine-tune thật và kiểm chứng
1. Chạy `scripts/train.py --config configs/train_config.yaml` (local nếu có GPU, hoặc `notebooks/finetune_colab.ipynb` trên Colab T4).
2. Theo dõi TensorBoard (`logs/`) để xác nhận loss giảm hợp lý, không NaN/divergence.
3. So sánh checkpoint mới với model gốc bằng `scripts/compare_models.py` — bắt buộc phải nghe được giọng, không nhiễu.
4. Lưu checkpoint tốt nhất, xoá/không commit các checkpoint kém.

### Giai đoạn C — Đánh giá khách quan (đóng lỗ hổng lớn nhất về đo lường)
1. Viết script `scripts/evaluate.py`:
   - WER/CER: dùng `faster-whisper` phiên âm lại output rồi so với văn bản gốc qua `jiwer`.
   - SECS: dùng `resemblyzer` trích embedding giọng mẫu và giọng sinh ra, tính cosine similarity.
2. Chạy trên một bộ câu test cố định (10–20 câu đa dạng độ dài) cho tất cả engine → bảng so sánh WER/CER/SECS/RMS/thời gian.
3. Đưa bảng này vào báo cáo đồ án — đây là bằng chứng định lượng thay vì chỉ nhận định "nghe có vẻ ổn".

### Giai đoạn D — Cải thiện chất lượng câu dài
1. Thử tăng NFE (32 → 48/64) và đo lại WER để xác nhận cải thiện có ý nghĩa (không chỉ là cảm nhận).
2. Thử cắt câu dài thành các câu con ngắn hơn trước khi đưa vào model (nếu model nhạy độ dài).
3. Chuẩn hoá thêm số → chữ, viết tắt tiếng Việt (có thể tận dụng `underthesea` đã có trong requirements nhưng dường như chưa dùng trong `app.py` — kiểm tra lại).

### Giai đoạn E — Đóng gói & test
1. Thêm test cơ bản (pytest) cho: load model không lỗi, inference không crash với input hợp lệ/không hợp lệ, engines.py xử lý lỗi mạng (Edge-TTS) gracefully.
2. Viết Dockerfile hoặc hướng dẫn deploy lên HuggingFace Spaces để demo không cần cài local.
3. Cập nhật README: gộp nội dung 2 file báo cáo hiện có (`README.md` + `BAO_CAO_TIEN_DO.md`) thành 1 nguồn sự thật duy nhất, tránh trùng lặp/lệch thông tin (có thể xoá `BAO_CAO_TIEN_DO.md` sau khi merge vào file này).

### Giai đoạn F — Hoàn thiện báo cáo đồ án
1. Tổng hợp: kiến trúc hệ thống, 2 bug lớn đã sửa (kèm số liệu trước/sau), bảng đánh giá 6 engine (giai đoạn C), quá trình fine-tune (giai đoạn B), giới hạn còn lại.
2. Chuẩn bị demo trực tiếp: app chạy ổn định, có sẵn audio mẫu sạch, kịch bản demo (2-3 câu, ngắn/dài, có dấu câu phức tạp).

### Ưu tiên nếu thời gian gấp
Nếu deadline gần, ưu tiên: **Giai đoạn A + B rút gọn** (dùng bộ dữ liệu mở có sẵn thay vì tự thu âm, để có ít nhất 1 checkpoint fine-tune thật hoạt động) + **Giai đoạn C** (bảng đánh giá khách quan — đây là phần "điểm" dễ bị hỏi nhất trong bảo vệ đồ án vì hiện tại chưa có số liệu WER/CER nào cả).
