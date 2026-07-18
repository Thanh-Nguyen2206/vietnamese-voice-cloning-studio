# Report 3 — Model & Results
### Vietnamese Voice Cloning Studio · Đồ án DSP391m — FPTU Summer 2026

> **Phạm vi Report 3:** Model Development · Model Evaluation and Fine-Tuning ·
> Results Interpretation and Visualization · Conclusion and Recommendations.
>
> Tài liệu này bám theo *Data Science Capstone Project Template* (DSP391m). Mọi số liệu
> trong báo cáo được sinh trực tiếp từ code trong repository:
> `scripts/evaluate.py` → `outputs/evaluation/results.json` → `scripts/plot_results.py`.
> Không có số liệu bịa; các giới hạn được nêu rõ ràng thay vì che giấu.

---

## Tóm tắt điều hành (Executive Summary)

Bài toán là **nhân bản giọng nói tiếng Việt (zero-shot voice cloning TTS)**: cho một đoạn
audio mẫu ngắn của người nói + một đoạn văn bản, hệ thống đọc văn bản đó bằng chính giọng
trong audio mẫu. Mô hình chính là **F5-TTS ViVoice** (kiến trúc DiT + Conditional Flow
Matching, vocoder Vocos 24 kHz). Ngoài mô hình chính, hệ thống tích hợp **5 engine so sánh**
(XTTS-v2, MMS-TTS, Piper, Edge-TTS, Bark) để đối chứng khách quan trên cùng một input.

Kết quả đánh giá định lượng (WER/CER qua ASR round-trip + speaker similarity + tốc độ) cho
thấy **F5-TTS đạt cân bằng tốt nhất** giữa hai yêu cầu đối nghịch của bài toán voice cloning:
*phát âm dễ hiểu* (WER **5.9 %**, thấp nhất trong 6 engine) và *giữ được đặc trưng giọng mẫu*
(speaker similarity **0.837**, chỉ sau XTTS-v2). Đây chính là hai trục mà một hệ voice cloning
phải tối ưu đồng thời — và chỉ F5-TTS và XTTS-v2 nằm ở góc "vừa dễ hiểu vừa giống giọng".

---

## 1. Model Development

### 1.1. Model Selection — vì sao chọn F5-TTS ViVoice

| Tiêu chí | Lý do chọn F5-TTS |
|---|---|
| **Zero-shot voice cloning** | Nhân bản giọng chỉ từ 1 đoạn mẫu ngắn (5–10 s), không cần fine-tune per-speaker |
| **Đã pre-train tiếng Việt** | `hynt/F5-TTS-Vietnamese-ViVoice` train trên ~1000 giờ tiếng Việt → phát âm dấu thanh tốt |
| **Kiến trúc hiện đại** | DiT (Diffusion Transformer) + Conditional Flow Matching — SOTA cho non-autoregressive TTS |
| **Chất lượng phổ giọng** | Không bị "robot"/đơn điệu như VITS một-giọng; giữ ngữ điệu tự nhiên |
| **License nghiên cứu** | CC-BY-NC-SA-4.0 — hợp lệ cho đồ án giáo dục phi thương mại |

5 engine còn lại được chọn làm **baseline đối chứng** trải trên phổ đánh đổi: XTTS-v2 (voice
cloning cạnh tranh), MMS-TTS & Piper (TTS một-giọng, cực nhanh trên CPU), Edge-TTS (baseline
neural thương mại chạy cloud), Bark (baseline yếu, không hỗ trợ tiếng Việt — dùng để thể hiện
"đáy" của thang đo).

### 1.2. Model Architecture — F5-TTS (DiT + CFM)

Kiến trúc được nạp **đúng theo `F5TTS_Base.yaml`** của thư viện gốc (không tự dựng lại), khớp
100 % với checkpoint pre-trained. Cấu hình trong [`configs/train_config.yaml`](../configs/train_config.yaml):

```
CFM (Conditional Flow Matching) wrapper
   └── DiT backbone
         dim              = 1024      # embedding dimension
         depth            = 22        # số DiT blocks
         heads            = 16        # số attention heads
         ff_mult          = 2         # hệ số feed-forward
         text_dim         = 512       # chiều text embedding
         conv_layers      = 4         # conv layers trong text encoder
         text_mask_padding = false    # ⚠ BẮT BUỘC — khớp forward pass ViVoice
         pe_attn_head     = 1         # ⚠ BẮT BUỘC — khớp forward pass ViVoice
   └── Vocoder: Vocos @ 24 kHz (mel → waveform)

Tổng tham số ≈ 336M · Vocab: config.json plain-text (~2.5k token, ký tự Việt có dấu + pinyin kế thừa)
Audio front-end: sr=24000, n_fft=1024, hop=256, win=1024, n_mel=100
```

> **Bài học kỹ thuật quan trọng (đã ghi vào code & test):** hai trường
> `text_mask_padding=false` và `pe_attn_head=1` **không được đổi**. Khi thiếu chúng, DiT lấy
> default của bản v1 → forward pass chạy sai dù trọng số nạp đúng shape → giọng méo/líu nhíu
> (spectral flatness tụt về 0.0019). Sau khi nạp đúng arch, spectral flatness trở lại vùng
> giọng người (~0.11–0.14). Đây là một trong hai bug lớn nhất đã sửa của dự án.

**Cơ chế voice cloning:** F5-TTS và XTTS-v2 học đặc trưng giọng (timbre, cao độ, ngữ điệu) từ
reference audio ngay lúc suy luận (in-context), *không* cần huấn luyện riêng cho từng người. Các
engine MMS/Piper/Edge/Bark dùng **giọng cố định** — đưa vào để đối chứng "TTS thường" vs "cloning".

### 1.3. Inference Pipeline (đường suy luận sản phẩm)

Toàn bộ đường F5-TTS dùng **utility chính thức** của thư viện (`preprocess_ref_audio_text`,
`infer_process`) để đảm bảo bước tokenize (`convert_char_to_pinyin`) và preprocessing khớp với
lúc model được train. Pipeline xử lý văn bản dài (trong [`voice_studio/text_processing.py`](../voice_studio/text_processing.py)):

```
Văn bản → NFC normalize + chuẩn hoá tiếng Việt (giữ dấu thanh)
        → boundary-aware chunking (tách ở ranh giới câu/mệnh đề, max ~280 ký tự,
          BẢO VỆ URL/email/số thập phân/ngày/viết tắt khỏi bị cắt)
        → suy luận tuần tự từng chunk (cùng seed → tái lập được)
        → chèn khoảng lặng 180 ms giữa các chunk
        → peak normalization (≤ 0.99, chống clipping) → WAV 24 kHz
```

Ổn định vận hành: mỗi engine lazy-load & cache riêng, khóa `RLock` + hàng đợi
`concurrency=1` (chống OOM khi hai request cùng nạp model), tự lùi về CPU nếu CUDA/MPS lỗi.

### 1.4. Training Procedure — pipeline fine-tune

Mục tiêu fine-tune: thích nghi mô hình nền với **một giọng cụ thể** (speaker adaptation) trong khi
tránh *catastrophic forgetting*. Toàn bộ pipeline nằm trong [`scripts/train.py`](../scripts/train.py):

```
metadata.csv → VoiceDataset (raw waveform + text, mel lens)
             → deterministic train/val split (random_split, generator seed=42)
             → collate_fn (pad theo hop_length=256)
             → CFM(DiT) forward → flow matching loss (scalar)
             → accelerator.backward (FP16 GradScaler) + gradient accumulation
             → clip_grad_norm_(1.0) → optimizer.step mỗi 8 micro-batch
             → manual LR scheduler: warmup tuyến tính (200 step) → cosine decay
             → validation loss cuối mỗi epoch → chọn best/ theo val loss
             → checkpoint mỗi 500 step (kèm metadata JSON đầy đủ)
```

**Đặc điểm kỹ thuật đã gia cố (production-grade):**
- **EMA weights:** nạp đúng `ema_model_state_dict` (bỏ prefix `ema_model.`) từ checkpoint chính thức.
- **Determinism:** `set_seed(42)`; split train/val tất định theo generator seed → tái lập được.
- **Guardrails:** phát hiện `loss` NaN/Inf và dừng ngay; từ chối resume nếu thiếu metadata hoặc sai base model; chặn dùng `data/_invalid_demo_data` trừ khi có cờ technical-test có tên rõ.
- **Checkpoint metadata JSON:** lưu base model, architecture, vocab size, config, **git commit**, step, validation loss và dataset summary → mỗi checkpoint tự mô tả, truy vết được.
- **`best/`** được chọn theo **validation loss** (fallback về train loss nếu không có tập validation).

> **Trạng thái trung thực:** pipeline fine-tune đã được gia cố và có unit test cho các thành phần
> thuần (collate, split, config, guardrails), **nhưng chưa được xác minh end-to-end với dữ liệu
> giọng người thật + GPU**. Điểm nghẽn là **thiếu 30–60 phút audio giọng người thật** (mono 24 kHz,
> transcript soát tay). Các checkpoint cũ (train trên sóng sin giả) đã bị cách ly và **không** được
> dùng để tuyên bố chất lượng. Vì vậy các kết quả ở Mục 3 là của **mô hình nền (zero-shot), chưa
> fine-tune** — điều này được nêu rõ để tránh over-claim.

---

## 2. Model Evaluation and Fine-Tuning

### 2.1. Evaluation Metrics — bộ chỉ số khách quan

TTS/voice cloning không có "nhãn đúng/sai" đơn giản như phân loại; ta đo bằng **proxy khách quan**
trên hai trục chất lượng cốt lõi + tốc độ. Cài đặt trong [`voice_studio/evaluation.py`](../voice_studio/evaluation.py):

| Nhóm | Chỉ số | Ý nghĩa | Cách đo | Chiều tốt |
|---|---|---|---|---|
| **Dễ hiểu** | **WER** (Word Error Rate) | Sai bao nhiêu % từ | ASR round-trip: Whisper phiên âm lại output rồi so với text gốc (edit distance từ) | ↓ thấp |
| **Dễ hiểu** | **CER** (Character Error Rate) | Sai bao nhiêu % ký tự | như trên nhưng theo ký tự (nhạy hơn với dấu thanh) | ↓ thấp |
| **Danh tính giọng** | **Speaker similarity (SECS)** | Giọng sinh ra có giống audio mẫu không | Resemblyzer embedding → cosine similarity | ↑ cao |
| **Tốc độ** | **RTF** (Real-Time Factor) | Chậm gấp mấy lần thời lượng audio | thời gian tạo ÷ thời lượng | ↓ thấp (<1 = nhanh hơn real-time) |
| **Chất lượng tín hiệu** | RMS, peak, clipping ratio, **spectral flatness** | Âm lượng / méo / "có cấu trúc giọng hay là nhiễu" | thống kê trực tiếp trên waveform | — (dùng để sàng lỗi) |

Chi tiết phương pháp:
- **ASR round-trip** dùng `faster-whisper` (model `small`, `beam_size=5`, `language="vi"`). Đây là
  cách biến "nghe có hiểu không" thành số đo lặp lại được.
- **WER/CER** ưu tiên thư viện `jiwer`, có **fallback edit-distance thuần** tương đương (không phụ
  thuộc mạng). Mặc định **giữ nguyên dấu thanh tiếng Việt**; cờ `--ignore-case-punctuation` bật chính
  sách relaxed (bỏ hoa/thường + dấu câu) để đối chiếu.
- **Speaker similarity là chỉ số *tương đối*, không phải bằng chứng sinh trắc học** — được ghi rõ
  trong report tự sinh để tránh diễn giải sai.
- **Fail-soft:** mỗi test case có trạng thái riêng; file hỏng/thiếu dependency không làm dừng cả batch.
  Exit code: `0` nếu ≥1 case thành công, `1` nếu tất cả fail, `2` nếu manifest sai định dạng.

### 2.2. Data Splitting & Cross-Validation

Với **fine-tune** (khi có dữ liệu thật), chiến lược kiểm định là:
- **Held-out validation split** tất định: `validation_split = 0.1`, tách bằng `random_split` với
  generator cố định seed 42 → mỗi lần chạy cho đúng một tập validation, so sánh được giữa các lần.
- **Model selection theo validation loss:** checkpoint `best/` được chọn theo val loss thấp nhất
  (không phải train loss) → chống overfit và chọn nhầm checkpoint "học vẹt" tập train.

> **Lưu ý học thuật:** k-fold cross-validation cổ điển **không phù hợp** với fine-tune một mô hình
> sinh 336M tham số trên vài chục phút audio (chi phí train × k lần là bất khả thi trên T4, và tín
> hiệu là generative chứ không phải nhãn phân loại). Ở bài toán này, "cross-validation" được thay thế
> hợp lý bằng: (1) **held-out validation loss** trong lúc train, và (2) **bộ test cố định độc lập**
> gồm nhiều câu đa dạng để đo WER/CER/SECS sau train (xem [`evaluation/test_cases.example.json`](../evaluation/test_cases.example.json):
> câu ngắn, số, thập phân, ngày, viết tắt, tên riêng, câu hỏi, câu ghép, câu dài).

### 2.3. Hyperparameter Tuning

Chia làm hai nhóm: **siêu tham số suy luận** (đã có thể tinh chỉnh ngay trên mô hình nền) và **siêu
tham số huấn luyện** (dùng khi fine-tune).

**(a) Siêu tham số suy luận — ảnh hưởng trực tiếp chất lượng đầu ra:**

| Tham số | Mặc định | Khuyến nghị / vai trò |
|---|---:|---|
| **NFE** (số bước ODE flow-matching) | 32 | Câu dài/khó → tăng **48–64** để giảm lệch từ nửa sau câu (đánh đổi tốc độ). Bộ test đặt NFE 32 cho câu ngắn, 48–64 cho câu dài. |
| **Seed** | 42 | Cố định → tái lập; cùng seed dùng chung cho mọi engine → **so sánh công bằng**. |
| **Chunk max_chars** | 280 | Ngưỡng tách câu dài; giảm nếu model nhạy độ dài. |
| **Silence giữa chunk** | 180 ms | Nhịp nghỉ tự nhiên khi ghép câu dài. |
| **Speed** | 1.0 | Tốc độ đọc; map sang length-scale/rate theo từng engine. |

**(b) Siêu tham số huấn luyện — [`configs/train_config.yaml`](../configs/train_config.yaml) (tối ưu cho T4 16 GB):**

| Tham số | Giá trị | Lý do |
|---|---:|---|
| learning_rate | 1e-5 | LR **nhỏ** cho fine-tune → tránh catastrophic forgetting mô hình nền |
| batch_size × grad_accum | 1 × 8 = **8** | VRAM T4 hạn chế → micro-batch 1 + tích lũy gradient |
| warmup_steps | 200 | Warm-up tuyến tính 0 → LR, ổn định giai đoạn đầu |
| scheduler | cosine | Cosine annealing về min_lr 1e-7 → hội tụ mượt |
| max_grad_norm | 1.0 | Gradient clipping → chống bùng nổ gradient |
| mixed_precision | fp16 | T4 (compute 7.5) hỗ trợ FP16; tự tắt khi không có CUDA |
| weight_decay / betas / eps | 0.01 / [0.9, 0.999] / 1e-8 | AdamW tiêu chuẩn |

> **Ràng buộc cứng:** siêu tham số **kiến trúc** (`dim/depth/heads/text_mask_padding/pe_attn_head`,
> cấu hình mel) **không được coi là tunable** — chúng phải khớp checkpoint pre-trained, nếu đổi sẽ hỏng
> forward pass. Đây là ranh giới rõ ràng giữa "tuning hợp lệ" và "phá vỡ tương thích".

---

## 3. Results Interpretation and Visualization

### 3.1. Điều kiện đánh giá (để kết quả tái lập & công bằng)

- **6 engine, cùng một câu, cùng seed 42, cùng audio mẫu** `reference_audio/sample_clean_vi.wav`.
- Câu test: *"Trí tuệ nhân tạo đang thay đổi cách con người giao tiếp và làm việc mỗi ngày."* (17 từ).
- Manifest: [`evaluation/audit_existing_outputs.json`](../evaluation/audit_existing_outputs.json);
  chạy `python scripts/evaluate.py --manifest ...` → [`outputs/evaluation/results.json`](../outputs/evaluation/results.json)
  → `python scripts/plot_results.py` → biểu đồ dưới đây.
- **Cỡ mẫu: n = 1 câu/engine** (đây là *pilot benchmark / audit*, không phải đánh giá quy mô lớn) —
  giới hạn này được nêu thẳng ở Mục 3.5 và là việc cần mở rộng đầu tiên.

### 3.2. Bảng kết quả tổng hợp

| Engine | Loại | WER ↓ | CER ↓ | Speaker sim ↑ | Inference (s) | RTF ↓ |
|---|---|---:|---:|---:|---:|---:|
| **F5-TTS** | voice cloning | **0.059** | **0.026** | 0.837 | 193.0 | 45.12 |
| **XTTS-v2** | voice cloning | 0.118 | 0.065 | **0.888** | 36.2 | 8.48 |
| MMS-TTS | 1 giọng cố định | 0.118 | 0.039 | 0.601 | 3.9 | 0.61 |
| Piper | 1 giọng cố định | 0.118 | 0.052 | 0.615 | **2.1** | **0.59** |
| Edge-TTS | cloud, 1 giọng | 0.118 | 0.065 | 0.561 | 3.0 | 0.65 |
| Bark | 1 giọng (không hỗ trợ VI) | 1.176 | 0.844 | 0.480 | 167.2 | 11.59 |

*(In đậm = tốt nhất mỗi cột. Nguồn: `outputs/evaluation/results.json`, không chỉnh tay.)*

### 3.3. Trực quan hoá

**Hình 1 — Độ chính xác phát âm (WER & CER).** F5-TTS thấp nhất rõ rệt (WER 5.9 %); nhóm
XTTS/MMS/Piper/Edge sát nhau (~11.8 %, chỉ lệch 2 từ); Bark >100 % vì không đọc được tiếng Việt
(phiên âm ra tiếng Anh vô nghĩa) — đóng vai "đáy thang đo".

![WER và CER theo engine](../outputs/evaluation/figures/wer_cer.png)

**Hình 2 — Speaker similarity (SECS).** Chỉ **F5-TTS (0.837)** và **XTTS-v2 (0.888)** vào được vùng
"nhân bản giọng" (>0.75). Các engine một-giọng (MMS/Piper/Edge ~0.56–0.62) về bản chất **không**
nhân bản giọng — điểm SECS của chúng phản ánh giọng cố định mặc định, không phải giọng mẫu.

![Speaker similarity theo engine](../outputs/evaluation/figures/speaker_similarity.png)

**Hình 3 — Tốc độ (RTF, thang log).** Piper/MMS/Edge chạy **nhanh hơn real-time** (RTF < 1) trên CPU;
F5-TTS chậm nhất (RTF ~45 trên CPU) — đây là chi phí của chất lượng + flow-matching nhiều bước, và
sẽ giảm mạnh trên GPU. Lưu ý: engine chạy **đầu tiên** trong tiến trình gánh cả thời gian nạp model
lần đầu, nên con số tuyệt đối chỉ mang tính tương đối.

![Real-time factor theo engine](../outputs/evaluation/figures/speed_rtf.png)

**Hình 4 — Biểu đồ đánh đổi (insight chính).** Trục X = giống giọng mẫu (SECS), trục Y = dễ hiểu
(1 − WER), kích thước bong bóng ∝ tốc độ. Voice cloning tốt phải nằm ở **góc trên-phải**. Chỉ
**F5-TTS và XTTS-v2** đạt cả hai; nhóm TTS một-giọng dễ hiểu nhưng lệch trái (không giống giọng mẫu);
Bark rơi hẳn xuống đáy.

![Đánh đổi dễ hiểu vs giống giọng](../outputs/evaluation/figures/tradeoff.png)

### 3.4. Diễn giải & insight

1. **F5-TTS thắng về cân bằng.** WER thấp nhất (5.9 %) *đồng thời* SECS cao (0.837). Nó là engine duy
   nhất vừa dễ hiểu nhất vừa giữ được danh tính giọng — đúng mục tiêu bài toán.
2. **XTTS-v2 là đối thủ đáng gờm & nhanh hơn.** SECS cao nhất (0.888) và nhanh hơn F5-TTS ~5× trên CPU,
   nhưng WER gấp đôi (11.8 %). → Lựa chọn tốt khi **ưu tiên độ giống giọng và tốc độ** hơn độ chính xác.
3. **TTS một-giọng ≠ voice cloning.** MMS/Piper/Edge dễ hiểu và cực nhanh (RTF < 1), nhưng SECS ~0.6 cho
   thấy chúng **không** tái tạo giọng mẫu. Phù hợp cho ứng dụng chỉ cần "đọc rõ tiếng Việt", không cần
   giọng riêng.
4. **CER < WER ở mọi engine "tốt".** Ví dụ F5-TTS: sai chủ yếu ở **1 từ đầu câu** ("Trí"→"Chí" — lẫn phụ
   âm đầu tr/ch, một hiện tượng phương ngữ), nên CER (2.6 %) nhỏ hơn WER (5.9 %) nhiều. Đây là lỗi *phát âm
   vùng miền*, không phải lỗi "không hiểu".
5. **Đo lường bắt được lỗi thật.** Bark bị gắn cờ tự động (WER >100 %, SECS thấp nhất) — chứng minh bộ
   metric thực sự phân biệt được engine hỏng khỏi engine tốt, không phải "đo cho có".

### 3.5. Uncertainty & Giới hạn (trung thực)

- **Cỡ mẫu nhỏ (n = 1 câu/engine):** đây là audit/pilot. Chưa tính được khoảng tin cậy hay độ lệch chuẩn.
  → Cần chạy trên bộ ≥12 câu đa dạng ([`test_cases.example.json`](../evaluation/test_cases.example.json)) để có mean ± std.
- **ASR bản thân có sai số:** Whisper cũng nhầm (nhất là dấu thanh) → WER/CER là *cận trên* của lỗi thật;
  nên đọc theo hướng **so sánh tương đối giữa engine**, không phải con số tuyệt đối.
- **SECS là tương đối:** cosine similarity giữa embedding, **không** phải xác thực sinh trắc học.
- **Tốc độ đo trên CPU** và gồm cả overhead nạp model lần đầu (với engine đầu tiên) → RTF tuyệt đối sẽ
  khác hẳn trên GPU.
- **Kết quả là của mô hình nền zero-shot, chưa fine-tune.** Chưa có bằng chứng end-to-end rằng fine-tune
  cải thiện thêm (bị chặn bởi thiếu dữ liệu giọng thật + GPU).

---

## 4. Conclusion and Recommendations

### 4.1. Key Findings

1. **Về chọn mô hình:** F5-TTS ViVoice là lựa chọn đúng cho voice cloning tiếng Việt — dẫn đầu về độ
   dễ hiểu (WER 5.9 %) và nằm trong top-2 về độ giống giọng (SECS 0.837), là engine duy nhất tối ưu
   *đồng thời* cả hai trục.
2. **Về đo lường:** đã xây dựng được **framework đánh giá khách quan, tái lập** (WER/CER + SECS + RTF +
   audio metrics) chạy được offline, fail-soft, và có biểu đồ tự sinh — biến nhận định cảm tính "nghe
   ổn" thành số đo kiểm chứng được.
3. **Về kỹ thuật:** hai bug lớn đã được chẩn đoán bằng số liệu và sửa dứt điểm (checkpoint giả tự đè +
   pipeline tokenize sai; thiếu `text_mask_padding`/`pe_attn_head`), khôi phục đầu ra thành tiếng Việt
   rõ ràng, kiểm chứng bằng ASR round-trip.
4. **Về sự đánh đổi:** không có engine "thắng mọi mặt" — F5-TTS trả giá bằng tốc độ (RTF cao trên CPU),
   XTTS-v2 giống giọng nhất nhưng kém chính xác hơn, nhóm một-giọng nhanh nhất nhưng không cloning được.

### 4.2. Actionable Recommendations

**Theo tình huống sử dụng:**
- **Cần cả dễ hiểu và giống giọng, chấp nhận chậm** → **F5-TTS** (khuyến nghị mặc định; bật GPU + tăng
  NFE 48–64 cho câu dài).
- **Ưu tiên giống giọng & tốc độ** → **XTTS-v2** (dùng audio mẫu thật sạch, một giọng).
- **Chỉ cần đọc rõ tiếng Việt, real-time, offline** → **Piper/MMS-TTS** (RTF < 1, không cần giọng riêng).
- **Không dùng Bark cho tiếng Việt** (không hỗ trợ ngôn ngữ; chỉ giữ làm baseline đối chứng).

**Kỹ thuật để nâng chất lượng ngay:**
- Với câu dài, **tăng NFE lên 48–64** và giữ audio mẫu sạch 5–10 s (giảm lệch từ nửa sau câu).
- Dùng **transcript mẫu chính xác** thay vì để Whisper tự nhận (ASR sai sẽ kéo chất lượng xuống).

### 4.3. Reflection & Future Work

**Nhìn lại quá trình:** giá trị lớn nhất của giai đoạn này không phải "làm cho nó chạy" mà là **thiết lập
được thước đo khách quan** — nhờ đó mọi cải tiến sau này đều chứng minh được bằng số, và các bug tinh vi
(forward pass sai dù shape đúng) mới lộ ra. Việc **trung thực về giới hạn** (chưa fine-tune end-to-end,
n=1) được ưu tiên hơn việc tạo số liệu đẹp nhưng không kiểm chứng được.

**Việc cần làm tiếp, theo thứ tự ưu tiên:**
1. **Mở rộng bộ test** từ 1 → ≥12 câu đa dạng (đã có sẵn `test_cases.example.json`) để báo cáo mean ± std,
   có ý nghĩa thống kê.
2. **Thu 30–60 phút audio giọng người thật** (mono 24 kHz, transcript soát tay) → chạy fine-tune
   **end-to-end trên GPU** → đo WER/CER/SECS **trước vs sau** fine-tune để chứng minh giá trị speaker adaptation.
3. **Đánh giá chủ quan (MOS):** bổ sung điểm nghe của con người (naturalness) song song với chỉ số khách quan.
4. **Tối ưu tốc độ F5-TTS:** benchmark trên GPU, thử giảm NFE có kiểm soát, cân nhắc distillation/vocoder nhanh hơn.
5. **Đóng gói & deploy** (Docker/HF Spaces đã có khung) để demo không cần setup local.

---

## Phụ lục — Cách tái lập kết quả

```bash
# 1. (Khi có audio đã sinh) đánh giá theo manifest → results.json + report.md
python scripts/evaluate.py \
    --manifest evaluation/audit_existing_outputs.json \
    --output-dir outputs/evaluation

# 2. Sinh biểu đồ từ results.json
python scripts/plot_results.py \
    --results outputs/evaluation/results.json \
    --output-dir outputs/evaluation/figures

# 3. (Tùy chọn) so sánh 6 engine không cần web
python scripts/compare_models.py --ref reference_audio/sample_clean_vi.wav
```

**Tài liệu liên quan:** [kiến trúc](ARCHITECTURE.md) · [fine-tuning](FINETUNING.md) ·
[evaluation](EVALUATION.md) · [deployment](DEPLOYMENT.md) · [licenses](LICENSES.md).
