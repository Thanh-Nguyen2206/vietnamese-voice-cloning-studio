# Kiến trúc

`app.py` chỉ orchestration/UI. `voice_studio/` chứa logic nhẹ có thể test offline. Model được lazy-load;
F5-TTS dùng `preprocess_ref_audio_text` và `infer_process` chính thức. Cấu hình DiT lấy từ
`F5TTS_Base.yaml`, với fallback vẫn giữ `text_mask_padding=False` và `pe_attn_head=1`.

Long text đi qua NFC normalization → boundary-aware chunking → inference tuần tự → silence → peak
normalization. URL, email, số thập phân, ngày và viết tắt thông dụng được bảo vệ khi tách.

Mỗi optional engine cache riêng. Orchestrator bắt lỗi theo card, map lỗi dependency/network/OOM và giữ
thứ tự registry dù thời gian hoàn thành khác nhau. Queue/concurrency là 1 để giảm rủi ro OOM.

Training: metadata → deterministic split → raw waveform/mel lens → CFM/DiT → validation → checkpoint có
metadata. Resume là explicit và bị từ chối nếu thiếu metadata hoặc sai base model.
