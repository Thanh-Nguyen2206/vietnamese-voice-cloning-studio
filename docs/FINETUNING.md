# Fine-tuning

Điều kiện: 30–60 phút giọng người thật, một người nói, mono 24 kHz, transcript được soát tay. Không dùng
audio tổng hợp để tuyên bố chất lượng. `data/_invalid_demo_data` bị chặn trừ cờ technical-test có tên rõ.

Pipeline: preprocess → nghe kiểm tra → điền transcript → `data_prep.py validate` → train. Config DiT không
được đổi hai field compatibility. Mixed precision tự tắt khi không có CUDA; CPU chỉ phù hợp smoke test.

Resume phải chỉ định `--resume`. Mỗi checkpoint lưu model/optimizer/training state và JSON gồm base model,
architecture, vocab size, config, git commit, step, validation loss và dataset summary. `best/` chọn theo
validation loss. End-to-end hiện bị block bởi dữ liệu thật và GPU trong môi trường audit.
