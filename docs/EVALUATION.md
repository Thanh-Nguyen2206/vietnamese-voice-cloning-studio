# Evaluation

Manifest JSON cần `id`, `reference_text`, `generated_audio`, `engine`; `reference_audio`, checkpoint,
seed, NFE, inference time và metadata là tùy chọn. Đường dẫn tương đối được resolve theo thư mục manifest.

WER/CER giữ dấu tiếng Việt mặc định; `--ignore-case-punctuation` bật chính sách relaxed. ASR dùng
faster-whisper. Speaker embedding dùng Resemblyzer, sau đó cosine similarity. Audio metrics gồm duration,
RMS, spectral flatness, peak và clipping ratio; RTF = inference time / duration.

Mỗi case có trạng thái riêng. File hỏng/dependency thiếu không dừng batch. Exit 0 khi có ít nhất một case
thành công, 1 khi tất cả thất bại, 2 khi manifest invalid. Không so report giữa các engine nếu text, seed,
reference hoặc môi trường khác nhau.
