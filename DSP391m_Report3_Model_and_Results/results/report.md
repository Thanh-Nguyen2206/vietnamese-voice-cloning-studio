# Báo cáo đánh giá TTS

Manifest: `outputs/benchmark/manifest.json`

> Speaker similarity là metric tương đối, không phải bằng chứng nhận dạng sinh trắc học.

| Engine | Mẫu | WER mean/median | CER mean/median | Similarity | Inference (s) | RTF | Lỗi |
|---|---:|---:|---:|---:|---:|---:|---:|
| bark | 7 | 1.4530 / 1.2500 | 1.2235 / 0.9643 | 0.4691 | 105.8400 | 7.4800 | 0 |
| edge | 7 | 0.2388 / 0.0909 | 0.2052 / 0.0392 | 0.5648 | 2.9457 | 0.5537 | 0 |
| f5tts | 7 | 0.2237 / 0.0870 | 0.1894 / 0.0260 | 0.8624 | 92.1071 | 22.7461 | 0 |
| mms | 7 | 0.4316 / 0.3636 | 0.2658 / 0.1553 | 0.6178 | 2.1729 | 0.3948 | 0 |
| piper | 7 | 0.2706 / 0.0909 | 0.2181 / 0.0683 | 0.6369 | 0.4557 | 0.1543 | 0 |
| xtts | 7 | 0.2180 / 0.1111 | 0.1911 / 0.0536 | 0.8767 | 13.2229 | 2.9221 | 0 |
