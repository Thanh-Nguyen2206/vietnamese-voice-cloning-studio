# Deployment

## Docker CPU

Chạy `docker compose build && docker compose up`, mở cổng 7860. Lần đầu có cold start vì model tải vào
volume cache. CPU RAM nên từ 8–16 GB tùy engine; chỉ bật F5/MMS/Piper khi tài nguyên thấp. Bark nên tắt.

## Hugging Face Spaces

1. Tạo Docker Space, copy repository và dùng Dockerfile hiện có.
2. Không commit audio, dataset hoặc checkpoint riêng. Dùng persistent storage cho `HF_HOME` nếu có.
3. Đặt `VVCS_DEVICE=cpu`, `VVCS_OUTPUT_DIR=/tmp/outputs`; tắt Bark mặc định và cân nhắc
   `VVCS_ENABLE_CLOUD_ENGINES=0` nếu Space không cho outbound/cloud.
4. Gradio dùng cổng 7860. Cold start/model download có thể vượt free-tier RAM/time.
5. Secret/token Hugging Face chỉ đặt trong Space Settings, không ghi vào source/log.
6. Hiển thị license CC-BY-NC-SA-4.0 và điều khoản đồng ý sử dụng giọng.

Chưa có deployment nào được thực hiện từ repository audit này; tài liệu chỉ là runbook.
