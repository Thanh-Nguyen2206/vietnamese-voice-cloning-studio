# ⚠️ Dữ liệu không hợp lệ — KHÔNG dùng để huấn luyện

Các file trong thư mục này được tạo tự động để chạy thử pipeline, **không phải
giọng người thật**:

- `processed/segment_*.wav` — thực chất là **sóng sin thuần** (~1392 Hz),
  độ phẳng phổ ≈ 0.0002. Không có cấu trúc giọng nói.
- `processed/synthetic_audio_*.wav` — tín hiệu tổng hợp dạng tông, không phải tiếng nói.
- `metadata.csv` — trỏ tới đường dẫn Colab `/content/...` không tồn tại trên máy này.

## Vì sao quan trọng

Các checkpoint trong `checkpoints/step_*/model.pt` đã được fine-tune **trên chính
dữ liệu giả này**. Vì vậy chúng làm hỏng mô hình gốc và cho ra **nhiễu tĩnh / gần
như im lặng** thay vì giọng người. Đó là lý do app phiên bản trước nghe như nhiễu.

Bản app hiện tại mặc định dùng **mô hình gốc** (sạch) và chỉ liệt kê các checkpoint
này để bạn **tự đối chiếu** trong giao diện so sánh.

## Muốn fine-tune thật?

Đặt **audio giọng người thật** (30–60 phút, 24kHz mono) vào `data/raw/`, rồi:

```bash
python scripts/data_prep.py --input_dir data/raw --output_dir data/processed
# điền transcript vào cột "text" của data/metadata/metadata.csv
python scripts/train.py --config configs/train_config.yaml
```
