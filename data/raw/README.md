# Đặt audio giọng THẬT của bạn vào đây

Thư mục này hiện **trống** — đây chính là nơi duy nhất trong dự án cần bạn tự
đưa dữ liệu vào. Không có gì ở đây được tự động tải hay tạo sẵn.

## Cần gì

- **Giọng của 1 người duy nhất** (chính bạn, hoặc người bạn muốn nhân bản giọng).
- **Tổng thời lượng 30–60 phút** audio có tiếng nói thực (chưa tính khoảng lặng).
  Có thể là nhiều file ngắn hoặc vài file dài — script sẽ tự gộp và cắt lại.
- **Sạch**: phòng yên tĩnh, không nhạc nền, không tiếng vọng/echo, không tiếng ồn
  nền (quạt, điều hoà, xe cộ...). Micro càng gần miệng càng tốt (không dùng mic
  laptop/phone để xa).
- **Định dạng bất kỳ trong**: `.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`, `.wma` —
  script tự chuyển hết về chuẩn 24kHz/mono/16-bit, không cần bạn tự convert.
- Có thể đặt trực tiếp ở đây hoặc trong thư mục con (script quét đệ quy).
- **Không cần cắt sẵn thành câu ngắn** — cứ để nguyên bản ghi dài, script sẽ tự
  cắt tại các điểm nghỉ hơi tự nhiên (đã kiểm chứng lại logic cắt này ngày
  14/07/2026, xem ghi chú trong `scripts/data_prep.py`).

## Sau khi đặt file vào đây

```bash
cd vietnamese-voice-cloning-studio
python scripts/data_prep.py --input_dir data/raw --output_dir data/processed
```

Lệnh này sẽ:
1. Chuẩn hoá & cắt audio thành đoạn 3–10 giây tại điểm nghỉ tự nhiên.
2. Ghi ra `data/processed/segment_*.wav`.
3. Tạo `data/metadata/metadata.csv` với cột `text` **để trống**.

## Bước tiếp theo — điền transcript (BẮT BUỘC, không có cách nào tự động hoá đúng)

Mở `data/metadata/metadata.csv`, nghe từng file `segment_*.wav` tương ứng, gõ
**chính xác** những gì bạn nói vào cột `text` (giữ dấu tiếng Việt đầy đủ). Đây là
bước tốn thời gian nhất nhưng quan trọng nhất — transcript sai ở đâu, mô hình học
sai chỗ đó.

Sau khi điền xong hết transcript, nén `data/processed/` + `data/metadata/metadata.csv`
thành 1 file zip và làm theo `notebooks/finetune_colab.ipynb` để train trên GPU miễn phí
của Google Colab (xem lý do bên dưới).

## ⚠️ Trước khi thu âm, cần biết: máy này KHÔNG train được

`scripts/train.py` bắt buộc GPU NVIDIA hỗ trợ FP16 (theo thiết kế cho T4). Máy
hiện tại (Mac, không CUDA) không thể chạy huấn luyện thật — chỉ dùng để chuẩn bị
dữ liệu (`data_prep.py`, không cần GPU nên chạy tốt ở đây). Khi có transcript đầy đủ,
mở **`notebooks/finetune_colab.ipynb`** trên [Google Colab](https://colab.research.google.com/)
(tải notebook lên hoặc mở trực tiếp từ GitHub) — notebook đã dựng sẵn toàn bộ quy trình:
mount Drive, tải dữ liệu bạn đã xử lý ở đây lên, chạy `train.py` trên GPU T4 miễn phí,
tự resume nếu bị ngắt phiên, và hướng dẫn mang checkpoint kết quả về máy này để dùng
trong `app.py`.
