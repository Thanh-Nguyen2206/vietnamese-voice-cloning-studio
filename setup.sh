#!/usr/bin/env bash
# =============================================================================
# Vietnamese Voice Cloning Studio - Environment Setup Script
#
# Mục đích: Tự động cài đặt toàn bộ môi trường cho dự án
# Hardware : NVIDIA T4 (16GB VRAM)
# CUDA     : Hỗ trợ cả 11.8 và 12.1
#
# Cách chạy:
#   chmod +x setup.sh
#   bash setup.sh
#
# Lưu ý: Script này cần chạy trên máy có GPU NVIDIA + CUDA driver
#         (Kaggle, Google Colab Pro, RunPod, hoặc local)
# =============================================================================

set -e  # Dừng ngay nếu có lệnh nào lỗi

# ========================== CẤU HÌNH ==========================
ENV_NAME="vvcs"             # Tên conda environment (Vietnamese Voice Cloning Studio)
PYTHON_VERSION="3.10"       # Python 3.10 ổn định nhất với PyTorch + F5-TTS
CUDA_VERSION="12.1"         # Đổi thành "11.8" nếu máy bạn dùng CUDA 11.8
# ===============================================================

echo "=============================================="
echo " Vietnamese Voice Cloning Studio - Environment Setup"
echo " Target GPU: NVIDIA T4 (16GB VRAM)"
echo " CUDA Version: ${CUDA_VERSION}"
echo "=============================================="

# -----------------------------------------------------------------------------
# Bước 1: Kiểm tra NVIDIA Driver & GPU
# Đảm bảo GPU khả dụng trước khi cài đặt
# -----------------------------------------------------------------------------
echo ""
echo "[1/6] Kiểm tra GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ nvidia-smi tìm thấy. Thông tin GPU:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "⚠️  nvidia-smi không tìm thấy!"
    echo "    Nếu bạn đang trên Kaggle/Colab, hãy bật GPU Accelerator trước."
    echo "    Script vẫn tiếp tục cài đặt (có thể chạy CPU-only để test)..."
fi

# -----------------------------------------------------------------------------
# Bước 2: Tạo Conda Environment (nếu dùng conda)
# Nếu không có conda (Kaggle/Colab), bỏ qua bước này và dùng pip trực tiếp
# -----------------------------------------------------------------------------
echo ""
echo "[2/6] Thiết lập Python environment..."

if command -v conda &> /dev/null; then
    echo "    Conda detected. Tạo environment '${ENV_NAME}'..."
    
    # Xóa env cũ nếu tồn tại (tránh conflict)
    conda deactivate 2>/dev/null || true
    conda env remove -n "${ENV_NAME}" -y 2>/dev/null || true
    
    # Tạo env mới
    conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
    
    # Activate environment
    eval "$(conda shell.bash hook)"
    conda activate "${ENV_NAME}"
    
    echo "✅ Conda environment '${ENV_NAME}' đã được tạo và kích hoạt"
else
    echo "    Conda không tìm thấy. Sử dụng pip trong môi trường hiện tại."
    echo "    (Phù hợp cho Kaggle/Colab/RunPod)"
    
    # Đảm bảo pip được cập nhật
    pip install --upgrade pip
fi

# -----------------------------------------------------------------------------
# Bước 3: Cài đặt PyTorch + torchaudio với CUDA
# Đây là bước QUAN TRỌNG NHẤT - phải khớp CUDA version với driver
# -----------------------------------------------------------------------------
echo ""
echo "[3/6] Cài đặt PyTorch + torchaudio (CUDA ${CUDA_VERSION})..."

if [ "${CUDA_VERSION}" = "12.1" ]; then
    # PyTorch 2.3.x với CUDA 12.1
    pip install torch==2.3.1 torchaudio==2.3.1 \
        --index-url https://download.pytorch.org/whl/cu121
elif [ "${CUDA_VERSION}" = "11.8" ]; then
    # PyTorch 2.3.x với CUDA 11.8  
    pip install torch==2.3.1 torchaudio==2.3.1 \
        --index-url https://download.pytorch.org/whl/cu118
else
    echo "❌ CUDA version '${CUDA_VERSION}' không được hỗ trợ."
    echo "   Chỉ hỗ trợ: 11.8, 12.1"
    exit 1
fi

echo "✅ PyTorch đã cài đặt thành công"

# -----------------------------------------------------------------------------
# Bước 4: Cài đặt các thư viện từ requirements.txt
# Bao gồm: f5-tts, gradio, librosa, accelerate, etc.
# -----------------------------------------------------------------------------
echo ""
echo "[4/6] Cài đặt dependencies từ requirements.txt..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pip install -r "${SCRIPT_DIR}/requirements.txt"

echo "✅ Tất cả dependencies đã cài đặt"

# -----------------------------------------------------------------------------
# Bước 5: Cài đặt ffmpeg (cần thiết cho pydub xử lý audio)
# ffmpeg là system package, cài qua apt hoặc conda
# -----------------------------------------------------------------------------
echo ""
echo "[5/6] Kiểm tra & cài đặt ffmpeg..."

if command -v ffmpeg &> /dev/null; then
    echo "✅ ffmpeg đã có sẵn: $(ffmpeg -version 2>&1 | head -1)"
else
    echo "    ffmpeg chưa cài. Đang cài đặt..."
    if command -v apt-get &> /dev/null; then
        # Ubuntu/Debian (Colab, Kaggle, RunPod)
        sudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg
    elif command -v conda &> /dev/null; then
        conda install -c conda-forge ffmpeg -y
    elif command -v brew &> /dev/null; then
        # macOS
        brew install ffmpeg
    else
        echo "⚠️  Không thể tự động cài ffmpeg."
        echo "    Hãy cài thủ công: https://ffmpeg.org/download.html"
    fi
fi

# -----------------------------------------------------------------------------
# Bước 6: Tạo cấu trúc thư mục dự án
# Chuẩn bị sẵn các folder cho các Phase tiếp theo
# -----------------------------------------------------------------------------
echo ""
echo "[6/6] Tạo cấu trúc thư mục dự án..."

PROJECT_DIR="${SCRIPT_DIR}"

# Tạo các thư mục con
mkdir -p "${PROJECT_DIR}/data/raw"           # Audio gốc chưa xử lý
mkdir -p "${PROJECT_DIR}/data/processed"     # Audio đã cắt + clean
mkdir -p "${PROJECT_DIR}/data/metadata"      # File CSV/JSON mapping text-audio
mkdir -p "${PROJECT_DIR}/configs"            # YAML config cho training
mkdir -p "${PROJECT_DIR}/scripts"            # Scripts: preprocess, train, eval
mkdir -p "${PROJECT_DIR}/checkpoints"        # Model checkpoints sau training
mkdir -p "${PROJECT_DIR}/outputs"            # Audio sinh ra (inference results)
mkdir -p "${PROJECT_DIR}/logs"               # Training logs & TensorBoard
mkdir -p "${PROJECT_DIR}/evaluation"         # Kết quả đánh giá (spectrogram, metrics)
mkdir -p "${PROJECT_DIR}/app"                # Gradio web application

echo "✅ Cấu trúc thư mục đã được tạo:"
echo ""
echo "  vietnamese-voice-cloning-studio/"
echo "  ├── data/"
echo "  │   ├── raw/              # Audio gốc (30-60 phút)"
echo "  │   ├── processed/        # Audio đã xử lý (3-10s segments)"
echo "  │   └── metadata/         # Transcription CSV/JSON"
echo "  ├── configs/              # Training YAML configs"
echo "  ├── scripts/              # Python scripts (preprocess, train, eval)"
echo "  ├── checkpoints/          # Saved model weights"
echo "  ├── outputs/              # Generated audio files"
echo "  ├── logs/                 # TensorBoard logs"
echo "  ├── evaluation/           # Metrics & spectrograms"
echo "  ├── app/                  # Gradio web demo"
echo "  ├── requirements.txt"
echo "  ├── setup.sh"
echo "  └── verify_env.py"

# =============================================================================
# HOÀN TẤT
# =============================================================================
echo ""
echo "=============================================="
echo " ✅ SETUP HOÀN TẤT!"
echo "=============================================="
echo ""
echo " Bước tiếp theo:"
echo "   1. Chạy: python verify_env.py"
echo "      → Kiểm tra tất cả thư viện đã cài đúng"
echo ""
echo "   2. Nếu dùng conda, nhớ activate trước khi làm việc:"
echo "      conda activate ${ENV_NAME}"
echo ""
echo "   3. Xác nhận với team rồi sang Phase 2:"
echo "      → Data Collection & Preprocessing"
echo "=============================================="
