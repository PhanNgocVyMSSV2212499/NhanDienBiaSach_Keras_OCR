# Chọn Python 3.10 (ổn định cho keras-ocr)
FROM python:3.10-slim

# Cài đặt các gói hệ thống cần thiết
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Nâng cấp pip
RUN pip install --upgrade pip

# Cài keras-ocr và các thư viện liên quan (chốt version ổn định)
RUN pip install keras-ocr==0.9.3 tensorflow==2.10.1 numpy==1.23.5 matplotlib opencv-python-headless

# Cài torch (CPU version) + easyocr để hỗ trợ tiếng Việt
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install easyocr

# Đặt thư mục làm việc trong container
WORKDIR /app

# Copy toàn bộ code từ máy vào container
COPY . /app

# Lệnh mặc định khi container chạy
CMD ["python", "testKeras.py"]