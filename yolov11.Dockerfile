FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# ✅ python 커맨드 보장 (너희 코드/스크립트에서 python을 쓰는 경우 대비)
RUN ln -sf /usr/bin/python3 /usr/bin/python

RUN python -m pip install --no-cache-dir -U pip

# ✅ torch/torchvision (CUDA 12.1 wheel)
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121

# ✅ ultralytics는 deps 포함 설치 (버전 고정 추천)
RUN pip install --no-cache-dir \
    ultralytics==8.4.13 \
    opencv-python-headless \
    PyYAML

RUN pip install --no-cache-dir fastapi uvicorn

# ✅ 빌드 타임에 바로 검증 (여기서 실패하면 이미지 빌드가 실패함)
RUN python - << 'EOF'
import yaml
import cv2
from ultralytics import YOLO
print("OK ultralytics")
EOF

EXPOSE 8020
CMD ["uvicorn", "yolov11_server:app", "--host", "0.0.0.0", "--port", "8020"]
