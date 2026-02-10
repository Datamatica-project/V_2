FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    libglib2.0-0 \
    libgl1 \
 && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

RUN python -m pip install --no-cache-dir -U pip

RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir \
    "ultralytics[export]==8.4.13" \
    opencv-python-headless \
    PyYAML

RUN pip install --no-cache-dir \
    fastapi uvicorn

RUN python - << 'EOF'
import yaml, cv2
from ultralytics import YOLO
print("OK ultralytics export-ready")
EOF

EXPOSE 8021
CMD ["uvicorn", "rtdetr_server:app", "--host", "0.0.0.0", "--port", "8021"]
