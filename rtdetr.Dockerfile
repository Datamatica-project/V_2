FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir \
    ultralytics \
    fastapi \
    uvicorn \
    opencv-python-headless

EXPOSE 8021
CMD ["uvicorn", "rtdetr_server:app", "--host", "0.0.0.0", "--port", "8021", "--reload"]
