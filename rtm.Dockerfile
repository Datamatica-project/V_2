FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    python3-dev \
    python-is-python3 \
    build-essential \
    ninja-build \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir -U pip setuptools wheel
RUN python3 -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu118 \
    torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2
COPY requirements/rtm-cu118-torch21.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt
RUN mim install "mmcv==2.1.0" \
    -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
RUN python3 -m pip install --no-cache-dir --force-reinstall "numpy<2"

RUN python3 - <<'PY'
import numpy as np
print("numpy:", np.__version__)
import torch
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "avail:", torch.cuda.is_available())
import mmcv, mmdet
print("mmcv:", mmcv.__version__)
print("mmdet:", mmdet.__version__)
from mmcv.ops import nms
print("mmcv ops OK")
PY

EXPOSE 8022
ENV PYTHONPATH="/app:${PYTHONPATH}"
CMD ["uvicorn", "api.rtm.server:app", "--host", "0.0.0.0", "--port", "8022"]
