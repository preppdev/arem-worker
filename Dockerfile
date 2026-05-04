# RunPod serverless image for the AREM editing pipeline.
# Bakes everything in: source code, Photomatix CLI, classifier weights.
# Restormer checkpoints (~778 MB) are downloaded from R2 at handler init.

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps (rawpy needs libraw; opencv-headless avoids GUI deps;
# rclone for Dropbox + R2 transfers; libgl is for cv2 even headless on
# some manylinux wheels). Photomatix is a static binary, no extra libs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip python3.11-venv \
        rclone curl ca-certificates libraw-dev \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

# Python deps
WORKDIR /workspace
COPY requirements.txt /workspace/
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

# Photomatix CLI (24 MB static binary + presets)
COPY photomatix/ /opt/photomatix/
ENV PMTX_STATIC=/opt/photomatix/PhotomatixCL/PhotomatixCL-static

# Pipeline source code (Restormer model, infer pipeline, upright, classifier)
COPY pipeline/ /workspace/pipeline/
ENV AREM_REPO=/workspace/pipeline \
    UPRIGHT_REPO=/workspace/pipeline \
    CLASSIFIER_PATH=/workspace/pipeline/classifier_v2.pth

# Restormer checkpoints land here at handler init (downloaded from R2).
ENV CHECKPOINT_INTERIOR=/workspace/checkpoints/interior_full_v1_latest.pth \
    CHECKPOINT_EXTERIOR=/workspace/checkpoints/exterior_full_v1_latest.pth \
    CHECKPOINT_R2_PREFIX=r2:arem-training-data/checkpoints

# Worker code + handler
COPY worker.py /workspace/worker.py
COPY handler.py /workspace/handler.py
COPY entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh

ENV WORK_ROOT=/tmp/arem-worker \
    PYTHON_BIN=/usr/bin/python3.11 \
    DASHBOARD_URL=https://arem-editing-dashboard.vercel.app

CMD ["/workspace/entrypoint.sh"]
