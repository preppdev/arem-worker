# RunPod serverless image for the AREM editing pipeline.
# Pipeline: RAW → Stage 1 NAFNet → routed Stage 2 Restormer → auto-upright + EXIF.
# Restormer + Stage-1 NAFNet checkpoints (~1.2 GB) are downloaded from R2 at
# handler init.

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps:
#   - rawpy  → libraw
#   - lensfunpy → liblensfun (and the runtime DB at /usr/share/lensfun)
#   - opencv-headless → libgl + glib (some manylinux wheels still need them)
#   - rclone for Dropbox + R2 transfers
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip python3.11-venv \
        rclone curl ca-certificates \
        libraw-dev liblensfun1 liblensfun-bin liblensfun-dev \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && lensfun-update-data || true \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

# Python deps
WORKDIR /workspace
COPY requirements.txt /workspace/
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

# Pipeline source code (NAFNet + Restormer + lens correction + classifier
# + auto-upright). classifier_v2.pth (~50 MB) is baked in; the larger
# Restormer + Stage-1 NAFNet ckpts come from R2 at startup.
COPY pipeline/ /workspace/pipeline/
ENV AREM_REPO=/workspace/pipeline \
    UPRIGHT_REPO=/workspace/pipeline \
    CLASSIFIER_PATH=/workspace/pipeline/classifier_v2.pth

ENV CHECKPOINT_STAGE1=/workspace/checkpoints/stage1_jxl_v1_best_lpips.pth \
    CHECKPOINT_INTERIOR=/workspace/checkpoints/may26_interior_w32_b4_4gpu_ep35_inference.pth \
    CHECKPOINT_EXTERIOR=/workspace/checkpoints/may26_exterior_w32_b4_4gpu_ep29_inference.pth \
    CHECKPOINT_R2_PREFIX=r2:arem-training-data/checkpoints

# Worker code + handler. scripts/ included so the same image can run
# one-shot backfill modes (Stage-1 retroactive coverage, future
# matched-thumbnail jobs, etc.) without affecting production handler.py.
COPY worker.py /workspace/worker.py
COPY handler.py /workspace/handler.py
COPY scripts/ /workspace/scripts/
COPY entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh

ENV WORK_ROOT=/tmp/arem-worker \
    PYTHON_BIN=/usr/bin/python3.11 \
    DASHBOARD_URL=https://arem-editing-dashboard.vercel.app

CMD ["/workspace/entrypoint.sh"]
