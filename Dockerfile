# Same render worker in one image — run it on GitHub Actions, a VPS, GCP,
# AWS, or your own machine. The code never depends on where it runs or on a
# stable inbound IP: it always talks OUTBOUND (HTTPS Cloudflare API + R2).
# Use the small CPU-oriented image to keep cold starts acceptable on ubuntu
# GitHub runners; a GPU-safe variant can swap python:3.11 with a CUDA stack.
FROM python:3.11-slim

# ffmpeg + ffprobe (required for scene encode + concat)
# git (clone SadTalker repo), curl (download checkpoints),
# libsm6/libxext6 (opencv headless GUI libs SadTalker/opencv import).
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    curl \
    libsm6 \
    libxext6 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first so code edits don't invalidate the model/package cache.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# ── SadTalker (talking-head from photo + audio) ──────────────────────────
# Provisioned once at image build time so each render job starts fast.
# torch CPU wheel (no GPU on GitHub free runners) + SadTalker python deps.
# NOTE: torch is installed with its own --index-url so it doesn't shadow
# PyPI for the other packages (opencv-python etc. are NOT on the PyTorch
# index). Installing in one pip call with --index-url would break those.
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir \
    opencv-python \
    scipy \
    tqdm \
    safetensors \
    facexlib

# Clone the SadTalker repo + download the checkpoints needed for the
# safetensor code path with preprocess='crop' (see renderer/sadtalker.py).
RUN git clone --depth 1 https://github.com/OpenTalker/SadTalker.git vendor/SadTalker \
  && mkdir -p vendor/SadTalker/checkpoints \
  && curl -fsSL -o vendor/SadTalker/checkpoints/SadTalker_V0.0.2_256.safetensors \
     https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_256.safetensors \
  && curl -fsSL -o vendor/SadTalker/checkpoints/mapping_00229-model.pth.tar \
     https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00229-model.pth.tar

COPY . .

# Long-running worker (VPS / dev). For GitHub Actions use
#   python -m renderer.main --max-jobs N --max-seconds S
ENTRYPOINT ["python", "-m", "renderer.main"]