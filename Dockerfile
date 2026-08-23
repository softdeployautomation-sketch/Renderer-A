# Same render worker in one image — run it on GitHub Actions, a VPS, GCP,
# AWS, or your own machine. The code never depends on where it runs or on a
# stable inbound IP: it always talks OUTBOUND (HTTPS Cloudflare API + R2).
# Use the small CPU-oriented image to keep cold starts acceptable on ubuntu
# GitHub runners; a GPU-safe variant can swap python:3.11 with a CUDA stack.
FROM python:3.11-slim

# ffmpeg + ffprobe (required for scene encode + concat)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first so code edits don't invalidate the model/package cache.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Long-running worker (VPS / dev). For GitHub Actions use
#   python -m renderer.main --max-jobs N --max-seconds S
ENTRYPOINT ["python", "-m", "renderer.main"]