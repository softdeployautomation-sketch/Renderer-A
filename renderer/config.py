"""Environment configuration for the render worker.

All values are read from the environment (GitHub Actions secrets /
Wrangler secrets / Docker -e flags). See .env.example for the full list.
Nothing in this module ever contains a secret value; it only reads names.
"""
from __future__ import annotations

import os


def _required(name: str, default: str | None = None) -> str:
    val = os.environ.get(name)
    if val is None or not val.strip():
        if default is not None:
            return default
        raise RuntimeError(f"missing required env var: {name}")
    return val.strip()


# ── Cloudflare backend / queue -------------------------------------------
# Base of the API Worker (e.g. https://api.channelryapp.sbs). The worker
# heartbeats, claims, reports progress and reports completion through this
# same host. This is an HTTPS https:// URL — outbound only.
CLOUD_API_URL = _required("CHANNELRY_CLOUD_API_URL")

# Server-to-server token the backend validates with checkWorker(). This is
# the credential that lets the renderer claim jobs / report — it does NOT
# grant admin access. Equivalent to the plan's "RENDERER_TOKEN"; the
# existing backend calls it WORKER_TOKEN. Keep it out of git (GitHub
# Actions secret).
WORKER_TOKEN = "".join(_required("CHANNELRY_RENDER_WORKER_TOKEN").split())

# Identify THIS ephemeral runner. A fresh value per job is fine and expected;
# the backend keys its render_workers heartbeat rows off this id.
WORKER_ID = _required("CHANNELRY_RENDER_WORKER_ID", "renderer-a")
WORKER_NAME = _required("CHANNELRY_RENDER_WORKER_NAME", WORKER_ID)

# ── R2 (asset download + final-video upload) -----------------------------
# S3-compatible R2 bucket "channelry-videos".
R2_ENDPOINT = _required("CHANNELRY_R2_ENDPOINT")
R2_BUCKET = _required("CHANNELRY_R2_BUCKET")
R2_ACCESS_KEY_ID = _required("CHANNELRY_R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = _required("CHANNELRY_R2_SECRET_ACCESS_KEY")
R2_PUBLIC_BASE_URL = _required("CHANNELRY_R2_PUBLIC_BASE_URL").rstrip("/")

# ── Loop / behavior -------------------------------------------------------
POLL_SECONDS = float(os.environ.get("CHANNELRY_RENDER_POLL_SECONDS", "3"))
# Max length of a video in seconds (safety, not a hard product limit).
MAX_VIDEO_SECONDS = float(os.environ.get("CHANNELRY_RENDER_MAX_SECONDS", "600"))

# Default output canvas (16:9). Scene backgrounds are cover-cropped to this.
OUTPUT_WIDTH = int(os.environ.get("CHANNELRY_OUTPUT_WIDTH", "1280"))
OUTPUT_HEIGHT = int(os.environ.get("CHANNELRY_OUTPUT_HEIGHT", "720"))

# Per-job aspect ratio (2026-08-29, owner request: an aspect-ratio picker
# ahead of script entry in the editor). Worker-full.ts validates and stores
# `aspect_ratio` on the job's settings as one of these three keys; anything
# else (missing, old jobs queued before this, an unknown value) falls back
# to the existing 16:9 default above — same behavior as before this feature.
ASPECT_RATIO_DIMS: dict[str, tuple[int, int]] = {
    "16:9": (OUTPUT_WIDTH, OUTPUT_HEIGHT),
    "9:16": (OUTPUT_HEIGHT, OUTPUT_WIDTH),
    "1:1": (1024, 1024),
}


def resolve_output_dims(settings: dict) -> tuple[int, int]:
    """Job's requested (width, height), falling back to the 16:9 default."""
    ratio = str((settings or {}).get("aspect_ratio") or "").strip()
    return ASPECT_RATIO_DIMS.get(ratio, (OUTPUT_WIDTH, OUTPUT_HEIGHT))

# When rembg is not installed (e.g. a quick local smoke test) the pipeline
# falls back to using the raw character image. Set to "1" to force that path.
FORCE_NO_REMBG = os.environ.get("CHANNELRY_FORCE_NO_REMBG", "0") == "1"

# edge-tts voice used for stylized/unknown voice ids (character:*, etc.).
DEFAULT_VOICE = "en-US-ChristopherNeural"