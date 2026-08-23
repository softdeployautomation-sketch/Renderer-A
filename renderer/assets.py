"""Asset handling: download, workspace, dedupe."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from . import cloud


def make_workspace(job_id: str) -> Path:
    """Fresh per-job working dir (GitHub runners are ephemeral anyway)."""
    ws = Path(os.environ.get("CHANNELRY_WORKSPACE", "/tmp")) / f"render-{job_id}"
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "assets").mkdir(exist_ok=True)
    (ws / "clips").mkdir(exist_ok=True)
    return ws


def download_asset(url: str, ws: Path, prefix: str) -> str:
    """Download to workspace/assets/{prefix}-{sha1[:12]}.{ext} (deduped)."""
    ext = _ext_of(url)
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    target = ws / "assets" / f"{prefix}-{digest}.{ext}"
    if not target.exists():
        cloud.download(url, str(ws / "assets"), target.name)
    return str(target)


def _ext_of(url: str) -> str:
    lower = (url.split("?", 1)[0] or "").lower()
    # Remove signed/query strings and take the path suffix.
    path = lower.split("/")[-1]
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        if path.endswith(ext):
            return ext.strip(".")
    return "png"  # safe default; Pillow reads by content anyway