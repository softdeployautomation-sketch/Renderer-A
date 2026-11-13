"""Cloudflare API client + R2 upload for the render worker.

Reuses the exact endpoints the existing worker already uses
(verified in worker-full.ts):
  POST /api/render-workers/heartbeat
  POST /api/render-workers/claim
  GET  /api/render-workers/jobs/{id}/status
  PATCH /api/render-workers/jobs/{id}      (progress / done / failed)
"""
from __future__ import annotations

import json
import logging
import shutil
import tempfile
import urllib.request
from pathlib import Path

import boto3

from . import config

log = logging.getLogger("render.cloud")


def _request(url: str, method: str = "GET", body: dict | None = None, timeout: int = 60) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "User-Agent": "renderer-a/1.0",
            "Accept": "application/json",
            "Content-Type": "application/json" if body is not None else "application/json",
            "Authorization": f"Bearer {config.WORKER_TOKEN}",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        if not raw:
            return {}
        return json.loads(raw)


def _api(path: str, method: str = "GET", body: dict | None = None) -> dict:
    return _request(f"{config.CLOUD_API_URL}{path}", method, body)


# ── Queue protocol ──────────────────────────────────────────────────────

def heartbeat(meta: dict | None = None) -> None:
    """Register/refresh this worker so claim() will accept it."""
    _api(
        "/api/render-workers/heartbeat",
        "POST",
        {"worker_id": config.WORKER_ID, "name": config.WORKER_NAME, "meta": meta or {}},
    )


def claim() -> dict | None:
    """Atomically claim one queued job, or None when the queue is empty.

    The backend's claim uses `UPDATE ... WHERE status='queued'` guarded by
    changes() so exactly one renderer ever wins a job.
    """
    res = _api("/api/render-workers/claim", "POST", {"worker_id": config.WORKER_ID})
    return res.get("job") if isinstance(res, dict) else None


def report(
    job_id: str,
    status: str,
    progress: int,
    progress_step: str = "",
    error: str = "",
    video_url: str = "",
    scene_snapshot: list | None = None,
) -> None:
    body = {
        "status": status,
        "progress": max(0, min(100, int(progress))),
        "progress_step": progress_step[:80],
        "error": error[:500],
        "video_url": video_url,
    }
    if scene_snapshot:
        body["scene_snapshot"] = scene_snapshot
    _api(f"/api/render-workers/jobs/{job_id}", "PATCH", body)


def check_status(job_id: str) -> str:
    """Cancel detection: returns the job status as the backend sees it."""
    res = _api(f"/api/render-workers/jobs/{job_id}/status", "GET")
    return str(res.get("status", ""))


# ── R2 upload ───────────────────────────────────────────────────────────

def _r2():
    return boto3.client(
        "s3",
        endpoint_url=config.R2_ENDPOINT,
        aws_access_key_id=config.R2_ACCESS_KEY_ID,
        aws_secret_access_key=config.R2_SECRET_ACCESS_KEY,
        region_name="auto",
    )


def upload_file(local_path: str, r2_key: str, content_type: str = "video/mp4") -> str:
    """Upload a local file to R2 and return its public URL."""
    path = Path(local_path)
    if not path.is_file():
        raise FileNotFoundError(f"upload source missing: {path}")
    _r2().upload_file(str(path), config.R2_BUCKET, r2_key, ExtraArgs={"ContentType": content_type})
    return f"{config.R2_PUBLIC_BASE_URL}/{r2_key}"


def upload_video(job_id: str, local_path: str) -> str:
    """final.mp4 -> R2 videos/{job_id}/final.mp4 -> public URL."""
    return upload_file(local_path, f"videos/{job_id}/final.mp4", "video/mp4")


def download(url: str, dest_dir: str, filename: str) -> str:
    """Download a public asset (character image / scene background) to disk."""
    dest = Path(dest_dir) / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not url:
        raise ValueError("empty download url")
    req = urllib.request.Request(url, headers={"User-Agent": "renderer-a/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as out:
        shutil.copyfileobj(resp, out)
    return str(dest)