"""Run the render worker loop.

Claims one job at a time via /api/render-workers/claim (worker must have
heartbeat first), renders it, reports progress/completion, then loops.
Compatible with the ephemeral GitHub-hosted runner model: a fresh
WORKER_ID per job is fine; the worker always talks outbound only.
"""
from __future__ import annotations

import logging
import time
import traceback

from . import cloud, config, render
from .background_removal import backend_name

log = logging.getLogger("render.worker")

# Methods this worker can render. Reported in the heartbeat meta so the queue
# can later select a capable renderer (plan §25) without the frontend caring.
SUPPORTED_METHODS = ("character_video", "still", "walk", "character_cutout")


def capabilities() -> dict:
    return {
        "runner": "github-actions",
        "renderer_id": "renderer-a",
        "rembg": backend_name(),
        "methods": list(SUPPORTED_METHODS),
        "renderer_type": "cpu-ffmpeg",
    }


def handle_job(job: dict) -> None:
    job_id = str(job.get("id") or "")
    mode = str(job.get("mode") or "")
    settings = render.parse_settings(job)
    kind = settings.get("kind") or ""
    method = settings.get("method") or ""

    log.info("claimed job %s mode=%s kind=%s method=%s", job_id, mode, kind, method)
    cloud.report(job_id, "rendering", 2, progress_step="asset download")

    try:
        if method == "character_cutout":
            # pure image cutout -> transparent PNG -> R2 (no video/voice/ffmpeg)
            url, snapshot = render.render_character_cutout(job_id, job)
            cloud.report(
                job_id, "done", 100, "uploading cutout",
                video_url=url, scene_snapshot=snapshot,
            )
            log.info("job %s CUTOUT COMPLETED -> %s", job_id, url)
            return
        if method == "character_video":
            url, snapshot = render.render_character_video(job_id, job)
        elif method in ("still", "walk"):
            url, snapshot = render.render_single_character(job_id, job, method)
        else:
            cloud.report(
                job_id, "failed", 100,
                error=f"unsupported render method '{method}' (supported: {', '.join(SUPPORTED_METHODS)})",
            )
            return
        cloud.report(
            job_id, "done", 100, "uploading final video",
            video_url=url, scene_snapshot=snapshot,
        )
        log.info("job %s COMPLETED -> %s", job_id, url)
    except render.RenderJobError as exc:
        log.error("job %s failed (job-level): %s", job_id, exc)
        cloud.report(job_id, "failed", 100, error=str(exc))
    except Exception as exc:  # noqa: BLE001
        log.error("job %s crashed:\n%s", job_id, traceback.format_exc())
        cloud.report(job_id, "failed", 100, error=f"worker error: {exc}")


def run_once() -> bool:
    """Heartbeat then attempt one claim. Returns True if a job was handled."""
    cloud.heartbeat(capabilities())
    job = cloud.claim()
    if not job:
        return False
    handle_job(job)
    return True


def run_forever(poll_seconds: float | None = None) -> None:
    poll = poll_seconds if poll_seconds is not None else config.POLL_SECONDS
    log.info(
        "render worker started (id=%s, rembg=%s, poll=%.1fs)",
        config.WORKER_ID, backend_name(), poll,
    )
    while True:
        try:
            run_once()
        except Exception:  # noqa: BLE001
            log.error("worker iteration failed:\n%s", traceback.format_exc())
        time.sleep(poll)


def run_batch(max_jobs: int, max_seconds: float, max_empty: int = 6) -> int:
    """Render up to `max_jobs` / `max_seconds` / `max_empty`, whichever hits first.

    Ideal for an ephemeral GitHub-hosted runner: heartbeat + claim + render
    whatever is available, then end so the job finishes and the VM is torn
    down. `max_empty` stops after that many consecutive empty claims so an
    empty queue exits in ~15s instead of burning the whole `max_seconds` cap
    polling nothing. Returns how many jobs were handled.
    """
    handled = 0
    empty_streak = 0
    start = time.monotonic()
    log.info(
        "render batch started (id=%s, rembg=%s, max_jobs=%d, max_seconds=%.0f, max_empty=%d)",
        config.WORKER_ID, backend_name(), max_jobs, max_seconds, max_empty,
    )
    while handled < max_jobs and (time.monotonic() - start) < max_seconds:
        try:
            if run_once():
                handled += 1
                empty_streak = 0
            else:
                empty_streak += 1
                if max_empty and empty_streak >= max_empty:
                    log.info(
                        "queue empty for %d consecutive polls — stopping early (id=%s)",
                        empty_streak, config.WORKER_ID,
                    )
                    break
        except Exception:  # noqa: BLE001
            log.error("worker iteration failed:\n%s", traceback.format_exc())
        time.sleep(config.POLL_SECONDS)
    log.info("batch finished — handled %d job(s), id=%s", handled, config.WORKER_ID)
    return handled