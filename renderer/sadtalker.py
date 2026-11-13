"""SadTalker talking-head render mode for Renderer-A.

CPU-only on GitHub free runners (~2 vCPU, 7 GB RAM, no GPU). Wraps
OpenTalker/SadTalker ``inference.py`` as a subprocess:

    still photo + driven audio  ->  talking-head mp4 (lip movement + head motion)

SadTalker is Apache 2.0 (fine for personal use). It fits in 7 GB RAM on CPU
but is slow (~2-5 min per clip), so we batch small jobs and use ``--still`` +
``--batch_size 1`` + ``--size 256`` to stay within the runner limits.

The SadTalker repo + checkpoints are provisioned once on the runner
(GitHub Actions installs torch CPU, clones the repo, downloads checkpoints).
This module only orchestrates the subprocess call; it never imports torch
itself, so the renderer imports cleanly even when SadTalker isn't provisioned.

Env vars
--------
CHANNELRY_SADTALKER_DIR
    Dir containing the cloned SadTalker repo. Defaults to
    <app>/vendor/SadTalker.
CHANNELRY_SADTALKER_CHECKPOINT_DIR
    Dir holding the downloaded checkpoints. Defaults to
    <sadtalker_dir>/checkpoints.
CHANNELRY_SADTALKER_SIZE
    256 (default) or 512. 256 keeps CPU/RAM usage low.
CHANNELRY_SADTALKER_PREPROCESS
    ``crop`` (default).
USE_SADTALKER
    Set to ``1`` to expose the sadtalker capability. Individual jobs opt in
    via ``settings.method == "sadtalker"`` (per-job opt-in; see worker.py).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

from . import config

log = logging.getLogger("render.sadtalker")

# OpenTalker/SadTalker release assets (v0.0.2-rc, safetensor / new code path).
_RELEASE_ROOT = (
    "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc"
)

# Checkpoints required for ``preprocess='crop'`` + safetensor path
# (see src/utils/init_path.py). Only mapping_00229 is needed for 'crop';
# mapping_00109 is for 'full' preprocess which we deliberately do not use.
_CHECKPOINTS: dict[str, str] = {
    "SadTalker_V0.0.2_256.safetensors": f"{_RELEASE_ROOT}/SadTalker_V0.0.2_256.safetensors",
    "mapping_00229-model.pth.tar": f"{_RELEASE_ROOT}/mapping_00229-model.pth.tar",
}

# Default output image size used by SadTalker (kept low for 7 GB CPU runners).
DEFAULT_SIZE = 256
DEFAULT_PREPROCESS = "crop"



def _download(url: str, dest: Path) -> None:
    """Download *url* to *dest* (no-op if already present)."""
    if dest.exists() and dest.stat().st_size > 0:
        log.debug("checkpoint already present: %s", dest)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    import urllib.request

    log.info("downloading SadTalker checkpoint %s -> %s", url, dest)
    with urllib.request.urlopen(url, timeout=300) as resp, open(dest, "wb") as out:
        shutil.copyfileobj(resp, out)


def ensure_sadtalker() -> tuple[Path, Path]:
    """Provision the SadTalker repo + checkpoints if missing.

    Returns ``(sadtalker_dir, checkpoint_dir)``. Raises SadTalkerError if the
    repo can't be cloned or a required checkpoint can't be downloaded.
    """
    sdir = sadtalker_dir()
    cdir = checkpoint_dir()
    inference = sdir / "inference.py"

    # 1) Clone the SadTalker repo (shallow) if not already present.
    if not inference.exists():
        if sdir.exists():
            log.warning("SadTalker dir exists but inference.py missing: %s", sdir)
        else:
            log.info("cloning SadTalker repo -> %s", sdir)
            sdir.parent.mkdir(parents=True, exist_ok=True)
            r = subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "https://github.com/OpenTalker/SadTalker.git",
                    str(sdir),
                ],
                capture_output=True,
                timeout=600,
            )
            if r.returncode != 0 or not inference.exists():
                raise SadTalkerError(
                    f"failed to clone SadTalker repo: {r.stderr.decode('utf-8','replace')[-800:]}"
                )

    # 2) Download required checkpoints into the checkpoint dir.
    for name, url in _CHECKPOINTS.items():
        _download(url, cdir / name)

    # 3) Sanity-check the entry point.
    if not inference.exists():
        raise SadTalkerError(f"SadTalker inference.py not found at {inference}")

    log.info("SadTalker ready (repo=%s, checkpoints=%s)", sdir, cdir)
    return sdir, cdir


class SadTalkerError(RuntimeError):
    """Raised when SadTalker cannot run or fails inference."""


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


def _find_output_mp4(result_dir: Path, before: float) -> Optional[Path]:
    """Find the newest mp4 in *result_dir* mtime >= *before* (SadTalker output)."""
    best: Optional[Path] = None
    best_mtime = before
    if result_dir.is_dir():
        for p in result_dir.rglob("*.mp4"):
            try:
                m = p.stat().st_mtime
            except OSError:
                continue
            if m >= best_mtime and (best is None or m > best_mtime):
                best, best_mtime = p, m
    return best


def render(photo_path: str, audio_path: str, work_dir: Path) -> str:
    """Run SadTalker inference on *photo_path* + *audio_path*.

    Returns the path to the generated talking-head mp4.
    """
    if not use_sadtalker():
        raise SadTalkerError("USE_SADTALKER is not set to '1'")

    sdir, cdir = ensure_sadtalker()

    inference = sdir / "inference.py"
    python = shutil.which("python3") or shutil.which("python") or "python3"

    result_dir = work_dir / "sadtalker-result"
    result_dir.mkdir(parents=True, exist_ok=True)
    before = time.time()

    cmd = [
        python,
        str(inference),
        "--source_image",
        str(photo_path),
        "--driven_audio",
        str(audio_path),
        "--result_dir",
        str(result_dir),
        "--checkpoint_dir",
        str(cdir),
        "--size",
        str(sadtalker_size()),
        "--preprocess",
        sadtalker_preprocess(),
        "--still",
        "--batch_size",
        "1",
        "--cpu",
        "--verbose",
    ]

    log.info(
        "running SadTalker inference (size=%s, preprocess=%s, cpu)",
        sadtalker_size(),
        sadtalker_preprocess(),
    )
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

    # Log a trimmed tail of SadTalker's stdout for debugging (no secrets).
    if r.stdout:
        log.debug("sadtalker stdout tail: %s", r.stdout[-1500:])
    if r.returncode != 0:
        raise SadTalkerError(
            f"SadTalker inference failed (rc={r.returncode}): "
            f"{r.stderr[-2000:] if r.stderr else ''}"
        )

    out = _find_output_mp4(result_dir, before)
    if out is None or not out.exists():
        raise SadTalkerError("SadTalker completed but produced no output mp4")

    log.info("SadTalker output: %s (%d bytes)", out, out.stat().st_size)
    return str(out)



def sadtalker_dir() -> Path:
    """Location of the cloned SadTalker repo (contains inference.py, src/)."""
    return Path(_env("CHANNELRY_SADTALKER_DIR", "vendor/SadTalker")).expanduser()


def checkpoint_dir() -> Path:
    """Location of downloaded SadTalker checkpoints."""
    return Path(
        _env("CHANNELRY_SADTALKER_CHECKPOINT_DIR", str(sadtalker_dir() / "checkpoints"))
    ).expanduser()


def sadtalker_size() -> int:
    try:
        return int(_env("CHANNELRY_SADTALKER_SIZE", str(DEFAULT_SIZE)))
    except ValueError:
        return DEFAULT_SIZE


def sadtalker_preprocess() -> str:
    return _env("CHANNELRY_SADTALKER_PREPROCESS", DEFAULT_PREPROCESS)


def use_sadtalker() -> bool:
    return os.environ.get("USE_SADTALKER", "0").strip() == "1"