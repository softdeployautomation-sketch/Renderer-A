"""RemBG-backed background removal with a safe passthrough fallback.

Production (the runtime Docker image) installs `rembg` + onnxruntime so this
cuts the character out of its studio background. In a minimal dev/smoke-test
environment where rembg isn't installed we fall back to returning the image
uncut (passthrough) rather than crash — the pipeline still renders, so you
can test composition/voice/ffmpeg without downloading ~170MB of models.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from PIL import Image

from .. import config

log = logging.getLogger("render.rembg")

_AVAILABLE = None


def backend_name() -> str:
    if _has_rembg():
        return "rembg"
    return "passthrough"


@lru_cache(maxsize=1)
def _has_rembg() -> bool:
    if config.FORCE_NO_REMBG:
        return False
    try:
        from rembg import remove  # noqa: F401
        from rembg.bg import _load_models  # noqa: F401
        return True
    except Exception:  # pragma: no cover - environment-dependent
        log.warning("rembg unavailable — using passthrough (no background removal)")
        return False


def remove_background(image: Image.Image) -> Image.Image:
    """Return a copy of `image` with the background removed (as RGBA).

    Falls back to the original when rembg is not available so a local
    smoke test can run without the model download.
    """
    if not _has_rembg():
        return image.convert("RGBA")

    import numpy as np
    from rembg import remove

    arr = np.asarray(image.convert("RGBA"))
    result = remove(arr, post_process_mask=True)
    return Image.fromarray(result).convert("RGBA")