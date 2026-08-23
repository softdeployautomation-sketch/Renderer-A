"""Video composition: build a scene frame from a background + characters.

Pure Pillow (no DOM/browser). A character's full generated image is
background-removed then placed on the scene at normalized x/y/scale —
the exact fields the backend accepts on `beats[].characters[]`
({image_url, x:0..1, y:0..1, scale:0.1..1}). x is a left-anchor fraction,
y a bottom-anchor fraction, scale relative to the character's own size.
"""
from __future__ import annotations

import logging
from typing import Iterable

from PIL import Image, ImageDraw

log = logging.getLogger("render.compose")


def _load(path: str) -> Image.Image:
    return Image.open(path).convert("RGBA")


def _letterbox_bg(path: str | None, w: int, h: int, color=(8, 8, 14, 255)) -> Image.Image:
    bg = Image.new("RGBA", (w, h), color)
    if not path:
        return bg
    img = _load(path)
    img.thumbnail((w, h), Image.LANCZOS)
    x = (w - img.width) // 2
    y = (h - img.height) // 2
    bg.alpha_composite(img, (x, y))
    return bg


def _place_character(
    bg: Image.Image,
    char_img: Image.Image,
    x: float,
    y: float,
    scale: float,
) -> None:
    """Composite `char_img` (RGBA, transparent bg) onto `bg`.

    x is a left-anchor (0..1) of the character's width position; y is a
    bottom-anchor (0..1) of where the character's feet sit relative to the
    background height.
    """
    w, h = bg.size
    # Character height is a fraction of the canvas driven by `scale`.
    target_h = max(20, int(scale * h))
    ratio = target_h / char_img.height
    cw = max(1, int(char_img.width * ratio))
    ch = max(1, target_h)
    resized = char_img.resize((cw, ch), Image.LANCZOS)

    px = int((w - cw) * max(0.0, min(1.0, x)))
    # bottom anchor: feet sit at y fraction of the canvas height.
    feet_y = int(h * (1.0 - max(0.0, min(1.0, y))))
    py = max(0, feet_y - ch)

    bg.alpha_composite(resized, (px, py))


def compose_scene(
    background_path: str | None,
    characters: Iterable[dict],
    width: int,
    height: int,
) -> Image.Image:
    """Render one full scene frame as RGB (flattened, ready for ffmpeg)."""
    bg = _letterbox_bg(background_path, width, height)
    for c in characters or []:
        img_path = (c or {}).get("path")
        if not img_path:
            continue
        try:
            char_img = _load(img_path)
        except Exception as exc:  # noqa: BLE001
            log.warning("failed to load character %s: %s", img_path, exc)
            continue
        x = max(0.0, min(1.0, float((c or {}).get("x", 0.5))))
        y = max(0.0, min(1.0, float((c or {}).get("y", 0.92))))
        scale = max(0.1, min(1.0, float((c or {}).get("scale", 0.5))))
        _place_character(bg, char_img, x, y, scale)
    return bg.convert("RGB")