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


def _wrap_text(text: str, font, max_w: int) -> list[str]:
    """Naive word-wrap using a PIL font's getlength (no external deps)."""
    words = text.split()
    if not words:
        return []
    lines: list[str] = []
    cur = ""
    for w_ in words:
        trial = f"{cur} {w_}".strip()
        if font.getbbox(trial)[2] <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w_
    if cur:
        lines.append(cur)
    return lines


def compose_title_card(
    background_path: str | None,
    title: str,
    subtitle: str = "",
    position: str = "center",
    w: int = 1280,
    h: int = 720,
) -> Image.Image:
    """Render a title-card frame: background + centered title/subtitle text.

    Mirrors the classic `sceneTitleCardCard` data model (title/subtitle/
    position: center | top | lower_third). No characters, no audio.
    """
    frame = _letterbox_bg(background_path, w, h)
    draw = ImageDraw.Draw(frame)

    # Scrim so text stays readable over any background.
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.rectangle([0, 0, w, h], fill=(8, 8, 14, 120))
    frame.alpha_composite(overlay)

    title_font = _load_font(max(20, w // 24))
    sub_font = _load_font(max(14, w // 42))

    def _text_h(font, text: str) -> int:
        try:
            return font.getbbox(text)[3] - font.getbbox(text)[1]
        except Exception:
            return int(font.size)

    def _draw_lines(lines, font, start_y, fill, gap) -> int:
        cur = start_y
        for ln in lines:
            bbox = font.getbbox(ln)
            tw = bbox[2] - bbox[0]
            draw.text(((w - tw) // 2, cur), ln, font=font, fill=fill)
            cur += _text_h(font, ln) + gap
        return cur

    title_lines = _wrap_text(title or "", title_font, int(w * 0.82))
    sub_lines = _wrap_text(subtitle or "", sub_font, int(w * 0.8))

    # Vertical placement depending on position.
    if position == "top":
        yd = int(h * 0.14)
    elif position == "lower_third":
        yd = int(h * 0.66)
    else:  # center
        # roughly center the title block around the middle
        title_h = sum(_text_h(title_font, ln) + 6 for ln in title_lines) if title_lines else 0
        yd = max(0, int((h - title_h) / 2) - int(h * 0.05))

    cur_y = _draw_lines(title_lines, title_font, yd, (255, 255, 255, 255), 6)
    if sub_lines:
        cur_y += _text_h(sub_font, "Myg")
        _draw_lines(sub_lines, sub_font, cur_y, (226, 220, 245, 255), 4)
    return frame.convert("RGB")


def _load_font(size: int):
    """Best-effort PIL truetype font fallback."""
    from PIL import ImageFont
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "Arial.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, max(14, size))
        except Exception:
            continue
    return ImageFont.load_default()


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