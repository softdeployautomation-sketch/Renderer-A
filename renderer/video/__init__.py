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
    """Cover-crop a background image onto the w×h frame (object-fit: cover).

    2026-08-26 (owner-regression fix): this used to LETTERBOX (contain) — scale
    to FIT then centre, leaving dark bars on the short dimension. For a square
    AI-generated background in a 16:9 scene that produced blank bars on BOTH
    left/right and shoved characters into dark gutters (\"it cut out the left
    and right leaving blank in both\"). The old sidecar engine used
    ``_fit_background_cover``; this now matches it: scale to COVER (so the
    smallest frame edge only just fills), then center-crop. No bars, no
    'reduced' background — the artwork fills the whole frame at all times.
    """
    bg = Image.new("RGBA", (w, h), color)
    if not path:
        return bg
    img = _load(path)
    iw, ih = img.size
    if iw <= 0 or ih <= 0:
        return bg
    scale = max(w / iw, h / ih)
    nw = max(1, round(iw * scale))
    nh = max(1, round(ih * scale))
    img = img.resize((nw, nh), Image.LANCZOS)
    left = (nw - w) // 2
    top = (nh - h) // 2
    img = img.crop((left, top, left + w, top + h))
    bg.alpha_composite(img.convert("RGBA"), (0, 0))
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


def visible_bbox(img: Image.Image) -> tuple[int, int, int, int]:
    """Return the opaque (alpha) bounding box of a character cutout.

    If the cutout has NO transparent pixels (fully opaque) it returns the whole
    image bounds. The painted character is what `scale` should really measure —
    a cutout can carry a lot of transparent padding (the person only fills part
    of the PNG), so sizing off the full image makes a character render smaller
    than its scale number. Every place we resize/place a character goes through
    this so `scale` is a true fraction of the ON-SCREEN body height, not of the
    padded cutout.
    """
    bb = img.getchannel("A").getbbox()
    if not bb:
        return (0, 0, img.width, img.height)
    return bb


def resize_cutout_to_visible_height(
    img: Image.Image, target_visible_px: int
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Resize a cutout so its VISIBLE (opaque) height equals `target_visible_px`.

    Returns (resized_img, visible_bbox_in_resized_space). Because the whole
    image is scaled by the same ratio, the character's proportional placement
    inside the frame is preserved — we just agree that `scale` names the visible
    person's height, not the transparent padding's.

    FIX (2026-08-31): vbox is now the ACTUAL alpha bbox measured on the
    resized image, not the pre-resize bbox coordinates projected by `ratio`.
    Confirmed live: LANCZOS resampling bleeds alpha beyond that theoretical
    projection on any real upscale (a hard-edged 10px-tall test rect upscaled
    2.8x measured 35px of real opaque height against a projected 28px) — a
    silent few-pixel drift on every character, worse the more a cutout gets
    scaled up. Measuring the real result instead of projecting from the
    source is exact regardless of resampling behavior.
    """
    l, t, r, b = visible_bbox(img)
    vis_h = max(1, b - t)
    ratio = target_visible_px / vis_h
    rw = max(1, int(img.width * ratio))
    rh = max(1, int(img.height * ratio))
    resized = img.resize((rw, rh), Image.LANCZOS)
    vbox = visible_bbox(resized)
    return resized, vbox


def _place_character(
    bg: Image.Image,
    char_img: Image.Image,
    x: float,
    y: float,
    scale: float,
) -> None:
    """Composite `char_img` (RGBA, transparent bg) onto `bg`.

    FIX (2026-08-25): x/y now use the SAME convention as the editors
    (classic dashboard.html + new video-tab.html placement editor).
    In the editor each character is drawn with CSS
    `left:{x*100}%; top:{y*100}%; transform:translate(-50%,-100%)`, so:
      - x is the horizontal CENTER of the character (0..1 fraction of width), and
      - y is the top-anchor of where the character's FEET sit (0..1 fraction of
        height DOWN from the top; y=0.92 => near the bottom of the frame).
    Previously this renderer treated y as a bottom-UP fraction (`h*(1-y)`) which
    mirrored the character vertically (y=0.92 rendered near the TOP) — the root
    cause of "dragged position not persisting in the video" since both editors
    send top-anchor values.

    Size (2026-08-26): `target_h` is now a fraction of the VISIBLE (opaque)
    body height, not the full cutout — a character already cut with transparent
    margins now renders at the same on-screen size as a tightly-cropped one for
    the same `scale` (and no longer "very small" at a high size number).
    """
    w, h = bg.size
    # FIX-12 (2026-08-31, owner-reported: "at 90 it's small... should start
    # at 50 with that size"). A straight scale*h mapping made even a 90%
    # slider read as modest on screen. Boosted 1.4x (capped at the full
    # frame) so a lower slider number still looks properly sized and 90%
    # reads as noticeably larger than before — matches the SAME boost the
    # editor's own placement preview applies (video-tab.html's
    # _visualScale), so what's previewed is what renders.
    target_h = max(20, int(min(1.0, scale * 1.4) * h))
    cx = max(0.0, min(1.0, float(x)))
    cy = max(0.0, min(1.0, float(y)))

    resized, vbox = resize_cutout_to_visible_height(char_img, target_h)
    # Centre on the VISIBLE character's horizontal centre; feet = visible bottom.
    v_center_x = (vbox[0] + vbox[2]) / 2.0
    v_bottom_y = vbox[3]

    px = int(w * cx - v_center_x)
    px = max(0, min(w - resized.width, px))

    feet_y = int(h * cy)
    py = max(0, min(h - resized.height, feet_y - v_bottom_y))

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