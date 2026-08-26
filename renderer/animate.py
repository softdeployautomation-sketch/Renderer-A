"""Character animation for Renderer-A (add 2026-08-26).

The Renderer-A ``render_character_video`` path used to composite ONE static
frame per scene and loop it with ffmpeg ``-loop 1`` -- so every ``character_video``
"action" job rendered with a frozen face (no mouth movement, no body motion).
The older sidecar scene engine had real animation (Rhubarb lip-sync + whole-body
motion styles); this module ports that *technique* into Renderer-A without the
heavy sidecar-only deps (MediaPipe / Rhubarb / isolated venvs).

What it adds, all CPU-only with the deps Renderer-A already ships (Pillow +
numpy + ffmpeg):

* **Audio-energy lip-sync** -- the speaker's dialogue MP3 is decoded to a mono
  PCM envelope; per-frame RMS is normalised to a mouth "openness" 0..1 curve.
  No external phoneme/viseme tool needed, and it degrades to silence=closed.
* **Jaw-drop mouth shape** -- port of the sidecar's ``_apply_mouth_shape`` (the
  lower half of the mouth region translates down, revealing a soft dark gap),
  applied to the speaker's cutout each frame.
* **Whole-body scene motion** -- the sidecar's ``SCENE_MOTION_STYLES`` table
  (idle / sway / bounce / nod / vibrate / still / walk_in_place) translated +
  rotated per frame, so characters stay "alive" even when a style isn't set.

Every step is defensive: if the mouth region heuristic misses (non-human /
profile / costume), if audio can't be decoded, or if `motion_style` is unknown,
the code gracefully animates what it can and never hard-fails the render.
"""

from __future__ import annotations

import logging
import math
import os
import subprocess

from PIL import Image, ImageDraw

# Reuse the same letterbox helper as the static composer so an animated scene
# matches (frame-for-frame) what the still path produced.
from .video import _letterbox_bg

log = logging.getLogger("render.animate")

# Same data-driven style table as the sidecar scene compositor. `intensity`
# (per-character, 0..1, default 0.5) scales whichever quantity a style uses;
# values below are each style's intensity=1.0 amplitude.
SCENE_MOTION_STYLES = {
    "idle":          {"bob_freq": 0.35, "bob_amp": 0.024, "rot_amp": 0.0,  "rot_freq": 0.0, "jitter_x": 0.0,   "jitter_y": 0.0},
    "vibrate":       {"bob_freq": 0.0,  "bob_amp": 0.0,   "rot_amp": 0.0,  "rot_freq": 0.0, "jitter_x": 0.024, "jitter_y": 0.020},
    "sway":          {"bob_freq": 0.0,  "bob_amp": 0.0,   "rot_amp": 6.0,  "rot_freq": 0.6, "jitter_x": 0.0,   "jitter_y": 0.0},
    "bounce":        {"bob_freq": 0.9,  "bob_amp": 0.045, "rot_amp": 0.0,  "rot_freq": 0.0, "jitter_x": 0.0,   "jitter_y": 0.0},
    "nod":           {"bob_freq": 0.0,  "bob_amp": 0.0,   "rot_amp": 10.0, "rot_freq": 0.9, "jitter_x": 0.0,   "jitter_y": 0.0},
    "still":         {"bob_freq": 0.0,  "bob_amp": 0.0,   "rot_amp": 0.0,  "rot_freq": 0.0, "jitter_x": 0.0,   "jitter_y": 0.0},
    "walk_in_place": {"bob_freq": 1.4,  "bob_amp": 0.03,  "rot_amp": 4.0,  "rot_freq": 1.4, "jitter_x": 0.0,   "jitter_y": 0.0},
}

# Map a beat's free-text `action` (the "[action: ...]" script tag; defaults to
# 'speaking') onto a real whole-body motion style so the user's action tags
# change what the character does, not just whether the mouth moves.
_ACTION_TO_STYLE = {
    "speaking": "sway",
    "talking": "sway",
    "speech": "sway",
    "gesturing": "sway",
    "gesture": "sway",
    "point": "sway",
    "pointing": "sway",
    "wave": "sway",
    "waving": "sway",
    "hand on head": "sway",
    "hand_on_head": "sway",
    "nod": "nod",
    "nodding": "nod",
    "bounce": "bounce",
    "bouncing": "bounce",
    "vibrate": "vibrate",
    "vibrating": "vibrate",
    "walk": "walk_in_place",
    "walking": "walk_in_place",
    "walk in place": "walk_in_place",
    "idle": "idle",
    "still": "still",
}


def resolve_motion_style(char: dict, is_speaker: bool) -> str:
    """Pick a whole-body motion style for a character.

    Priority: an explicit `motion_style` (if it's one we know) -> a mapped
    `action` word -> a speaker/bystander default. Unknown names fall back to
    "speaker sway" / "idle" (never hard-fail, never freeze).
    """
    ms = str((char or {}).get("motion_style") or "").strip().lower()
    if ms in SCENE_MOTION_STYLES:
        return ms
    act = str((char or {}).get("action") or "").strip().lower()
    for key, style in _ACTION_TO_STYLE.items():
        if key in act:
            return style
    return "sway" if is_speaker else "idle"


def motion_offset(style: str, intensity: float, t: float, duration: float,
                  phase: float, target_w: int, target_h: int) -> tuple:
    """Return (tx, ty, rotation_deg) for one character at normalized time t.

    Ported from the sidecar ``_scene_character_motion_offset``; `t` is 0..1
    progress through the clip, `phase` staggers multiple characters so they
    don't all bob in lockstep.
    """
    cfg = SCENE_MOTION_STYLES.get(style, SCENE_MOTION_STYLES["idle"])
    tt = t * duration + phase
    tx = ty = angle = 0.0
    if cfg["bob_amp"]:
        ty = -abs(math.sin(2 * math.pi * cfg["bob_freq"] * tt)) * (target_h * cfg["bob_amp"] * intensity)
    if cfg["rot_amp"]:
        angle = cfg["rot_amp"] * intensity * math.sin(2 * math.pi * cfg["rot_freq"] * tt)
    if cfg["jitter_x"] or cfg["jitter_y"]:
        tx += (math.sin(tt * 41.0) + 0.6 * math.sin(tt * 67.0)) * (target_w * cfg["jitter_x"] * intensity)
        ty += (math.sin(tt * 53.0 + 1.1) + 0.6 * math.sin(tt * 29.0)) * (target_h * cfg["jitter_y"] * intensity)
    return tx, ty, angle

def estimate_mouth_region(char_img: Image.Image):
    """Heuristic mouth rect from the cutout's alpha bbox (no face detector).

    Returns (mx0, my0, mx1, my1) in the cutout's own pixel space, or None.
    Assumes a roughly-centered, camera-facing figure: the head occupies the
    top ~1/3 of the visible cutout and the mouth sits in the lower half of
    that head band, horizontally centred. This is intentionally coarse -- if it
    misses (profile / helmet / non-human) `apply_jaw_drop` is a harmless no-op
    and the body motion + audio still render.
    """
    bbox = char_img.getchannel("A").getbbox()
    if not bbox:
        return None
    left, top, right, bottom = bbox
    bw, bh = right - left, bottom - top
    if bh <= 12:
        return None
    # Head = top ~32% of the visible figure.
    head_bottom = top + bh * 0.32
    mx0 = left + bw * 0.30
    mx1 = left + bw * 0.70
    my0 = top + bh * 0.16          # below the eyes
    my1 = head_bottom * 0.86       # just under the mouth -> upper lip zone
    if mx1 <= mx0 or my1 <= my0:
        return None
    return (round(mx0), round(my0), round(mx1), round(my1))


def apply_jaw_drop(char_img: Image.Image, mouth_rect, openness: float) -> None:
    """Open the mouth by translating the lower half of the mouth region down.

    Ported from the sidecar ``_apply_mouth_shape`` (v3): the lower half of the
    crop TRANSLATES down proportional to ``openness`` and a soft dark gap is
    revealed where it used to sit -- the same unambiguous "jaw drops, a dark gap
    appears" cue a hand-drawn mouth-open frame uses. Mutates ``char_img`` in
    place.
    """
    mx0, my0, mx1, my1 = mouth_rect
    if mx1 <= mx0 or my1 <= my0:
        return
    openness = max(0.0, min(1.0, float(openness)))
    mw, mh = mx1 - mx0, my1 - my0
    mid_y = round((my0 + my1) / 2.0)
    max_drop = mh * 0.85
    drop = round(max_drop * openness)

    overlay = Image.new("RGBA", char_img.size, (0, 0, 0, 0))
    if drop >= 1:
        gap_h = (mid_y - my0) + drop + round(mh * 0.15)
        interior = Image.new("RGBA", (mw, max(1, gap_h)), (0, 0, 0, 0))
        idraw = ImageDraw.Draw(interior)
        idraw.ellipse(
            [round(mw * 0.10), 0, round(mw * 0.90), gap_h],
            fill=(25, 12, 12, round(190 * openness)),
        )
        overlay.paste(interior, (mx0, my0), interior)

    lower_crop = char_img.crop((mx0, mid_y, mx1, my1))
    overlay.paste(lower_crop, (mx0, mid_y + drop), lower_crop)
    char_img.alpha_composite(overlay)


def audio_openness_curve(audio_path: str, fps: int):
    """Per-frame mouth openness (0..1) driven by the dialogue audio's RMS.

    Decodes the MP3 to mono 16 kHz f32 PCM via ffmpeg (no audio lib needed),
    windows it into ~1-frame hops and normalises RMS against the 10th/99th
    percentiles. Silence (and any trailing tail) maps to ~0 -> mouth closed.
    Returns a numpy float array of length = number of frames, or None if the
    audio can't be decoded / is too short.
    """
    if not audio_path or not os.path.exists(audio_path):
        return None
    try:
        import numpy as np
        r = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", os.path.abspath(audio_path),
             "-f", "f32le", "-ac", "1", "-ar", "16000", "-"],
            capture_output=True, timeout=60,
        )
        if r.returncode != 0 or not r.stdout:
            return None
        data = np.frombuffer(r.stdout, dtype=np.float32)
        sr = 16000
        hop = max(1, sr // max(1, int(fps)))
        n = len(data) // hop
        if n < 3:
            return None
        env = data[: n * hop].reshape(n, hop)
        rms = np.sqrt(np.mean(env.astype(np.float64) ** 2, axis=1) + 1e-9)
        p_low = float(np.percentile(rms, 10))
        p_hi = float(np.percentile(rms, 99))
        span = p_hi - p_low
        if span <= 1e-7:
            return np.zeros(n, dtype=np.float64)
        openness = np.clip((rms - p_low) / span, 0.0, 1.0)
        # Light moving-average smooth so the mouth doesn't twitch inter-frame.
        k = max(1, int(round(0.05 * fps)))
        kern = np.ones(k, dtype=np.float64) / k
        return np.convolve(openness, kern, mode="same")
    except Exception:                                   # noqa: BLE001
        log.warning("lip-sync audio envelope failed; proceeding with body motion only")
        return None


def render_beat_frames(bg_path, chars, width: int, height: int, fps: int,
                       duration: float, openness, frame_dir: str, prefix: str):
    """Render every frame of one scene into `frame_dir` as PNGs.

    `chars` is a list of dicts with: path, x, y, scale, name, motion_style,
    action, intensity, is_speaker. `openness` is the audio-openness array (or
    None). Returns the list of written frame paths.
    """
    n_frames = max(1, int(round(duration * fps)))

    # Precompute each character's resized base cutout + mouth rect + motion cfg.
    preps = []
    for i, c in enumerate(chars or []):
        try:
            img = Image.open(c["path"]).convert("RGBA")
        except Exception as exc:                        # noqa: BLE001
            log.warning("animate: failed to load char %s: %s", c.get("name"), exc)
            continue
        target_h = max(20, int(float(c.get("scale", 0.5)) * height))
        ratio = target_h / img.height
        img = img.resize((max(1, int(img.width * ratio)), max(1, target_h)), Image.LANCZOS)
        is_speaker = bool(c.get("is_speaker"))
        mouth = estimate_mouth_region(img)
        style = resolve_motion_style(c, is_speaker)
        intensity = float(c.get("intensity", 0.5))
        phase = i * 0.7
        preps.append({
            "img": img, "mouth": mouth, "is_speaker": is_speaker,
            "style": style, "intensity": intensity, "phase": phase,
            "x": float(c.get("x", 0.5)), "y": float(c.get("y", 0.92)),
        })

    paths = []
    for f in range(n_frames):
        t = (f / (n_frames - 1)) if n_frames > 1 else 0.0
        bg = _letterbox_bg(bg_path, width, height)
        ov = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        for p in preps:
            work = p["img"].copy()
            if p["is_speaker"] and p["mouth"] and openness is not None:
                opn = float(openness[min(f, len(openness) - 1)])
                apply_jaw_drop(work, p["mouth"], opn)
            tx, ty, angle = motion_offset(
                p["style"], p["intensity"], t, duration, p["phase"], width, height,
            )
            if angle:
                work = work.rotate(float(angle), resample=Image.BICUBIC, expand=True)
            # Feet-anchor placement, same convention as _place_character /
            # the editor (y is a top-anchor for the feet line).
            px = int(width * p["x"] - work.width / 2.0 + tx)
            py = int(height * p["y"] - work.height + ty)
            px = max(-work.width, min(width, px))
            py = max(-work.height, min(height, py))
            ov.alpha_composite(work, (px, py))
        bg.alpha_composite(ov)
        fp = os.path.join(frame_dir, "%s%05d.png" % (prefix, f))
        bg.convert("RGB").save(fp)
        paths.append(fp)
    return paths

