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
* **Mouth-opening lip-sync** -- a real open-**mouth** shape (lips part, a dark
  opening ellipse whose height/darkness scale with openness) is drawn centred on
  the mouth band each frame. This is applied to the SPEAKER only. (2026-08-26:
  replaced the jaw-translate approach that read as "mouth moving on the jaw".)
* **Whole-body scene motion** -- the sidecar's ``SCENE_MOTION_STYLES`` table
  (idle / sway / bounce / nod / vibrate / still / walk_in_place) translated +
  rotated per frame. Body motion is OPT-IN: plain speaking maps to ``idle`` so a
  talking character's body stays still (only lips move); bounce/nod/walk/etc.
  only happen via an explicit action tag (owner: "that should be an added
  action, it shouldn't just happen").

Every step is defensive: if the mouth region heuristic misses (non-human /
profile / costume), if audio can't be decoded, or if `motion_style` is unknown,
the code gracefully animates what it can and never hard-fails the render.
"""

from __future__ import annotations

import logging
import math
import os
import subprocess

import numpy as np

from PIL import Image, ImageDraw

# Reuse the same letterbox helper as the static composer so an animated scene
# matches (frame-for-frame) what the still path produced. `visible_bbox` /
# `resize_cutout_to_visible_height` keep the animated path's size semantics in
# lock-step with the static composer: `scale` names the VISIBLE body height so
# a padded cutout doesn't render small, and the feet anchor on the visible
# character's bottom (not the transparent margin).
from .video import _letterbox_bg, resize_cutout_to_visible_height

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
# change what the character does.
#
# 2026-08-26 (owner-reported regression): talking/unknown defaults to **idle**,
# NOT sway — the speaker's whole body should not bounce just because they talk
# ("the speaker keeps bouncing — that should be an added action, it shouldn't
# just happen"). Bouncing/nodding/walking/etc. only happen via an explicit
# action tag; plain speaking leaves the body still and only the mouth moves
# (the audio-driven lip-sync).
_ACTION_TO_STYLE = {
    # Speech family -> body still; mouth animates only. No auto-bounce/sway.
    "speaking": "idle",
    "speak": "idle",
    "talking": "idle",
    "speech": "idle",
    "talk": "idle",
    # Explicit gesture-y actions -> gentle whole-body motion (opt-in).
    "gesturing": "sway",
    "gesture": "sway",
    "point": "sway",
    "pointing": "sway",
    "wave": "sway",
    "waving": "sway",
    "hand on head": "sway",
    "hand_on_head": "sway",
    # Explicit body actions.
    "nod": "nod",
    "nodding": "nod",
    "bounce": "bounce",
    "bouncing": "bounce",
    "dance": "bounce",
    "dancing": "bounce",
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
    stillness (`idle`) for BOTH roles — body motion is opt-in via an action tag,
    and speaking alone must not make a character bounce/sway. (2026-08-26). The
    speaker's lips still move from the audio; only their whole body stays still.
    """
    ms = str((char or {}).get("motion_style") or "").strip().lower()
    if ms in SCENE_MOTION_STYLES:
        return ms
    act = str((char or {}).get("action") or "").strip().lower()
    for key, style in _ACTION_TO_STYLE.items():
        if key in act:
            return style
    return "idle"


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
    """Adaptive mouth rect from the cutout's alpha silhouette (no face detector).

    Owner-reported regression (2026-08-26): \"the mouth movement moved to the
    eyes.\" The old fixed-fraction heuristic assumed the head is the top ~1/3 of
    the figure and put the mouth at ~22% of height — for a Waist/Portrait cast
    image (head filling most of the frame) that lands on the forehead/EYES. This
    version estimates the HEAD from the alpha **width profile** (the skull is the
    top wide band; the neck is the first narrow valley below it), then places the
    mouth in the LOWER third of that head (well below the eye line) and sizes it
    to the head/face — adapting to full-body vs waist-up vs close-up framing.

    Returns (mx0, my0, mx1, my1) in the cutout's own pixel space, or None.
    Coarse on purpose: if it misses (profile / helmet / non-human) `apply_mouth_open`
    is a harmless no-op and the body motion + audio still render.
    """
    bbox = char_img.getchannel("A").getbbox()
    if not bbox:
        return None
    left, top, right, bottom = bbox
    bw, bh = right - left, bottom - top
    if bh <= 24 or bw <= 12:
        return None

    alpha = np.asarray(char_img.getchannel("A"), dtype=np.uint8)
    band = alpha[top:bottom, left:right] > 90
    row_w = band.sum(axis=1)  # opaque pixel count per row (the width profile)

    first = int(np.argmax(row_w > 0)) if np.any(row_w > 0) else 0
    # The skull (top of the head) is the widest band right at the top; the neck
    # is the first narrower band just below it. Key nominal per-row pixel counts,
    # but rather than hardcode a pixel value we derive both from the silhouette.
    skull_span = max(2, int(bh * 0.06))
    skull = row_w[first:first + skull_span]
    head_ref = float(skull.max()) if skull.size else 0.0
    head_h = bh * 0.22                      # fallback if no neck pinch found
    if head_ref > 0:
        neck_rows = np.flatnonzero(row_w[first + 2:] < head_ref * 0.68)
        if neck_rows.size:
            neck = first + 2 + int(neck_rows[0])
            if neck - first >= 8:
                head_h = neck - first
    head_h = max(8.0, min(float(head_h), bh * 0.80))

    # Mouth sits ~70% down the head; band height ~a fifth of the head; width ~ a
    # third of the face width — clearly BELOW the eyes either way.
    mouth_cy = top + head_h * 0.70
    mouth_half = max(2.0, head_h * 0.10)
    mouth_w = max(6.0, bw * 0.34)
    cx = left + bw * 0.5
    my0 = int(round(mouth_cy - mouth_half))
    my1 = int(round(mouth_cy + mouth_half))
    mx0 = int(round(cx - mouth_w / 2.0))
    mx1 = int(round(cx + mouth_w / 2.0))
    if mx1 <= mx0 or my1 <= my0:
        return None
    return (mx0, my0, mx1, my1)


def apply_mouth_open(char_img: Image.Image, mouth_rect, openness: float) -> None:
    """Animate the MOUTH (lips parting), not the jaw translating down.

    Owner-reported regression (2026-08-26): "mouth movement is happening on the
    jaw not the mouth." The previous approach physically translated the lower
    mouth strip downward (a jaw-drop cue), which read as the jaw moving. This
    version instead draws a real open-**mouth** shape (a dark, horizontally-wide
    ellipse centred on the mouth's midline, whose height and darkness scale with
    `openness`) — a closed mouth at 0, a clear open mouth at 1. It stays inside
    the head/mouth band, so only the mouth changes. Mutates ``char_img`` in
    place and is a harmless no-op if the mouth rect is degenerate.
    """
    mx0, my0, mx1, my1 = mouth_rect
    if mx1 <= mx0 or my1 <= my0:
        return
    openness = max(0.0, min(1.0, float(openness)))
    mw, mh = mx1 - mx0, my1 - my0
    mid_y = round((my0 + my1) / 2.0)
    # Opening height: a slim closed slit at 0, a clear open mouth at 1.
    open_h = max(1, round(mh * (0.06 + 0.94 * openness)))

    lip_h = max(1, round(mh * 0.30))   # soft lip frame around the opening
    canvas_w = mw + 4
    canvas_h = open_h + 2 * lip_h

    patch = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(patch)

    # Lips: a slightly larger rounded outline in a dark tone.
    lip_alpha = round(150 * openness)
    d.ellipse([1, 0, canvas_w - 2, canvas_h - 1], outline=(80, 40, 40, lip_alpha), width=2)

    # Interior open mouth (dark); height grows with openness.
    inner_alpha = round(120 + 135 * openness)
    d.ellipse(
        [2, lip_h, canvas_w - 3, lip_h + open_h - 1],
        fill=(15, 7, 7, inner_alpha),
    )

    # Centre the patch on the mouth band's vertical midline, horizontally on it.
    paste_y = mid_y - open_h // 2 - lip_h
    overlay = Image.new("RGBA", char_img.size, (0, 0, 0, 0))
    overlay.paste(patch, (mx0 - 2, paste_y), patch)
    char_img.alpha_composite(overlay)


def apply_jaw_drop(char_img: Image.Image, mouth_rect, openness: float) -> None:
    """Backward-compatible alias for :func:`apply_mouth_open`.

    Kept so any external callers / older tests that referenced the original name
    keep working; the rendered result is the new mouth-open (not a jaw
    translation) — see :func:`apply_mouth_open`.
    """
    apply_mouth_open(char_img, mouth_rect, openness)


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
        # Size the VISIBLE (opaque) character to `target_h`, not the padded
        # cutout, and keep the feet anchored to the visible bottom so the size
        # number matches the static composer exactly (see _place_character).
        img, vbox = resize_cutout_to_visible_height(img, target_h)
        # Horizontal centre + vertical bottom of the visible character, in the
        # resized image's pixel space (used to centre/feet-anchor below).
        v_center_x = (vbox[0] + vbox[2]) / 2.0
        v_bottom_y = vbox[3]
        is_speaker = bool(c.get("is_speaker"))
        mouth = estimate_mouth_region(img)
        style = resolve_motion_style(c, is_speaker)
        intensity = float(c.get("intensity", 0.5))
        phase = i * 0.7
        preps.append({
            "img": img, "mouth": mouth, "is_speaker": is_speaker,
            "style": style, "intensity": intensity, "phase": phase,
            "x": float(c.get("x", 0.5)), "y": float(c.get("y", 0.92)),
            "v_center_x": v_center_x, "v_bottom_y": v_bottom_y,
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
                apply_mouth_open(work, p["mouth"], opn)
            tx, ty, angle = motion_offset(
                p["style"], p["intensity"], t, duration, p["phase"], width, height,
            )
            if angle:
                work = work.rotate(float(angle), resample=Image.BICUBIC, expand=True)
            # Feet-anchor placement, same convention as _place_character / the
            # editor (y is a top-anchor for the feet line). We centre on the
            # VISIBLE character's horizontal centre and pin the visible bottom
            # to `y` — not the transparent padding.
            px = int(width * p["x"] - p["v_center_x"] + tx)
            py = int(height * p["y"] - p["v_bottom_y"] + ty)
            px = max(-work.width, min(width, px))
            py = max(-work.height, min(height, py))
            ov.alpha_composite(work, (px, py))
        bg.alpha_composite(ov)
        fp = os.path.join(frame_dir, "%s%05d.png" % (prefix, f))
        bg.convert("RGB").save(fp)
        paths.append(fp)
    return paths

