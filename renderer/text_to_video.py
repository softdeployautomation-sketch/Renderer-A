"""Text-to-video: script -> TTS audio -> per-scene backgrounds -> Ken Burns
-> timed captions -> ffmpeg concat -> final MP4.

Runs entirely on CPU (GitHub free runner). Uses edge-tts (free, no API key)
for voice synthesis and ffmpeg zoompan/drawtext for motion + captions.

Opt-in per job: ``settings.method == "text_to_video"``.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from . import assets, cloud, config, ffmpeg, voice
from .render import parse_settings

log = logging.getLogger("render.text_to_video")


class TextToVideoError(Exception):
    """A job-level failure reported FAILED, not a worker crash."""


# ── Scene splitting ────────────────────────────────────────────────────────

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_scenes(script: str) -> list[str]:
    """Split a script into scenes by sentence boundaries."""
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(script.strip()) if p.strip()]
    merged: list[str] = []
    for p in parts:
        if merged and len(p) < 3:
            merged[-1] = f"{merged[-1]} {p}"
        else:
            merged.append(p)
    return merged


# ── Background generation (PIL) ────────────────────────────────────────────

_BG_STYLES = {
    "gradient": ("#1a1a2e", "#16213e"),
    "sunset": ("#4b1248", "#141e30"),
    "forest": ("#0f2027", "#203a43", "#2c5364"),
    "ocean": ("#000000", "#0f0c29", "#302b63", "#24243e"),
    "midnight": ("#0f0c29", "#302b63", "#24243e"),
}


def _load_font(size: int):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for c in candidates:
        if os.path.exists(c):
            try:
                return ImageFont.truetype(c, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _wrap_text(draw, text, font, max_w):
    words = text.split()
    lines: list[str] = []
    cur = ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if draw.textlength(trial, font=font) <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _make_background(scene_idx, total, text, out_path, width, height, bg_style):
    colors = _BG_STYLES.get(bg_style, _BG_STYLES["gradient"])
    img = Image.new("RGB", (width, height), colors[0])
    draw = ImageDraw.Draw(img)

    if len(colors) > 1:
        for y in range(height):
            t = y / max(height - 1, 1)
            idx2 = min(int(t * (len(colors) - 1)), len(colors) - 2)
            frac = (t * (len(colors) - 1)) - idx2
            c1 = colors[idx2]
            c2 = colors[idx2 + 1]
            r = int(int(c1[1:3], 16) * (1 - frac) + int(c2[1:3], 16) * frac)
            g = int(int(c1[3:5], 16) * (1 - frac) + int(c2[3:5], 16) * frac)
            b = int(int(c1[5:7], 16) * (1 - frac) + int(c2[5:7], 16) * frac)
            draw.line([(0, y), (width, y)], fill=(r, g, b))

    title = f"Scene {scene_idx + 1} of {total}"
    title_font = _load_font(max(28, height // 24))
    body_font = _load_font(max(20, height // 36))

    title_w = draw.textlength(title, font=title_font)
    draw.text(((width - title_w) / 2, height * 0.35), title, font=title_font, fill="white")

    body_lines = _wrap_text(draw, text, body_font, width * 0.85)
    line_h = getattr(body_font, "size", 30)
    total_h = line_h * len(body_lines) * 1.2
    start_y = height * 0.52
    for i, line in enumerate(body_lines):
        lw = draw.textlength(line, font=body_font)
        draw.text(((width - lw) / 2, start_y + i * line_h * 1.2), line, font=body_font, fill="#e8ecf4")

    img.save(out_path, "PNG")

# ── FFmpeg helpers ─────────────────────────────────────────────────────────

def _ffprobe_duration(path: str) -> float:
    """Duration of a media file in seconds (0.0 on failure)."""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", path],
        capture_output=True, timeout=30,
    )
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def _encode_scene_with_captions(bg_path, audio_path, out_path, width, height,
                                caption, duration):
    """Encode one scene: Ken Burns zoompan on background + timed caption + audio."""
    duration = max(duration + 0.3, 2.0)

    zoompan = (f"zoompan=z='if(lte(zoom,1.0),1.05,zoom+0.0015)':d={duration*30}:"
               f"s={width}x{height}:fps=30")

    safe_caption = caption.replace("\\", "\\\\").replace("'", "\\'")
    drawtext = (f"drawtext=fontfile=/usr/share/fonts/truetype/dejavu/"
                f"DejaVuSans-Bold.ttf:fontsize={max(24, height // 28)}:"
                f"fontcolor=white:shadowcolor=black@0.7:shadowx=2:shadowy=2:"
                f"x=(w-text_w)/2:y=h*0.82:text='{safe_caption}':"
                f"enable='between(t,0,{duration - 0.3})'")

    vf = f"{zoompan},{drawtext},format=yuv420p,fps=30"

    cmd = [
        "ffmpeg", "-y", "-loop", "1", "-i", bg_path,
        "-i", audio_path,
        "-vf", vf,
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
        "-t", f"{duration:.2f}", "-movflags", "+faststart",
        out_path,
    ]
    r = subprocess.run(cmd, capture_output=True, timeout=600)
    if r.returncode != 0 or not os.path.exists(out_path):
        raise RuntimeError(
            "ffmpeg scene encode failed (rc=%s): %s"
            % (r.returncode, r.stderr.decode("utf-8", "replace")[-1500:])
        )
    return out_path


# ── Main entry point ───────────────────────────────────────────────────────

def render_text_to_video(job_id: str, job: dict) -> tuple[str, list]:
    """Render a narrated text-to-video from a script string.

    Settings expected:
        script        — the text to narrate (required)
        voice         — edge-tts voice id (optional, default config.DEFAULT_VOICE)
        bg_style      — background style name (optional, default "gradient")
        channel_id    — optional channel id for the job report

    Returns (final_video_url, scene_snapshot).
    """
    settings = parse_settings(job)
    ws = assets.make_workspace(job_id)

    script = str(settings.get("script") or "").strip()
    if not script:
        raise TextToVideoError("text_to_video job has no script to narrate")

    voice_id = str(settings.get("voice") or config.DEFAULT_VOICE).strip()
    bg_style = str(settings.get("bg_style") or "gradient").strip()
    if bg_style not in _BG_STYLES:
        bg_style = "gradient"

    scenes = _split_scenes(script)
    if not scenes:
        raise TextToVideoError("script produced no narratable scenes")

    w, h = config.OUTPUT_WIDTH, config.OUTPUT_HEIGHT
    clips: list[str] = []
    snapshot: list[dict] = []

    for idx, scene_text in enumerate(scenes):
        # 1) TTS -> audio for this scene.
        audio_path = str(ws / "clips" / f"tts-{idx}.mp3")
        try:
            voice.synthesize(scene_text, voice_id, audio_path)
        except Exception as exc:
            raise TextToVideoError(f"TTS failed for scene {idx}: {exc}") from exc
        audio_dur = _ffprobe_duration(audio_path)

        # 2) Background image (PIL gradient + text).
        bg_path = str(ws / "assets" / f"bg-{idx}.png")
        _make_background(idx, len(scenes), scene_text, bg_path, w, h, bg_style)

        # 3) Encode scene clip: zoompan + drawtext caption + audio.
        clip_path = str(ws / "clips" / f"scene-{idx}.mp4")
        try:
            _encode_scene_with_captions(bg_path, audio_path, clip_path, w, h,
                                        scene_text, audio_dur)
        except Exception as exc:
            raise TextToVideoError(f"scene {idx} encode failed: {exc}") from exc
        clips.append(clip_path)

        snapshot.append({
            "index": idx,
            "text": scene_text,
            "duration": round(audio_dur, 2),
            "voice": voice_id,
        })
        log.info("scene %d/%d done (%.1fs)", idx + 1, len(scenes), audio_dur)

    if not clips:
        raise TextToVideoError("no scenes produced usable clips")

    # 4) Concat scenes -> final.mp4
    final_path = str(ws / "final.mp4")
    ffmpeg.concat_clips(clips, final_path, w, h)

    # 5) Upload to R2 -> public URL.
    url = cloud.upload_video(job_id, final_path)
    return url, snapshot
    return out_path