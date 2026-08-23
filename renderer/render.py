"""Main render pipeline (plan §20 order).

Receive job -> validate -> download assets -> prepare workspace
-> per character: background removal -> per scene: load bg, place chars,
   apply x/y/scale/layers, voice, dialogue/audio sync, render scene
-> concat scenes -> final ffmpeg -> upload to R2 -> report COMPLETED.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from PIL import Image

from . import assets, background_removal, cloud, config, ffmpeg, voice
from .video import compose_scene

log = logging.getLogger("render.pipeline")


class RenderJobError(Exception):
    """A job-level failure that should be reported FAILED, not crash the worker."""


def parse_settings(job: dict) -> dict:
    raw = job.get("settings") or "{}"
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _download(url: str, ws: Path, prefix: str) -> str:
    return assets.download_asset(url, ws, prefix)


def _cut_character(char: dict, ws: Path, cache: dict):
    """Background-remove a character image (cached by URL)."""
    url = (char or {}).get("image_url") or ""
    if not url:
        return None
    if url in cache:
        return cache[url]

    raw = _download(url, ws, "char")
    img = Image.open(raw).convert("RGBA")
    removed = background_removal.remove_background(img)
    out = ws / "assets" / f"cut-{len(cache)}.png"
    removed.save(out)

    entry = {
        "path": str(out),
        "name": (char or {}).get("name") or "character",
        "x": float((char or {}).get("x", 0.5)),
        "y": float((char or {}).get("y", 0.92)),
        "scale": float((char or {}).get("scale", 0.6)),
    }
    cache[url] = entry
    return entry
def _scene_characters(beat: dict, ws: Path, cache: dict) -> list:
    """Resolve the characters placed in a scene."""
    chars = beat.get("characters")
    scene_chars = []
    if isinstance(chars, list) and chars:
        for c in chars:
            entry = _cut_character(c, ws, cache)
            if entry:
                scene_chars.append(entry)
        if scene_chars:
            return scene_chars

    # Legacy single-beat shape.
    entry = _cut_character(
        {"image_url": beat.get("image_url", ""), "x": 0.5, "y": 0.92, "scale": 0.6},
        ws,
        cache,
    )
    if entry:
        scene_chars.append(entry)
    return scene_chars


def _beat_dialogue(beat: dict, scene_chars: list) -> str:
    """Line for a scene: beat-level first, else a character's own line."""
    line = str(beat.get("dialogue") or "").strip()
    if line:
        return line
    for c in beat.get("characters") or []:
        d = str((c or {}).get("dialogue") or "").strip()
        if d:
            return d
    return ""


def _beat_voice(beat: dict) -> str:
    v = str(beat.get("voice") or "").strip()
    return v or config.DEFAULT_VOICE


def _single_character_entry(settings: dict, ws: Path) -> dict | None:
    """Build a bg-removed single-character entry from image_url settings."""
    url = str(settings.get("image_url") or "").strip()
    if not url:
        return None
    return _cut_character({"image_url": url, "x": 0.5, "y": 0.92, "scale": 0.6}, ws, {})


def render_single_character(job_id: str, job: dict, method: str) -> tuple[str, list]:
    """Render classic single-character methods (still / walk).

    Both carry a single image_url (+ optional background_url) and an optional
    dialogue + voice + duration. Produces one composed clip (image + TTS),
    uploaded as final.mp4. Motion (walk_across / intensity) is currently
    rendered as a static composed scene — action/motion plugins are future
    hooks, exact backend settings shape is preserved.
    """
    settings = parse_settings(job)
    ws = assets.make_workspace(job_id)

    bg_url = str(settings.get("background_url") or "").strip()
    bg_path = _download(bg_url, ws, "bg") if bg_url else None

    entry = _single_character_entry(settings, ws)
    if not entry:
        raise RenderJobError(f"{method} job has no image_url to render")

    w, h = config.OUTPUT_WIDTH, config.OUTPUT_HEIGHT
    frame = compose_scene(
        bg_path,
        [{"path": entry["path"], "x": entry["x"], "y": entry["y"], "scale": entry["scale"]}],
        w,
        h,
    )
    frame_path = ws / "clips" / "frame-0.png"
    frame.save(frame_path)

    dialogue = str(settings.get("dialogue") or "").strip()
    audio_path = None
    if dialogue:
        audio_path = voice.synthesize(dialogue, settings.get("voice"), str(ws / "clips" / "voice-0.mp3"))

    clip_path = ws / "clips" / "scene-0.mp4"
    ffmpeg.encode_scene_clip(
        str(frame_path), audio_path, str(clip_path), w, h,
        duration=settings.get("duration"),
    )
    final_path = str(ws / "final.mp4")
    ffmpeg.concat_clips([clip_path], final_path, w, h)
    url = cloud.upload_video(job_id, final_path)
    snapshot = [{"index": 0, "background": bg_url, "characters": [entry["name"]], "line": dialogue}]
    return url, snapshot
def render_character_video(job_id: str, job: dict) -> tuple[str, list]:
    """Render a character_video + auto_cast job.

    Returns (final_video_url, scene_snapshot). Raises RenderJobError on a
    job-level failure so the caller reports FAILED.
    """
    settings = parse_settings(job)
    beats = settings.get("beats") or []
    if not beats:
        raise RenderJobError("job has no resolvable beats")

    ws = assets.make_workspace(job_id)
    character_cache: dict = {}
    clips: list[str] = []
    snapshot: list[dict] = []
    w, h = config.OUTPUT_WIDTH, config.OUTPUT_HEIGHT

    for idx, beat in enumerate(beats):
        # 1) background
        bg_url = str(beat.get("background_url") or "").strip()
        bg_path = _download(bg_url, ws, "bg") if bg_url else None

        # 2) characters in this scene (background-removed).
        scene_chars = _scene_characters(beat, ws, character_cache)
        if not scene_chars:
            log.warning("beat %d produced no usable character — skipping", idx)
            continue

        # 3) dialogue + voice
        dialogue = _beat_dialogue(beat, scene_chars)
        audio_path = None
        if dialogue:
            voice_id = _beat_voice(beat)
            audio_path = voice.synthesize(
                dialogue, voice_id, str(ws / "clips" / f"voice-{idx}.mp3")
            )

        # 4) compose the full scene frame.
        frame = compose_scene(
            bg_path,
            [
                {"path": e["path"], "x": e["x"], "y": e["y"], "scale": e["scale"]}
                for e in scene_chars
            ],
            w,
            h,
        )
        frame_path = ws / "clips" / f"frame-{idx}.png"
        frame.save(frame_path)

        # 5) encode this scene as a clip (frame + dialogue audio).
        clip_path = ws / "clips" / f"scene-{idx}.mp4"
        ffmpeg.encode_scene_clip(str(frame_path), audio_path, str(clip_path), w, h)
        clips.append(str(clip_path))

        snapshot.append(
            {
                "index": idx,
                "background": bg_url,
                "characters": [e["name"] for e in scene_chars],
                "line": _beat_dialogue(beat, scene_chars),
            }
        )

    if not clips:
        raise RenderJobError("no scenes produced usable clips")

    # 6) concat scenes -> final.mp4
    final_path = str(ws / "final.mp4")
    ffmpeg.concat_clips(clips, final_path, w, h)

    # 7) upload to R2 -> public URL
    url = cloud.upload_video(job_id, final_path)
    return url, snapshot