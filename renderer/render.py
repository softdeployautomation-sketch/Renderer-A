"""Main render pipeline (plan §20 order).

Receive job -> validate -> download assets -> prepare workspace
-> per character: background removal -> per scene: load bg, place chars,
   apply x/y/scale/layers, voice, dialogue/audio sync, render scene
-> concat scenes -> final ffmpeg -> upload to R2 -> report COMPLETED.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path

from PIL import Image

from . import animate, assets, background_removal, cloud, config, ffmpeg, music, voice
from .video import compose_scene, compose_title_card

log = logging.getLogger("render.pipeline")

# Frames-per-second for animated scenes. Kept in sync with ffmpeg's hardcoded
# fps=30 so an animated clip concats cleanly with the still/title clips.
ANIM_FPS = 30


def _video_duration(path: str) -> float:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", path],
            capture_output=True, timeout=30,
        )
        return max(0.0, float(r.stdout.strip()))
    except Exception:
        return 0.0


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


def _character_entry(path: str, char: dict) -> dict:
    """Build a placed-character entry from a cut image + the CURRENT beat's char.

    x/y/scale/motion are ALWAYS taken from the caller's `char`, never from a cache,
    because the same character image URL appears in many scenes with DIFFERENT
    per-scene placements. Returning a cached copy of x/y would render every later
    scene at the character's FIRST-seen position (see _cut_character below).
    """
    base_name = (char or {}).get("name") or "character"
    return {
        "path": path,
        "name": base_name,
        "x": float((char or {}).get("x", 0.5)),
        "y": float((char or {}).get("y", 0.92)),
        "scale": float((char or {}).get("scale", 0.6)),
        "motion_style": (char or {}).get("motion_style") or "idle",
        "gesture_start": float((char or {}).get("gesture_start") or 0),
        "gesture_duration": float((char or {}).get("gesture_duration") or 0),
    }


def _cut_character(char: dict, ws: Path, cache: dict):
    """Background-remove a character image (the cut-IMAGE is cached by URL).

    Skips re-cutting when the image URL already points at a transparent cutout
    (contains '/character-cutout/' or 'character-cutouts/') so a character pre-cut
    in a prior flow is reused — the backend enqueues `character_cutout` jobs for
    fresh auto-cast characters, so at render time prefer that asset instead of
    re-removing the background.

    FIX (2026-08-25): previously the WHOLE entry (including x/y/scale) was cached by
    URL and returned on a cache hit. The same character URL recurs across scenes, so
    a later scene got the FIRST scene's placement back — "dragged position doesn't
    reflect; video shows the first position." Now only the cut-asset PATH is reused
    from the cache; the returned entry is rebuilt from the current beat so each scene
    keeps its own x/y/scale/motion.
    """
    url = (char or {}).get("image_url") or ""
    if not url:
        return None

    # Reuse an already-cut asset for this URL but ALWAYS apply the current
    # beat's per-scene placement/action.
    cached = cache.get(url) if isinstance(cache, dict) else None
    if cached and cached.get("path"):
        return _character_entry(cached["path"], char)

    is_precut = "/character-cutout/" in url or "/character-cutouts/" in url
    if is_precut:
        raw = _download(url, ws, "char")
        entry = _character_entry(str(raw), char)
        cache[url] = entry
        return entry

    raw = _download(url, ws, "char")
    img = Image.open(raw).convert("RGBA")
    removed = background_removal.remove_background(img)
    out = ws / "assets" / f"cut-{len(cache)}.png"
    removed.save(out)

    entry = _character_entry(str(out), char)
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


def render_character_cutout(job_id: str, job: dict) -> tuple[str, list]:
    """Background-remove a raw character image and store a transparent PNG.

    Cutout-first (2026-08-24): the backend enqueues a `character_cutout` job per
    auto-generated character (settings carry `image_url` / `r2_key` / `gallery_id`).
    We download the raw image, run RemBG, and upload the transparent PNG back to
    R2 at `settings.r2_key` (so it lands in the character gallery as a real
    cutout, not the plain-background raw). No voice / ffmpeg involved — this is a
    pure image -> cutout -> R2 step. The backend PATCH completion handler points
    that saved_artworks row at the new key. Returns (uploaded_public_url, []).
    """
    settings = parse_settings(job)
    url = str(settings.get("image_url") or "").strip()
    if not url:
        raise RenderJobError("character_cutout job has no image_url to cut")
    r2_key = str(settings.get("r2_key") or "").strip() or (
        f"uploads/character-cutout/{job_id}.png"
    )

    ws = assets.make_workspace(job_id)
    raw = _download(url, ws, "char")
    img = Image.open(raw).convert("RGBA")
    removed = background_removal.remove_background(img)
    out = ws / "cutout.png"
    removed.save(out)

    public = cloud.upload_file(str(out), r2_key, content_type="image/png")
    log.info("character_cutout %s -> %s", job_id, r2_key)
    return public, []


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
        # Title-card first (2026-08-25): a beat flagged is_title_card has no
        # characters — render the background + centered title/subtitle text as a
        # static intro scene FIRST (the backend unshifts it to the head).
        if beat.get("is_title_card"):
            bg_url = str(beat.get("background_url") or "").strip()
            bg_path = _download(bg_url, ws, "bg") if bg_url else None
            title = str(beat.get("title") or beat.get("text") or "").strip() or "Your story"
            subtitle = str(beat.get("subtitle") or "").strip()
            position = str(beat.get("position") or "center")
            duration = max(2, min(20, int(float(beat.get("duration") or 4))))
            frame = compose_title_card(bg_path, title, subtitle, position, w, h)
            frame_path = ws / "clips" / f"title-{idx}.png"
            frame.save(frame_path)
            clip_path = ws / "clips" / f"title-{idx}.mp4"
            ffmpeg.encode_scene_clip(str(frame_path), None, str(clip_path), w, h, duration=duration)
            clips.append(str(clip_path))
            snapshot.append({"index": idx, "kind": "title_card", "title": title})
            continue

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

        # 4/5) compose + encode this scene as a clip (face + audio).
        #    Animated path (2026-08-26): when a character speaks we render a
        #    real frame sequence -- mouth lip-sync (audio-energy jaw-drop) plus
        #    whole-body motion from the beat's action/motion_style -- instead of
        #    a single frozen still. Silent/title scenes still use the static loop.
        clip_path = ws / "clips" / f"scene-{idx}.mp4"
        scene_dur_raw = beat.get("duration")
        dur_override = float(scene_dur_raw) if scene_dur_raw else None

        if dialogue and audio_path and scene_chars:
            # Per-scene timing (Class B): honor beat.duration, else audio+
            # a short silent tail (mirrors encode_scene_clip's default).
            scene_dur = dur_override or (_video_duration(str(audio_path)) + 0.3)
            scene_dur = max(0.4, min(config.MAX_VIDEO_SECONDS * 60, scene_dur))

            # Which character is talking: the one with its own per-character
            # dialogue, else the first character (beat-level dialogue belongs to
            # the scene's speaker in the new auto_cast flow). scene_chars keeps
            # the same order as beat.characters (see _scene_characters).
            raw_chars = beat.get("characters") or []
            speaker_idx = 0
            for i, c in enumerate(raw_chars):
                if str((c or {}).get("dialogue") or "").strip():
                    speaker_idx = i
                    break
            speaker_ch = raw_chars[speaker_idx] if speaker_idx < len(raw_chars) else {}
            try:
                speaker_pos = scene_chars.index(scene_chars[speaker_idx]) if len(scene_chars) > speaker_idx else 0
            except (IndexError, ValueError):
                speaker_pos = 0
            # Action defaults to the beat's (auto_cast sets action='speaking').
            beat_action = str(beat.get("action") or speaker_ch.get("action") or "").strip()

            anim_chars = []
            for pos, e in enumerate(scene_chars):
                anim_chars.append({
                    "path": e["path"], "x": e["x"], "y": e["y"], "scale": e["scale"],
                    "name": e["name"],
                    "action": beat_action or e.get("action") or "speaking",
                    "intensity": float(e.get("intensity", 0.5)),
                    "is_speaker": (pos == speaker_pos),
                })

            frame_dir = ws / "clips" / f"frames-{idx}"
            frame_dir.mkdir(parents=True, exist_ok=True)
            openness = animate.audio_openness_curve(str(audio_path), ANIM_FPS)
            animate.render_beat_frames(
                bg_path, anim_chars, w, h, ANIM_FPS,
                float(scene_dur), openness, str(frame_dir), "frame",
            )
            ffmpeg.encode_frame_sequence(
                str(frame_dir), "frame%05d.png", ANIM_FPS,
                str(audio_path), str(clip_path), w, h,
            )
        else:
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
            ffmpeg.encode_scene_clip(
                str(frame_path), audio_path, str(clip_path), w, h,
                duration=dur_override,
            )
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

    # 6b) Background music (Class B): mix the mood bed under the dialogue.
    #     Synthesized to the exact final duration and amixed at the level the
    #     frontend chose (`settings.music_volume`, 0..1) or the default.
    music_mood = str(settings.get("music") or "").strip()
    if music_mood:
        bed_path = str(ws / "music-bed.wav")
        if music.synth_bed(_video_duration(final_path), music_mood, bed_path):
            mixed_path = str(ws / "final-with-music.mp4")
            mv = settings.get("music_volume")
            music.mix_music(final_path, bed_path, mixed_path,
                            volume=None if mv is None else float(mv))
            shutil.copyfile(mixed_path, final_path)

    # 7) upload to R2 -> public URL
    url = cloud.upload_video(job_id, final_path)
    return url, snapshot