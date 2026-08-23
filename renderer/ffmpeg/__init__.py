"""FFmpeg subsystem: encode a scene frame+audio into a clip, then concat.

Concat uses the concat FILTER (-filter_complex concat=) with per-input
normalization, not the concat demuxer — the demuxer fed heterogeneous
clips through one shared decoder and desynced real jobs (see the matching
comment in the original cloud-render-worker.py). Each clip is normalized
to the same canvas/fps/pix_fmt before the splice point.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess

log = logging.getLogger("render.ffmpeg")


def _probe_audio(path: str) -> bool:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
         "-of", "csv=p=0", path],
        capture_output=True,
    )
    return bool(r.stdout.strip())


def _duration_of(path: str) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", path],
        capture_output=True,
        timeout=30,
    )
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def encode_scene_clip(
    frame_path: str,
    audio_path: str | None,
    out_path: str,
    width: int,
    height: int,
    duration: float | None = None,
) -> str:
    """Loop a still scene + optional audio into an H.264/AAC clip."""
    cmd = ["ffmpeg", "-y", "-loop", "1", "-i", frame_path]
    has_audio = audio_path is not None and os.path.exists(audio_path) and _probe_audio(audio_path)
    audio_dur = 0.0
    if has_audio:
        cmd += ["-i", audio_path]
        audio_dur = _duration_of(audio_path)

    if duration is None or duration <= 0:
        duration = audio_dur or 5.0
    # Keep a tiny tail of silence so the clip never cuts the audio short.
    duration = max(duration, audio_dur + 0.3)

    cmd += [
        "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
               f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=#08080e,setsar=1,fps=30,format=yuv420p",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
    ]
    if has_audio:
        cmd += ["-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2"]
    cmd += ["-t", f"{duration:.2f}", "-movflags", "+faststart", out_path]
    r = subprocess.run(cmd, capture_output=True, timeout=600)
    if r.returncode != 0 or not os.path.exists(out_path):
        raise RuntimeError("ffmpeg encode failed (rc=%s): %s" % (r.returncode, r.stderr.decode('utf-8', 'replace')[-1500:]))
    return out_path


def concat_clips(clip_paths: list, out_path: str, width: int, height: int) -> str:
    """Splice normalized clips into one output via the concat filter.

    Note: when some clips carry audio and some do not, the concat filter
    cannot splice a mixed a/v graph in one pass. We normalize each clip up
    front in a first pass (guaranteeing every clip has stereo audio), then
    concat over the normalized set. This avoids the shared-decoder desync
    the old single-pass demuxer approach hit with heterogeneous inputs.
    """
    clips = [c for c in clip_paths if c and os.path.exists(c)]
    if not clips:
        raise RuntimeError("no clips to concat")
    if len(clips) == 1:
        shutil.copyfile(clips[0], out_path)
        return out_path

    import tempfile
    tmpdir = tempfile.mkdtemp(prefix="concat-norm-")
    norm = []
    try:
        # 1) Normalize every clip to a uniform canvas/fps AND give each one
        #    stereo audio so the single concat filter handles all inputs.
        for i, p in enumerate(clips):
            np_ = os.path.join(tmpdir, f"clip{i}.mp4")
            _normalize_clip(p, np_, width, height)
            norm.append(np_)

        # 2) Single pass: both streams available on every input.
        cmd = ["ffmpeg", "-y"]
        n = len(norm)
        for p in norm:
            cmd += ["-i", p]
        parts = []
        refs = []
        for i in range(n):
            parts.append(
                f"[{i}:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,"
                f"fps=30,format=yuv420p[v{i}]"
            )
            parts.append(
                f"[{i}:a]aresample=44100,aformat=sample_fmts=fltp:channel_layouts=stereo[a{i}]"
            )
            refs.append(f"[v{i}][a{i}]")
        parts.append("".join(refs) + f"concat=n={n}:v=1:a=1[vout][aout]")
        cmd += [
            "-filter_complex", ";".join(parts),
            "-map", "[vout]", "-map", "[aout]",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
            "-movflags", "+faststart", out_path,
        ]
        r = subprocess.run(cmd, capture_output=True, timeout=1200)
        if r.returncode != 0 or not os.path.exists(out_path):
            raise RuntimeError(
                "ffmpeg concat failed (rc=%s): %s"
                % (r.returncode, r.stderr.decode('utf-8', 'replace')[-1500:])
            )
        return out_path
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _normalize_clip(src: str, dst: str, width: int, height: int) -> None:
    """Uniform re-encode so a mixed set of clips concats without desync."""
    vf = (f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
          f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1")
    has_audio = _probe_audio(src)
    cmd = ["ffmpeg", "-y", "-i", src]
    if not has_audio:
        cmd += ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100"]
    cmd += ["-vf", vf, "-r", "30", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "medium", "-crf", "20",
            "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
            "-movflags", "+faststart"]
    if not has_audio:
        cmd += ["-shortest"]
    cmd += [dst]
    r = subprocess.run(cmd, capture_output=True, timeout=300)
    if r.returncode != 0 or not os.path.exists(dst):
        raise RuntimeError(
            "ffmpeg normalize failed (rc=%s): %s"
            % (r.returncode, r.stderr.decode('utf-8', 'replace')[-1500:])
        )