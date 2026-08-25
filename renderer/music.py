"""Background-music layer for Renderer-A (Class B, 2026-08-25).

The renderer has no music track today — only per-beat dialogue. The backend
passes a music *mood* on the job (`settings.music`, e.g. horror_piano /
eight_bit / tension / orchestral / dramatic / calm / upbeat / corporate / "").
We synthesize a soft generative ambient bed with ffmpeg lavfi (no bundled
audio, no external CDN — the runner only talks outbound to the Cloudflare API /
R2) and mix it *under* the dialogue with amix at a low volume so speech stays
clear.

Empty / "none" moods produce no bed (returns None → no music).
"""

from __future__ import annotations

import logging
import os
import subprocess

log = logging.getLogger("render.music")

SAMPLE_RATE = 44100

# Mood -> deep drone frequencies (Hz). Falsy/unknown maps to no music.
MOOD_DRONES: dict[str, list[float]] = {
    "horror_piano": [110.0, 164.81, 220.0],
    "dramatic":      [110.0, 164.81, 220.0],
    "tension":       [110.0, 116.47],
    "orchestral":    [110.0, 220.0, 329.63, 440.0],
    "calm":          [196.0, 293.66, 392.0],
    "upbeat":        [220.0, 329.63, 440.0, 554.37],
    "corporate":     [261.63, 329.63, 392.0],
    "eight_bit":     [220.0, 329.63, 440.0],
}

# Level the bed is mixed at under the dialogue.
MUSIC_MIX_VOLUME = 0.22


def _bed_src(freqs: list[float], dur: float) -> str:
    """aevalsrc lavfi source: summed sine drones with a slow LFO swell.

    Exposed for unit-testing the drone math.
    """
    n = max(1, len(freqs))
    terms = " + ".join(f"({round(1.0 / n, 3)}*sin(2*PI*{f}*t))" for f in freqs)
    expr = f"({terms})*(0.7+0.3*sin(2*PI*0.13*t))"
    return f"aevalsrc={expr}|{expr}:s={SAMPLE_RATE}:d={dur:.2f}"


def synth_bed(duration: float, mood: str, out_path: str) -> str | None:
    """Generate `duration` s of ambient music for `mood` into out_path (.wav)."""
    freqs = MOOD_DRONES.get((mood or "").strip().lower()) or []
    if not freqs:
        return None
    dur = max(1.0, float(duration))
    fade_out = max(0.0, dur - 1.2)
    filt = (
        "lowpass=f=700,tremolo=f=0.14:d=0.4,"
        f"afade=t=in:st=0:d=0.8,afade=t=out:st={fade_out:.2f}:d=1.2"
    )
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", _bed_src(freqs, dur),
        "-filter_complex", f"[0:a]{filt},volume=0.55[a]",
        "-map", "[a]",
        "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le",
        out_path,
    ]
    r = subprocess.run(cmd, capture_output=True, timeout=300)
    if r.returncode != 0 or not os.path.exists(out_path):
        log.warning("music bed generation failed: %s", r.stderr.decode("utf8", "replace")[-400:])
        return None
    return out_path


def mix_music(video_path: str, music_path: str | None, out_path: str) -> str:
    """amix the music bed under the video's dialogue audio at low volume."""
    if not music_path or not os.path.exists(music_path):
        return video_path
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", music_path,
        "-filter_complex",
        f"[1:a]volume={MUSIC_MIX_VOLUME}[mus];[0:a][mus]amix=inputs=2:duration=first:dropout_transition=2[a]",
        "-map", "0:v", "-map", "[a]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-ar", "44100",
        "-movflags", "+faststart",
        out_path,
    ]
    r = subprocess.run(cmd, capture_output=True, timeout=600)
    if r.returncode != 0 or not os.path.exists(out_path):
        raise RuntimeError("music mix failed: " + r.stderr.decode("utf8", "replace")[-400:])
    return out_path