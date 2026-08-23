#!/usr/bin/env python3
"""Offline smoke test: exercise render_character_video without Cloudflare/R2.

Runs in the /tmp venv (Pillow + edge-tts; rembg NOT installed, so the
passthrough rembg fallback runs). Monkeypatches only the network-facing
functions (asset download + video upload) so the whole compose -> voice ->
ffmpeg -> concat pipeline runs against local files.
"""
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("CHANNELRY_FORCE_NO_REMBG", "1")
os.environ.setdefault("CHANNELRY_CLOUD_API_URL", "http://__offline__")
os.environ.setdefault("CHANNELRY_RENDER_WORKER_TOKEN", "test")
os.environ.setdefault("CHANNELRY_R2_ENDPOINT", "http://__offline__")
os.environ.setdefault("CHANNELRY_R2_BUCKET", "b")
os.environ.setdefault("CHANNELRY_R2_ACCESS_KEY_ID", "k")
os.environ.setdefault("CHANNELRY_R2_SECRET_ACCESS_KEY", "s")
os.environ.setdefault("CHANNELRY_R2_PUBLIC_BASE_URL", "https://r2.test")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Stub boto3 so the offline test can import renderer.cloud without it
# (cloud.upload_video is replaced below; boto3 is never actually invoked).
import sys as _sys
import types as _types
if "boto3" not in _sys.modules:
    _fake = _types.ModuleType("boto3")
    _fake.client = lambda *a, **k: None
    _sys.modules["boto3"] = _fake

from renderer import assets, cloud, render
from PIL import Image


def main():
    tmp = Path("/tmp/chrender-smoke")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    bg_path = tmp / "bg.png"
    Image.new("RGB", (960, 540), (20, 24, 48)).save(bg_path)
    ch_path = tmp / "char.png"
    Image.new("RGB", (256, 256), (70, 60, 120)).save(ch_path)

    def local_download(url, ws, prefix):
        if url.startswith("file://"):
            src = url[len("file://"):]
            name = prefix + "-local" + os.path.splitext(src)[1]
            dst = str(Path(ws) / "assets" / name)
            shutil.copyfile(src, dst)
            return dst
        raise RuntimeError("offline: unexpected download " + url)

    def local_upload_video(job_id, local_path):
        dst = str(tmp / f"{job_id}.mp4")
        shutil.copyfile(local_path, dst)
        return "file://" + dst

    assets.download_asset = local_download
    cloud.upload_video = local_upload_video

    job = {
        "id": "smoke-test-job",
        "mode": "motion_video",
        "settings": {
            "kind": "motion_video",
            "method": "character_video",
            "beats": [
                {
                    "background_url": "file://" + str(bg_path),
                    "characters": [
                        {"image_url": "file://" + str(ch_path), "x": 0.5, "y": 0.9, "scale": 0.6},
                    ],
                    "dialogue": "Hello, this is a Channelry test render.",
                    "voice": "en-US-JennyNeural",
                },
                {
                    "background_url": "file://" + str(bg_path),
                    "characters": [
                        {"image_url": "file://" + str(ch_path), "x": 0.35, "y": 0.85, "scale": 0.5},
                    ],
                    "dialogue": "And here is the second scene.",
                    "voice": "en-GB-SoniaNeural",
                },
            ],
        },
    }

    url, snapshot = render.render_character_video(job["id"], job)
    print("FINAL_URL:", url)
    print("SNAPSHOT:", json.dumps(snapshot, indent=2))
    final = tmp / "smoke-test-job.mp4"
    print("FINAL_SIZE:", os.path.getsize(final) if final.exists() else 0)

    # Also exercise a classic single-character (still/walk) job.
    still_job = {
        "id": "smoke-still-job",
        "mode": "motion_video",
        "settings": {
            "kind": "motion_video",
            "method": "still",
            "image_url": "file://" + str(ch_path),
            "background_url": "file://" + str(bg_path),
            "dialogue": "A classic still render check.",
            "voice": "en-US-AriaNeural",
        },
    }
    url2, snap2 = render.render_single_character(still_job["id"], still_job, "still")
    still_out = tmp / "smoke-still-job.mp4"
    print("STILL_URL:", url2)
    print("STILL_SNAPSHOT:", json.dumps(snap2, indent=2))
    print("STILL_SIZE:", os.path.getsize(still_out) if still_out.exists() else "missing")


if __name__ == "__main__":
    main()