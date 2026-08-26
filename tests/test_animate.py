#!/usr/bin/env python3
"""Offline unit test for renderer/animate.py (mouth + body motion).

Purely Pillow-based (no ffmpeg / audio / edge-tts / network):

  - resolve_motion_style: action/motion_style -> whole-body style, with sane
    speaker vs bystander defaults and a graceful unknown-name fallback.
  - motion_offset: returns finite, non-zero, time-varying offsets for an
    animated style and all-zeros for "still".
  - estimate_mouth_region + apply_jaw_drop: a transparent-background "head"
    image gets a candid mouth rect, and opening it higher actually changes the
    pixels (the jaw-drop is visible by construction, not a silent no-op).
  - render_beat_frames: renders exactly round(duration*fps) frames.

Run: python tests/test_animate.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from renderer.animate import (  # noqa: E402
    estimate_mouth_region,
    apply_jaw_drop,
    motion_offset,
    resolve_motion_style,
    render_beat_frames,
    SCENE_MOTION_STYLES,
)


def _head_image(w=300, h=200, body_alpha=255):
    """A fully opaque upright figure (alpha bbox = whole image)."""
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    px = img.load()
    for y in range(h):
        for x in range(w):
            px[x, y] = (200, 120, 80, body_alpha)
    return img


def _bbox(img):
    return img.getchannel("A").getbbox()


def test_style_mapping():
    assert resolve_motion_style({"action": "speaking"}, True) == "sway"
    assert resolve_motion_style({"action": "nodding"}, True) == "nod"
    assert resolve_motion_style({"action": "walking"}, False) == "walk_in_place"
    assert resolve_motion_style({"motion_style": "bounce"}, True) == "bounce"
    # Unknown action -> speaker sway / bystander idle (never errors).
    assert resolve_motion_style({"action": "totally unknown"}, True) == "sway"
    assert resolve_motion_style({"action": "totally unknown"}, False) == "idle"
    assert resolve_motion_style({}, False) == "idle"
    print("test_style_mapping OK")


def test_motion_offset():
    # still never moves.
    tx0, ty0, a0 = motion_offset("still", 1.0, 0.2, 8.0, 0.0, 100, 100)
    assert (tx0, ty0, a0) == (0.0, 0.0, 0.0)
    # sway has rotation.
    _, _, a_sway = motion_offset("sway", 1.0, 0.1, 8.0, 0.0, 100, 100)
    assert abs(a_sway) > 0.0
    # bob moves vertically (negative = upward from the feet line).
    _, t_early, _ = motion_offset("bounce", 1.0, 0.0, 8.0, 0.0, 100, 100)
    _, t_later, _ = motion_offset("bounce", 1.0, 0.25, 8.0, 0.0, 100, 100)
    assert t_early != 0.0 or t_later != 0.0
    print("test_motion_offset OK")


def test_mouth_region_and_jaw():
    img = _head_image()
    # estimate a mouth rect on the synthetic head.
    rect = estimate_mouth_region(img)
    assert rect is not None, "no mouth rect from a plain head image"
    mx0, my0, mx1, my1 = rect
    assert mx1 > mx0 and my1 > my0
    # jaw-drop must change pixels when openness goes up.
    closed = img.copy()
    open_ = img.copy()
    apply_jaw_drop(closed, rect, 0.0)
    apply_jaw_drop(open_, rect, 1.0)
    # Count differing pixels within an extended mouth band.
    band = (max(0, mx0 - 10), max(0, my0 - 10), mx1 + 10, min(img.height, my1 + 30))
    c_crop = closed.crop(band).convert("RGB")
    o_crop = open_.crop(band).convert("RGB")
    diff = 0
    cpix = c_crop.getdata()
    opix = o_crop.getdata()
    for a, b in zip(cpix, opix):
        if a != b:
            diff += 1
    assert diff > 0, "jaw-drop at openness=1.0 produced zero pixel change (mouth not opening)"
    print(f"test_mouth_region_and_jaw OK (mouth rect {rect}, {diff} px changed)")


def test_render_beat_frames(tmp="/tmp/animate_test"):
    import os
    import glob
    import shutil
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    bg_path = os.path.join(tmp, "bg.png")
    Image.new("RGB", (320, 180), (10, 16, 30)).save(bg_path)
    ch_path = os.path.join(tmp, "char.png")
    _head_image(60, 120).save(ch_path)

    chars = [{
        "path": ch_path, "x": 0.5, "y": 0.9, "scale": 0.6, "name": "A",
        "action": "speaking", "intensity": 0.5, "is_speaker": True,
    }, {
        "path": ch_path, "x": 0.25, "y": 0.85, "scale": 0.4, "name": "B",
        "action": "", "intensity": 0.3, "is_speaker": False,
    }]
    frames = render_beat_frames(bg_path, chars, 320, 180, fps=10, duration=1.2,
                                openness=None, frame_dir=tmp, prefix="frame")
    assert len(frames) == round(1.2 * 10) == 12, f"expected 12 frames, got {len(frames)}"
    assert len(glob.glob(os.path.join(tmp, "frame*.png"))) == 12
    assert os.path.exists(frames[0])
    print("test_render_beat_frames OK (12 frames at 1.2s@10fps)")


def main():
    test_style_mapping()
    test_motion_offset()
    test_mouth_region_and_jaw()
    test_render_beat_frames()
    print("ALL ANIMATE TESTS PASS")


if __name__ == "__main__":
    main()