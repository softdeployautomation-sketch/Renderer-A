"""Offline unit test for character placement (proxy-anchor regression guard).

Verifies `_place_character` uses the SAME coordinate convention as the editors
(classic dashboard.html + new video-tab.html placement canvas):

  - x = horizontal CENTER of the character (0..1 fraction of frame width)
  - y = top-anchor position of the character's FEET (0..1 fraction of height
        measured DOWN from the top; y=0.92 => feet near the BOTTOM of the frame)

This guards the 2026-08-25 fix: the renderer previously treated y as a bottom-UP
anchor (feet_y = h*(1-y)), which vertically mirrored the character so a character
dragged to the bottom (y=0.92) rendered near the TOP — the root cause of
"dragged positions don't persist in the video".

Run: python tests/test_placement.py   (needs Pillow)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from renderer.video import _place_character


def _bbox(img):
    minx = miny = 999
    maxx = maxy = -1
    for yy in range(img.height):
        for xx in range(img.width):
            if img.getpixel((xx, yy))[0] > 200:
                minx = min(minx, xx)
                maxx = max(maxx, xx)
                miny = min(miny, yy)
                maxy = max(maxy, yy)
    return minx, maxx, miny, maxy


def main():
    bg = Image.new("RGBA", (100, 100), (0, 0, 0, 255))
    ch = Image.new("RGBA", (10, 10), (255, 0, 0, 255))
    SCALE = 0.2  # target_h = 20px, small so placement is observable

    # y=0.92: feet/bottom edge should be near the BOTTOM (~92).
    b = bg.copy()
    _place_character(b, ch, 0.5, 0.92, SCALE)
    _, _, _, maxy = _bbox(b)
    print(f"y=0.92 feet edge: {maxy} (expected ~92)")
    assert abs(maxy - 92) <= 3, f"y=0.92 feet={maxy}"

    # y=0.5: feet in the middle (~50).
    b = bg.copy()
    _place_character(b, ch, 0.5, 0.5, SCALE)
    _, _, _, maxy = _bbox(b)
    print(f"y=0.5 feet edge: {maxy} (expected ~50)")
    assert abs(maxy - 50) <= 3, f"y=0.5 feet={maxy}"

    # y=0.1: feet near the TOP (clamps to top given 20px char).
    b = bg.copy()
    _place_character(b, ch, 0.5, 0.1, SCALE)
    _, _, miny, _ = _bbox(b)
    print(f"y=0.1 top edge: {miny} (expected 0, char pinned to top)")
    assert miny == 0, f"y=0.1 miny={miny}"

    # x=0.85: horizontally centered near the right (~85).
    b = bg.copy()
    _place_character(b, ch, 0.85, 0.5, SCALE)
    minx, maxx, _, _ = _bbox(b)
    cx = (minx + maxx) / 2
    print(f"x=0.85 center: {cx} (expected ~85)")
    assert abs(cx - 85) <= 3, f"x=0.85 center={cx}"

    # Regression guard: THE OLD buggy formula put y=0.92 at the TOP.
    old = int(100 * (1.0 - 0.92))  # = 8 (near top)
    assert old < 50, "sanity: old bottom-up formula placed y=0.92 near top"

    print("ALL PLACEMENT TESTS PASS")


if __name__ == "__main__":
    main()