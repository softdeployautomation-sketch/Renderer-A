"""Offline unit test for character placement (proxy-anchor regression guard).

Verifies `_place_character` uses the SAME coordinate convention as the editors
(classic dashboard.html + new video-tab.html placement canvas):

  - x = horizontal CENTER of the character (0..1 fraction of frame width)
  - y = top-anchor position of the character's FEET (0..1 fraction of height
        measured DOWN from the top; y=0.92 => feet near the BOTTOM of the frame)

This guards the 2026-08-25 fix: the renderer previously treated y as a bottom-UP
anchor (feet_y = h*(1-y)), which vertically mirrored the character so a character
dragged to the bottom (y=0.92) rendered near the TOP — the root cause of
"dragged positions don't persist in the video". It also guards the 2026-08-25
repeat-character cache fix (the same character URL across scenes keeps its own
per-scene x/y/scale instead of always using the first scene's).

Run: python tests/test_placement.py   (needs Pillow)
"""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from renderer.video import _place_character, _letterbox_bg

# Stub out the heavy runtime deps that renderer/render.py pulls in transitively
# (cloud->boto3, ffmpeg, music, voice, assets, background_removal, config) so this
# test runs offline against the real _cut_character/_character_entry code — none
# of them are exercised by the cache logic under test.
import renderer  # noqa: E402
for _name in ("assets", "background_removal", "cloud", "config", "ffmpeg", "music", "voice"):
    sys.modules.setdefault(f"renderer.{_name}", types.ModuleType(f"renderer.{_name}"))

from renderer.render import _cut_character  # noqa: E402


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

    # ── 2026-08-25 repeat-character cache regression ────────────────────────
    # The SAME character image URL recurs across scenes, but each scene has its
    # OWN x/y/scale. Previously _cut_character cached the WHOLE entry (incl. x/y)
    # keyed by URL, so a later scene got the FIRST scene's placement back.
    url = "/character-cutout/hero.png"
    # Seed the cache as if the character was first rendered at a far-left/top spot.
    cache = {url: {"path": "/tmp/already-cut.png"}}
    first = _cut_character(
        {"image_url": url, "name": "Hero", "x": 0.1, "y": 0.1, "scale": 0.5},
        None,  # ws unused on the cache-hit branch
        cache,
    )
    # Now the "same" character appears in a LATER scene dragged to the bottom-right.
    later = _cut_character(
        {"image_url": url, "name": "Hero", "x": 0.9, "y": 0.92, "scale": 0.8},
        None,
        cache,
    )
    # The cached branch must rebuild the entry from the CURRENT beat, not reuse
    # the first scene's placement (the old code would return first's x/y here).
    print(f"cache-hit: x={later['x']} y={later['y']} scale={later['scale']} "
          f"(expected x=0.9 y=0.92 s=0.8, NOT first scene 0.1/0.1/0.5)")
    assert later["x"] == 0.9 and later["y"] == 0.92 and later["scale"] == 0.8, \
        "cache hit returned the character's FIRST-scene placement, not the current scene's"
    assert first["x"] == 0.1, "first (cache-miss) placement should keep its own values"

    # ── 2026-08-26 size regression: `scale` must reflect the VISIBLE body ────
    # A cutout that carries transparent padding must render the character at the
    # same on-screen height as a tightly-cropped one for the SAME scale, and the
    # feet must anchor on the VISIBLE bottom. Before this fix the renderer sized
    # off the full (padded) image, so a character looked smaller than its size
    # number ("size 90 renders very small").
    def _padded_cutout(inner=(0, 30, 20, 40)):
        """40x40 cutout whose opaque character is ONLY the rect `inner`."""
        im = Image.new("RGBA", (40, 40), (0, 0, 0, 0))
        l, t, r, b = inner
        px = im.load()
        for yy in range(t, b):
            for xx in range(l, r):
                px[xx, yy] = (255, 0, 0, 255)
        return im

    padded = _padded_cutout()          # visible char height = 10px (t..b)
    tight = Image.new("RGBA", (10, 10), (255, 0, 0, 255))  # visible 10px
    bg_p = Image.new("RGBA", (100, 100), (0, 0, 0, 255))
    bgt = Image.new("RGBA", (100, 100), (0, 0, 0, 255))
    TARGET_SCALE = 0.2   # target visible height = 20px on the 100px bg
    _place_character(bg_p, padded, 0.5, 0.5, TARGET_SCALE)
    _place_character(bgt, tight, 0.5, 0.5, TARGET_SCALE)
    # Visible (opaque) height is identical regardless of padding — THE fix.
    vh_p = _bbox(bg_p)[3] - _bbox(bg_p)[2]
    vh_t = _bbox(bgt)[3] - _bbox(bgt)[2]
    print(f"padded visible height={vh_p}px, tight visible height={vh_t}px (both ~20)")
    assert abs(vh_p - 20) <= 4, f"padded cutout visible height={vh_p}, expected ~20"
    assert abs(vh_t - 20) <= 4, f"tight cutout visible height={vh_t}, expected ~20"
    assert abs(vh_p - vh_t) <= 2, "padding changed the character's on-screen size"
    # Feet (opaque bottom) anchored at y=0.5 -> ~50 for the tight char; the
    # padded char must also render fully inside the frame (bottom stays <= h).
    for name, b in (("tight", bgt), ("padded", bg_p)):
        bottom = _bbox(b)[3]
        print(f"{name} feet bottom={bottom} (frame height=100)")
        assert 0 < bottom <= 100, f"{name} feet bottom={bottom} outside frame"

    # ── 2026-08-26 background regression: square AI art must COVER a 16:9 frame ─
    # Owner: "the background fit the landscape but the video cut out the left
    # and right leaving blank in both, looked like it reduced the background."
    # Old _letterbox_bg LETTERBOXED (contain): a 1:1 square bg in a 16:9 frame →
    # dark bars on both sides. It now cover-crops so the bg fills the whole frame.
    sq = Image.new("RGB", (100, 100), (200, 30, 30))          # square red bg
    # Rebuild _letterbox_bg result via its own cover path using a temp file.
    import tempfile, os as _os
    _tmpbg = _os.path.join(tempfile.gettempdir(), "bg_cover_test.png")
    sq.save(_tmpbg)
    covered = _letterbox_bg(_tmpbg, 100, 56)
    px = covered.convert("RGB")
    print(f"bg corners: TL={px.getpixel((2,2))} TR={px.getpixel((97,2))} "
          f"BL={px.getpixel((2,53))} BR={px.getpixel((97,53))}")
    for corner in ((2, 2), (97, 2), (2, 53), (97, 53)):
        r, g, b = px.getpixel(corner)
        assert (r, g, b) == (200, 30, 30), \
            f"background corner {corner} is dark/letterboxed, got {(r,g,b)}"
    try:
        _os.remove(_tmpbg)
    except OSError:
        pass
    print("test_background_cover OK (square bg fully covers 16:9 frame, no bars)")

    print("ALL PLACEMENT TESTS PASS")


if __name__ == "__main__":
    main()