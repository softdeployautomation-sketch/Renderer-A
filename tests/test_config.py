"""Offline unit test for the per-job aspect ratio (2026-08-29 feature).

Guards `config.resolve_output_dims`: the job's `settings.aspect_ratio` (set
by worker-full.ts from the editor's new aspect-ratio picker, chosen before
script entry) must map to real canvas pixel dimensions, and anything missing
or unrecognized (old jobs queued before this feature existed, or a bad value)
must fall back to the existing 16:9 default so nothing already in flight
regresses.

Run: python tests/test_config.py   (or via pytest)
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# config.py requires several CHANNELRY_* env vars at import time (real
# credentials in production) — dummy values are fine here, this test only
# exercises the pure aspect-ratio -> pixel-dims mapping, not anything that
# actually talks to R2/the Worker.
for _var, _default in (
    ("CHANNELRY_CLOUD_API_URL", "http://example.invalid"),
    ("CHANNELRY_RENDER_WORKER_TOKEN", "test-token"),
    ("CHANNELRY_R2_ENDPOINT", "http://example.invalid"),
    ("CHANNELRY_R2_BUCKET", "test-bucket"),
    ("CHANNELRY_R2_ACCESS_KEY_ID", "test-key"),
    ("CHANNELRY_R2_SECRET_ACCESS_KEY", "test-secret"),
    ("CHANNELRY_R2_PUBLIC_BASE_URL", "http://example.invalid"),
):
    os.environ.setdefault(_var, _default)

from renderer import config  # noqa: E402


def test_default_16_9_matches_existing_output_dims():
    w, h = config.resolve_output_dims({})
    assert (w, h) == (config.OUTPUT_WIDTH, config.OUTPUT_HEIGHT), \
        "no aspect_ratio set must still render at the pre-existing default size"


def test_9_16_is_portrait():
    w, h = config.resolve_output_dims({"aspect_ratio": "9:16"})
    assert h > w, f"9:16 must be taller than wide, got {w}x{h}"


def test_1_1_is_square():
    w, h = config.resolve_output_dims({"aspect_ratio": "1:1"})
    assert w == h, f"1:1 must be square, got {w}x{h}"


def test_unknown_value_falls_back_to_default():
    w, h = config.resolve_output_dims({"aspect_ratio": "not-a-real-ratio"})
    assert (w, h) == (config.OUTPUT_WIDTH, config.OUTPUT_HEIGHT), \
        "an unrecognized aspect_ratio must not crash or produce a weird size"


def main():
    test_default_16_9_matches_existing_output_dims()
    test_9_16_is_portrait()
    test_1_1_is_square()
    test_unknown_value_falls_back_to_default()
    print("ALL CONFIG TESTS PASS")


if __name__ == "__main__":
    main()
