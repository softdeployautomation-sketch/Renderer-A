"""Renderer-A — Channelry video render worker (GitHub Actions disposable compute).

Drives the Channelry render queue from the *outside*:
  claim job -> download assets -> background removal -> composite scenes
  -> voice + FFmpeg -> upload to R2 -> report completion.

The runner only ever talks OUTBOUND over HTTPS to the Cloudflare API and
R2. It never depends on a stable/allowlisted inbound IP (ephemeral
GitHub-hosted runners change address every job — that is intentional).
"""
__version__ = "0.1.0"