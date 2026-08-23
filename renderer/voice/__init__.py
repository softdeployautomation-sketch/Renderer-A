"""Voice synthesis subsystem.

Edge-tts (Microsoft Edge neural voices, free, no API key) is used because
the voice IDs in the Channelry voice catalog are exactly the Azure neural
voice names edge-tts uses (en-US-ChristopherNeural, en-GB-SoniaNeural, ...).
Each beat's `voice` is passed straight through; stylized `character:*` ids
map onto a sensible default.

Isolated under voice/ so a paid provider (ElevenLabs etc.) can replace it
later without changing the queue or editor."""
from __future__ import annotations

import asyncio
import logging
import os

from .. import config

log = logging.getLogger("render.voice")

# stylized/unknown voice ids -> a real edge-tts voice
_VOICE_FALLBACKS = {
    "character:kid": "en-US-AnaNeural",
    "character:kid_playful": "en-US-AriaNeural",
    "character:storyteller": "en-GB-RyanNeural",
    "character:storyteller_cozy": "en-GB-SoniaNeural",
    "character:small_critter": "en-US-AriaNeural",
    "character:big_animal": "en-US-EricNeural",
    "character:elder_male": "en-US-GuyNeural",
    "character:elder_female": "en-US-JennyNeural",
    "character:teen_male": "en-US-MichelleNeural",
    "character:teen_female": "en-US-AnaNeural",
    "character:tiny_animal": "en-US-AriaNeural",
    "character:wise_owl": "en-GB-RyanNeural",
}


def resolve_voice(voice_id: str | None) -> str:
    v = (voice_id or "").strip()
    if not v:
        return config.DEFAULT_VOICE
    if v in _VOICE_FALLBACKS:
        return _VOICE_FALLBACKS[v]
    # en-*-*Neural ids are passed straight through to edge-tts.
    if v.startswith("en-"):
        return v
    return config.DEFAULT_VOICE


def synthesize(text: str, voice_id: str | None, out_path: str) -> str:
    """Synthesize dialogue to an audio file; returns the created path."""
    import edge_tts

    text = (text or "").strip()
    if not text:
        raise ValueError("no dialogue to synthesize")
    voice = resolve_voice(voice_id)
    out_path = os.path.abspath(out_path)
    # edge-tts is async; run within this thread's loop.
    asyncio.run(edge_tts.Communicate(text, voice).save(out_path))
    if not os.path.exists(out_path):
        raise RuntimeError(f"voice synthesis produced no file: {out_path}")
    log.info("synthesized %s bytes via edge-tts (%s)", os.path.getsize(out_path), voice)
    return out_path