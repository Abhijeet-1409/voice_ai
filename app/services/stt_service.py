"""
stt_service.py — Speech-to-Text using Cartesia Ink REST API

No model loads at import time — Cartesia Ink is a cloud REST endpoint.
Each transcription is a single POST request, naturally concurrent across calls.

Input:  16kHz mono WAV file path  (produced by audio_utils.webm_to_wav)
Output: plain transcript string
"""

import httpx

from config.settings import settings

CARTESIA_STT_URL = "https://api.cartesia.ai/stt/transcribe"
CARTESIA_VERSION = "2025-04-16"
STT_MODEL        = "ink-whisper"


def transcribe(wav_path: str) -> str:
    """
    Transcribe a 16kHz mono WAV file using Cartesia Ink.

    Args:
        wav_path: Path to WAV file produced by audio_utils.webm_to_wav

    Returns:
        Plain transcript string. Empty string if nothing recognised.
    """
    try:
        with open(wav_path, "rb") as f:
            audio_bytes = f.read()

        with httpx.Client(timeout=30) as client:
            res = client.post(
                CARTESIA_STT_URL,
                headers={
                    "X-API-Key": settings.cartesia_api_key,
                    "Cartesia-Version": CARTESIA_VERSION,
                },
                files={
                    "clip": ("audio.wav", audio_bytes, "audio/wav"),
                },
                data={
                    "model":       STT_MODEL,
                    "language":    "en",
                    "encoding":    "pcm_s16le",
                    "sample_rate": "16000",
                },
            )

        if res.status_code != 200:
            print(f"[stt_service] Cartesia Ink error [{res.status_code}]: {res.text}")
            return ""

        data = res.json()
        return data.get("transcript", "").strip()

    except Exception as e:
        print(f"[stt_service] Transcription failed: {e}")
        return ""