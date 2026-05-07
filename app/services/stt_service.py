"""
stt_service.py — Speech-to-Text using Cartesia Ink REST API (Batch)

No model loads at import time — Cartesia Ink is a cloud REST endpoint.
Each transcription is a single POST request, naturally concurrent across calls.

API endpoint: POST https://api.cartesia.ai/stt
Auth:         Authorization: Bearer <api_key>
Input:        16kHz mono WAV file path (produced by audio_utils.webm_to_wav)
Output:       plain transcript string
"""

import httpx

from config.settings import settings
from utils.logger import stt_logger

CARTESIA_STT_URL = "https://api.cartesia.ai/stt"
CARTESIA_VERSION = "2026-03-01"
STT_MODEL        = "ink-whisper"


def transcribe(wav_path: str, session_id: str) -> str:
    """
    Transcribe a 16kHz mono WAV file using Cartesia Ink batch endpoint.

    Args:
        wav_path:   Path to WAV file produced by audio_utils.webm_to_wav
        session_id: Session ID for log tracing

    Returns:
        Plain transcript string. Empty string if nothing recognised.
    """
    stt_logger.debug(f"[{session_id}] Sending WAV to Cartesia Ink: {wav_path}")

    try:
        with open(wav_path, "rb") as f:
            audio_bytes = f.read()

        with httpx.Client(timeout=30) as client:
            res = client.post(
                CARTESIA_STT_URL,
                headers={
                    # Cartesia uses Bearer token auth — NOT X-API-Key
                    "Authorization"   : f"Bearer {settings.cartesia_api_key}",
                    "Cartesia-Version": CARTESIA_VERSION,
                },
                # encoding and sample_rate go as query params — not form data
                # only needed when uploading raw PCM without container header.
                # Since we send a proper WAV file (has container header),
                # Cartesia auto-detects format — but we pass them for safety.
                params={
                    "encoding"   : "pcm_s16le",
                    "sample_rate": 16000,
                },
                # model and language go in the multipart form body
                # field name is "file" — NOT "clip"
                files={
                    "file": ("audio.wav", audio_bytes, "audio/wav"),
                },
                data={
                    "model"   : STT_MODEL,
                    "language": "en",
                },
            )

        if res.status_code != 200:
            stt_logger.error(
                f"[{session_id}] Cartesia Ink error [{res.status_code}]: {res.text}"
            )
            return ""

        data = res.json()

        # Response field is "text" — NOT "transcript"
        transcript = data.get("text", "").strip()

        stt_logger.info(f"[{session_id}] Transcript: {transcript}")
        return transcript

    except Exception as e:
        stt_logger.exception(f"[{session_id}] Transcription failed: {e}")
        return ""