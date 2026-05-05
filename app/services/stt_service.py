"""
stt_service.py — Speech-to-Text using Cartesia Ink REST API

No model loads at import time — Cartesia Ink is a cloud REST endpoint.
Each transcription is a single POST request, naturally concurrent across calls.

Input:  16kHz mono WAV file path  (produced by audio_utils.webm_to_wav)
Output: plain transcript string
"""

import httpx

from config.settings import settings
from utils.logger import stt_logger

CARTESIA_STT_URL = "https://api.cartesia.ai/stt/transcribe"
CARTESIA_VERSION = "2025-04-16"
STT_MODEL        = "ink-whisper"


def transcribe(wav_path: str, session_id: str) -> str:
    """
    Transcribe a 16kHz mono WAV file using Cartesia Ink.

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
                    "X-API-Key"        : settings.cartesia_api_key,
                    "Cartesia-Version" : CARTESIA_VERSION,
                },
                files={
                    "clip": ("audio.wav", audio_bytes, "audio/wav"),
                },
                data={
                    "model"      : STT_MODEL,
                    "language"   : "en",
                    "encoding"   : "pcm_s16le",
                    "sample_rate": "16000",
                },
            )

        if res.status_code != 200:
            stt_logger.error(f"[{session_id}] Cartesia Ink error [{res.status_code}]: {res.text}")
            return ""

        data       = res.json()
        transcript = data.get("transcript", "").strip()

        stt_logger.info(f"[{session_id}] Transcript: {transcript}")
        return transcript

    except Exception as e:
        stt_logger.exception(f"[{session_id}] Transcription failed: {e}")
        return ""