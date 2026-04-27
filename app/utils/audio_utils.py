import os
import tempfile

import numpy as np
import resampy
import soundfile as sf


# ── Constants ─────────────────────────────────────────────────────────────────

TARGET_SR   = 16000   # faster-whisper requires 16kHz
TARGET_CH   = 1       # mono
TARGET_SUBTYPE = "PCM_16"


# ── Main function ─────────────────────────────────────────────────────────────

def webm_to_wav(webm_bytes: bytes) -> str:
    """
    Convert raw WebM audio bytes from browser into a 16kHz mono WAV file.
    Returns the path to the temp WAV file.
    Caller is responsible for deleting the file after use.
    """

    # Step 1 — write WebM bytes to a temp file so soundfile can read it
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp_webm:
        tmp_webm.write(webm_bytes)
        webm_path = tmp_webm.name

    try:
        # Step 2 — read audio data and sample rate from WebM
        audio, sr = sf.read(webm_path)

        # Step 3 — convert stereo to mono by averaging channels
        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        # Step 4 — resample to 16kHz if needed
        if sr != TARGET_SR:
            audio = resampy.resample(audio, sr, TARGET_SR)

        # Step 5 — write to a temp WAV file and return the path
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            wav_path = tmp_wav.name

        sf.write(wav_path, audio, TARGET_SR, subtype=TARGET_SUBTYPE)
        return wav_path

    finally:
        # Always clean up the temp WebM file
        os.unlink(webm_path)