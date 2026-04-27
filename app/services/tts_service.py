import io

import numpy as np
import sherpa_onnx
import soundfile as sf

# ── Constants ─────────────────────────────────────────────────────────────────

VOICE_SID   = 3      # af_heart — warm friendly female voice
VOICE_SPEED = 1.0    # normal speaking speed
MAX_CHARS   = 800    # max characters per synthesis call

TTS_MODEL_DIR = "/app/tts_models/kokoro-en-v0_19"


# ── Model loads ONCE at import time ───────────────────────────────────────────

print("[tts_service] Loading Kokoro TTS model...")

_tts = sherpa_onnx.OfflineTts(
    sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                model    = f"{TTS_MODEL_DIR}/model.onnx",
                voices   = f"{TTS_MODEL_DIR}/voices.bin",
                tokens   = f"{TTS_MODEL_DIR}/tokens.txt",
                data_dir = f"{TTS_MODEL_DIR}/espeak-ng-data",
            ),
            num_threads=2,
        ),
        rule_fsts="",
    )
)

print("[tts_service] Model ready.")


# ── Synthesize ────────────────────────────────────────────────────────────────

def synthesize(text: str) -> bytes:
    """
    Convert text to WAV audio bytes using Kokoro TTS.

    Args:
        text: Sentence to synthesize. Trimmed to MAX_CHARS if needed.

    Returns:
        WAV audio as raw bytes ready to send over WebSocket.
        Returns empty bytes if synthesis fails.
    """
    if not text or not text.strip():
        return b""

    # Trim to max chars to avoid very long synthesis calls
    text = text.strip()[:MAX_CHARS]

    try:
        # Generate audio samples — returns a sherpa_onnx.GeneratedAudio object
        audio = _tts.generate(
            text,
            sid=VOICE_SID,
            speed=VOICE_SPEED,
        )

        # Convert samples to float32 numpy array
        samples = np.array(audio.samples, dtype=np.float32)

        # Write to in-memory WAV buffer — no disk write needed
        buffer = io.BytesIO()
        sf.write(buffer, samples, audio.sample_rate, format="WAV", subtype="PCM_16")
        buffer.seek(0)

        return buffer.read()

    except Exception as e:
        print(f"[tts_service] Synthesis failed: {e}")
        return b""