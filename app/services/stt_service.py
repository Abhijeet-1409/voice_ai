from faster_whisper import WhisperModel

# ── Model loads ONCE at import time ───────────────────────────────────────────
# main.py imports this module at startup so the model is ready
# before the first customer call arrives. No download delay —
# model is pre-warmed into ~/.cache/huggingface/ during docker build.

print("[stt_service] Loading faster-whisper model...")

_model = WhisperModel(
    "base.en",
    device="cpu",
    compute_type="int8",
)

print("[stt_service] Model ready.")


# ── Transcribe ────────────────────────────────────────────────────────────────

def transcribe(wav_path: str) -> str:
    """
    Transcribe a 16kHz mono WAV file to text.

    Args:
        wav_path: Path to the WAV file (produced by audio_utils.webm_to_wav)

    Returns:
        Plain transcript string. Empty string if nothing detected.
    """
    segments, _ = _model.transcribe(
        wav_path,
        beam_size=5,
        language="en",
    )

    transcript = " ".join(segment.text.strip() for segment in segments)
    return transcript.strip()