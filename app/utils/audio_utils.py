import os
import subprocess
import tempfile


# ── Constants ─────────────────────────────────────────────────────────────────

TARGET_SR = 16000   # faster-whisper requires 16kHz
TARGET_CH = 1       # mono


# ── Main function ─────────────────────────────────────────────────────────────

def webm_to_wav(webm_bytes: bytes) -> str:
    """
    Convert raw WebM audio bytes from browser into a 16kHz mono WAV file.
    Returns the path to the temp WAV file.
    Caller is responsible for deleting the file after use.

    Why ffmpeg instead of soundfile?
      soundfile uses libsndfile under the hood — it only handles uncompressed
      formats (WAV, FLAC, AIFF). It cannot decode WebM/Opus or WebM/Vorbis,
      which is what the browser MediaRecorder sends.
      ffmpeg handles every format the browser can produce.
    """

    # Step 1 — write WebM bytes to a temp file so ffmpeg can read it
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp_webm:
        tmp_webm.write(webm_bytes)
        webm_path = tmp_webm.name

    # Step 2 — create a temp path for the output WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        wav_path = tmp_wav.name

    try:
        # Step 3 — use ffmpeg to convert WebM → 16kHz mono WAV
        #
        # ffmpeg flags explained:
        #   -y              overwrite output file if it exists (temp file was just created)
        #   -i webm_path    input file (the WebM from the browser)
        #   -ar 16000       set output audio sample rate to 16000 Hz (Whisper requirement)
        #   -ac 1           set output to 1 channel (mono)
        #   -f wav          force output format to WAV
        #   wav_path        output file path
        #
        # stdout and stderr are captured so ffmpeg output doesn't pollute app logs.
        # If ffmpeg fails, CalledProcessError is raised with the full error message.

        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", webm_path,
                "-ar", str(TARGET_SR),
                "-ac", str(TARGET_CH),
                "-f", "wav",
                wav_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,         # raises CalledProcessError if ffmpeg exits non-zero
        )

        return wav_path

    except subprocess.CalledProcessError as e:
        # ffmpeg failed — clean up the empty WAV file and re-raise with context
        if os.path.exists(wav_path):
            os.unlink(wav_path)
        raise RuntimeError(
            f"ffmpeg failed to convert WebM to WAV.\n"
            f"ffmpeg stderr: {e.stderr.decode(errors='replace')}"
        ) from e

    finally:
        # Always clean up the input WebM temp file
        if os.path.exists(webm_path):
            os.unlink(webm_path)