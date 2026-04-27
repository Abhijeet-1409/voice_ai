# ==============================================================================
# INTELICS VOICE AI AGENT — test_stt.py
# Tests the faster-whisper Speech-to-Text service.
#
# HOW TO RUN (from inside the container):
#   docker exec -it <app_container_name> python tests/test_stt.py
#
# HOW TO RUN (locally, outside Docker — if you have faster-whisper installed):
#   python app/tests/test_stt.py
#
# BEFORE RUNNING:
#   1. Set AUDIO_FILE below to the path of a real .wav or .mp3 file.
#   2. The audio should contain clear English speech — a sentence or two is enough.
#   3. If running locally, the model will download to ~/.cache/huggingface/
#      (same place as in Docker — first run takes ~30 seconds to download).
# ==============================================================================

import sys
import os
import time

# ------------------------------------------------------------------------------
# AUDIO FILE TO TRANSCRIBE
# Change this path to point to any audio file on your machine.
# Supported formats: WAV, MP3, M4A, FLAC (faster-whisper handles all via ffmpeg)
# For best results: use a WAV file recorded at 16kHz mono (same as production).
# If you don't have an audio file handy:
#   - Record one quickly using your phone's voice memo app and transfer it
#   - Or use any short YouTube clip downloaded as MP3
# ------------------------------------------------------------------------------
AUDIO_FILE = "/app/tests/sample_audio.wav"   # ← change this path


def check_audio_file(path: str) -> bool:
    """
    Checks if the audio file exists before we bother loading the model.
    Loading the model takes ~5 seconds — no point doing it if the file isn't there.
    """
    if not os.path.exists(path):
        print(f"\n❌  Audio file not found: {path}")
        print("    Please update AUDIO_FILE at the top of this script.")
        print("    The file should be a WAV, MP3, or M4A containing English speech.\n")
        return False

    file_size_kb = os.path.getsize(path) / 1024
    print(f"✅  Audio file found: {path}")
    print(f"    Size: {file_size_kb:.1f} KB")
    return True


def load_model():
    """
    Loads the faster-whisper WhisperModel.
    In Docker: model is already cached at ~/.cache/huggingface/ from build time → instant.
    Locally: downloads ~145MB on first run, then cached for all future runs.

    Model config:
      base.en   — English-only model, faster and more accurate than multilingual "base"
      cpu       — we run on CPU (no GPU in this POC)
      int8      — 8-bit integer quantization — 2x faster than float32, minimal accuracy loss
    """
    print("\n⏳  Loading faster-whisper model (base.en, int8)...")
    t0 = time.time()

    from faster_whisper import WhisperModel
    model = WhisperModel("base.en", device="cpu", compute_type="int8")

    elapsed = time.time() - t0
    print(f"✅  Model loaded in {elapsed:.2f}s")
    return model


def run_transcription(model, audio_path: str):
    """
    Runs transcription on the provided audio file and prints detailed results.

    faster-whisper returns:
      segments  — list of timed transcript segments (each has start, end, text)
      info      — metadata: detected language, language probability, audio duration

    We join all segment texts into a single string — same as stt_service.py does
    in production. This lets us verify the output matches what the real service produces.

    beam_size=5:
      Controls how many candidate sequences the model explores at each step.
      Higher = more accurate but slower. 5 is a good balance for production.

    language="en":
      Forces English transcription — skips language detection step.
      Slightly faster and more accurate when we know the language upfront.
    """
    print(f"\n⏳  Transcribing: {audio_path}")
    print("    (beam_size=5, language=en — same settings as production)")
    t0 = time.time()

    segments, info = model.transcribe(audio_path, beam_size=5, language="en")

    # Collect all segment texts — segments is a generator, we must exhaust it
    # before measuring time (timing the generator creation is meaningless)
    segment_list = list(segments)
    elapsed = time.time() - t0

    return segment_list, info, elapsed


def print_results(segment_list, info, elapsed):
    """
    Prints the full transcription output in a readable format.
    Shows: full transcript, timing per segment, language confidence, audio duration.
    """
    print(f"\n{'='*60}")
    print("TRANSCRIPTION RESULTS")
    print(f"{'='*60}")

    # --- Full transcript (what stt_service.py returns in production) ---
    full_transcript = " ".join(seg.text.strip() for seg in segment_list)
    print(f"\n📝  Full transcript:\n    \"{full_transcript}\"")

    # --- Per-segment breakdown ---
    # In production we only use the full joined text.
    # But during testing it's useful to see each segment with its timestamps.
    # This helps identify if there are long pauses or silence being transcribed oddly.
    if len(segment_list) > 1:
        print(f"\n🔍  Segment breakdown ({len(segment_list)} segments):")
        for i, seg in enumerate(segment_list, 1):
            print(f"    [{i}] {seg.start:.1f}s → {seg.end:.1f}s : \"{seg.text.strip()}\"")

    # --- Performance metrics ---
    print(f"\n⚡  Performance:")
    print(f"    Transcription time : {elapsed:.2f}s")
    print(f"    Audio duration     : {info.duration:.2f}s")

    # Real-time factor: how long transcription took vs audio length
    # RTF < 1.0 means faster-than-realtime (ideal)
    # RTF > 1.0 means slower than realtime (would cause lag in production)
    rtf = elapsed / info.duration if info.duration > 0 else 0
    rtf_label = "✅ faster than real-time" if rtf < 1.0 else "⚠️  slower than real-time"
    print(f"    Real-time factor   : {rtf:.2f}x ({rtf_label})")

    # --- Language detection info ---
    # Even though we forced language="en", info still shows detected language
    # and the model's confidence. Should be "en" with high probability.
    print(f"\n🌐  Language detection:")
    print(f"    Detected language  : {info.language}")
    print(f"    Confidence         : {info.language_probability:.1%}")

    if info.language != "en":
        print(f"    ⚠️  Warning: detected '{info.language}' — expected 'en'")
        print(f"       Check if your audio file contains English speech.")

    print(f"\n{'='*60}")

    # --- Final verdict ---
    if full_transcript.strip():
        print("✅  STT service is working correctly.")
        print("    The transcript above is what websocket_handler.py will receive")
        print("    on each customer exchange in production.")
    else:
        print("❌  Transcription returned empty text.")
        print("    Possible causes:")
        print("      - Audio file contains only silence")
        print("      - Audio file is corrupt or in an unsupported format")
        print("      - Very short audio clip (< 0.5 seconds)")

    print(f"{'='*60}\n")


def main():
    print("=" * 60)
    print("INTELICS VOICE AI — STT SERVICE TEST")
    print("faster-whisper base.en | int8 | CPU")
    print("=" * 60)

    # Step 1: Check the audio file exists before loading the heavy model
    if not check_audio_file(AUDIO_FILE):
        sys.exit(1)

    # Step 2: Load the Whisper model
    model = load_model()

    # Step 3: Run transcription
    segment_list, info, elapsed = run_transcription(model, AUDIO_FILE)

    # Step 4: Print detailed results
    print_results(segment_list, info, elapsed)


if __name__ == "__main__":
    main()