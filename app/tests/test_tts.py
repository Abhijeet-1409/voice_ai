# ==============================================================================
# INTELICS VOICE AI AGENT — test_tts.py
# Tests the Kokoro TTS service by generating a sample sentence for all 20 voices.
# Saves each voice as a WAV file so you can listen and compare.
#
# HOW TO RUN (from inside the container):
#   docker exec -it <app_container_name> python tests/test_tts.py
#
# HOW TO RUN (locally):
#   python app/tests/test_tts.py
#   (Kokoro model must already be at /app/tts_models/kokoro-en-v0_19/)
#
# OUTPUT:
#   Creates: app/tests/test_output_voices/
#   Saves:   test_output_voices/test_output_00_af_alloy.wav
#            test_output_voices/test_output_01_af_aoede.wav
#            ... (20 files total, one per voice)
#
# AFTER RUNNING:
#   Copy the test_output_voices/ folder to your machine and listen.
#   Use VLC, QuickTime, or any audio player.
#   Compare voices and confirm sid=3 (af_heart) sounds best for sales calls.
# ==============================================================================

import os
import sys
import time
import struct
import wave
import io


# ------------------------------------------------------------------------------
# MODEL PATH
# Must match the path used in tts_service.py and baked into the Docker image.
# Change this if running locally with a different model location.
# ------------------------------------------------------------------------------
MODEL_DIR = "/app/tts_models/kokoro-en-v0_19"

# ------------------------------------------------------------------------------
# OUTPUT DIRECTORY
# All 20 voice WAV files will be saved here.
# Created automatically if it doesn't exist.
# Excluded from Docker image via .dockerignore (tests/test_output_voices/)
# ------------------------------------------------------------------------------
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_output_voices")

# ------------------------------------------------------------------------------
# TEST SENTENCE
# This is a realistic sales greeting — same style as the agent will use in production.
# A good test sentence covers:
#   - Natural pacing (commas create pauses — good for testing prosody)
#   - Numbers (rupees, percent) — tests how the voice handles digits
#   - A question — tests rising intonation
# ------------------------------------------------------------------------------
TEST_SENTENCE = (
    "Hello! Welcome to Intelics Cloud Solutions. "
    "I can help you find the right cloud plan for your needs. "
    "Are you looking for a Linux or Windows virtual machine today?"
)

# ------------------------------------------------------------------------------
# ALL 20 KOKORO VOICES
# sid = speaker ID (0-19)
# Each entry: (sid, internal_name, notes)
# af_ prefix = American Female
# am_ prefix = American Male
# ------------------------------------------------------------------------------
VOICES = [
    (0,  "af_alloy",   "American female — neutral"),
    (1,  "af_aoede",   "American female — soft"),
    (2,  "af_bella",   "American female — bright"),
    (3,  "af_heart",   "American female — warm ★ CHOSEN FOR PRODUCTION"),
    (4,  "af_jessica", "American female — professional"),
    (5,  "af_kore",    "American female — calm"),
    (6,  "af_nicole",  "American female — friendly"),
    (7,  "af_nova",    "American female — energetic"),
    (8,  "af_river",   "American female — smooth"),
    (9,  "af_sarah",   "American female — clear (good alternative)"),
    (10, "af_sky",     "American female — airy"),
    (11, "am_adam",    "American male — deep"),
    (12, "am_echo",    "American male — resonant"),
    (13, "am_eric",    "American male — natural"),
    (14, "am_fenrir",  "American male — strong"),
    (15, "am_liam",    "American male — conversational (good alternative)"),
    (16, "am_michael", "American male — confident (good alternative)"),
    (17, "am_onyx",    "American male — rich"),
    (18, "am_puck",    "American male — light"),
    (19, "am_santa",   "American male — warm"),
]

# Voice speed — same as production setting in tts_service.py
VOICE_SPEED = 1.0


def check_model_exists() -> bool:
    """
    Checks that the Kokoro model folder exists with all required files.
    The model is downloaded during docker build — this check catches
    cases where someone runs the test before building the image.
    """
    required_files = [
        "model.onnx",
        "voices.bin",
        "tokens.txt",
    ]

    if not os.path.isdir(MODEL_DIR):
        print(f"\n❌  Model directory not found: {MODEL_DIR}")
        print("    The Kokoro model is downloaded during docker build.")
        print("    Run: docker-compose up --build")
        print("    Or manually: wget the kokoro-en-v0_19.tar.bz2 and extract it.\n")
        return False

    for f in required_files:
        full_path = os.path.join(MODEL_DIR, f)
        if not os.path.exists(full_path):
            print(f"\n❌  Missing model file: {full_path}")
            print("    The model may be incomplete. Try re-running docker build.\n")
            return False

    # Check espeak-ng-data folder (needed for phoneme processing)
    espeak_dir = os.path.join(MODEL_DIR, "espeak-ng-data")
    if not os.path.isdir(espeak_dir):
        print(f"\n❌  Missing espeak-ng-data folder: {espeak_dir}")
        print("    Kokoro needs this for phoneme conversion.")
        print("    Also ensure espeak-ng is installed: apt-get install espeak-ng\n")
        return False

    print(f"✅  Model found at: {MODEL_DIR}")
    return True


def load_tts_model():
    """
    Loads the Kokoro TTS model using sherpa-onnx.

    OfflineTtsKokoroModelConfig — specific to Kokoro ONNX models.
    Do NOT use OfflineTtsVitsModelConfig (that was for the old Piper TTS).

    num_threads=2:
      How many CPU threads sherpa-onnx uses for inference.
      2 is a good balance — enough parallelism without starving other services.

    The TTS model itself is speaker-independent — all 20 voices come from
    the same model.onnx file. The voices.bin file contains the 20 speaker
    embedding vectors. We pass the sid (speaker ID) at synthesis time.
    """
    print("\n⏳  Loading Kokoro TTS model...")
    t0 = time.time()

    import sherpa_onnx

    model_config = sherpa_onnx.OfflineTtsKokoroModelConfig(
        model   = os.path.join(MODEL_DIR, "model.onnx"),
        voices  = os.path.join(MODEL_DIR, "voices.bin"),
        tokens  = os.path.join(MODEL_DIR, "tokens.txt"),
        data_dir= os.path.join(MODEL_DIR, "espeak-ng-data"),
    )

    tts_config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            kokoro=model_config,
            num_threads=2,
            debug=False,        # Set True to see sherpa-onnx internal logs
        ),
        rule_fsts="",           # No custom pronunciation rules
        max_num_sentences=1,    # Process one sentence at a time
    )

    tts = sherpa_onnx.OfflineTts(tts_config)
    elapsed = time.time() - t0
    print(f"✅  Model loaded in {elapsed:.2f}s")
    print(f"    Sample rate: {tts.sample_rate} Hz")
    return tts


def samples_to_wav_bytes(samples, sample_rate: int) -> bytes:
    """
    Converts raw float32 audio samples to WAV bytes in memory.
    Same approach used in tts_service.py — no audio files written to disk in production.

    Parameters:
      samples     — list or numpy array of float32 values in range [-1.0, 1.0]
      sample_rate — typically 22050 Hz for Kokoro

    WAV format:
      16-bit PCM (int16) — standard, compatible with all audio players and browsers
      Mono — single channel
      Little-endian byte order

    struct.pack('<h', ...) converts each float32 sample to a 16-bit signed integer.
    Clamping to [-32767, 32767] prevents clipping artifacts from any out-of-range samples.
    """
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)           # Mono
        wf.setsampwidth(2)           # 2 bytes = 16-bit
        wf.setframerate(sample_rate) # e.g. 22050 Hz

        # Convert float32 samples [-1.0, 1.0] → int16 [-32767, 32767]
        pcm_data = b""
        for s in samples:
            clamped = max(-1.0, min(1.0, s))          # Clamp to [-1.0, 1.0]
            int_sample = int(clamped * 32767)          # Scale to int16 range
            pcm_data += struct.pack('<h', int_sample)  # Pack as little-endian int16

        wf.writeframes(pcm_data)

    return buf.getvalue()


def synthesize_voice(tts, sid: int, voice_name: str, sentence: str) -> tuple:
    """
    Generates audio for a single voice and returns timing + WAV bytes.

    tts.generate() returns an object with:
      .samples     — list of float32 audio samples
      .sample_rate — audio sample rate in Hz (typically 22050 for Kokoro)

    sid = speaker ID (0-19) — selects which of the 20 voices to use
    speed = 1.0 — natural speed; <1.0 = slower, >1.0 = faster
    """
    t0 = time.time()
    audio = tts.generate(sentence, sid=sid, speed=VOICE_SPEED)
    elapsed = time.time() - t0

    wav_bytes = samples_to_wav_bytes(audio.samples, audio.sample_rate)
    duration_seconds = len(audio.samples) / audio.sample_rate

    return elapsed, wav_bytes, duration_seconds, audio.sample_rate


def save_wav(wav_bytes: bytes, sid: int, voice_name: str) -> str:
    """
    Saves WAV bytes to the output directory.
    Filename format: test_output_XX_voicename.wav
    XX = zero-padded sid (00, 01, ... 19) for alphabetical sort order.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = f"test_output_{sid:02d}_{voice_name}.wav"
    filepath = os.path.join(OUTPUT_DIR, filename)

    with open(filepath, "wb") as f:
        f.write(wav_bytes)

    return filepath


def main():
    print("=" * 65)
    print("INTELICS VOICE AI — TTS SERVICE TEST")
    print("Kokoro af_heart (sid=3) | sherpa-onnx | All 20 voices")
    print("=" * 65)

    # --- Step 1: Check model files exist ---
    if not check_model_exists():
        sys.exit(1)

    # --- Step 2: Load the TTS model (done once — shared across all 20 voices) ---
    tts = load_tts_model()

    print(f"\n📝  Test sentence:")
    print(f'    "{TEST_SENTENCE}"')
    print(f"\n🎙️   Generating {len(VOICES)} voices → {OUTPUT_DIR}/\n")
    print(f"{'─'*65}")
    print(f"{'SID':<5} {'Voice':<15} {'Time':>7} {'Duration':>10} {'Size':>8}   Notes")
    print(f"{'─'*65}")

    total_t0 = time.time()
    success_count = 0
    production_voice_info = None

    # --- Step 3: Generate audio for each of the 20 voices ---
    for sid, voice_name, notes in VOICES:
        try:
            elapsed, wav_bytes, duration_sec, sample_rate = synthesize_voice(
                tts, sid, voice_name, TEST_SENTENCE
            )

            # Save the WAV file to disk
            filepath = save_wav(wav_bytes, sid, voice_name)

            size_kb = len(wav_bytes) / 1024
            marker = " ★" if sid == 3 else ""   # Highlight the production voice

            print(
                f"  {sid:02d}   {voice_name:<15} {elapsed:>5.2f}s  "
                f"{duration_sec:>7.2f}s  {size_kb:>6.1f}KB{marker}"
            )

            success_count += 1

            # Store production voice stats for summary
            if sid == 3:
                production_voice_info = {
                    "elapsed": elapsed,
                    "duration": duration_sec,
                    "size_kb": size_kb,
                    "sample_rate": sample_rate,
                }

        except Exception as e:
            print(f"  {sid:02d}   {voice_name:<15}  ❌ FAILED: {e}")

    total_elapsed = time.time() - total_t0

    # --- Step 4: Print summary ---
    print(f"{'─'*65}")
    print(f"\n✅  {success_count}/{len(VOICES)} voices generated in {total_elapsed:.1f}s")
    print(f"    Output directory: {OUTPUT_DIR}/")

    if production_voice_info:
        print(f"\n★   Production voice (sid=3, af_heart):")
        print(f"    Synthesis time : {production_voice_info['elapsed']:.2f}s")
        print(f"    Audio duration : {production_voice_info['duration']:.2f}s")
        print(f"    File size      : {production_voice_info['size_kb']:.1f} KB")
        print(f"    Sample rate    : {production_voice_info['sample_rate']} Hz")

        # Check if synthesis is faster than real-time (it should be)
        rtf = production_voice_info['elapsed'] / production_voice_info['duration']
        if rtf < 1.0:
            print(f"    Real-time factor: {rtf:.2f}x ✅ faster than real-time")
        else:
            print(f"    Real-time factor: {rtf:.2f}x ⚠️  slower than real-time")
            print(f"    This may cause lag in production. Consider:")
            print(f"      - Increasing num_threads in tts_service.py")
            print(f"      - Using a faster host machine")

    print(f"\n📂  To listen: copy {OUTPUT_DIR}/ to your machine")
    print(f"    scp <container>:{OUTPUT_DIR}/ ./test_voices/")
    print(f"    Or: docker cp <container>:{OUTPUT_DIR} ./test_voices/")
    print()


if __name__ == "__main__":
    main()