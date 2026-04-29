# ==============================================================================
# INTELICS VOICE AI AGENT — Dockerfile
# Build context: app/ folder (docker-compose sets build: ./app)
#
# Models are NOT downloaded during build — they are mounted as volumes
# from your host machine via docker-compose.yml at runtime.
#
# What this Dockerfile does:
#   1. Installs system dependencies (ffmpeg, espeak-ng)
#   2. Installs Python packages
#   3. Pre-warms faster-whisper base.en (145MB — small, fast, auto-caches)
#   4. Copies app code
#
# Models on your host machine (downloaded once manually):
#   voice_ai/tts_models/kokoro-en-v0_19/     → mounted to /app/tts_models/
#   voice_ai/rag_models/all-MiniLM-L6-v2/    → mounted to /app/rag_models/
#   voice_ai/data/rate_card.xlsx             → mounted to /app/data/
# ==============================================================================

# ------------------------------------------------------------------------------
# STAGE: Base image
# python:3.11-slim is a minimal Debian-based image with Python 3.11 pre-installed.
# We use slim (not full) to keep image size smaller — we only add what we need.
# ------------------------------------------------------------------------------
FROM python:3.11-slim

# ------------------------------------------------------------------------------
# Set the working directory inside the container.
# All subsequent commands (COPY, RUN, CMD) use /app as the base path.
# When the container runs, the process also starts in /app.
# ------------------------------------------------------------------------------
WORKDIR /app

# ------------------------------------------------------------------------------
# SYSTEM DEPENDENCIES
#   ffmpeg    — needed by pydub (used in audio_utils.py) to decode WebM audio
#               browsers send audio in WebM format — ffmpeg converts it to WAV
#   espeak-ng — needed by Kokoro TTS for phoneme processing (text → phonemes)
#               sherpa-onnx calls espeak-ng internally to convert text to sounds
#
# wget, git, git-lfs are NOT needed here anymore — models are volume mounted,
# not downloaded during build.
#
# We clean up apt lists after install to reduce image size.
# ------------------------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    ffmpeg \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------------------
# PYTHON DEPENDENCIES
# We copy requirements.txt first (before copying app code) — this is important!
# Docker caches each layer. If only your code changes (not requirements.txt),
# Docker reuses the cached pip install layer → much faster rebuilds.
# If we copied all code first, every code change would re-trigger pip install.
# ------------------------------------------------------------------------------

# Step 1: Install CPU-only torch FIRST from PyTorch's own index.
# This prevents sentence-transformers from pulling the 530MB GPU build from PyPI.
RUN pip install --no-cache-dir \
    torch==2.3.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Step 2: Install all remaining Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------------------------
# MODEL: faster-whisper base.en — Speech-to-Text (only model downloaded at build)
# Unlike Kokoro and all-MiniLM-L6-v2, faster-whisper is small (145MB) and
# manages its own versioned cache format via HuggingFace Hub.
# It's safer and simpler to let the library handle its own download.
#
# We pre-warm it here so the model is cached at ~/.cache/huggingface/ inside
# the image. On first real customer call, model is already loaded — no delay.
#
# stt_service.py loads: WhisperModel("base.en", device="cpu", compute_type="int8")
# ------------------------------------------------------------------------------
ENV CURL_CA_BUNDLE=""
ENV REQUESTS_CA_BUNDLE=""
ENV HF_HUB_DISABLE_SSL_VERIFICATION=1

RUN python -c "from faster_whisper import WhisperModel; WhisperModel('base.en', device='cpu', compute_type='int8')"

# ------------------------------------------------------------------------------
# APPLICATION CODE
# COPY . . copies everything from build context (app/) into /app in the image.
# Done LAST so code changes don't invalidate the pip install cache layer above.
# If this were first, every code edit would re-trigger pip install.
#
# .dockerignore (in app/ folder) prevents __pycache__, *.pyc from being copied.
# ------------------------------------------------------------------------------
COPY . .

# ------------------------------------------------------------------------------
# EXPOSE port 8000
# This is documentation — tells Docker "this container listens on 8000".
# The actual port binding happens in docker-compose.yml (ports: 8000:8000).
# ------------------------------------------------------------------------------
EXPOSE 8000

# ------------------------------------------------------------------------------
# STARTUP COMMAND
# Runs the FastAPI app using uvicorn when the container starts.
#   main:app        → file main.py, FastAPI instance named "app"
#   --host 0.0.0.0  → listen on all interfaces (not just localhost inside container)
#   --port 8000     → match the EXPOSE above
#
# When main.py starts, it imports stt_service, tts_service, rag_service at top.
# This triggers all 3 models to load into memory immediately at startup —
# NOT on the first customer call. First call is instant.
#
# Kokoro and all-MiniLM-L6-v2 are loaded from mounted volumes at runtime:
#   /app/tts_models/  → voice_ai/tts_models/ on your machine
#   /app/rag_models/  → voice_ai/rag_models/ on your machine
# ------------------------------------------------------------------------------
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]