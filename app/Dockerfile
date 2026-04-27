# ==============================================================================
# INTELICS VOICE AI AGENT — Dockerfile
# Build context: app/ folder (docker-compose sets build: ./app)
# This image downloads ALL models at build time — no downloads on first call.
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
# We install these with apt-get before anything else because:
#   wget     — needed to download the Kokoro TTS model from GitHub
#   git      — needed to clone the all-MiniLM-L6-v2 embedding model from HuggingFace
#   git-lfs  — HuggingFace stores large model files (.onnx, .bin) using Git LFS
#              without this, git clone gives you pointer files, not actual weights
#   ffmpeg   — needed by pydub (used in audio_utils.py) to decode WebM audio
#              browsers send audio in WebM format — ffmpeg converts it to WAV
#   espeak-ng — needed by Kokoro TTS for phoneme processing (text → phonemes)
#               sherpa-onnx calls espeak-ng internally to convert text to sounds
#
# We clean up apt lists after install to reduce image size.
# ------------------------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    wget \
    git \
    git-lfs \
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
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------------------------
# MODEL 1: Kokoro TTS — af_heart voice (sid=3)
# Downloaded from sherpa-onnx GitHub releases as a .tar.bz2 archive.
# Extracted to /app/tts_models/kokoro-en-v0_19/
# Removed the archive after extraction to save disk space in the image.
#
# After extraction, the folder contains:
#   model.onnx         — the neural TTS model weights
#   voices.bin         — all 20 voice embeddings (af_heart is sid=3)
#   tokens.txt         — phoneme-to-token mapping
#   espeak-ng-data/    — espeak language data used internally by Kokoro
#
# tts_service.py points to these exact paths at /app/tts_models/kokoro-en-v0_19/
# ------------------------------------------------------------------------------
RUN mkdir -p /app/tts_models && \
    cd /app/tts_models && \
    wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-en-v0_19.tar.bz2 && \
    tar xf kokoro-en-v0_19.tar.bz2 && \
    rm kokoro-en-v0_19.tar.bz2

# ------------------------------------------------------------------------------
# MODEL 2: all-MiniLM-L6-v2 — Sentence Embedding Model for RAG
# Cloned from HuggingFace via git-lfs.
# git lfs install activates LFS so actual weight files are pulled (not pointers).
# Cloned to /app/rag_models/all-MiniLM-L6-v2/
#
# rag_service.py loads it with:
#   SentenceTransformer("/app/rag_models/all-MiniLM-L6-v2")
#
# ~90MB total. Used to embed pricing chunks and customer queries for semantic search.
# ------------------------------------------------------------------------------
RUN git lfs install && \
    git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 \
    /app/rag_models/all-MiniLM-L6-v2

# ------------------------------------------------------------------------------
# MODEL 3: faster-whisper base.en — Speech-to-Text Model
# Unlike the others, faster-whisper downloads its own model via HuggingFace Hub.
# We trigger the download by loading the model once at build time (pre-warming).
# The model is cached at ~/.cache/huggingface/ inside the container.
#
# Why pre-warm instead of manual download?
#   faster-whisper manages its own versioned cache format — it's safer to let
#   the library handle the download than to manually place files.
#
# On first real customer call, the model is already in cache → instant load.
# stt_service.py loads: WhisperModel("base.en", device="cpu", compute_type="int8")
# ------------------------------------------------------------------------------
RUN python -c "from faster_whisper import WhisperModel; \
    WhisperModel('base.en', device='cpu', compute_type='int8')"

# ------------------------------------------------------------------------------
# RATE CARD DATA
# Copies the Intelics pricing Excel file from voice_ai/data/ into the image.
# Note the path: ../data — this is RELATIVE TO THE BUILD CONTEXT (app/ folder)
# Since build context is app/, ../data means voice_ai/data/
#
# Inside the image, it lands at /app/data/rate_card.xlsx
# rag_service.py reads: /app/data/rate_card.xlsx
#
# If prices change → update rate_card.xlsx → rebuild the image.
# This is intentional: pricing is baked in, not a live mount.
# ------------------------------------------------------------------------------
COPY ../data /app/data

# ------------------------------------------------------------------------------
# APPLICATION CODE
# COPY . . means: copy everything from the build context (app/) into /app in image.
# This is done LAST so that code changes don't invalidate model download cache layers.
# If this were first, every code edit would re-download all 3 models.
#
# .dockerignore (in app/ folder) prevents __pycache__, *.pyc, .env from being copied.
# ------------------------------------------------------------------------------
COPY . .

# ------------------------------------------------------------------------------
# EXPOSE port 8000
# This is documentation — it tells Docker "this container listens on 8000".
# The actual port binding happens in docker-compose.yml (ports: 8000:8000).
# ------------------------------------------------------------------------------
EXPOSE 8000

# ------------------------------------------------------------------------------
# STARTUP COMMAND
# Runs the FastAPI app using uvicorn when the container starts.
#   main:app   → file main.py, FastAPI instance named "app"
#   --host 0.0.0.0  → listen on all interfaces (not just localhost inside container)
#   --port 8000     → match the EXPOSE above
#
# When main.py starts, it imports stt_service, tts_service, rag_service at top level.
# This triggers all 3 models to load into memory immediately at startup —
# NOT on the first customer call. First call is instant.
# ------------------------------------------------------------------------------
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]