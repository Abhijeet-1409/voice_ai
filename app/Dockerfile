# ==============================================================================
# INTELICS VOICE AI AGENT — Dockerfile
# Build context: app/ folder (docker-compose sets build: ./app)
#
# STT: Cartesia Ink   — cloud REST API, no model download needed
# TTS: Cartesia Sonic — cloud WebSocket API, no model download needed
# RAG: all-MiniLM-L6-v2 — volume mounted from host (unchanged)
#
# What this Dockerfile does:
#   1. Installs system dependencies (ffmpeg only)
#   2. Installs Python packages
#   3. Copies app code
#
# Models on your host machine (downloaded once manually):
#   voice_ai/rag_models/all-MiniLM-L6-v2/  → mounted to /app/rag_models/
#   voice_ai/data/rate_card.xlsx            → mounted to /app/data/
#
# NOTE: tts_models/ folder and faster-whisper are no longer needed.
#       espeak-ng and torch are no longer needed.
#       Docker builds are now significantly faster and smaller.
# ==============================================================================

FROM python:3.11-slim

WORKDIR /app

# ------------------------------------------------------------------------------
# SYSTEM DEPENDENCIES
#   ffmpeg — converts WebM audio from browser → 16kHz mono WAV
#            needed by audio_utils.py before sending audio to Cartesia Ink
#
# espeak-ng is no longer needed (was for Kokoro TTS — now using Cartesia Sonic)
# wget, git, git-lfs are no longer needed (no models downloaded at build time)
# ------------------------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------------------
# PYTHON DEPENDENCIES
# Copy requirements.txt first so Docker caches this layer separately.
# Code changes won't re-trigger pip install unless requirements.txt changes.
#
# No torch install needed — sentence-transformers for RAG uses CPU numpy only.
# cartesia[websockets] handles both Ink (REST) and Sonic (WebSocket SDK).
# ------------------------------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------------------------
# APPLICATION CODE
# Copied last so code edits don't invalidate the pip install cache above.
# .dockerignore excludes __pycache__, *.pyc, .env
# ------------------------------------------------------------------------------
COPY . .

# ------------------------------------------------------------------------------
# EXPOSE + STARTUP
# ------------------------------------------------------------------------------
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]