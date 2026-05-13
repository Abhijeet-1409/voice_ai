FROM python:3.11-slim

WORKDIR /app

# ------------------------------------------------------------------------------
# SYSTEM DEPENDENCIES
#   ffmpeg — converts WebM audio from browser → 16kHz mono WAV
#   wget   — downloads VAD static assets at build time
# ------------------------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    ffmpeg \
    wget \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------------------
# PYTHON DEPENDENCIES
# ------------------------------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------------------------
# APPLICATION CODE
# ------------------------------------------------------------------------------
COPY . .

# ------------------------------------------------------------------------------
# VAD STATIC ASSETS
# Downloaded once at build time into static/ so FastAPI can serve them.
# Browser loads main bundle from CDN but worklet + model + wasm must be
# served from same origin due to browser cross-origin restrictions.
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# EXPOSE + STARTUP
# ------------------------------------------------------------------------------
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]