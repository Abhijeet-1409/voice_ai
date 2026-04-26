# Intelics Voice AI Agent

A real-time AI voice agent that handles inbound sales calls via browser. Built with FastAPI, WebSocket, faster-whisper STT, Gemini 2.5 Flash LLM, and Kokoro TTS — fully local audio processing with zero cloud cost for speech, STT, or RAG.

---

## Architecture

```
Browser (Client)
─────────────────────────────────────────────────────
  MediaDevices API     →  Captures mic audio (echo cancellation built-in)
  MediaRecorder API    →  Encodes audio into chunks
  WebSocket            →  Sends audio chunks to server (full duplex)
  Web Audio API        →  Plays streamed audio response
  Barge-in logic       →  Monitors mic volume, sends interrupt signal

          ↕  WebSocket (full duplex, persistent connection)

FastAPI Server
─────────────────────────────────────────────────────
  faster-whisper       →  STT — transcribes audio locally (~200ms)
  RAG pipeline         →  Hybrid keyword + embedding search on rate card Excel
  Gemini 2.5 Flash     →  LLM — streams reply sentence by sentence
  Kokoro TTS           →  TTS — generates audio per sentence (~300ms)
  af_heart (sid=3)     →  Warm friendly female voice

          ↕

Session & Storage
─────────────────────────────────────────────────────
  Redis                →  Session store — exchanges stored by session_id
  PostgreSQL           →  Persistent storage — calls + exchanges tables
  Gmail SMTP           →  Email notification on call end

Call End (browser disconnects WebSocket)
─────────────────────────────────────────────────────
  WebSocketDisconnect  →  Save full session to PostgreSQL
                       →  Send email summary to sales team
                       →  Delete session from Redis
```

---

## Barge-in Handling

The browser always monitors mic even while agent audio is playing:

```
MediaDevices echo cancellation    →  removes agent audio echo automatically
300ms mute window                 →  blocks interrupt detection at audio start
500ms minimum duration filter     →  ignores short words like "hmm", "yeah"
Sustained volume trend check      →  confirms real human voice
{"type": "interrupt"} over WS     →  server stops TTS stream immediately
```

---

## Session Management

Each call gets a unique `session_id` at start. All exchanges are stored in Redis during the call. On disconnect, the full session is dumped to PostgreSQL once and Redis is cleared.

```
session_id  →  generated at call start (e.g. sess_a3f9b2c1d4e5)
exchanges   →  stored in Redis as JSON list during conversation
call ends   →  Redis → PostgreSQL (one write) → Redis cleared
```

---

## File Structure

```
voice_ai/                            ← root — config, compose, data, docs only
│
├── .env                             ← all secrets and config (never in Docker image)
├── sample.env                       ← example env file (no real secrets)
├── docker-compose.yml               ← runs app + Redis + PostgreSQL
│
├── data/
│   └── rate_card.xlsx               ← Intelics pricing Excel — 6 sheets
│                                       copied into image via Dockerfile COPY
│
├── docs/
│   ├── architecture.md              ← system architecture overview
│   ├── setup.md                     ← installation and setup guide
│   ├── websocket_protocol.md        ← WebSocket message types and flow
│   ├── session_management.md        ← Redis session lifecycle
│   ├── api.md                       ← API endpoints reference
│   └── rag.md                       ← RAG design, chunk structure, category map
│
└── app/                             ← ALL CODE LIVES HERE — copied into Docker image
    │
    ├── main.py                      ← FastAPI app + WebSocket endpoint + startup
    ├── requirements.txt             ← all Python dependencies
    ├── Dockerfile                   ← builds image — downloads all 3 models inside
    ├── .dockerignore                ← excludes __pycache__, *.pyc
    │
    ├── config/
    │   └── settings.py              ← pydantic settings — loads .env
    │
    ├── models/
    │   └── db_models.py             ← SQLAlchemy Call + Exchange tables
    │
    ├── utils/
    │   ├── session_store.py         ← Redis — create/get/update/delete sessions
    │   ├── audio_utils.py           ← WebM bytes → 16kHz mono WAV conversion
    │   └── email_utils.py           ← Gmail SMTP email notification
    │
    ├── services/
    │   ├── stt_service.py           ← faster-whisper — transcribe(wav_path) → str
    │   ├── tts_service.py           ← Kokoro — synthesize(text) → WAV bytes
    │   ├── rag_service.py           ← hybrid RAG — keyword filter + embeddings on rate card
    │   └── llm_service.py           ← Gemini streaming + RAG context injection
    │
    ├── handlers/
    │   └── websocket_handler.py     ← all WebSocket message type handling
    │
    ├── static/
    │   └── index.html               ← browser UI
    │
    └── tests/
        ├── test_tts.py              ← generates all 20 Kokoro voice samples
        ├── test_stt.py              ← tests STT with a local audio file
        └── test_rag.py              ← tests RAG queries across all 6 rate card sheets
```

> **Note:** All models (Kokoro TTS, all-MiniLM-L6-v2, faster-whisper) are downloaded **inside the Docker image** during `docker build`. Nothing needs to exist on the host machine.

---

## Tech Stack

| Component | Technology | Notes |
|---|---|---|
| Backend | FastAPI | Async, WebSocket support |
| STT | faster-whisper base.en | Local, ~200ms, no cloud cost |
| LLM | Gemini 2.5 Flash | Streaming, sentence by sentence |
| TTS | Kokoro af_heart (sid=3) | Local, warm human voice, ~300ms |
| RAG | sentence-transformers | Local, ~90MB, no cloud cost |
| Embedding Model | all-MiniLM-L6-v2 | Semantic search on pricing chunks |
| Transport | WebSocket | Full duplex, persistent connection |
| Session Store | Redis | Fast, TTL support, shared across workers |
| Database | PostgreSQL | Production ready, scalable |
| Email | Gmail SMTP | No third party service needed |
| Containers | Docker + docker-compose | App + Redis + PostgreSQL |

---

## Quick Start

```bash
# 1. Clone and enter project
git clone <repo>
cd voice_ai

# 2. Copy env file and fill in your values
cp sample.env .env

# 3. Build and start all services
#    (models are downloaded automatically inside the image during build)
docker-compose up --build

# 4. Open browser
http://localhost:8000
```

---

## Environment Variables

See `sample.env` for all required variables.

---

## Testing

```bash
# Test all 20 Kokoro voices
python tests/test_tts.py

# Test STT with a local audio file
python tests/test_stt.py

# Test RAG queries across all 6 rate card sheets
python tests/test_rag.py
```