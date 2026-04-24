# Architecture Overview

## Summary

Intelics Voice AI Agent is a real-time voice agent that handles inbound sales calls via browser. A customer opens the browser, speaks into their mic, and the AI agent responds with natural human-like speech. The entire audio pipeline runs locally — no cloud cost for STT or TTS. Only Gemini is called externally for the LLM reasoning.

---

## Why These Choices

### Why WebSocket instead of HTTP
Each customer exchange requires sending audio to the server and receiving audio back. HTTP would require a new connection per exchange adding latency. WebSocket keeps a single persistent full-duplex connection open for the entire conversation — audio flows both ways over the same connection with minimal overhead.

### Why faster-whisper instead of OpenAI Whisper API
Runs locally — no per-minute billing, no data sent to cloud, ~200ms transcription time on CPU. The base.en model (145MB) gives a good balance of speed and accuracy for English sales calls.

### Why Kokoro instead of ElevenLabs or Piper
Kokoro runs locally (no cloud cost), produces the most human-sounding speech of any free offline TTS engine, and is already supported by sherpa-onnx which we use anyway. The af_heart voice (sid=3) is warm and friendly — ideal for sales calls.

### Why Gemini 2.5 Flash instead of GPT-4o
Fast streaming support, competitive quality, and cost effective for high-volume sales call scenarios.

### Why Redis for sessions
Each customer call is a multi-exchange conversation. State (caller name, exchanges, interest level) must persist across multiple WebSocket messages. Redis is fast, supports TTL (auto-expiry), and is shared across all workers and servers — unlike in-memory Python dicts which break with multiple workers.

### Why session_id instead of phone number as Redis key
Phone number is not always known at call start. Two calls from the same number could happen simultaneously. Web calls have no phone number. Session ID is generated immediately at call start and is always unique.

### Why PostgreSQL instead of SQLite
SQLite is a local file — it cannot handle concurrent writes from multiple workers and is not suitable for production. PostgreSQL is production-ready, scalable, and handles concurrent connections properly.

### Why Docker
Ensures Redis and PostgreSQL run identically on every machine without manual installation. docker-compose brings up all three services (app, Redis, PostgreSQL) with a single command.

---

## Full Flow

### Call Start
```
Browser opens page
    ↓
JavaScript creates WebSocket connection to server
    ↓
Server generates unique session_id (e.g. sess_a3f9b2c1d4e5)
    ↓
Server creates empty session in Redis
    ↓
Server sends session_id to browser
    ↓
Browser stores session_id for all future messages
```

### Each Exchange
```
Customer speaks into mic
    ↓
MediaDevices API captures audio (echo cancellation on)
    ↓
MediaRecorder encodes into chunks
    ↓
Chunks sent over WebSocket to server
    ↓
Customer stops speaking → browser sends {"type": "audio_end"}
    ↓
Server converts WebM chunks → 16kHz mono WAV (audio_utils.py)
    ↓
faster-whisper transcribes WAV → transcript string (stt_service.py)
    ↓
Gemini 2.5 Flash receives transcript + conversation history
    ↓
Gemini streams reply sentence by sentence (llm_service.py)
    ↓
Each sentence → Kokoro generates WAV bytes (tts_service.py)
    ↓
WAV bytes sent immediately over WebSocket to browser
    ↓
Web Audio API plays each chunk as it arrives
    ↓
Server updates Redis session with new exchange + extracted info
```

### Barge-in
```
Agent audio playing in browser
    ↓
Browser always monitoring mic volume (even during playback)
    ↓
Echo cancellation removes agent audio from mic input
    ↓
300ms mute window blocks false triggers at audio start
    ↓
Voice detected for 500ms+ with sustained volume
    ↓
Browser sends {"type": "interrupt"} over WebSocket
    ↓
Browser stops audio playback immediately
    ↓
Server cancels remaining TTS chunks
    ↓
Server cancels Gemini generation if still running
    ↓
Server switches to listening mode
    ↓
Server sends {"type": "listening"} to browser
    ↓
Normal flow resumes
```

### Call End
```
Customer closes browser tab or clicks end call
    ↓
WebSocket connection closes
    ↓
FastAPI catches WebSocketDisconnect automatically
    ↓
Server fetches full session from Redis
    ↓
Saves complete session to PostgreSQL (calls + exchanges tables) — once
    ↓
Sends email summary to sales team via Gmail SMTP — once
    ↓
Deletes session from Redis
```

---

## Data Flow Diagram

```
Browser                          Server
───────                          ──────
mic audio chunks    →→→→→→→→    audio_utils  →  stt_service
                                                      ↓
                                               llm_service (Gemini)
                                                      ↓
                    ←←←←←←←←    tts_service  ←  sentence stream
audio plays         ←←←←←←←←    WAV bytes

{"type":"interrupt"} →→→→→→→→   cancel TTS + cancel Gemini
                    ←←←←←←←←    {"type":"listening"}

WebSocketDisconnect              → Redis → PostgreSQL
                                 → Gmail SMTP
                                 → Redis cleared
```

---

## WebSocket Message Types

| Direction | Message | Meaning |
|---|---|---|
| Browser → Server | `{"type": "audio_chunk", "data": "..."}` | Audio chunk bytes |
| Browser → Server | `{"type": "audio_end"}` | Customer stopped speaking |
| Browser → Server | `{"type": "interrupt"}` | Customer is speaking — stop agent |
| Server → Browser | `{"type": "audio_chunk", "data": "..."}` | TTS audio chunk |
| Server → Browser | `{"type": "audio_end"}` | Agent finished speaking |
| Server → Browser | `{"type": "listening"}` | Ready to receive customer audio |
| Server → Browser | `{"type": "transcript", "text": "..."}` | What customer said |
| Server → Browser | `{"type": "reply_text", "text": "..."}` | What agent said |

---

## Database Schema

### `calls` table
One row per completed call.

| Column | Type | Description |
|---|---|---|
| id | integer | Auto increment primary key |
| session_id | string | Unique session identifier |
| caller_phone | string | Extracted from conversation (nullable) |
| caller_name | string | Extracted by Gemini (nullable) |
| caller_email | string | Extracted by Gemini (nullable) |
| caller_need | text | What the caller was asking about |
| interest_level | string | high / medium / low |
| call_start | datetime | When call began |
| call_end | datetime | When call ended |
| call_duration | integer | Duration in seconds |
| exchange_count | integer | Number of back-and-forths |
| created_at | datetime | Record creation time |

### `exchanges` table
One row per message exchange within a call.

| Column | Type | Description |
|---|---|---|
| id | integer | Auto increment primary key |
| call_id | integer | Foreign key → calls.id |
| exchange_number | integer | Order of exchange (1, 2, 3...) |
| caller_message | text | What the customer said |
| agent_reply | text | What the AI agent said |
| timestamp | datetime | When this exchange happened |

---

## Redis Session Structure

```json
{
    "session_id"     : "sess_a3f9b2c1d4e5",
    "caller_phone"   : null,
    "caller_name"    : null,
    "caller_email"   : null,
    "caller_need"    : null,
    "interest_level" : null,
    "start_time"     : "2024-01-15T10:30:00",
    "end_time"       : null,
    "exchange_count" : 0,
    "is_ended"       : false,
    "exchanges"      : []
}
```

Fields are updated exchange by exchange as Gemini extracts information from the conversation. Phone number, name, email, and need are all null at start and filled in as the customer mentions them.

---

## Docker Services

| Service | Image | Port | Purpose |
|---|---|---|---|
| app | custom (Dockerfile) | 8000 | FastAPI voice agent |
| redis | redis:7-alpine | 6379 | Session store |
| postgres | postgres:15-alpine | 5432 | Persistent database |

The `tts_models/` folder is mounted as a Docker volume — not baked into the image. This keeps the image small and fast to rebuild.
