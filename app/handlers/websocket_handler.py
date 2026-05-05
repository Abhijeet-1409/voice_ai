"""
websocket_handler.py — Full WebSocket call lifecycle handler

One handle_websocket() coroutine runs per active call.
Each call gets its own CartesiaTTS instance (its own WebSocket to Cartesia).
Barge-in works by cancelling the speak_task asyncio.Task mid-stream.
"""

import asyncio
import base64
import json
import os
from enum import Enum

from fastapi import WebSocket, WebSocketDisconnect

from models.db_models import Call, Exchange, SessionLocal
from services.tts_service import CartesiaTTS
from services import stt_service
from services.llm_service import stream_reply
from utils.audio_utils import webm_to_wav
from utils.email_utils import send_email_notification
from utils.session_store import (
    add_exchange,
    delete_session,
    end_session,
    get_session,
)


# ── Call state machine ────────────────────────────────────────────────────────

class CallState(Enum):
    LISTENING  = "listening"   # waiting for customer to speak
    PROCESSING = "processing"  # STT + RAG + LLM running
    SPEAKING   = "speaking"    # TTS streaming to browser


# ── Main handler ──────────────────────────────────────────────────────────────

async def handle_websocket(websocket: WebSocket, session_id: str) -> None:
    """
    Main WebSocket handler — manages the full lifecycle of one customer call.

    One CartesiaTTS instance is created here and lives for the entire call.
    speak_task holds the current _process_exchange Task so it can be
    cancelled instantly when the customer interrupts (barge-in).

    Args:
        websocket:  FastAPI WebSocket connection
        session_id: Unique session ID — already created in Redis by main.py
    """
    state        = CallState.LISTENING
    audio_buffer = bytearray()   # accumulates WebM chunks during one utterance
    speak_task   = None          # currently running _process_exchange Task

    # ── One Cartesia TTS connection per call ──────────────────────────────────
    tts = CartesiaTTS()
    await tts.connect()

    try:
        # Tell browser we are ready
        await _send(websocket, {"type": "listening"})

        while True:
            raw      = await websocket.receive_text()
            msg      = json.loads(raw)
            msg_type = msg.get("type")

            # ── audio_chunk — browser sending mic audio ────────────────────
            if msg_type == "audio_chunk":
                if state == CallState.LISTENING:
                    chunk_bytes = base64.b64decode(msg["data"])
                    audio_buffer.extend(chunk_bytes)

            # ── audio_end — customer finished speaking ─────────────────────
            elif msg_type == "audio_end":
                if state == CallState.LISTENING and len(audio_buffer) > 0:
                    state = CallState.SPEAKING

                    # Run the full STT → LLM → TTS pipeline as a cancellable Task
                    speak_task = asyncio.create_task(
                        _process_exchange(
                            websocket  = websocket,
                            session_id = session_id,
                            webm_bytes = bytes(audio_buffer),
                            tts        = tts,
                        )
                    )
                    audio_buffer.clear()

                    try:
                        await speak_task
                    except asyncio.CancelledError:
                        # Barge-in cancelled this task — that is expected
                        pass

                    # Only go back to LISTENING if we weren't interrupted
                    # (interrupt handler sets state itself)
                    if state == CallState.SPEAKING:
                        state = CallState.LISTENING
                        await _send(websocket, {"type": "listening"})

            # ── interrupt — customer spoke during agent audio ───────────────
            elif msg_type == "interrupt":
                # Cancel TTS streaming immediately
                if speak_task and not speak_task.done():
                    speak_task.cancel()
                    try:
                        await speak_task
                    except asyncio.CancelledError:
                        pass

                audio_buffer.clear()
                state = CallState.LISTENING
                await _send(websocket, {"type": "listening"})

    except WebSocketDisconnect:
        await _handle_call_end(session_id)

    except Exception as e:
        print(f"[websocket_handler] Unexpected error: {e}")
        await _handle_call_end(session_id)

    finally:
        # Always close the Cartesia TTS WebSocket when the call ends
        await tts.close()


# ── Process one exchange ──────────────────────────────────────────────────────

async def _process_exchange(
    websocket  : WebSocket,
    session_id : str,
    webm_bytes : bytes,
    tts        : CartesiaTTS,
) -> None:
    """
    Full pipeline for one customer utterance:
      WebM bytes → STT → LLM (with RAG) → TTS sentence-by-sentence → browser

    This runs as an asyncio.Task so it can be cancelled mid-stream on barge-in.
    asyncio.CancelledError propagates naturally — no special handling needed here.

    Args:
        websocket:  FastAPI WebSocket
        session_id: Redis session key
        webm_bytes: Complete WebM audio blob from browser
        tts:        CartesiaTTS instance for this call
    """
    wav_path = None

    try:
        # ── Step 1: WebM → 16kHz mono WAV ─────────────────────────────────
        wav_path   = webm_to_wav(webm_bytes)

        # ── Step 2: WAV → transcript (Cartesia Ink) ────────────────────────
        transcript = stt_service.transcribe(wav_path)

        if not transcript.strip():
            return

        # ── Step 3: Send transcript to browser ────────────────────────────
        await _send(websocket, {"type": "transcript", "text": transcript})

        # ── Step 4: Get conversation history from Redis ────────────────────
        session = get_session(session_id)
        history = session.get("exchanges", []) if session else []

        # ── Step 5: Stream LLM reply sentence by sentence ─────────────────
        extracted_info = {}
        full_reply     = []

        async for sentence in stream_reply(transcript, history, extracted_info):
            full_reply.append(sentence)

            # ── Step 6: Synthesize sentence → PCM bytes (Cartesia Sonic) ───
            # synthesize() is an async generator — yields chunks as they arrive
            # If speak_task is cancelled, CancelledError raises here and
            # propagates up naturally, stopping mid-sentence immediately
            audio_chunks = bytearray()
            async for pcm_chunk in tts.synthesize(sentence):
                audio_chunks.extend(pcm_chunk)

            if audio_chunks:
                # ── Step 7: Send audio to browser (base64 encoded) ─────────
                await _send(websocket, {
                    "type": "audio_chunk",
                    "data": base64.b64encode(bytes(audio_chunks)).decode("utf-8"),
                })

        # ── Step 8: Signal agent finished speaking ─────────────────────────
        await _send(websocket, {"type": "audio_end"})

        # ── Step 9: Send full reply text for display ───────────────────────
        agent_reply = " ".join(full_reply)
        await _send(websocket, {"type": "reply_text", "text": agent_reply})

        # ── Step 10: Save exchange to Redis ────────────────────────────────
        add_exchange(
            session_id     = session_id,
            caller_message = transcript,
            agent_reply    = agent_reply,
            extracted_info = extracted_info,
        )

    finally:
        # Always clean up temp WAV file even if cancelled mid-stream
        if wav_path and os.path.exists(wav_path):
            os.unlink(wav_path)


# ── Call end ──────────────────────────────────────────────────────────────────

async def _handle_call_end(session_id: str) -> None:
    """
    Called on WebSocketDisconnect (or unexpected error).
    Saves full session to PostgreSQL, sends email, cleans up Redis.
    All steps run independently — one failure does not skip the rest.
    """
    print(f"[websocket_handler] Call ended: {session_id}")

    try:
        session = end_session(session_id)

        if not session:
            print(f"[websocket_handler] Session not found in Redis: {session_id}")
            return

        _save_to_db(session)
        send_email_notification(session)
        delete_session(session_id)

        print(f"[websocket_handler] Cleanup complete: {session_id}")

    except Exception as e:
        print(f"[websocket_handler] Call end error: {e}")


# ── Save to PostgreSQL ────────────────────────────────────────────────────────

def _save_to_db(session: dict) -> None:
    """
    Write full session to PostgreSQL in one transaction.
    One Call row + one Exchange row per exchange.
    """
    db = SessionLocal()

    try:
        call_start = _parse_dt(session.get("start_time"))
        call_end   = _parse_dt(session.get("end_time"))
        duration   = None

        if call_start and call_end:
            duration = int((call_end - call_start).total_seconds())

        call = Call(
            session_id     = session["session_id"],
            caller_phone   = session.get("caller_phone"),
            caller_name    = session.get("caller_name"),
            caller_email   = session.get("caller_email"),
            caller_need    = session.get("caller_need"),
            interest_level = session.get("interest_level"),
            call_start     = call_start,
            call_end       = call_end,
            call_duration  = duration,
            exchange_count = session.get("exchange_count", 0),
        )
        db.add(call)
        db.flush()

        for ex in session.get("exchanges", []):
            exchange = Exchange(
                call_id         = call.id,
                exchange_number = ex.get("exchange_number", 0),
                caller_message  = ex.get("caller_message", ""),
                agent_reply     = ex.get("agent_reply", ""),
                timestamp       = _parse_dt(ex.get("timestamp")),
            )
            db.add(exchange)

        db.commit()
        print(f"[websocket_handler] Saved to DB: {session['session_id']}")

    except Exception as e:
        db.rollback()
        print(f"[websocket_handler] DB save failed: {e}")

    finally:
        db.close()


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _send(websocket: WebSocket, data: dict) -> None:
    """Send JSON over WebSocket. Silently ignores errors (client may have disconnected)."""
    try:
        await websocket.send_text(json.dumps(data))
    except Exception:
        pass


def _parse_dt(value: str | None):
    """Parse ISO datetime string → datetime object. Returns None if invalid."""
    from datetime import datetime
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None