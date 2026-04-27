import asyncio
import base64
import json
import os
from enum import Enum

from fastapi import WebSocket, WebSocketDisconnect

from models.db_models import SessionLocal, Call, Exchange
from services import stt_service, tts_service
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
    LISTENING   = "listening"    # waiting for customer to speak
    PROCESSING  = "processing"   # STT + RAG + LLM running
    SPEAKING    = "speaking"     # TTS streaming to browser


# ── Main handler ──────────────────────────────────────────────────────────────

async def handle_websocket(websocket: WebSocket, session_id: str) -> None:
    """
    Main WebSocket handler. Manages the full lifecycle of one customer call.

    Args:
        websocket:  FastAPI WebSocket connection
        session_id: Unique session ID — already created in Redis by main.py
    """
    state        = CallState.LISTENING
    audio_buffer = bytearray()       # accumulates WebM chunks during speech
    speak_task   = None              # holds the current speaking asyncio Task

    try:
        # Notify browser we are ready to listen
        await _send(websocket, {"type": "listening"})

        while True:
            # ── Receive next message ───────────────────────────────────────
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type")

            # ── audio_chunk — customer is speaking ─────────────────────────
            if msg_type == "audio_chunk":
                if state == CallState.LISTENING:
                    chunk_bytes = base64.b64decode(msg["data"])
                    audio_buffer.extend(chunk_bytes)

            # ── audio_end — customer finished speaking ─────────────────────
            elif msg_type == "audio_end":
                if state == CallState.LISTENING and len(audio_buffer) > 0:
                    state = CallState.PROCESSING
                    await _process_exchange(
                        websocket, session_id,
                        bytes(audio_buffer),
                    )
                    audio_buffer.clear()
                    state = CallState.LISTENING
                    await _send(websocket, {"type": "listening"})

            # ── interrupt — customer started speaking during agent audio ────
            elif msg_type == "interrupt":
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


# ── Process one exchange ──────────────────────────────────────────────────────

async def _process_exchange(
    websocket  : WebSocket,
    session_id : str,
    webm_bytes : bytes,
) -> None:
    """
    Full pipeline for one customer exchange:
    WebM bytes → STT → LLM (with RAG) → TTS → WebSocket stream
    """
    wav_path = None

    try:
        # ── Step 1: Convert WebM → WAV ─────────────────────────────────────
        wav_path   = webm_to_wav(webm_bytes)

        # ── Step 2: Transcribe WAV → text ──────────────────────────────────
        transcript = stt_service.transcribe(wav_path)

        if not transcript.strip():
            # Nothing heard — go back to listening silently
            return

        # ── Step 3: Send transcript to browser for display ─────────────────
        await _send(websocket, {"type": "transcript", "text": transcript})

        # ── Step 4: Get conversation history from Redis ────────────────────
        session = get_session(session_id)
        history = session.get("exchanges", []) if session else []

        # ── Step 5: Stream LLM reply sentence by sentence ──────────────────
        extracted_info = {}
        full_reply     = []

        async for sentence in stream_reply(transcript, history, extracted_info):
            full_reply.append(sentence)

            # ── Step 6: Synthesize each sentence to WAV bytes ──────────────
            wav_bytes = tts_service.synthesize(sentence)

            if wav_bytes:
                # ── Step 7: Send TTS audio chunk to browser ─────────────────
                await _send(websocket, {
                    "type": "audio_chunk",
                    "data": base64.b64encode(wav_bytes).decode("utf-8"),
                })

        # ── Step 8: Signal agent finished speaking ─────────────────────────
        await _send(websocket, {"type": "audio_end"})

        # ── Step 9: Send full reply text for display ───────────────────────
        agent_reply = " ".join(full_reply)
        await _send(websocket, {"type": "reply_text", "text": agent_reply})

        # ── Step 10: Update Redis session with exchange + extracted info ────
        add_exchange(
            session_id     = session_id,
            caller_message = transcript,
            agent_reply    = agent_reply,
            extracted_info = extracted_info,
        )

    finally:
        # Always clean up the temp WAV file
        if wav_path and os.path.exists(wav_path):
            os.unlink(wav_path)


# ── Call end ──────────────────────────────────────────────────────────────────

async def _handle_call_end(session_id: str) -> None:
    """
    Called on WebSocketDisconnect.
    Saves full session to PostgreSQL, sends email, deletes Redis session.
    All steps run even if one fails.
    """
    print(f"[websocket_handler] Call ended: {session_id}")

    try:
        # ── Step 1: Mark session ended in Redis ────────────────────────────
        session = end_session(session_id)

        if not session:
            print(f"[websocket_handler] Session not found in Redis: {session_id}")
            return

        # ── Step 2: Save to PostgreSQL ─────────────────────────────────────
        _save_to_db(session)

        # ── Step 3: Send email notification ───────────────────────────────
        send_email_notification(session)

        # ── Step 4: Delete from Redis ──────────────────────────────────────
        delete_session(session_id)

        print(f"[websocket_handler] Call cleanup complete: {session_id}")

    except Exception as e:
        print(f"[websocket_handler] Call end error: {e}")


# ── Save to PostgreSQL ────────────────────────────────────────────────────────

def _save_to_db(session: dict) -> None:
    """
    Write full session to PostgreSQL in one transaction.
    One Call row + one Exchange row per exchange.
    """
    from datetime import datetime

    db = SessionLocal()

    try:
        # Parse timestamps
        call_start = _parse_dt(session.get("start_time"))
        call_end   = _parse_dt(session.get("end_time"))

        duration = None
        if call_start and call_end:
            duration = int((call_end - call_start).total_seconds())

        # ── Insert Call row ────────────────────────────────────────────────
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
        db.flush()   # get call.id without committing yet

        # ── Insert Exchange rows ───────────────────────────────────────────
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
    """Send JSON message over WebSocket. Silently ignores send errors."""
    try:
        await websocket.send_text(json.dumps(data))
    except Exception:
        pass


def _parse_dt(value: str | None):
    """Parse ISO datetime string to datetime object. Returns None if invalid."""
    from datetime import datetime
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None