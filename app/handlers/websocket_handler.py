"""
websocket_handler.py — Full WebSocket call lifecycle handler

One handle_websocket() coroutine runs per active call.
Each call gets its own CartesiaSTT and CartesiaTTS instance.

STT: Cartesia Ink WebSocket — audio streamed in real time while customer speaks.
     finalize() is called after VAD signals speech end — transcript ready instantly.
TTS: Cartesia Sonic WebSocket — synthesizes sentence by sentence.

Barge-in works by cancelling the speak_task asyncio.Task mid-stream.
"""

import asyncio
import base64
import json
import struct
import time
from enum import Enum

from fastapi import WebSocket, WebSocketDisconnect

from models.db_models import Call, Exchange, SessionLocal
from services.tts_service import CartesiaTTS
from services.stt_service import CartesiaSTT,STTError
from services.llm_service import stream_reply,extract_info
from utils.email_utils import send_email_notification
from utils.logger import ws_logger, db_logger
from utils.session_store import (
    add_exchange,
    delete_session,
    end_session,
    get_session,
)


# ── Call state machine ────────────────────────────────────────────────────────

class CallState(Enum):
    LISTENING  = "listening"   # waiting for customer to speak
    PROCESSING = "processing"  # STT finalize + RAG + LLM running
    SPEAKING   = "speaking"    # TTS streaming to browser


class CallContext:
    def __init__(self):
        self._state: CallState = CallState.LISTENING

    @property
    def state(self) -> CallState:
        return self._state

    @state.setter
    def state(self, value: CallState) -> None:
        if not isinstance(value, CallState):
            raise ValueError(f"Expected CallState, got {type(value).__name__}: {value}")
        ws_logger.info(f"State transition → {self._state.value} → {value.value}")
        self._state = value


# ── PCM conversion helper ─────────────────────────────────────────────────────

def float32_to_pcm16(wav_base64: str) -> bytes:
    """
    Convert base64-encoded WAV (from VAD float32ToWav) back to raw int16 PCM bytes.
    Strips the 44-byte WAV header — Cartesia STT needs raw pcm_s16le.

    The browser's float32ToWav() wraps Float32→int16 PCM in a WAV container.
    We strip that header here so Cartesia receives pure PCM bytes.
    """
    wav_bytes = base64.b64decode(wav_base64)
    # WAV header is always 44 bytes — PCM data starts at byte 44
    return wav_bytes[44:]


# ── Main handler ──────────────────────────────────────────────────────────────

async def handle_websocket(websocket: WebSocket, session_id: str) -> None:
    """
    Main WebSocket handler — manages the full lifecycle of one customer call.

    One CartesiaSTT and one CartesiaTTS instance live for the entire call.
    speak_task holds the current _process_exchange Task so it can be
    cancelled instantly when the customer interrupts (barge-in).

    Args:
        websocket:  FastAPI WebSocket connection
        session_id: Unique session ID — already created in Redis by main.py
    """
    ctx          = CallContext()
    speak_task   = None

    ws_logger.info(f"[{session_id}] Call started — WebSocket connected")

    # ── One STT + one TTS connection per call ─────────────────────────────────
    stt = CartesiaSTT()
    tts = CartesiaTTS()

    await stt.connect(session_id)
    await tts.connect(session_id)

    try:
        # Tell browser we are ready
        await _send(websocket, {"type": "listening"})
        ws_logger.info(f"[{session_id}] Sent listening signal to browser")

        while True:
            raw      = await websocket.receive_text()
            msg      = json.loads(raw)
            msg_type = msg.get("type")

            # ── audio_chunk — VAD sending PCM while customer speaks ─────────
            if msg_type == "audio_chunk":
                if ctx.state == CallState.LISTENING:
                    # Strip WAV header → raw PCM → stream to Cartesia STT
                    pcm_bytes = float32_to_pcm16(msg["data"])
                    await stt.send_audio(pcm_bytes, session_id=session_id)
                    ws_logger.debug(f"[{session_id}] PCM streamed to STT — {len(pcm_bytes) / 1024:.1f}KB")

            # ── audio_end — VAD says customer finished speaking ────────────
            elif msg_type == "audio_end":
                if ctx.state == CallState.LISTENING:
                    ws_logger.info(f"[{session_id}] Audio end — starting processing")
                    ctx.state = CallState.PROCESSING

                    speak_task = asyncio.create_task(
                        _process_exchange(
                            websocket  = websocket,
                            session_id = session_id,
                            stt        = stt,
                            tts        = tts,
                            ctx        = ctx,
                        )
                    )

                    try:
                        await speak_task
                    except asyncio.CancelledError:
                        ws_logger.info(f"[{session_id}] speak_task cancelled — barge-in")

            # ── ready — browser finished playing all audio chunks ──────────
            elif msg_type == "ready":
                if ctx.state == CallState.SPEAKING:
                    ctx.state = CallState.LISTENING
                    await _send(websocket, {"type": "listening"})
                    ws_logger.info(f"[{session_id}] Browser ready — transitioning to LISTENING")

            # ── interrupt — barge-in, customer spoke over agent ────────────
            elif msg_type == "interrupt":
                ws_logger.info(f"[{session_id}] Barge-in triggered — cancelling agent speech")
                ctx.state = CallState.LISTENING

                if speak_task and not speak_task.done():
                    speak_task.cancel()
                    try:
                        await speak_task
                    except asyncio.CancelledError:
                        ws_logger.info(f"[{session_id}] speak_task cancelled — barge-in")

                await _send(websocket, {"type": "listening"})
                ws_logger.info(f"[{session_id}] Barge-in complete — back to LISTENING")

    except WebSocketDisconnect:
        ws_logger.info(f"[{session_id}] WebSocket disconnected")
        await _handle_call_end(session_id)

    except Exception as e:
        ws_logger.exception(f"[{session_id}] Unexpected error — {e}")
        await _handle_call_end(session_id)

    finally:
        await stt.close(session_id)
        await tts.close(session_id)
        ws_logger.info(f"[{session_id}] STT + TTS connections closed")


# ── Process one exchange ──────────────────────────────────────────────────────

async def _process_exchange(
    websocket  : WebSocket,
    session_id : str,
    stt        : CartesiaSTT,
    tts        : CartesiaTTS,
    ctx        : CallContext,
) -> None:
    """
    Full pipeline for one customer utterance:
      STT finalize → LLM (with RAG) → TTS sentence-by-sentence → browser

    STT has already been processing audio in real time — finalize() just
    flushes remaining audio and returns the final transcript instantly.

    This runs as an asyncio.Task so it can be cancelled mid-stream on barge-in.
    """
    start_time = time.time()
    ws_logger.info(f"[{session_id}] _process_exchange started")

    try:
        # ── Step 1: Finalize STT — get transcript ──────────────────────────
        # Audio was already streamed in real time — this just flushes remainder
        transcript = await stt.finalize(session_id=session_id)

        # ── Step 2: Send transcript to browser ────────────────────────────
        await _send(websocket, {"type": "transcript", "text": transcript})
        ws_logger.info(f"[{session_id}] Transcript sent to browser")

        # ── Step 3: Get conversation history from Redis ────────────────────
        session = get_session(session_id)
        history = session.get("exchanges", []) if session else []
        ws_logger.info(f"[{session_id}] History loaded — {len(history)} exchanges")

        # ── Step 4: Stream LLM reply sentence by sentence ─────────────────
        full_reply     = []
        sentence_count = 0

        async for sentence in stream_reply(transcript, history, extracted_info, session_id=session_id):
            full_reply.append(sentence)
            sentence_count += 1

            # Transition to SPEAKING on first sentence only
            if ctx.state == CallState.PROCESSING:
                ctx.state = CallState.SPEAKING

            # ── Step 5: Synthesize sentence → PCM bytes ────────────────────
            audio_chunks = bytearray()
            async for pcm_chunk in tts.synthesize(sentence, session_id=session_id):
                audio_chunks.extend(pcm_chunk)

            if audio_chunks:
                # ── Step 6: Send audio to browser ──────────────────────────
                await _send(websocket, {
                    "type": "audio_chunk",
                    "data": base64.b64encode(bytes(audio_chunks)).decode("utf-8"),
                })
                ws_logger.info(f"[{session_id}] Audio chunk sent — sentence {sentence_count} — {len(audio_chunks) / 1024:.1f}KB")

        # ── Step 7: Signal agent finished speaking ─────────────────────────
        await _send(websocket, {"type": "audio_end"})

        # ── Step 8: Send full reply text for display ───────────────────────
        agent_reply = " ".join(full_reply)
        await _send(websocket, {"type": "reply_text", "text": agent_reply})

        # ── Step 9: Extract customer info from this exchange ───────────────
        extracted_info = await extract_info(transcript, agent_reply, session_id=session_id)

        # ── Step 10: Save exchange to Redis ────────────────────────────────
        add_exchange(
            session_id     = session_id,
            caller_message = transcript,
            agent_reply    = agent_reply,
            extracted_info = extracted_info,
        )

        elapsed = time.time() - start_time
        ws_logger.info(f"[{session_id}] Exchange complete — {sentence_count} sentences — {elapsed:.2f}s total")

    except asyncio.CancelledError:
        ws_logger.info(f"[{session_id}] Exchange cancelled mid-stream — barge-in")
        raise   # re-raise so speak_task sees CancelledError

    except STTError as e:
        ws_logger.error(f"[{session_id}] STT failed — {e}")

        # Tell browser what happened
        await _send(websocket, {"type": "transcript", "text": "..."})
        ctx.state = CallState.SPEAKING

        # Apologize to customer via TTS
        apology = "I'm sorry, I didn't quite catch that. Could you please repeat yourself?"

        audio_chunks = bytearray()
        async for pcm_chunk in tts.synthesize(apology, session_id=session_id):
            audio_chunks.extend(pcm_chunk)

        if audio_chunks:
            await _send(websocket, {
                "type": "audio_chunk",
                "data": base64.b64encode(bytes(audio_chunks)).decode("utf-8"),
            })

        await _send(websocket, {"type": "audio_end"})
        await _send(websocket, {"type": "reply_text", "text": apology})

        ctx.state = CallState.LISTENING
        await _send(websocket, {"type": "listening"})

    except Exception as e:
        ws_logger.exception(f"[{session_id}] Exchange failed — {e}")
        ctx.state = CallState.LISTENING
        await _send(websocket, {"type": "listening"})


# ── Call end ──────────────────────────────────────────────────────────────────

async def _handle_call_end(session_id: str) -> None:
    """
    Called on WebSocketDisconnect (or unexpected error).
    Saves full session to PostgreSQL, sends email, cleans up Redis.
    """
    ws_logger.info(f"[{session_id}] Handling call end")

    try:
        session = end_session(session_id)

        if not session:
            ws_logger.warning(f"[{session_id}] Session not found in Redis — skipping cleanup")
            return

        await asyncio.to_thread(_save_to_db, session)
        send_email_notification(session)
        delete_session(session_id)

        ws_logger.info(f"[{session_id}] Call cleanup complete — {session.get('exchange_count', 0)} exchanges")

    except Exception as e:
        ws_logger.error(f"[{session_id}] Call end error — {e}")


# ── Save to PostgreSQL ────────────────────────────────────────────────────────

def _save_to_db(session: dict) -> None:
    """
    Write full session to PostgreSQL in one transaction.
    One Call row + one Exchange row per exchange.
    """
    session_id = session.get("session_id", "unknown")
    db         = SessionLocal()

    db_logger.debug(f"[{session_id}] Saving session to PostgreSQL")

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
        db_logger.info(f"[{session_id}] Saved to PostgreSQL — {session.get('exchange_count', 0)} exchanges | duration: {duration}s")

    except Exception as e:
        db.rollback()
        db_logger.error(f"[{session_id}] PostgreSQL save failed — {e}")

    finally:
        db.close()


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _send(websocket: WebSocket, data: dict) -> None:
    """Send JSON over WebSocket. Silently ignores errors."""
    try:
        await websocket.send_text(json.dumps(data))
    except Exception as e:
        ws_logger.debug(f"WebSocket send failed — type={data.get('type')} — {e}")


def _parse_dt(value: str | None):
    """Parse ISO datetime string → datetime object. Returns None if invalid."""
    from datetime import datetime
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None