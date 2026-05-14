"""
stt_service.py — Speech-to-Text using Cartesia Ink WebSocket (Streaming)

One CartesiaSTT instance is created per call in websocket_handler.py.
Audio is streamed in real time as the customer speaks — STT processes
it as it arrives so the transcript is ready almost instantly when
the customer stops speaking.

Flow:
  connect()     → open WebSocket to Cartesia, start receiver task
  send_audio()  → send raw PCM chunks as binary messages (while customer speaks)
  finalize()    → send "finalize" → wait for is_final transcript → return text
  close()       → send "done" → close WebSocket

WebSocket URL:
  wss://api.cartesia.ai/stt/websocket
    ?model=ink-whisper
    &language=en
    &encoding=pcm_s16le
    &sample_rate=16000
    &cartesia_version=2026-03-01
    &access_token=<api_key>

Performance:
  Cartesia recommends sending audio in pcm_s16le format at 16kHz.
  VAD (Silero) already produces Float32Array at 16kHz — convert to
  int16 PCM directly in websocket_handler before calling send_audio().
  No ffmpeg, no temp files, no WAV headers needed.
"""

import asyncio
import json

import websockets

from config.settings import settings
from utils.logger import stt_logger

# ── Constants ─────────────────────────────────────────────────────────────────

CARTESIA_STT_WS_URL = "wss://api.cartesia.ai/stt/websocket"
CARTESIA_VERSION    = "2026-03-01"
STT_MODEL           = "ink-whisper"


# ── CartesiaSTT ───────────────────────────────────────────────────────────────

class CartesiaSTT:
    """
    Manages one Cartesia Ink WebSocket STT connection per call.

    One instance lives for the entire call — connect() on call start,
    close() on call end. Each customer utterance calls send_audio()
    while they speak, then finalize() to get the final transcript.
    """

    def __init__(self):
        self._ws              = None
        self._receiver_task   = None
        self._transcript      = ""          # accumulates partial transcripts
        self._final_event     = asyncio.Event()   # set when is_final=true received
        self._flush_event     = asyncio.Event()   # set when flush_done received
        self._final_transcript = ""         # holds the last final transcript


    # ── Connect ───────────────────────────────────────────────────────────────

    async def connect(self, session_id: str = "") -> None:
        """
        Open WebSocket connection to Cartesia Ink STT.
        Starts a background receiver task to process incoming messages.
        Called once at call start.
        """
        url = (
            f"{CARTESIA_STT_WS_URL}"
            f"?model={STT_MODEL}"
            f"&language=en"
            f"&encoding=pcm_s16le"
            f"&sample_rate=16000"
            f"&cartesia_version={CARTESIA_VERSION}"
            f"&access_token={settings.cartesia_api_key}"
        )

        self._ws = await websockets.connect(url)
        stt_logger.info(f"[{session_id}] Cartesia Ink STT WebSocket connected")

        # Start background task to receive transcription messages
        self._receiver_task = asyncio.create_task(
            self._receive_loop(session_id)
        )


    # ── Send audio ────────────────────────────────────────────────────────────

    async def send_audio(self, pcm_bytes: bytes, session_id: str = "") -> None:
        """
        Send raw PCM audio bytes to Cartesia Ink for real-time transcription.
        Called continuously while customer is speaking.

        Args:
            pcm_bytes:  Raw int16 PCM bytes at 16kHz mono (pcm_s16le)
            session_id: For log tracing
            chunk size: 8KB per message (well under Cartesia's 32KB limit)
        """
        
        if self._ws is None:
            stt_logger.error(f"[{session_id}] send_audio called before connect()")
            return

        # ── Split into 8KB chunks — well under Cartesia's 32KB limit ─────────
        CHUNK_SIZE = 8 * 1024   # 8KB per message

        try:
            for i in range(0, len(pcm_bytes), CHUNK_SIZE):
                chunk = pcm_bytes[i : i + CHUNK_SIZE]
                await self._ws.send(chunk)
                stt_logger.debug(
                    f"[{session_id}] STT chunk sent — "
                    f"{len(chunk) / 1024:.1f}KB "
                    f"({i // CHUNK_SIZE + 1}/{(len(pcm_bytes) + CHUNK_SIZE - 1) // CHUNK_SIZE})"
                )
        except Exception as e:
            stt_logger.error(f"[{session_id}] STT send_audio failed — {e}")

    # ── Finalize ──────────────────────────────────────────────────────────────

    async def finalize(self, session_id: str = "") -> str:
        """
        Flush remaining audio and wait for the final transcript.

        Sends "finalize" text message → Cartesia flushes remaining audio
        and sends back a flush_done acknowledgment along with the final
        transcript. Returns the complete transcript text.

        Args:
            session_id: For log tracing

        Returns:
            Final transcript string. Empty string if nothing recognised.
        """
        if self._ws is None:
            stt_logger.error(f"[{session_id}] finalize called before connect()")
            return ""

        try:
            # Reset events and transcript for this utterance
            self._final_event.clear()
            self._flush_event.clear()
            self._final_transcript = ""

            # Send finalize command
            await self._ws.send("finalize")
            stt_logger.debug(f"[{session_id}] STT finalize sent — waiting for flush_done")

            # Wait for flush_done acknowledgment (max 5 seconds)
            await asyncio.wait_for(self._flush_event.wait(), timeout=5.0)

            transcript = self._final_transcript.strip()
            stt_logger.info(f"[{session_id}] Transcript: {transcript}")
            return transcript

        except asyncio.TimeoutError:
            stt_logger.error(f"[{session_id}] STT finalize timed out after 5s")
            return ""

        except Exception as e:
            stt_logger.error(f"[{session_id}] STT finalize failed — {e}")
            return ""


    # ── Close ─────────────────────────────────────────────────────────────────

    async def close(self, session_id: str = "") -> None:
        """
        Send "done" and close the WebSocket connection.
        Called once at call end.
        """
        if self._ws is None:
            return

        try:
            await self._ws.send("done")
            stt_logger.debug(f"[{session_id}] STT done sent")
        except Exception:
            pass

        # Cancel receiver task
        if self._receiver_task and not self._receiver_task.done():
            self._receiver_task.cancel()
            try:
                await self._receiver_task
            except asyncio.CancelledError:
                pass

        try:
            await self._ws.close()
            stt_logger.info(f"[{session_id}] Cartesia Ink STT WebSocket closed")
        except Exception as e:
            stt_logger.debug(f"[{session_id}] STT close error — {e}")

        self._ws = None


    # ── Receiver loop ─────────────────────────────────────────────────────────

    async def _receive_loop(self, session_id: str) -> None:
        """
        Background task — receives transcription messages from Cartesia.

        Message types:
          transcript  → partial or final transcription result
          flush_done  → acknowledgment that finalize was processed
          done        → session closing acknowledgment
          error       → error from Cartesia
        """
        try:
            async for raw in self._ws:
                msg = json.loads(raw)
                msg_type = msg.get("type")

                if msg_type == "transcript":
                    text     = msg.get("text", "").strip()
                    is_final = msg.get("is_final", False)

                    stt_logger.debug(
                        f"[{session_id}] STT transcript — "
                        f"is_final={is_final} — \"{text}\""
                    )

                    # Accumulate transcript text
                    if text:
                        self._final_transcript = text

                    # If final, signal finalize() that transcript is ready
                    if is_final:
                        self._final_event.set()

                elif msg_type == "flush_done":
                    stt_logger.debug(f"[{session_id}] STT flush_done received")
                    # flush_done signals finalize() the flush is complete
                    self._flush_event.set()

                elif msg_type == "done":
                    stt_logger.debug(f"[{session_id}] STT done received — closing")
                    break

                elif msg_type == "error":
                    stt_logger.error(
                        f"[{session_id}] STT error — "
                        f"{msg.get('message', 'unknown error')} "
                        f"[{msg.get('error_code', '')}]"
                    )
                    # Signal finalize() to stop waiting on error
                    self._flush_event.set()

        except asyncio.CancelledError:
            pass
        except Exception as e:
            stt_logger.error(f"[{session_id}] STT receiver loop error — {e}")
            # Unblock finalize() if it's waiting
            self._flush_event.set()