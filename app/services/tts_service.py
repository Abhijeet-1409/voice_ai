"""
tts_service.py — Text-to-Speech using Cartesia Sonic SDK WebSocket

One CartesiaTTS instance is created per active call in websocket_handler.py.
The WebSocket connection to Cartesia opens when the call starts and closes
when the call ends — giving full isolation between concurrent callers.

Each call to synthesize() creates a new context on the same connection,
streams audio chunks back as raw bytes, and closes the context when done.
"""

from typing import AsyncGenerator

from cartesia import AsyncCartesia

from config.settings import settings
from utils.logger import tts_logger

TTS_MODEL = "sonic-2"
VOICE_ID  = "a0e99841-438c-4a64-b679-ae501e7d6091"   # override via settings if needed


class CartesiaTTS:
    """
    Per-call Cartesia Sonic TTS client.

    Lifecycle (managed by websocket_handler.py):
        tts = CartesiaTTS()
        await tts.connect(session_id)    # call once when call starts
        ...
        async for chunk in tts.synthesize(sentence, session_id):
            send_to_browser(chunk)       # called once per sentence
        ...
        await tts.close(session_id)      # call once when call ends
    """

    def __init__(self):
        self._client = AsyncCartesia(api_key=settings.cartesia_api_key)
        self._ws     = None

    async def connect(self, session_id: str) -> None:
        """Open the Cartesia Sonic WebSocket connection for this call."""
        self._ws = await self._client.tts.websocket()
        tts_logger.info(f"[{session_id}] Cartesia Sonic WebSocket connected")

    async def synthesize(self, text: str, session_id: str) -> AsyncGenerator[bytes, None]:
        """
        Stream TTS audio for one sentence.

        Creates a new context per sentence so each sentence is independent.
        Yields raw PCM bytes (pcm_s16le, 16kHz mono) as they arrive.
        Cancellation-safe — if the task is cancelled mid-stream, the
        async generator simply stops yielding.

        Args:
            text:       Sentence to synthesize (from LLM sentence stream)
            session_id: Session ID for log tracing

        Yields:
            Raw PCM audio bytes ready to base64-encode and send to browser
        """
        if not text or not text.strip():
            return

        if self._ws is None:
            tts_logger.error(f"[{session_id}] synthesize() called before connect()")
            return

        tts_logger.debug(f"[{session_id}] Synthesizing: {text[:80]}")

        try:
            ctx = self._ws.context(
                model_id=TTS_MODEL,
                voice={
                    "mode": "id",
                    "id"  : settings.cartesia_voice_id or VOICE_ID,
                },
                output_format={
                    "container"  : "raw",
                    "encoding"   : "pcm_s16le",
                    "sample_rate": 16000,
                },
            )

            await ctx.send(text.strip())
            await ctx.no_more_inputs()

            chunk_count = 0
            async for response in ctx.receive():
                if response.type == "chunk" and response.audio:
                    chunk_count += 1
                    yield response.audio

            tts_logger.debug(f"[{session_id}] Synthesis complete — {chunk_count} audio chunks")

        except Exception as e:
            tts_logger.exception(f"[{session_id}] Synthesis error: {e}")

    async def close(self, session_id: str) -> None:
        """Close the Cartesia Sonic WebSocket and the underlying HTTP client."""
        try:
            if self._ws:
                await self._ws.close()
            await self._client.close()
            tts_logger.info(f"[{session_id}] Cartesia Sonic WebSocket closed")
        except Exception as e:
            tts_logger.error(f"[{session_id}] Close error: {e}")