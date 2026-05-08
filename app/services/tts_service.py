"""
tts_service.py — Text-to-Speech using Cartesia Sonic 3.5 SDK WebSocket

One CartesiaTTS instance is created per active call in websocket_handler.py.
The WebSocket connection to Cartesia opens when the call starts and closes
when the call ends — giving full isolation between concurrent callers.

Each call to synthesize() creates a new context on the same connection,
streams audio chunks back as raw bytes, and closes the context when done.

Model: sonic-3.5 (latest stable — released May 4, 2026)
Voice: Katie (f786b574-daa5-4673-aa0c-cbe3e8534c02)
       Recommended by Cartesia for voice agents — stable, realistic, American English

WebSocket generation request structure (from Cartesia docs):
  context()              → creates a context ID only — no params
  ctx.send(              → generation request goes here:
      model_id           → model to use
      transcript         → text to synthesize
      voice              → voice config
      output_format      → audio format
      language           → language code
  )
  ctx.no_more_inputs()   → signal no more text is coming
  ctx.receive()          → async generator of WebSocketTtsOutput objects
                           access audio directly via response.audio
                           (no .type attribute on WebSocketTtsOutput)
"""

from typing import AsyncGenerator

from cartesia import AsyncCartesia

from config.settings import settings
from utils.logger import tts_logger

# sonic-3.5 — latest stable model as of May 2026
# Improvements over sonic-2: higher naturalness, lower latency, 42 languages
TTS_MODEL = "sonic-3.5"

# Katie — Cartesia's recommended voice for voice agents
# Stable, realistic American English — better than emotive/studio voices for agents
# Override via settings.cartesia_voice_id in .env if needed
VOICE_ID  = "f786b574-daa5-4673-aa0c-cbe3e8534c02"


class CartesiaTTS:
    """
    Per-call Cartesia Sonic TTS client.

    Lifecycle (managed by websocket_handler.py):
        tts = CartesiaTTS()
        await tts.connect(session_id)          # call once when call starts
        ...
        async for chunk in tts.synthesize(sentence, session_id):
            send_to_browser(chunk)             # called once per sentence
        ...
        await tts.close(session_id)            # call once when call ends
    """

    def __init__(self):
        self._client = AsyncCartesia(api_key=settings.cartesia_api_key)
        self._ws     = None

    async def connect(self, session_id: str) -> None:
        """Open the Cartesia Sonic WebSocket connection for this call."""
        self._ws = await self._client.tts.websocket()
        tts_logger.info(f"[{session_id}] Cartesia Sonic WebSocket connected (model={TTS_MODEL})")

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
            raise RuntimeError(f"[{session_id}] TTS synthesize() called before connect()")

        tts_logger.debug(f"[{session_id}] Synthesizing: {text[:80]}")

        try:
            # context() takes no params — just creates a context ID
            # All generation params go into ctx.send()
            ctx = self._ws.context()

            await ctx.send(
                model_id      = TTS_MODEL,
                transcript    = text.strip(),
                voice         = {
                    "mode": "id",
                    "id"  : settings.cartesia_voice_id or VOICE_ID,
                },
                output_format = {
                    "container"  : "raw",
                    "encoding"   : "pcm_s16le",
                    "sample_rate": 16000,
                },
                language = "en",
            )

            await ctx.no_more_inputs()

            chunk_count = 0
            async for response in ctx.receive():
                # WebSocketTtsOutput has no .type attribute
                # audio is accessed directly via response.audio
                if response.audio is not None:
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