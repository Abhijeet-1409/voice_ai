from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from models.db_models import init_db
from utils.logger import ws_logger
from utils.session_store import create_session, generate_session_id, get_session
from handlers.websocket_handler import handle_websocket

# ── Import all services at startup so models load ONCE ────────────────────────
# These imports trigger model loading at module level inside each service file.
# All 3 models are ready before the first customer call arrives.
import services.stt_service   # noqa: F401  loads cartesia ink whisper STT
import services.tts_service   # noqa: F401  loads cartesia sonic
import services.rag_service   # noqa: F401  loads all-MiniLM-L6-v2 + rate card


# ── Lifespan — startup and shutdown ──────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    ws_logger.info("Starting Intelics Voice AI Agent...")
    init_db()
    ws_logger.info("Database tables ready")
    ws_logger.info("All models loaded — server ready")
    yield
    # Shutdown
    await services.stt_service._http_client.aclose()
    ws_logger.info("Shutting down Intelics Voice AI Agent")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Intelics Voice AI Agent",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static files ──────────────────────────────────────────────────────────────

app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def serve_ui():
    """Serve the browser UI."""
    return FileResponse("static/index.html")


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for voice calls.

    Two modes:
      /ws/new        → generate new session_id, create Redis session, send to browser
      /ws/{existing} → look up existing session in Redis, resume if found
    """
    await websocket.accept()

    # ── New call ───────────────────────────────────────────────────────────
    if session_id == "new":
        session_id = generate_session_id()
        create_session(session_id)
        ws_logger.info(f"[{session_id}] New call — session created")

        await websocket.send_text(
            f'{{"type": "session_id", "session_id": "{session_id}"}}'
        )

    # ── Existing call ──────────────────────────────────────────────────────
    else:
        session = get_session(session_id)
        if not session:
            # Session expired or invalid — treat as new
            ws_logger.warning(f"[{session_id}] Session not found or expired — creating new session")
            session_id = generate_session_id()
            create_session(session_id)
            ws_logger.info(f"[{session_id}] New session created as replacement")
            await websocket.send_text(
                f'{{"type": "session_id", "session_id": "{session_id}"}}'
            )
        else:
            ws_logger.info(f"[{session_id}] Existing session resumed")

    # ── Hand off to handler ────────────────────────────────────────────────
    await handle_websocket(websocket, session_id)