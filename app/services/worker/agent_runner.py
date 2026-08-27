import asyncio
import json
import logging

from livekit.agents import AgentServer, JobProcess, JobContext, cli

from agent.job_entrypoint import entrypoint as _entrypoint
from config.worker_settings import get_worker_settings

from shared.infra.postgres import db_init
from shared.infra.redis import ping_redis

from rag import get_embedding_model

settings = get_worker_settings()

logger = logging.getLogger("worker.agent_runner")
logger.setLevel(settings.LOG_LEVEL)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setLevel(settings.LOG_LEVEL)
    _handler.setFormatter(logging.Formatter(settings.LOG_FORMAT, settings.DATA_FORMAT))
    logger.addHandler(_handler)


# Load environment-specific worker configurations and initialize the LiveKit AgentServer.
# Explicitly passing the API key, secret, and URL ensures the worker connects to the
# correct LiveKit instance, bypassing the need for default environment variable fallbacks.
server = AgentServer(
    api_key=settings.LIVEKIT_API_KEY,
    api_secret=settings.LIVEKIT_API_SECRET,
    ws_url=settings.LIVEKIT_URL
)


async def _prewarm_async(proc: JobProcess):
    """
    Initializes global resources before the worker begins accepting jobs.

    This function runs exactly once per worker process during the startup
    phase. It establishes essential connections to PostgreSQL and Redis.
    If these dependencies fail to connect, the exception will crash the
    process early, preventing the worker from accepting calls it cannot handle.

    Args:
        proc (JobProcess): The LiveKit process context managing this worker.

    Raises:
        RuntimeError: If Redis fails the ping test.
        Exception: Propagates any unexpected database initialization errors.
    """
    try:
        logger.info("Starting worker prewarm sequence: Initializing database...")
        await db_init()

        logger.info("Database initialized successfully. Verifying Redis connection...")
        if not await ping_redis():
            raise RuntimeError("Redis unreachable at startup — transcript storage unavailable")

        logger.info("Redis connection verified. Prewarm sequence complete.")

        logger.info("Loading embedding model...")
        get_embedding_model()
        logger.info("Embedding model loaded and cached.")

    except Exception as e:
        logger.critical(f"Fatal error during prewarm: {e}")
        raise


def prewarm(proc: JobProcess):
    """Sync entrypoint required by setup_fnc — runs the async logic to completion."""
    asyncio.run(_prewarm_async(proc))


server.setup_fnc = prewarm


@server.rtc_session(agent_name=settings.AGENT_NAME)
async def entrypoint(ctx: JobContext):
    """
    The main WebRTC session entrypoint for all incoming LiveKit jobs.

    This is triggered every time a new caller connects to the room. It
    delegates the actual state management, agent initialization, and
    conversation orchestration to the internal `_entrypoint` handler.

    Args:
        ctx (JobContext): The runtime context for the current job, providing
            access to the room, network data, and shutdown signals.
    """
    logger.debug(f"New RTC session started for job: {ctx.job.id}")
    await _entrypoint(ctx)


if __name__ == "__main__":
    logger.info("Starting LiveKit Agent worker in CLI mode...")
    cli.run_app(server)