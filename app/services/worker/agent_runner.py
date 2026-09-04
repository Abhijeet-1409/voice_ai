from livekit.agents import AgentServer, JobProcess, JobContext, cli

from shared.logging_setup import get_logger

from agent.job_entrypoint import entrypoint as _entrypoint
from config.worker_settings import get_worker_settings

from rag import get_embedding_model


_LOGGER = "worker.agent_runner"


settings = get_worker_settings()
logger = get_logger(_LOGGER)


# Load environment-specific worker configurations and initialize the LiveKit AgentServer.
# Explicitly passing the API key, secret, and URL ensures the worker connects to the
# correct LiveKit instance, bypassing the need for default environment variable fallbacks.
server = AgentServer(
    api_key=settings.LIVEKIT_API_KEY,
    api_secret=settings.LIVEKIT_API_SECRET,
    ws_url=settings.LIVEKIT_URL
)


def prewarm(proc: JobProcess):
    """
    Pre-warms the worker process before it starts accepting jobs.

    This synchronous entrypoint is executed by the LiveKit agent framework 
    during worker initialization. It is used to preload heavy, CPU-bound 
    assets (such as the embedding model) into memory to eliminate cold-start 
    latency on the first incoming call.

    Note: Only load CPU-bound assets here. Avoid initializing network-bound 
    clients (like database connection pools or Redis clients) in this step, 
    as they can cause connection drops or corruption when the process forks.

    Args:
        proc (JobProcess): The worker process context provided by the framework.
    """
    get_embedding_model()  # Preload the embedding model to reduce latency on first use


server.setup_fnc = prewarm


@server.rtc_session()
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