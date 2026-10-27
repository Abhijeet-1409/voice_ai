from livekit.agents import AgentServer, JobProcess, JobContext, cli
from livekit.agents import ToolContext

from shared.logging_setup import get_logger
from shared.config import CallType, Track, TicketStatus, TicketPriority, PromptCacheType

from agent import entrypoint as _entrypoint
from config import get_worker_settings
from schemas import ToolDependencies
from agent import select_tools
from utils import describe_all
from utils import build_instruction, build_voice_agent_guide

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

    settings = get_worker_settings()

    enum_reference = describe_all(
        Track,
        TicketPriority,
        TicketStatus,
    )

    voice_agent_guide = build_voice_agent_guide(enum_reference, settings.VOICE_AGENT_GUIDE_VERSION)

    tool_dependencies = ToolDependencies(
        agent_name=settings.AGENT_NAME,
        company_name=settings.COMPANY_NAME
    )

    cache_call_map = {
        PromptCacheType.INBOUND_PROMPT_CACHE: CallType.INBOUND,
        PromptCacheType.OUTBOUND_PROMPT_CACHE: CallType.OUTREACH,
        PromptCacheType.DEFAULT_PROMPT_CACHE: None
    }

    # Sync prompt cache client object creation will be add here


    for cache_type, call_type in cache_call_map.items():
        instruction = build_instruction(
            call_type=call_type,
            version=settings.SYSTEM_INSTRUCTION_VERSION
        )

        tool_list = select_tools(tool_dependencies, call_type)

        tool_ctx = ToolContext(tools = tool_list)
        tool_schema = tool_ctx.parse_function_tools(format='google')

        payload = {
            "prompt_cache_type": cache_type,
            "instruction": instruction,
            "voice_agent_guide": voice_agent_guide,
            "tool_schema": tool_schema
        }

        try:
            # TODO: Once the synchronous prompt-cache client is implemented,
            # send `payload` to the service here. Keep the request inside this
            # try block so failures can be logged per cache type.
            logger.info(
                "Prepared prompt-cache request data for cache_type=%s "
                "(instruction_chars=%d, guide_chars=%d, tool_count=%d).",
                cache_type.value,
                len(instruction),
                len(voice_agent_guide),
                len(tool_list),
            )
        except Exception:
            logger.exception(
                "Failed to prepare prompt-cache data for cache_type=%s.",
                cache_type.value,
            )

    cli.run_app(server)