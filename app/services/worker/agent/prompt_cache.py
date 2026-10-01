from typing import Optional, Sequence

from livekit.plugins.google.utils import create_tools_config
from livekit.agents.llm.tool_context import Tool, Toolset, ToolContext

from shared.logging_setup import get_logger
from shared.config import Track, TicketPriority, TicketStatus

from google.genai import types
from google.genai.client import Client

from schemas import UserData
from config import get_worker_settings
from domain import VOICE_AGENT_GUIDE
from utils import describe_all


_LOGGER = "worker.agent.prompt_cache"
logger = get_logger(_LOGGER)


def create_prompt_cache(
    user_data: UserData,
    system_instruction: str,
    agent_tools: Sequence[Tool | Toolset],
) -> Optional[str]:
    """
    Creates a context cache on Vertex AI containing:

    - System-level behavioral instructions
    - Static business and qualification guidance
    - Gemini-compatible tool definitions

    Args:
        user_data (UserData):
            The current session context containing call type and
            session metadata.

        system_instruction (str):
            The pre-rendered system instruction containing the
            agent's identity, personality, conversational behavior,
            language behavior, and general guardrails.

        agent_tools (Sequence[Tool | Toolset]):
            The LiveKit tools or toolsets available to the agent.

    Returns:
        Optional[str]:
            The resource name of the created Vertex AI cache,
            for example:
            'cachedContents/1234567890'

            Returns None if cache creation fails.
    """
    settings = get_worker_settings()

    try:
        client = Client(
            enterprise=True,
            project=settings.PROJECT,
            location=settings.LOCATION,
        )

        system_instruction_content = types.Content(
            parts=[
                types.Part.from_text(
                    text=system_instruction,
                )
            ],
        )

        enum_reference = describe_all(
            Track,
            TicketPriority,
            TicketStatus,
        )

        voice_agent_guide = VOICE_AGENT_GUIDE.format(
            company_name=settings.COMPANY_NAME,
            enum_reference=enum_reference,
        )

        guide_contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=(
                            "Use the following voice agent guide as "
                            "business and conversation guidance.\n\n"
                            f"{voice_agent_guide}"
                        ),
                    )
                ],
            )
        ]

        # Count the exact guide content being cached
        total_tokens: types.CountTokensResponse = (
            client.models.count_tokens(
                model=settings.GEMINI_MODEL,
                contents=guide_contents,
            )
        )

        logger.info(
            f"Cached guide content token count: "
            f"{total_tokens.total_tokens}"
        )

        # Convert LiveKit tools to Gemini tools
        tool_context = ToolContext(
            tools=agent_tools,
        )

        google_tools: list[types.Tool]
        google_tools, _ = create_tools_config(
            tool_ctx=tool_context,
        )

        cache_display_name = (
            f"{user_data.call_type.value}_voice_agent_cache"
        )

        ttl = f"{settings.CACHED_CONTENT_TTL}s"

        # Create Vertex AI context cache
        cached_content: types.CachedContent = (
            client.caches.create(
                model=settings.GEMINI_MODEL,
                config=types.CreateCachedContentConfig(
                    display_name=cache_display_name,
                    system_instruction=system_instruction_content,
                    contents=guide_contents,
                    tools=google_tools,
                    ttl=ttl,
                ),
            )
        )

        logger.info(
            f"Successfully created Vertex AI context cache "
            f"'{cached_content.name}' "
            f"[display_name={cache_display_name}, "
            f"stream_sid={user_data.stream_sid}]"
        )

        return cached_content.name

    except Exception as err:
        logger.exception(
            f"Failed to create Vertex AI context cache "
            f"for call_type='{user_data.call_type}' "
            f"[stream_sid={user_data.stream_sid}]: {err}"
        )

        return None