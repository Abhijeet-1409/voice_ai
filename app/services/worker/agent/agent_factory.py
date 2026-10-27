from livekit.agents import ChatContext
from livekit.agents.llm import LLM

from shared.config import CallType
from shared.logging_setup import get_logger

from .assistant import Assistant
from .tool_registry import select_tools
from config import get_worker_settings
from schemas import UserData, ToolDependencies


_LOGGER = "worker.agent.agent_factory"
logger = get_logger(_LOGGER)


def build_agent(
    user_data: UserData,
    instructions: str,
    chat_ctx: ChatContext | None = None,
    task_llm: LLM | None = None,
) -> Assistant:
    """
    Construct and configure an Assistant for the current call.

    Creates runtime tool dependencies, selects the tools available
    for the call type, and initializes the Assistant with the
    resulting configuration.

    Args:
        user_data: Session data for the active call.
        instructions: System instructions for the assistant.
        chat_ctx: Optional conversation context to restore.
        task_llm: Optional LLM used by specialized subtasks.

    Returns:
        Assistant: A configured assistant ready for the voice session.
    """
    settings = get_worker_settings()
    call_type: CallType = user_data.call_type

    tool_dependencies = ToolDependencies(
        task_llm=task_llm,
        agent_name=settings.AGENT_NAME,
        company_name=settings.COMPANY_NAME,
    )

    tool_list = select_tools(call_type, tool_dependencies)

    assistant = Assistant(
        instructions=instructions,
        tools=tool_list,
        name=settings.AGENT_NAME,
        company_name=settings.COMPANY_NAME,
        user_data=user_data,
        chat_ctx=chat_ctx,
    )

    logger.debug(
        f"Built Assistant for call_type='{call_type}' with "
        f"{len(tool_list)} configured tools."
    )

    return assistant
