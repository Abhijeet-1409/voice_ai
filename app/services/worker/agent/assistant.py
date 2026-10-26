from livekit.agents.llm import FunctionTool
from livekit.agents import Agent, ChatContext

from shared.logging_setup import get_logger

from schemas import UserData


_LOGGER = "worker.agent.assistant"
logger = get_logger(_LOGGER)


class Assistant(Agent):
    """Assistant agent for managing interaction flows and meeting scheduling.

    The Assistant owns the overall conversation and acts as the parent
    supervisor for bounded AgentTask workflows such as ConfirmEmailTask
    and ChooseSlotTask.
    """

    def __init__(
        self,
        name: str,
        company_name: str,
        instructions: str,
        tools: list[FunctionTool],
        user_data: UserData | None = None,
        chat_ctx: ChatContext | None = None,
    ) -> None:
        """Initialize the supervisor Assistant."""
        logger.info("Initializing Assistant agent")

        self.name = name
        self.company_name = company_name
        self.user_data = user_data

        super().__init__(
            instructions=instructions,
            tools=tools,
            chat_ctx=chat_ctx,
        )
