from livekit.agents.llm import FunctionTool, LLM
from livekit.agents import Agent, ChatContext, function_tool, RunContext

from shared.config import Track
from shared.logging_setup import get_logger

from schemas import UserData
from utils import build_user_context_block
from tasks import ConfirmEmailTask, ChooseSlotTask

_LOGGER = "worker.agent.assistant"
logger = get_logger(_LOGGER)


class Assistant(Agent):
    """Assistant agent for managing interaction flows and meeting scheduling.

    A container holding session-specific configurations, dynamic execution
    instructions, and tool sets assembled by `agent_factory.build_agent`.
    Maintains `schedule_meeting` as a bound method to allow sub-tasks to access
    conversation history (`self.chat_ctx`).

    Attributes:
        name (str): Display name of the assistant.
        company_name (str): Name of the organization represented by the assistant.
        user_data (UserData | None): User metadata for the current session.
        task_llm (LLM | None): Specialized LLM instance dedicated to sub-task
            executions.
    """

    def __init__(
        self, 
        name: str, 
        company_name: str, 
        instructions: str, 
        tools: list[FunctionTool], 
        user_data: UserData | None = None,
        chat_ctx: ChatContext | None = None,
        task_llm: LLM | None = None
    ) -> None:
        """Initializes an Assistant agent instance.

        Args:
            name (str): Display name of the assistant.
            company_name (str): Name of the company/organization.
            instructions (str): System prompt instructions driving assistant behavior.
            tools (list[FunctionTool]): Collection of free-function tools available to
                the agent.
            user_data (UserData | None, optional): User metadata and contextual information.
                Defaults to None.
            chat_ctx (ChatContext | None, optional): Initial chat history/context object.
                Defaults to None.
            task_llm (LLM | None, optional): LLM instance reserved for nested sub-tasks.
                Defaults to None.
        """
        logger.info("Initializing Assistant agent")
        self.name = name
        self.company_name = company_name
        self.user_data = user_data
        self.task_llm = task_llm
        
        super().__init__(
            instructions=instructions,
            tools=tools,
            chat_ctx=chat_ctx
        )

    @function_tool()
    async def get_contact_info(self, ctx: RunContext) -> str:
        """Collects and confirms the contact information required for scheduling.

        This tool handles the contact-information step separately from meeting
        scheduling. It confirms the caller's existing email when available or
        collects a new email address when it is not available.

        Args:
            ctx (RunContext): LiveKit agent execution context containing session state
                and userdata.

        Returns:
            str: A status message indicating that the caller's email was confirmed
                and is ready to be used for scheduling.
        """
        logger.info("Starting get_contact_info")

        logger.info("Executing ConfirmEmailTask")
        confirmed_email = await ConfirmEmailTask(
            candidate_email=ctx.userdata.email,
            chat_ctx=self.chat_ctx.copy(
                exclude_function_call=True,
                exclude_instructions=True,
            ),
            task_llm=self.task_llm,
        )

        ctx.userdata.email = confirmed_email
        logger.info("Email confirmed: %s", confirmed_email)

        return f"Contact information confirmed for {confirmed_email}."

    @function_tool()
    async def schedule_meeting(self, ctx: RunContext, track: Track) -> str:
        """Schedules a meeting by selecting and booking an available time slot.

        This tool assumes the caller's contact information has already been
        collected and confirmed by get_contact_info.

        Args:
            ctx (RunContext): LiveKit agent execution context containing session state
                and userdata.
            track (Track): The product or service offering track for the meeting based
                on user input.

        Returns:
            str: A natural-language status message indicating either successful booking
                details or instructions to follow up via email if booking failed.
        """
        logger.info("Starting schedule_meeting with track: %s", track)

        contact_email = ctx.userdata.email
        if not contact_email:
            logger.warning(
                "schedule_meeting called before contact information was collected"
            )
            return "Contact information must be collected and confirmed before scheduling the meeting."

        logger.info("Executing ChooseSlotTask for %s", contact_email)
        booked_slot = await ChooseSlotTask(
            contact_email=contact_email,
            track=track,
            chat_ctx=self.chat_ctx.copy(
                exclude_function_call=True, 
                exclude_instructions=True, 
            ),
            task_llm=self.task_llm,
        )

        if booked_slot:
            logger.info("Meeting successfully booked for slot: %s", booked_slot)
            ctx.userdata.meeting_scheduled = True
            ctx.userdata.meeting_slot = booked_slot
            return f"Meeting scheduled for {booked_slot} and confirmed. Will send email to {contact_email} with the invite."
        else:
            logger.warning("Could not book a meeting slot for %s", contact_email)
            return f"Could not book a meeting. Follow up with {contact_email} by email instead."
