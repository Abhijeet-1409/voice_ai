from livekit.agents import Agent, function_tool, RunContext
from livekit.agents.llm import FunctionTool

from shared.config import Track
from shared.logging_setup import get_logger

from tasks import ConfirmEmailTask, ChooseSlotTask


_LOGGER = "worker.agent.assistant"
logger = get_logger(_LOGGER)


class Assistant(Agent):
    """
    Thin container holding this call's instructions and free-function
    tool list (assembled by agent_factory.build_agent). Carries exactly
    one bound method of its own, schedule_meeting — it has to be bound
    (not a free function in domain/tools/) because it needs
    self.chat_ctx to hand conversation history into the Tasks it
    triggers, which a free function taking only ctx: RunContext can't
    access.
    """

    def __init__(self, instructions: str, tools: list[FunctionTool]) -> None:
        logger.info("Initializing Assistant agent")
        super().__init__(
            instructions=instructions,
            tools=tools,
        )

    @function_tool()
    async def schedule_meeting(self, ctx: RunContext, track: Track) -> str:
        """
        Use this when the user wants to schedule a meeting. Confirms
        their email (reading back whatever's already known, or
        collecting it fresh if not), then walks them through picking
        and booking an available time slot.

        Args:
            ctx (RunContext): The LiveKit agent execution context.
            track: Which offering track the meeting is for, based on
                what the user has told you.
        """
        logger.info("Starting schedule_meeting with track: %s", track)

        logger.info("Executing ConfirmEmailTask")
        confirmed_email = await ConfirmEmailTask(
            candidate_email=ctx.userdata.email,
            chat_ctx=self.chat_ctx.copy(exclude_instructions=True),
        )
        ctx.userdata.email = confirmed_email
        logger.info("Email confirmed: %s", confirmed_email)

        logger.info("Executing ChooseSlotTask for %s", confirmed_email)
        booked_slot = await ChooseSlotTask(
            contact_email=confirmed_email,
            track=track,
            chat_ctx=self.chat_ctx.copy(exclude_instructions=True),
        )

        if booked_slot:
            logger.info("Meeting successfully booked for slot: %s", booked_slot)
            ctx.userdata.meeting_scheduled = True
            ctx.userdata.meeting_slot = booked_slot
            return f"Meeting scheduled for {booked_slot} and confirmed. Will send email to {confirmed_email} with the invite."
        else:
            logger.warning("Could not book a meeting slot for %s", confirmed_email)
            return f"Could not book a meeting. Follow up with {confirmed_email} by email instead."