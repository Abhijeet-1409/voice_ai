from livekit.agents.llm import FunctionTool, LLM
from livekit.agents import Agent, ChatContext, function_tool, RunContext

from shared.config import Track
from shared.logging_setup import get_logger
from shared.infra.calendar.mock import MeetingSlot

from schemas import UserData
from utils import describe
from tasks import ConfirmEmailTask, ChooseSlotTask


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
        task_llm: LLM | None = None,
    ) -> None:
        """Initialize the supervisor Assistant."""
        logger.info("Initializing Assistant agent")

        self.name = name
        self.company_name = company_name
        self.user_data = user_data
        self.task_llm = task_llm

        self.meeting_slot_description = describe(MeetingSlot)

        super().__init__(
            instructions=instructions,
            tools=tools,
            chat_ctx=chat_ctx,
        )

    def _supervisor_role(self) -> str:
        """Return the role description injected into bounded subtasks."""
        return (
            f"{self.name} is the primary voice assistant for "
            f"{self.company_name}. The supervisor owns the overall caller "
            "conversation, business workflow, and decides what happens "
            "before and after each bounded subtask."
        )

    @function_tool()
    async def get_contact_info(self, ctx: RunContext) -> str:
        """Collect and confirm the contact information required for scheduling."""
        logger.info("Starting get_contact_info")
        logger.info("Executing ConfirmEmailTask")

        confirmed_email = await ConfirmEmailTask(
            candidate_email=ctx.userdata.email,
            chat_ctx=self.chat_ctx.copy(
                exclude_function_call=True,
                exclude_instructions=True,
            ),
            task_llm=self.task_llm,

            supervisor_name=self.name,
            supervisor_role=self._supervisor_role(),

            invocation_reason=(
                "The supervisor needs to verify the caller's contact "
                "email before any meeting can be scheduled."
            ),

            known_context=(
                f"Caller name: {ctx.userdata.name or 'Unknown'}\n"
                f"Current email candidate: "
                f"{ctx.userdata.email or 'Not yet provided'}\n"
                f"Call type: {ctx.userdata.call_type}\n"
                f"Track: {ctx.userdata.track or 'Not yet determined'}"
            ),

            expected_result=(
                "Return one email address that the caller has explicitly "
                "confirmed."
            ),

            already_completed=(
                "Do not schedule a meeting, qualify the caller, create a "
                "ticket, or perform any other business action."
            ),

            previous_step=(
                "The supervisor has already established that contact "
                "information needs to be confirmed."
            ),

            current_step="ConfirmEmailTask",

            next_step=(
                "Return the confirmed email to the supervisor. "
                "The supervisor will decide what happens next."
            ),
        )

        ctx.userdata.email = confirmed_email

        logger.info(
            "Email confirmed: %s",
            confirmed_email,
        )

        return f"Contact information confirmed for {confirmed_email}."

    @function_tool()
    async def schedule_meeting(
        self,
        ctx: RunContext,
        track: Track,
    ) -> str:
        """Schedule a meeting by selecting and booking an available slot."""
        logger.info(
            "Starting schedule_meeting with track: %s",
            track,
        )

        if ctx.userdata.meeting_scheduled:
            logger.warning(
                "schedule_meeting called after a meeting was already scheduled: %s",
                ctx.userdata.meeting_slot,
            )

            return (
                f"A meeting is already scheduled for "
                f"{ctx.userdata.meeting_slot}. "
                "Do not schedule another meeting."
            )

        contact_email = ctx.userdata.email

        if not contact_email:
            logger.warning(
                "schedule_meeting called before contact information "
                "was collected"
            )

            return (
                "Contact information must be collected and confirmed "
                "before scheduling the meeting."
            )

        logger.info(
            "Executing ChooseSlotTask for %s",
            contact_email,
        )

        booked_slot = await ChooseSlotTask(
            contact_email=contact_email,
            track=track,
            chat_ctx=self.chat_ctx.copy(
                exclude_function_call=True,
                exclude_instructions=True,
            ),
            task_llm=self.task_llm,

            supervisor_name=self.name,
            supervisor_role=self._supervisor_role(),

            invocation_reason=(
                f"The caller wants to schedule a meeting for the "
                f"{track.value} track. Their contact information has "
                "already been confirmed."
            ),

            known_context=(
                f"Caller name: {ctx.userdata.name or 'Unknown'}\n"
                f"Confirmed email: {contact_email}\n"
                f"Requested track: {track.value}\n"
                f"Call type: {ctx.userdata.call_type}"
            ),

            expected_result=(
                "Select one currently available meeting slot, obtain "
                "the caller's explicit confirmation, book that exact "
                "slot, and return the booking result."
            ),

            already_completed=(
                "Do not reconfirm the caller's email. "
                "Do not qualify the caller again. "
                "Do not create a ticket. "
                "Do not perform any other business action."
            ),

            previous_step=(
                "The supervisor has already collected and confirmed "
                "the caller's contact email."
            ),

            current_step="ChooseSlotTask",

            next_step=(
                "Return the booking result to the supervisor. "
                "The supervisor will continue the conversation."
            ),

            meeting_slot_description=self.meeting_slot_description,
        )

        if booked_slot:
            logger.info(
                "Meeting successfully booked for slot: %s",
                booked_slot,
            )

            ctx.userdata.meeting_scheduled = True
            ctx.userdata.meeting_slot = booked_slot

            return (
                f"Meeting scheduled for {booked_slot} and confirmed. "
                f"Will send email to {contact_email} with the invite."
            )

        logger.warning(
            "Could not book a meeting slot for %s",
            contact_email,
        )

        return (
            f"Could not book a meeting. "
            f"Follow up with {contact_email} by email instead."
        )