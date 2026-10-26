from livekit.agents import FunctionTool, function_tool, RunContext

from shared.logging_setup import get_logger
from shared.config.constants import Track
from shared.infra.calendar import MeetingSlot

from tasks import ChooseSlotTask
from schemas import ToolDependencies
from utils import describe


_LOGGER = "worker.domain.tools.schedule_meeting"
logger = get_logger(_LOGGER)


def create_schedule_meeting_tool(deps: ToolDependencies) -> FunctionTool:
    """
    Create a function tool for scheduling a meeting with the caller.

    The factory captures runtime dependencies required by ChooseSlotTask
    and returns a LiveKit FunctionTool. These dependencies are kept out
    of the parameters exposed to the LLM.

    Args:
        deps: Runtime dependencies containing the task LLM, agent name,
            and company name used to configure ChooseSlotTask.

    Returns:
        FunctionTool: A tool that selects and books a meeting slot,
        updates session user data, and reports the outcome.
    """

    @function_tool()
    async def schedule_meeting(ctx: RunContext, track: Track) -> str:
        """
        Schedule a meeting for the caller on the requested track.

        Call this tool when the caller is ready to schedule a meeting and
        their contact email has already been confirmed. The tool checks
        whether a meeting is already scheduled and whether contact
        information is available before delegating slot selection and
        booking to ChooseSlotTask.

        After the task returns, the tool updates the session's meeting
        state if a booking was successful. It does not reconfirm the
        caller's email or perform unrelated business actions. The
        supervisor decides what to do next based on the result.

        Args:
            ctx: LiveKit tool execution context containing the session
                and the caller's user data.
            track: The track or service category for which the caller
                wants to schedule a meeting.

        Returns:
            str: A message describing the booking outcome or explaining
            why scheduling could not proceed.
        """
        logger.info("Starting schedule_meeting with track: %s", track)

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

        logger.info("Executing ChooseSlotTask for %s", contact_email)

        supervisor_chat_ctx = ctx.session.current_agent.chat_ctx.copy(
            exclude_function_call=True,
            exclude_instructions=True,
        )

        meeting_slot_description = describe(MeetingSlot)

        booked_slot = await ChooseSlotTask(
            contact_email=contact_email,
            track=track,
            chat_ctx=supervisor_chat_ctx,
            task_llm=deps.task_llm,
            supervisor_name=deps.agent_name,
            supervisor_role=(
                f"{deps.agent_name} is the primary voice assistant for "
                f"{deps.company_name}. The supervisor owns the overall caller "
                "conversation and business workflow, and decides what happens "
                "before and after each bounded subtask."
            ),
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
            meeting_slot_description=meeting_slot_description,
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
                f"An invite will be sent to {contact_email} by email."
            )

        logger.warning(
            "Could not book a meeting slot for %s",
            contact_email,
        )

        return (
            f"Could not book a meeting. "
            f"Follow up with {contact_email} by email instead."
        )

    return schedule_meeting
