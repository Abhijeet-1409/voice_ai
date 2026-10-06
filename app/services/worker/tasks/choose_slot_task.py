import asyncio
from typing import Optional

from livekit.agents import AgentTask, RunContext, ToolError, function_tool, ChatContext
from livekit.agents.llm import LLM

from shared.config import Track
from shared.logging_setup import get_logger

from utils import get_slots, confirm_booking
from domain import CHOOSE_SLOT_TASK_PROMPT

_LOGGER = "worker.tasks.choose_slot_task"
logger = get_logger(_LOGGER)


_BOOKING_TIMEOUT_SECONDS = 10


class ChooseSlotTask(AgentTask[Optional[str]]):
    """Task that guides the user through selecting and confirming a meeting slot.

    Retrieves available slots upon entering the task, communicates options to
    the user, enforces mandatory explicit confirmation, and attempts booking
    via `confirm_booking`.

    Handles failures gracefully: returns `None` if no slots are available or if
    the booking call is rejected by the underlying calendar client. Raises a
    `ToolError` on timeout (hanging network requests), treating it as an execution
    failure rather than a standard decline.

    Attributes:
        contact_email (str): Confirmed email address where the invite will be sent.
        track (Optional[Track]): Offering track passed down to the calendar client.
        available_slots (list[str]): List of retrieved open slot strings.
        task_llm (Optional[LLM]): LLM instance dedicated to this task workflow.
    """

    def __init__(
        self,
        contact_email: str,
        track: Optional[Track] = None,
        chat_ctx: Optional[ChatContext] = None,
        task_llm: Optional[LLM] = None
    ) -> None:
        """Initializes a new ChooseSlotTask instance.

        Args:
            contact_email (str): Previously confirmed email address to receive
                the meeting invitation.
            track (Optional[Track]): Product or service offering track context.
                Defaults to None.
            chat_ctx (Optional[ChatContext]): Pre-existing chat history context.
                Defaults to None.
            task_llm (Optional[LLM]): Specialized LLM instance to execute this task.
                Defaults to None.
        """
        self.contact_email = contact_email
        self.track = track
        self.available_slots: list[str] = []
        self.task_llm = task_llm

        logger.info("Initializing ChooseSlotTask for email: %s, track: %s", contact_email, track)

        super().__init__(
            instructions=CHOOSE_SLOT_TASK_PROMPT,
            chat_ctx=chat_ctx,
            task_llm=task_llm
        )

    async def on_enter(self) -> None:
        """Lifecycle hook executed when entering the task workflow.

        Fetches available meeting slots and presents them to the user. If no
        slots are available, generates an apology reply and completes the task
        with `None`.
        """
        logger.info("Entering ChooseSlotTask, fetching available slots...")
        self.available_slots = await get_slots(track=self.track)

        if not self.available_slots:
            logger.warning("No available slots found for track: %s", self.track)
            await self.session.generate_reply(
                instructions="""
                Apologize that there are no available meeting slots right
                now, and let the user know you'll follow up by email
                instead.
                """
            )
            self.complete(None)
            return

        logger.info("Retrieved %d available slot(s): %s", len(self.available_slots), self.available_slots)
        slots_text = "\n".join(f"- {s}" for s in self.available_slots)
        await self.session.generate_reply(
            instructions=f"""
            Tell the caller briefly that you can help them choose a meeting time.
            Available slots: {slots_text}
            Read the available options naturally and ask which one works best.
            """
        )

    @function_tool
    async def submit_slot(self, ctx: RunContext, slot: str, read_back: bool) -> str:
        """Submits and books the user's selected meeting slot.

        Validates slot availability and ensures user confirmation before
        disabling interrupts and submitting the booking request to the calendar.

        Args:
            ctx (RunContext): LiveKit execution context for managing call state.
            slot (str): Selected slot, matching an entry from `available_slots`.
            read_back (bool): Must be `True` to confirm that the slot was read back
                to the user and explicitly acknowledged before calling this tool.

        Returns:
            str: A natural-language status or error message directing the LLM's next response.

        Raises:
            ToolError: If the calendar booking request times out.
        """
        logger.info("submit_slot called with slot: '%s', read_back: %s", slot, read_back)

        if slot not in self.available_slots:
            logger.warning("Selected slot '%s' is not in available slots: %s", slot, self.available_slots)
            return (
                f"'{slot}' is not in the list of available slots. "
                f"Choose one from: {', '.join(self.available_slots)}"
            )

        if not read_back:
            logger.warning("submit_slot invoked without user confirmation (read_back=False)")
            return "Read the chosen slot back to the user and get explicit confirmation before calling this tool again."

        ctx.disallow_interruptions()

        try:
            logger.info("Confirming booking for slot '%s' with email %s...", slot, self.contact_email)
            booked = await asyncio.wait_for(
                confirm_booking(
                    slot=slot,
                    contact_email=self.contact_email,
                    track=self.track,
                ),
                timeout=_BOOKING_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError as e:
            logger.error("Booking timed out after %d seconds for slot: '%s'", _BOOKING_TIMEOUT_SECONDS, slot)
            raise ToolError(
                "The booking system is taking too long to respond. Apologize "
                "to the user and let them know you'll follow up by email "
                "with the meeting details instead."
            ) from e

        if not booked:
            logger.warning("Booking failed on calendar client for slot: '%s'", slot)
            self.complete(None)
            return (
                "Booking failed on our end. Apologize to the user, let them "
                "know you'll follow up by email, and do not try booking again."
            )

        logger.info("Successfully booked slot '%s' for %s", slot, self.contact_email)
        self.complete(slot)
        return f"Slot booked: {slot}"