import asyncio
from typing import Optional

from livekit.agents import AgentTask, RunContext, ToolError, function_tool, ChatContext
from livekit.agents.llm import LLM

from shared.config import Track
from shared.logging_setup import get_logger
from shared.infra.calendar import MeetingSlot

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
    `ToolError` on timeout.

    Attributes:
        contact_email: Confirmed email address where the invite will be sent.
        track: Offering track passed down to the calendar client.
        available_slots: List of retrieved MeetingSlot objects.
        task_llm: LLM instance dedicated to this task workflow.
    """

    def __init__(
        self,
        contact_email: str,
        track: Optional[Track] = None,
        chat_ctx: Optional[ChatContext] = None,
        task_llm: Optional[LLM] = None,
    ) -> None:
        """Initializes a new ChooseSlotTask instance.

        Args:
            contact_email: Previously confirmed email address to receive
                the meeting invitation.
            track: Product or service offering track context.
            chat_ctx: Pre-existing chat history context.
            task_llm: Specialized LLM instance to execute this task.
        """
        self.contact_email = contact_email
        self.track = track
        self.available_slots: list[MeetingSlot] = []
        self.task_llm = task_llm

        logger.info(
            "Initializing ChooseSlotTask for email: %s, track: %s",
            contact_email,
            track,
        )

        super().__init__(
            instructions=CHOOSE_SLOT_TASK_PROMPT,
            chat_ctx=chat_ctx,
            llm=task_llm,
        )

    async def on_enter(self) -> None:
        """Fetch and present available meeting slots."""
        logger.info("Entering ChooseSlotTask, fetching available slots...")

        self.available_slots = await get_slots(track=self.track)

        if not self.available_slots:
            logger.warning(
                "No available slots found for track: %s",
                self.track,
            )

            await self.session.generate_reply(
                instructions="""
                Apologize that there are no available meeting slots right
                now, and let the user know you'll follow up by email instead.
                """
            )

            self.complete(None)
            return

        logger.info(
            "Retrieved %d available slot(s): %s",
            len(self.available_slots),
            self.available_slots,
        )

        slots_text = "\n".join(
            f"- ID: {slot.id} | Time: {slot.value}"
            for slot in self.available_slots
        )

        await self.session.generate_reply(
            instructions=f"""
            Tell the caller briefly that you can help them choose a meeting time.

            Available slots:
            {slots_text}

            Read the available times naturally to the caller and ask which
            one works best.

            When calling submit_slot, use the exact slot ID associated with
            the time the caller selected. Never pass the human-readable time
            as the slot argument.
            """
        )

    @function_tool
    async def submit_slot(
        self,
        ctx: RunContext,
        slot_id: str,
        read_back: bool,
    ) -> str:
        """Submits and books the user's selected meeting slot.

        Args:
            ctx: LiveKit execution context for managing call state.
            slot_id: ID of the selected slot from the available slots.
            read_back: Must be True to confirm that the slot was read back
                to the user and explicitly acknowledged before booking.

        Returns:
            Natural-language status or error message directing the LLM's
            next response.

        Raises:
            ToolError: If the calendar booking request times out.
        """
        logger.info(
            "submit_slot called with slot_id: '%s', read_back: %s",
            slot_id,
            read_back,
        )

        selected_slot = next(
            (
                slot
                for slot in self.available_slots
                if slot.id == slot_id
            ),
            None,
        )

        if selected_slot is None:
            available_slots_text = ", ".join(
                f"{slot.id} ({slot.value})"
                for slot in self.available_slots
            )

            logger.warning(
                "Selected slot_id '%s' is not in available slots: %s",
                slot_id,
                available_slots_text,
            )

            return (
                f"'{slot_id}' is not a valid slot ID. "
                f"Choose one from: {available_slots_text}"
            )

        if not read_back:
            logger.warning(
                "submit_slot invoked without user confirmation "
                "(read_back=False)"
            )

            return (
                f"Read the chosen slot back to the user as "
                f"'{selected_slot.value}' and get explicit confirmation "
                f"before calling this tool again."
            )

        ctx.disallow_interruptions()

        try:
            logger.info(
                "Confirming booking for slot_id='%s', slot='%s' "
                "with email %s...",
                selected_slot.id,
                selected_slot.value,
                self.contact_email,
            )

            booked = await asyncio.wait_for(
                confirm_booking(
                    slot=selected_slot,
                    contact_email=self.contact_email,
                    track=self.track,
                ),
                timeout=_BOOKING_TIMEOUT_SECONDS,
            )

        except asyncio.TimeoutError as e:
            logger.error(
                "Booking timed out after %d seconds for slot_id='%s', "
                "slot='%s'",
                _BOOKING_TIMEOUT_SECONDS,
                selected_slot.id,
                selected_slot.value,
            )

            raise ToolError(
                "The booking system is taking too long to respond. "
                "Apologize to the user and let them know you'll follow "
                "up by email with the meeting details instead."
            ) from e

        if not booked:
            logger.warning(
                "Booking failed on calendar client for slot_id='%s', "
                "slot='%s'",
                selected_slot.id,
                selected_slot.value,
            )

            self.complete(None)

            return (
                "Booking failed on our end. Apologize to the user, let them "
                "know you'll follow up by email, and do not try booking again."
            )

        logger.info(
            "Successfully booked slot_id='%s', slot='%s' for %s",
            selected_slot.id,
            selected_slot.value,
            self.contact_email,
        )

        self.complete(selected_slot.value)

        return f"Slot booked: {selected_slot.value}"