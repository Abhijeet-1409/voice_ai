import asyncio
from typing import Optional
from textwrap import dedent

from livekit.agents import AgentTask, ChatContext, RunContext, ToolError, function_tool
from livekit.agents.llm import LLM

from shared.config import Track
from shared.logging_setup import get_logger
from shared.infra.calendar.mock import MeetingSlot

from utils import get_slots, confirm_booking
from utils import build_choose_slot_task_prompt


_LOGGER = "worker.tasks.choose_slot_task"
logger = get_logger(_LOGGER)

_BOOKING_TIMEOUT_SECONDS = 10


class ChooseSlotTask(AgentTask[Optional[str]]):
    """Task for selecting, confirming, and booking one meeting slot.

    The actual available slots are runtime state. They are fetched in
    ``on_enter()`` and added to the task's ChatContext as a system message.
    """

    def __init__(
        self,
        contact_email: str,
        track: Optional[Track] = None,
        chat_ctx: Optional[ChatContext] = None,
        task_llm: Optional[LLM] = None,
        *,
        supervisor_name: str,
        supervisor_role: str,
        invocation_reason: str,
        known_context: str,
        expected_result: str,
        already_completed: str,
        previous_step: str,
        current_step: str,
        next_step: str,
        meeting_slot_description: str,
    ) -> None:
        self.contact_email = contact_email
        self.track = track
        self.available_slots: list[MeetingSlot] = []
        self.task_llm = task_llm
        self._booking_completed = False

        instructions = build_choose_slot_task_prompt(
            supervisor_name=supervisor_name,
            supervisor_role=supervisor_role,
            invocation_reason=invocation_reason,
            known_context=known_context,
            expected_result=expected_result,
            already_completed=already_completed,
            previous_step=previous_step,
            current_step=current_step,
            next_step=next_step,
            meeting_slot_description=meeting_slot_description,
        )

        logger.info(
            "Initializing ChooseSlotTask for email: %s, track: %s",
            contact_email,
            track,
        )

        super().__init__(
            instructions=instructions,
            chat_ctx=chat_ctx,
            llm=task_llm,
        )

    async def on_enter(self) -> None:
        """Fetch current slots, persist them in context, and present them."""
        logger.info("Entering ChooseSlotTask, fetching available slots...")

        self.available_slots = await get_slots(track=self.track)

        if not self.available_slots:
            logger.warning(
                "No available slots found for track: %s",
                self.track,
            )

            await self.session.generate_reply(
                instructions=(
                    "Tell the caller briefly that there are no available "
                    "meeting slots right now and that you will follow up "
                    "by email instead."
                )
            )

            self.complete(None)
            return

        logger.info(
            "Retrieved %d available slot(s): %s",
            len(self.available_slots),
            self.available_slots,
        )

        slots_text = "\n".join(
            f"- ID: {slot.id} | Value: {slot.value}"
            for slot in self.available_slots
        )

        self.chat_ctx.add_message(
            role="system",
            content=dedent(
                f"""
                # CURRENT AVAILABLE MEETING SLOTS

                The following slots were retrieved from the calendar for this task:

                {slots_text}

                These slots are authoritative runtime data.

                Rules:
                - Offer only these slots.
                - Preserve each ID-to-value mapping exactly.
                - The caller should hear the human-readable value.
                - `submit_slot` must receive the exact corresponding ID.
                - Never invent, modify, or reinterpret a slot.
                - If the caller requests a time not present here, explain that it is
                unavailable and offer the actual available slots.
                """
            ).strip(),
        )

        await self.update_chat_ctx(self.chat_ctx)

        await self.session.generate_reply(
            instructions=(
                "Briefly tell the caller that you can help them choose a "
                "meeting time. Present all currently available times naturally "
                "and ask which one works best."
            )
        )

    @function_tool
    async def submit_slot(
        self,
        ctx: RunContext,
        slot_id: str,
        read_back: bool,
    ) -> str:
        """Book the selected meeting slot after explicit confirmation."""
        logger.info(
            "submit_slot called with slot_id: '%s', read_back: %s",
            slot_id,
            read_back,
        )

        if self._booking_completed:
            logger.warning("submit_slot called after task completion")
            return (
                "This meeting-slot task has already been completed. "
                "Return the result to the supervisor."
            )

        selected_slot = next(
            (slot for slot in self.available_slots if slot.id == slot_id),
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
                "submit_slot invoked without explicit confirmation"
            )

            return (
                f"Read the chosen slot back to the user as "
                f"'{selected_slot.value}' and get explicit confirmation "
                "before calling this tool again."
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
                "Booking failed for slot_id='%s', slot='%s'",
                selected_slot.id,
                selected_slot.value,
            )

            self._booking_completed = True
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

        self._booking_completed = True
        self.complete(selected_slot.value)

        return f"Slot booked: {selected_slot.value}"
