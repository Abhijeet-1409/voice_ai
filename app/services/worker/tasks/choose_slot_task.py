import asyncio
from typing import Optional

from livekit.agents import AgentTask, RunContext, ToolError, function_tool

from shared.config import Track
from shared.logging_setup import get_logger

from utils import get_slots, confirm_booking


_LOGGER = "worker.tasks.choose_slot_task"
logger = get_logger(_LOGGER)


_BOOKING_TIMEOUT_SECONDS = 10


class ChooseSlotTask(AgentTask[Optional[str]]):
    """Helps the user pick a meeting slot and books it on confirmation.

    Fetches available slots via `get_slots` on enter, walks the user
    through picking one, requires explicit confirmation before booking,
    and books via `confirm_booking`. Fails soft for a normal decline (no
    slots available, or the calendar client reports booking failed): the
    task completes `None` and the caller is expected to offer a
    follow-up instead of treating it as an error. A genuine timeout
    (the booking call hung) is treated as unexpected and raises
    ToolError instead — distinct from a normal decline.

    Result:
        The booked slot string on success, `None` if no slots were
        available or booking was declined by the calendar client.
    """

    def __init__(
        self,
        contact_email: str,
        track: Optional[Track] = None,
        chat_ctx=None,
    ):
        """
        Args:
            contact_email: Previously confirmed email to send the invite to.
            track: Optional track context, passed through to the calendar client.
        """
        self.contact_email = contact_email
        self.track = track
        self.available_slots: list[str] = []

        logger.info("Initializing ChooseSlotTask for email: %s, track: %s", contact_email, track)

        super().__init__(
            instructions="""
            Help the user pick a meeting time from the available slots
            you'll be given shortly.

            Read out the available slots in natural, conversational
            language. Ask which one works for them.

            Once they pick one, read it back clearly and ask them to
            confirm it's the one they want. Only call `submit_slot` after
            they have explicitly confirmed. If they change their mind,
            help them pick again.

            Only offer slots from the list you were given. If the user
            asks for a time that isn't in the list, let them know it's
            not available and offer the closest options instead.
            """,
            chat_ctx=chat_ctx,
        )

    async def on_enter(self) -> None:
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
            Let the user know you'll help them find a meeting time.
            Available slots:
            {slots_text}
            Read these out naturally and ask which works best.
            """
        )

    @function_tool
    async def submit_slot(self, ctx: RunContext, slot: str, read_back: bool) -> str:
        """Submit and book the chosen slot.

        Args:
            ctx (RunContext): The LiveKit agent execution context.
            slot: The selected slot, exactly as given in the available
                slots list.
            read_back: Set to True only after you have read the chosen
                slot back to the user in natural language and they have
                explicitly confirmed it's the one they want.
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