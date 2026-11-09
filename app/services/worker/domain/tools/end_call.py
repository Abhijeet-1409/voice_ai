from typing import Literal

from livekit.agents import RunContext, function_tool


@function_tool
async def end_call(
    ctx: RunContext,
    confirmation: Literal["request", "confirm", "cancel"] = "request",
) -> str:
    """
    End the current call only after the caller explicitly confirms.

    Confirmation flow:
    1. When the caller asks to end the call, call this tool with the default
       confirmation="request". It records that confirmation is pending.
    2. Ask the caller: "Are you sure you want me to end this call?"
    3. If the caller clearly says yes, call this tool with confirmation="confirm".
    4. If the caller says no, call this tool with confirmation="cancel".

    Do not use confirmation="confirm" unless the caller has explicitly said
    yes after being asked for confirmation.

    Args:
        ctx (RunContext): The LiveKit agent execution context.
        confirmation (Literal["request", "confirm", "cancel"]):
            "request" starts the confirmation loop.
            "confirm" ends the call, but only if a confirmation is pending.
            "cancel" clears a pending request and keeps the call active.

    Returns:
        str: The result of the confirmation or call-ending operation.
    """

    # This field must be defined on the per-call UserData object.
    pending = ctx.userdata.end_call_confirmation_pending

    if confirmation == "request":
        ctx.userdata.end_call_confirmation_pending = True
        return (
            "Call-ending confirmation is now pending. Ask the caller: "
            "'Are you sure you want me to end this call?' "
            "Do not end the call until the caller explicitly confirms."
        )

    if confirmation == "cancel":
        if not pending:
            return "There is no pending call-ending confirmation."

        ctx.userdata.end_call_confirmation_pending = False
        return "Call ending cancelled. Continue the conversation normally."

    # confirmation == "confirm"
    if not pending:
        return (
            "There is no pending call-ending confirmation. "
            "Do not end the call. If the caller still wants to leave, "
            "start the confirmation process first."
        )

    # Prevent the confirmation tool operation from being interrupted.
    ctx.disallow_interruptions()

    # Clear the flag before shutdown so the action cannot be repeated.
    ctx.userdata.end_call_confirmation_pending = False

    # Speak the goodbye and wait until it has finished playing.
    await ctx.session.say(
        "Thank you for calling. Goodbye.",
        allow_interruptions=False,
    )

    # Request a graceful session shutdown after the goodbye has played.
    ctx.session.shutdown(drain=True)

    return "The caller confirmed. The goodbye message was played and session shutdown was requested."
