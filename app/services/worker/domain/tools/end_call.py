from typing import Literal

from livekit.agents import RunContext, function_tool


@function_tool
async def end_call(
    ctx: RunContext,
    confirmation: Literal["request", "confirm", "cancel"] = "request",
) -> str:
    """End the current call only after the caller explicitly confirms.

    Confirmation flow:
    1. When the caller asks to end the call, call this tool with
       confirmation="request" (the default).
    2. Ask: "Are you sure you want me to end this call?"
    3. If the caller clearly says yes, call with confirmation="confirm".
    4. If the caller says no, call with confirmation="cancel".

    Never call with confirmation="confirm" unless the caller explicitly
    confirms after being asked.

    Args:
        ctx: LiveKit function-tool context. Its userdata must be the per-call
            UserData model containing end_call_confirmation_pending.
        confirmation: "request" starts the confirmation flow; "confirm"
            ends the call when confirmation is pending; "cancel" clears the
            pending request and leaves the call active.

    Returns:
        A customer-facing instruction/result for the agent.
    """
    userdata = ctx.userdata

    if confirmation == "request":
        userdata.end_call_confirmation_pending = True
        return (
            "Call-ending confirmation is pending. Ask the caller: "
            "'Are you sure you want me to end this call?' Do not end the call "
            "until the caller explicitly confirms."
        )

    if confirmation == "cancel":
        if not userdata.end_call_confirmation_pending:
            return "There is no pending call-ending confirmation. Continue the conversation normally."

        userdata.end_call_confirmation_pending = False
        return "Call ending cancelled. Continue the conversation normally."

    # confirmation == "confirm"
    if not userdata.end_call_confirmation_pending:
        return (
            "There is no pending call-ending confirmation. Do not end the call. "
            "If the caller still wants to leave, ask for confirmation first."
        )

    # Make the goodbye speech uninterruptible so the caller hears it before
    # the session shutdown is requested.
    ctx.disallow_interruptions()
    await ctx.session.say(
        "Thank you for calling. Goodbye.",
        allow_interruptions=False,
    )

    # Clear the state after the goodbye has been delivered, then request a
    # graceful shutdown. shutdown(drain=True) is non-blocking in LiveKit.
    userdata.end_call_confirmation_pending = False
    ctx.session.shutdown(drain=True)

    return "The caller confirmed. The goodbye message was played and session shutdown was requested."
