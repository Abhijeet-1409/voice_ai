from typing import Optional

from livekit.agents import function_tool, RunContext


@function_tool
async def update_caller_info(
    ctx: RunContext,
    name: Optional[str] = None,
    email: Optional[str] = None,
    read_back: bool = False,
) -> str:
    """
    Records or corrects the caller's name and/or email, in memory, for
    this call. Covers every scenario where this comes up: a new caller
    stating their name for the first time, an existing caller
    correcting a name or email already on file, or a caller providing
    either unprompted mid-conversation.

    Only call this AFTER reading back whatever value(s) you're about to
    record and the caller has confirmed them are correct — do not call
    it based on a single unconfirmed mention, since a misheard name or
    email (common over voice/STT) would otherwise get silently
    recorded as fact.

    Args:
        ctx (RunContext): The LiveKit agent execution context.
        name (str, optional): The caller's name, exactly as they stated
            and you read back to them. Omit if only updating email.
        email (str, optional): The caller's email, exactly as they
            stated and you read back to them (spell it out letter by
            letter if needed to confirm). Omit if only updating name.
        read_back (bool): Set to True only after you have read back
            every value you're passing here and the caller has
            confirmed it's correct. Defaults to False.

    Returns:
        str: Confirmation, or an instruction to read back first if
            read_back was not set.
    """
    if not read_back:
        return "Read back the name/email to the caller first, get their confirmation, then call this tool again with read_back=True."

    if name is None and email is None:
        return "No name or email provided — nothing to update."

    if name is not None:
        ctx.userdata.name = name
    if email is not None:
        ctx.userdata.email = email

    return "Got it, recorded."