from livekit.agents import function_tool, RunContext

from shared.logging_setup import get_logger
from shared.config import Track, LifecycleStage


_LOGGER = "worker.domain.tools.sales_qualification"
logger = get_logger(_LOGGER)


@function_tool
async def qualify_lead(
    ctx: RunContext,
    track: Track,
    qualification_summary: str,
) -> str:
    """
    Marks the current caller as a qualified lead for one of the three
    AWS partner tracks. Only call this on a clear, specific signal from
    the caller — never a guess. Do not call this if the caller context
    already shows they're previously qualified.

    Identity is read from ctx.userdata.customer_id, not supplied by the
    LLM — the caller's identity is always already resolved (SIP-only
    phone channel, resolved by job_entrypoint.py before the agent joins).

    Args:
        ctx (RunContext): The LiveKit agent execution context.
        track (Track): Which of the three AWS partner tracks the caller
            is interested in.
        qualification_summary (str): A brief summary of the caller's
            stated needs or interests. Stored on the call record
            (call_log), not the contact record — this captures why THIS
            call resulted in qualification, not a permanent contact
            attribute.

    Returns:
        str: A confirmation message.
    """
    if ctx.userdata.qualified:
        logger.debug(
            f"qualify_lead called for already-qualified customer_id={ctx.userdata.customer_id} — skipping."
        )
        return "This caller is already a qualified lead — no action needed."

    ctx.userdata.track = track
    ctx.userdata.qualified = True
    ctx.userdata.lifecyclestage = LifecycleStage.SALES_QUALIFIED_LEAD
    ctx.userdata.qualification_summary = qualification_summary

    logger.info(
        f"Qualified lead (in-memory, synced to CRM at call end) — "
        f"customer_id={ctx.userdata.customer_id} track={track.value} "
        f"summary={qualification_summary}"
    )
    return "Successfully logged the qualified lead."