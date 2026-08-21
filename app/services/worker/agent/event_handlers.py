import asyncio
from datetime import datetime, timezone

from livekit.agents.voice import (
    ConversationItemAddedEvent,
    CloseEvent,
    ErrorEvent,
    FunctionToolsExecutedEvent,
)
from livekit.agents.llm.chat_context import ChatMessage

from shared.logging_setup import get_logger
from shared.infra.redis import append_turn, get_transcript, delete_transcript
from shared.infra.postgres import save_call_log, save_transcript, save_tool_log

from schemas import UserData, ToolCallRecord
from utils import create_customer, update_customer


_LOGGER = "worker.agent.event_handlers"
logger = get_logger(_LOGGER)


# ── Background task exception handling ─────────────────────────────────────────
# asyncio.create_task() silently swallows exceptions unless the returned Task's
# result/exception is retrieved somewhere. Since these tasks are fire-and-forget
# (nothing ever awaits or checks them directly), a done-callback is the only way
# to surface a failure — without it, a bug here would fail completely silently.

def _log_task_exception(task: asyncio.Task, *, context: str) -> None:
    """
    Done-callback for fire-and-forget asyncio.create_task() calls.
    Logs any exception the task raised, since nothing else observes it.
    """
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error(f"Background task failed [{context}]: {exc}", exc_info=exc)


def _create_tracked_task(coro, *, context: str) -> asyncio.Task:
    """
    Wraps asyncio.create_task() with a done-callback so exceptions in
    fire-and-forget tasks are logged instead of silently swallowed.
    """
    task = asyncio.create_task(coro)
    task.add_done_callback(lambda t: _log_task_exception(t, context=context))
    return task


# ── Transcript handler ────────────────────────────────────────────────────────

def on_conversation_item_added(
    event: ConversationItemAddedEvent,
    stream_sid: str
) -> None:
    """
    Fires on every new conversation item (agent or caller utterance).
    Appends the turn to the Redis transcript for this call.
    Only processes ChatMessage items with text content — ignores tool
    calls, system messages, and any non-text content.
    """
    if not isinstance(event.item, ChatMessage):
        return

    transcription = event.item.text_content
    if not transcription:
        return

    role = str(event.item.role)
    _create_tracked_task(
        append_turn(
            stream_sid=stream_sid,
            speaker=role,
            text=transcription
        ),
        context=f"append_turn stream_sid={stream_sid}",
    )


# ── Tool call handler ─────────────────────────────────────────────────────────

def on_function_tools_executed(
    event: FunctionToolsExecutedEvent,
    userdata: UserData
) -> None:
    """
    Fires after every batch of tool calls completes.
    Accumulates each tool call as a ToolCallRecord on userdata.tool_call_log
    for bulk write to Postgres at call end. Does NOT write to Postgres here —
    the call is still live and the event handler must not block.
    """
    for call, output in zip(event.function_calls, event.function_call_outputs):
        record = ToolCallRecord(
            tool_name=call.name,
            arguments=call.arguments if isinstance(call.arguments, dict) else {},
            result=output.output,
            is_error=output.is_error,
            error_message=output.error_message if output.is_error else None,
        )
        userdata.tool_call_log.append(record)
        logger.debug(
            f"Tool call recorded — tool={call.name} "
            f"is_error={output.is_error}"
        )


# ── Error handler ─────────────────────────────────────────────────────────────

def on_error(
    event: ErrorEvent,
    stream_sid: str
) -> None:
    """
    Fires on session-level errors (STT / LLM / TTS / realtime model failures).
    Logs the error. Does not raise — a crash here could interfere with
    cleanup.
    """
    logger.error(
        f"Session error [stream_sid={stream_sid}] "
        f"source={event.source} error={event.error}"
    )


# ── Close handler ─────────────────────────────────────────────────────────────

def on_close(
    event: CloseEvent,
    stream_sid: str,
    userdata: UserData
) -> None:
    """
    Fires when the session closes (caller hangs up, timeout, or error).
    Stamps end metadata onto userdata and schedules the bulk end-of-call
    write as a background task so the handler itself does not block shutdown.
    """
    userdata.end_reason = event.reason
    userdata.end_at = datetime.fromtimestamp(
        event.created_at, tz=timezone.utc
    )

    if event.error is not None:
        userdata.error_detail = str(event.error)

    logger.info(
        f"Session closed [stream_sid={stream_sid}] "
        f"reason={event.reason} error={event.error}"
    )

    _create_tracked_task(
        _end_of_call_writes(stream_sid, userdata),
        context=f"_end_of_call_writes stream_sid={stream_sid}",
    )


# ── End-of-call bulk write ────────────────────────────────────────────────────

async def _end_of_call_writes(stream_sid: str, userdata: UserData) -> None:
    """
    Runs as a background task after the session closes.
    Order:
      1. CRM create or update (deferred from mid-call)
      2. Post-call follow-up email (TODO — not yet implemented)
      3. save_call_log
      4. save_transcript (flush Redis → Postgres)
      5. save_tool_log (bulk write accumulated ToolCallRecords)

    All steps are non-raising — a failure in any one step is logged and
    execution continues so the remaining steps still run.
    """

    # ── 1. CRM write ──────────────────────────────────────────────────────────
    try:
        if userdata.customer_id is None:
            await create_customer(userdata)
        else:
            await update_customer(userdata)
    except Exception as e:
        logger.error(
            f"CRM write failed at call end [stream_sid={stream_sid}]: {e}"
        )

    # ── 2. Post-call follow-up email ──────────────────────────────────────────
    # TODO: send post-call follow-up email once email_sender util is built
    # if userdata.email and userdata.meeting_scheduled:
    #     await send_followup_email(
    #         to_email=userdata.email,
    #         recipient_name=userdata.name,
    #         track=userdata.track,
    #         meeting_slot=userdata.meeting_slot,
    #     )

    # ── 3. save_call_log ──────────────────────────────────────────────────────
    try:
        await save_call_log(
            started_at=userdata.started_at,
            channel=userdata.channel,
            call_type=userdata.call_type,
            stream_sid=stream_sid,
            caller_phone=userdata.phone,
            call_sid=userdata.call_sid,
            user_id=userdata.user_id,
            ended_at=userdata.end_at,
            end_reason=str(userdata.end_reason) if userdata.end_reason else None,
            duration_secs=(
                int((userdata.end_at - userdata.started_at).total_seconds())
                if userdata.end_at and userdata.started_at
                else None
            ),
            qualification_summary=userdata.qualification_summary,
        )
    except Exception as e:
        logger.error(
            f"save_call_log failed [stream_sid={stream_sid}]: {e}"
        )

    # ── 4. save_transcript ────────────────────────────────────────────────────
    try:
        turns = await get_transcript(stream_sid)
        if turns:
            await save_transcript(stream_sid=stream_sid, turns=turns)
            await delete_transcript(stream_sid)
        else:
            logger.warning(
                f"No transcript turns found [stream_sid={stream_sid}]"
            )
    except Exception as e:
        logger.error(
            f"save_transcript failed [stream_sid={stream_sid}]: {e}"
        )

    # ── 5. save_tool_log ──────────────────────────────────────────────────────
    try:
        for record in userdata.tool_call_log:
            await save_tool_log(
                stream_sid=stream_sid,
                tool_name=record.tool_name,
                arguments=record.arguments,
                called_at=record.called_at,
                result=record.result,
                is_error=record.is_error,
                error_message=record.error_message,
            )
    except Exception as e:
        logger.error(
            f"save_tool_log failed [stream_sid={stream_sid}]: {e}"
        )

    logger.info(f"End-of-call writes complete [stream_sid={stream_sid}]")