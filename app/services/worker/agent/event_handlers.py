import asyncio
from datetime import datetime, timezone
from functools import partial
from typing import Coroutine, Any

from livekit.agents.voice import (
    ConversationItemAddedEvent,
    CloseEvent,
    ErrorEvent,
    FunctionToolsExecutedEvent,
)
from livekit.agents.llm.chat_context import ChatMessage
from livekit.agents import AgentSession

from shared.logging_setup import get_logger
from shared.infra.redis import append_turn, get_transcript, delete_transcript
from shared.infra.postgres import save_call_log, save_transcript, save_tool_log

from schemas import UserData, ToolCallRecord
from utils import create_customer, update_customer


_LOGGER = "worker.agent.event_handlers"
logger = get_logger(_LOGGER)


class BackgroundTaskTracker:
    """Tracks per-call background work so shutdown can wait for it.

    Tasks are still asynchronous and do not block LiveKit event callbacks.
    Their exceptions are always retrieved and logged. The tracker is created
    once per call, rather than using a process-global task collection.
    """

    def __init__(self, stream_sid: str) -> None:
        self.stream_sid = stream_sid
        self._tasks: set[asyncio.Task] = set()
        self._transcript_append_tasks: set[asyncio.Task] = set()
        self._cleanup_started = False

    def create_task(
        self,
        coroutine: Coroutine[Any, Any, Any],
        *,
        context: str,
        transcript_append: bool = False,
    ) -> asyncio.Task:
        task = asyncio.create_task(coroutine, name=f"{context}:{self.stream_sid}")
        self._tasks.add(task)
        if transcript_append:
            self._transcript_append_tasks.add(task)
        task.add_done_callback(partial(self._on_task_done, context=context))
        return task

    def _on_task_done(self, task: asyncio.Task, *, context: str) -> None:
        self._tasks.discard(task)
        self._transcript_append_tasks.discard(task)
        if task.cancelled():
            logger.warning(
                "Background task was cancelled [stream_sid=%s context=%s]",
                self.stream_sid,
                context,
            )
            return

        # Retrieving the exception prevents 'Task exception was never retrieved'.
        exception = task.exception()
        if exception is not None:
            logger.error(
                "Background task failed [stream_sid=%s context=%s]",
                self.stream_sid,
                context,
                exc_info=(type(exception), exception, exception.__traceback__),
            )

    def start_call_cleanup(self, userdata: UserData) -> None:
        """Schedule end-of-call writes once, even if close is emitted twice."""
        if self._cleanup_started:
            logger.warning(
                "Ignoring duplicate call cleanup request [stream_sid=%s]",
                self.stream_sid,
            )
            return

        self._cleanup_started = True
        self.create_task(
            self._finish_call_cleanup(userdata),
            context="end_of_call_writes",
        )

    async def _finish_call_cleanup(self, userdata: UserData) -> None:
        # A close event can arrive while Redis append operations from the last
        # conversation events are still running. Let those writes finish before
        # reading the transcript for the final Postgres flush.
        pending_appends = set(self._transcript_append_tasks)
        if pending_appends:
            await asyncio.gather(*pending_appends, return_exceptions=True)
        await _end_of_call_writes(self.stream_sid, userdata)

    async def wait_for_completion(self, timeout: float = 8.0) -> None:
        """Wait briefly for tracked work during LiveKit job shutdown.

        The timeout leaves headroom for LiveKit's own shutdown deadline. Work
        that cannot finish in that window is reported rather than silently
        abandoned. No retry or recovery is attempted here.
        """
        if not self._tasks:
            return

        logger.info(
            "Waiting for %d background task(s) [stream_sid=%s]",
            len(self._tasks),
            self.stream_sid,
        )

        _, pending = await asyncio.wait(
            set(self._tasks),
            timeout=timeout,
            return_when=asyncio.ALL_COMPLETED,
        )

        if pending:
            pending_contexts = [task.get_name() for task in pending]
            logger.error(
                "Background cleanup did not finish before shutdown timeout "
                "[stream_sid=%s pending=%s]",
                self.stream_sid,
                pending_contexts,
            )
            # Do not cancel these tasks here: cancellation during a database
            # write could interrupt a transaction. LiveKit/process shutdown
            # will ultimately own the remaining task lifetime.


# ── Transcript handler ────────────────────────────────────────────────────────

def on_conversation_item_added(
    event: ConversationItemAddedEvent,
    stream_sid: str,
    task_tracker: BackgroundTaskTracker,
) -> None:
    """Append textual conversation messages to the Redis call transcript."""
    if not isinstance(event.item, ChatMessage):
        return

    transcription = event.item.text_content
    if not transcription:
        return

    task_tracker.create_task(
        append_turn(
            stream_sid=stream_sid,
            speaker=str(event.item.role),
            text=transcription,
        ),
        context="append_transcript_turn",
        transcript_append=True,
    )


# ── Tool call handler ─────────────────────────────────────────────────────────

def on_function_tools_executed(
    event: FunctionToolsExecutedEvent,
    userdata: UserData,
    stream_sid: str,
) -> None:
    """Keep tool execution records in memory for persistence after the call."""
    calls = event.function_calls
    outputs = event.function_call_outputs

    if len(calls) != len(outputs):
        logger.warning(
            "Tool call/output count mismatch [stream_sid=%s calls=%d outputs=%d]",
            stream_sid,
            len(calls),
            len(outputs),
        )

    for call, output in zip(calls, outputs):
        try:
            record = ToolCallRecord(
                tool_name=call.name,
                arguments=call.arguments if isinstance(call.arguments, dict) else {},
                result=output.output,
                is_error=output.is_error,
                error_message=output.output if output.is_error else None,
            )
            userdata.tool_call_log.append(record)
        except Exception:
            logger.exception(
                "Could not record tool execution [stream_sid=%s tool=%s]",
                stream_sid,
                getattr(call, "name", "unknown"),
            )


# ── Error handler ─────────────────────────────────────────────────────────────

def on_error(event: ErrorEvent, stream_sid: str, userdata: UserData) -> None:
    """Record LiveKit component errors without changing session behavior.

    This handler intentionally does not retry, recreate providers, or close
    the call. LiveKit's CloseEvent remains the source of call-end metadata.
    """
    error = event.error
    recoverable = getattr(error, "recoverable", None)
    source = event.source

    # Preserve the latest component error on the call record. A later
    # CloseEvent with its own error will replace this with the terminal error.
    userdata.error_detail = (
        f"source={source}; recoverable={recoverable}; "
        f"type={type(error).__name__}; error={error}"
    )

    log_message = (
        "LiveKit session error [stream_sid=%s source=%s recoverable=%s "
        "error_type=%s error=%s]"
    )
    log_args = (
        stream_sid,
        source,
        recoverable,
        type(error).__name__,
        error,
    )

    if recoverable is False:
        logger.error(log_message, *log_args)
    elif recoverable is True:
        logger.warning(log_message, *log_args)
    else:
        # Some event/error implementations may not expose recoverable.
        logger.error(log_message, *log_args)


# ── Close handler ─────────────────────────────────────────────────────────────

def on_close(
    event: CloseEvent,
    stream_sid: str,
    userdata: UserData,
    task_tracker: BackgroundTaskTracker,
) -> None:
    """Capture final call metadata and schedule persistence exactly once."""
    userdata.end_reason = event.reason
    userdata.end_at = datetime.fromtimestamp(event.created_at, tz=timezone.utc)

    if event.error is not None:
        userdata.error_detail = str(event.error)

    logger.info(
        "LiveKit session closed [stream_sid=%s reason=%s error=%s]",
        stream_sid,
        event.reason,
        event.error,
    )
    task_tracker.start_call_cleanup(userdata)


# ── Independent persistence operations ───────────────────────────────────────

async def _persist_crm(userdata: UserData, stream_sid: str) -> None:
    try:
        if userdata.customer_id is None:
            await create_customer(userdata)
        else:
            await update_customer(userdata)
    except Exception:
        logger.exception("CRM persistence failed [stream_sid=%s]", stream_sid)


async def _persist_call_log(userdata: UserData, stream_sid: str) -> None:
    try:
        duration_secs = None
        if userdata.end_at and userdata.started_at:
            duration_secs = max(
                0,
                int((userdata.end_at - userdata.started_at).total_seconds()),
            )

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
            duration_secs=duration_secs,
            qualification_summary=userdata.qualification_summary,
        )
    except Exception:
        logger.exception("Call-log persistence failed [stream_sid=%s]", stream_sid)


async def _persist_transcript(stream_sid: str) -> None:
    """Move transcript to Postgres; only delete Redis data after save succeeds."""
    try:
        turns = await get_transcript(stream_sid)
        if not turns:
            logger.info("No transcript turns to persist [stream_sid=%s]", stream_sid)
            return

        await save_transcript(stream_sid=stream_sid, turns=turns)
        await delete_transcript(stream_sid)
        logger.info("Transcript persisted [stream_sid=%s turns=%d]", stream_sid, len(turns))
    except Exception:
        logger.exception("Transcript persistence failed [stream_sid=%s]", stream_sid)


async def _persist_tool_logs(userdata: UserData, stream_sid: str) -> None:
    """Persist each tool record independently so one failure doesn't stop others."""
    for index, record in enumerate(userdata.tool_call_log):
        try:
            await save_tool_log(
                stream_sid=stream_sid,
                tool_name=record.tool_name,
                arguments=record.arguments,
                called_at=record.called_at,
                result=record.result,
                is_error=record.is_error,
                error_message=record.error_message,
            )
        except Exception:
            logger.exception(
                "Tool-log persistence failed [stream_sid=%s index=%d tool=%s]",
                stream_sid,
                index,
                record.tool_name,
            )


async def _end_of_call_writes(stream_sid: str, userdata: UserData) -> None:
    """Persist call-end data; each category is isolated from the others."""
    logger.info("Starting end-of-call persistence [stream_sid=%s]", stream_sid)

    # Keep these operations sequential. The repository helpers may share an
    # async database engine/session infrastructure, and AsyncSession instances
    # must not be used concurrently. Each helper catches its own exception, so
    # a failure in one category does not prevent the next category from running.
    await _persist_crm(userdata, stream_sid)
    await _persist_call_log(userdata, stream_sid)
    await _persist_transcript(stream_sid)
    await _persist_tool_logs(userdata, stream_sid)

    logger.info("End-of-call persistence finished [stream_sid=%s]", stream_sid)


# ── Event registration ────────────────────────────────────────────────────────

def register_event_handlers(
    session: AgentSession,
    stream_sid: str,
    user_data: UserData,
) -> BackgroundTaskTracker:
    """Register call event handlers and return the per-call task tracker."""
    task_tracker = BackgroundTaskTracker(stream_sid)

    session.on(
        "conversation_item_added",
        partial(
            on_conversation_item_added,
            stream_sid=stream_sid,
            task_tracker=task_tracker,
        ),
    )
    session.on(
        "close",
        partial(
            on_close,
            stream_sid=stream_sid,
            userdata=user_data,
            task_tracker=task_tracker,
        ),
    )
    session.on(
        "function_tools_executed",
        partial(on_function_tools_executed, userdata=user_data, stream_sid=stream_sid),
    )
    session.on(
        "error",
        partial(on_error, stream_sid=stream_sid, userdata=user_data),
    )

    return task_tracker
