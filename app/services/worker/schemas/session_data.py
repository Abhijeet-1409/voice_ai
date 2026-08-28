from dataclasses import dataclass, field as dc_field
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field

from livekit.agents.voice import CloseReason

from shared.config import Channel, CallType, Track, LifecycleStage


@dataclass(frozen=True)
class ToolCallRecord:
    """
    Immutable record of a single tool invocation during a call session.
    Accumulated in-memory on UserData.tool_call_log throughout the call,
    then bulk-written to the tool_log table at call end (on_close) —
    avoids a DB round-trip on every tool call during the live,
    latency-sensitive conversation.
    """
    tool_name: str
    arguments: dict[str, Any]
    result: str
    is_error: bool
    error_message: Optional[str] = None
    called_at: datetime = dc_field(default_factory=lambda: datetime.now(timezone.utc))


class UserData(BaseModel):
    """
    Tracks the comprehensive runtime state, routing metadata, and
    CRM-mirrored contact details for an individual voice call session.

    CRM-mirrored fields (customer_id, name, email, track, qualified,
    lifecyclestage) intentionally mirror shared.infra.postgres.contact.Contact
    one-to-one. Tools mutate these fields directly, in-memory, during the
    call — there is NO mid-call CRM write. The actual CRM record is only
    created or updated ONCE, at call end (event_handlers.py's on_close),
    via utils/customer_lookup.py's create_customer (new caller) or
    update_customer (existing caller, customer_id already set). This
    keeps every CRM write off the live, latency-sensitive call path.

    Population differs by channel (see design decision):
      - PHONE: channel, phone, and all CRM-mirrored fields are populated
        by job_entrypoint.py BEFORE the agent joins, via lookup_customer()
        using the phone number from room metadata. If not found,
        customer_id stays None for the whole call — creation is deferred
        to call end, not attempted mid-call.
      - WEB (not yet in scope — SIP/phone only for now): would start
        with CRM-mirrored fields empty and populate them mid-call.
    """

    model_config = {
        "arbitrary_types_allowed": True,  # required for CloseReason (livekit type, not Pydantic-native)
        "validate_assignment": True,      # re-validate on mid-call mutation, not just construction
    }

    # --- Routing metadata — always known at construction, from room metadata ---
    channel: Channel
    call_type: CallType
    stream_sid: str

    # --- Identity ---
    clerk_id: Optional[str] = None       # web channel only, not yet in scope
    phone: Optional[str] = None          # always set for phone channel from SIP metadata
    user_id: Optional[str] = None

    # --- CRM-mirrored fields (mirrors shared.infra.postgres.contact.Contact) ---
    # None/False/LEAD if no matching contact was found — customer_id
    # stays None until call end for a genuinely new caller.
    customer_id: Optional[str] = None
    name: Optional[str] = None
    email: Optional[str] = None
    track: Optional[Track] = None
    qualified: bool = False
    lifecyclestage: LifecycleStage = LifecycleStage.LEAD
    call_sid: Optional[str] = None
    # --- Call-specific qualification context (NOT on Contact — see
    # domain/tools/sales_qualification.py's qualify_lead docstring.
    # Written to call_log at call end, not the contact record, since
    # it describes why THIS call resulted in qualification, not a
    # permanent contact attribute) ---
    qualification_summary: Optional[str] = None

    # --- Scheduling state (in-session; not on Contact itself) ---
    meeting_scheduled: bool = False
    meeting_slot: Optional[str] = None

    # --- Tool call log (accumulated in-memory, bulk-written on call end) ---
    tool_call_log: list[ToolCallRecord] = Field(default_factory=list)

    # --- Determines whether the session uses a realtime (speech-to-speech)
    # model instead of a cascaded STT/LLM/TTS pipeline. Defaults to False —
    # cascaded pipeline is the current standard path; realtime is not yet
    # compatible with the Task-based email/meeting confirmation flow. ---
    is_realtime_model: bool = False
    
    # --- Lifecycle ---
    end_reason: Optional[CloseReason] = None
    error_detail: Optional[str] = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    end_at: Optional[datetime] = None