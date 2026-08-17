from dataclasses import dataclass, field as dc_field
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field

from livekit.agents.voice import CloseReason

from shared.config.constants import Channel, CallType, Track, LifecycleStage


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
    called_at: datetime = dc_field(default_factory=lambda: datetime.now(timezone.utc))


class UserData(BaseModel):
    """
    Tracks the comprehensive runtime state, routing metadata, and
    CRM-mirrored contact details for an individual voice call session.

    CRM-mirrored fields (customer_id, name, email, track, qualified,
    lifecyclestage) intentionally mirror shared.infra.postgres.contact.Contact
    one-to-one. This lets the agent read/update contact state in-memory
    during the call without a CRM round-trip on every check, while
    domain/tools/crm.py is responsible for keeping the actual CRM record
    in sync via update_contact whenever these fields change mid-call.

    Population differs by channel (see design decision):
      - PHONE: channel, phone, and all CRM-mirrored fields are populated
        by job_entrypoint.py BEFORE the agent joins, via get_contact()/
        create_contact() using the phone number from room metadata.
      - WEB: channel and clerk_id are set at construction; phone and all
        CRM-mirrored fields start None and are populated mid-call, once
        get_customer_profile (domain/tools/crm.py) collects a phone
        number from the caller and performs the same CRM lookup.
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
    clerk_id: Optional[str] = None       # web channel only
    phone: Optional[str] = None          # always set for phone channel; set mid-call for web

    # --- CRM-mirrored fields (mirrors shared.infra.postgres.contact.Contact) ---
    customer_id: Optional[str] = None
    name: Optional[str] = None
    email: Optional[str] = None
    track: Optional[Track] = None
    qualified: bool = False
    lifecyclestage: LifecycleStage = LifecycleStage.LEAD

    # --- Scheduling state (in-session; not on Contact itself) ---
    meeting_scheduled: bool = False
    meeting_slot: Optional[str] = None

    # --- Tool call log (accumulated in-memory, bulk-written on call end) ---
    tool_call_log: list[ToolCallRecord] = Field(default_factory=list)

    # --- Lifecycle ---
    end_reason: Optional[CloseReason] = None
    error_detail: Optional[str] = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    end_at: Optional[datetime] = None