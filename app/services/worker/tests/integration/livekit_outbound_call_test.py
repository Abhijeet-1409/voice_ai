"""
Test an agent-initiated outbound PSTN call through a configured LiveKit SIP
outbound trunk.

The AWS-referred prospect profile is loaded from outbound_call_test.json and
attached to:
1. Agent dispatch metadata, readable in the worker through ctx.job.metadata.
2. SIP participant metadata, readable from the SIP participant after it joins.

This script does not create an outbound carrier connection. A valid outbound
SIP trunk and an outbound-capable telephony provider are prerequisites.
"""

import asyncio
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

from livekit import api


# File location in the repository:
# app/services/worker/tests/integration/test_livekit_outbound_dispatch.py
# app/data/outbound_call_test.json
APP_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = APP_ROOT / "data" / "outbound_call_test.json"
ROOM_PREFIX = "worker-playground-outbound"


def load_config() -> dict[str, Any]:
    """Load and validate the call configuration and prospect profile."""
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = json.load(file)

    destination = config.get("destination_phone_number")
    prospect = config.get("prospect")

    if not isinstance(prospect, dict):
        raise ValueError("'prospect' must be a JSON object.")

    contact = prospect.get("contact")
    if not isinstance(contact, dict):
        raise ValueError("'prospect.contact' must be a JSON object.")

    if not destination:
        destination = contact.get("phone_e164")

    if not isinstance(destination, str) or not re.fullmatch(
        r"\+[1-9]\d{7,14}", destination
    ):
        raise ValueError(
            "destination_phone_number (or prospect.contact.phone_e164) "
            "must be in E.164 format, e.g. +14155550123."
        )

    if contact.get("phone_e164") and contact["phone_e164"] != destination:
        raise ValueError(
            "destination_phone_number and prospect.contact.phone_e164 "
            "must contain the same number."
        )

    config["destination_phone_number"] = destination
    return config


def build_call_metadata(
    config: dict[str, Any],
    room_name: str,
) -> dict[str, Any]:
    """
    Build a compact top-level call context plus the complete prospect summary.

    The top-level fields preserve compatibility with the existing UserData
    construction pattern. The nested 'prospect' object carries additional
    AWS referral, contact, company, and discovery context.
    """
    prospect = config["prospect"]
    contact = prospect["contact"]
    company = prospect.get("company") or {}
    destination = config["destination_phone_number"]

    return {
        "metadata_version": "1.0",
        "stream_sid": f"outbound-{uuid.uuid4().hex[:12]}",
        "channel": "phone",
        "call_type": "outreach",
        "direction": "outbound",
        "room_name": room_name,

        # Top-level fields map cleanly to the existing UserData model.
        "phone": destination,
        "user_id": None,
        "customer_id": prospect.get("customer_id"),
        "name": contact.get("full_name"),
        "email": contact.get("email"),
        "company": company.get("name"),
        "job_title": contact.get("job_title"),
        "track": None,
        "qualified": False,
        "qualification_summary": None,
        "lifecyclestage": "lead",

        "campaign": config.get("campaign") or {},

        # Keep the complete profile available to the job entrypoint.
        "prospect": prospect,
    }


async def main() -> None:
    livekit_url = os.environ["LIVEKIT_URL"]
    api_key = os.environ["LIVEKIT_API_KEY"]
    api_secret = os.environ["LIVEKIT_API_SECRET"]
    agent_name = os.environ["AGENT_NAME"]
    outbound_trunk_id = os.environ["SIP_OUTBOUND_TRUNK_ID"]
    room_timeout = int(os.getenv("LIVEKIT_ROOM_TIMEOUT", "300"))

    if not outbound_trunk_id.startswith("ST_"):
        raise ValueError(
            "SIP_OUTBOUND_TRUNK_ID must be the ID of a configured outbound "
            "SIP trunk (normally starts with ST_)."
        )

    config = load_config()
    destination = config["destination_phone_number"]
    room_name = f"{ROOM_PREFIX}-{uuid.uuid4().hex[:10]}"

    call_metadata = build_call_metadata(config, room_name)
    metadata_json = json.dumps(
        call_metadata,
        separators=(",", ":"),
        ensure_ascii=False,
    )

    async with api.LiveKitAPI(
        url=livekit_url,
        api_key=api_key,
        api_secret=api_secret,
    ) as lkapi:
        # Create one room for both the agent and the called SIP participant.
        room = await lkapi.room.create_room(
            api.CreateRoomRequest(
                name=room_name,
                empty_timeout=room_timeout,
                metadata=json.dumps({
                    "stream_sid": call_metadata["stream_sid"],
                    "channel": "phone",
                    "call_type": "outreach",
                    "direction": "outbound",
                }),
            )
        )
        print(f"Room created: {room.name}")

        # The worker can read this from ctx.job.metadata as soon as the job starts.
        dispatch = await lkapi.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                room=room_name,
                agent_name=agent_name,
                metadata=metadata_json,
            )
        )
        print(f"Agent dispatch created: {dispatch.id}")

        contact_name = call_metadata.get("name") or "Outbound Contact"
        participant_identity = f"outbound-{call_metadata['stream_sid']}"

        # This creates the actual outbound PSTN leg through the configured trunk.
        sip_request = api.CreateSIPParticipantRequest(
            sip_trunk_id=outbound_trunk_id,
            sip_call_to=destination,
            room_name=room_name,
            participant_identity=participant_identity,
            participant_name=contact_name,
            participant_metadata=metadata_json,
            wait_until_answered=True,
        )

        try:
            sip_participant = await lkapi.sip.create_sip_participant(sip_request)
        except api.SipCallError as exc:
            print(f"Outbound SIP call failed: {exc}")
            raise
        except api.ServerError as exc:
            print(f"LiveKit API error: {exc.code}: {exc.message}")
            raise

        print("\nOutbound call answered successfully.")
        print(f"Room: {room_name}")
        print(f"Agent dispatch: {dispatch.id}")
        print(f"SIP participant: {sip_participant.participant_identity}")
        print(f"Destination: {destination}")
        print(f"Prospect: {contact_name}")


if __name__ == "__main__":
    asyncio.run(main())
