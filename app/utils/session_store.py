import json
import uuid
from datetime import datetime

import redis

from config.settings import settings


# ── Redis connection ───────────────────────────────────────────────────────────

client = redis.Redis(
    host=settings.redis_host,
    port=settings.redis_port,
    decode_responses=True,
)

SESSION_TTL = 3600  # 1 hour auto expiry


# ── Session ID ────────────────────────────────────────────────────────────────

def generate_session_id() -> str:
    return "sess_" + uuid.uuid4().hex[:12]


# ── Create ────────────────────────────────────────────────────────────────────

def create_session(session_id: str) -> dict:
    session = {
        "session_id"     : session_id,
        "caller_phone"   : None,
        "caller_name"    : None,
        "caller_email"   : None,
        "caller_need"    : None,
        "interest_level" : None,
        "start_time"     : datetime.utcnow().isoformat(),
        "end_time"       : None,
        "exchange_count" : 0,
        "is_ended"       : False,
        "exchanges"      : [],
    }
    _save(session_id, session)
    return session


# ── Read ──────────────────────────────────────────────────────────────────────

def get_session(session_id: str) -> dict | None:
    data = client.get(session_id)
    if data is None:
        return None
    return json.loads(data)


# ── Add exchange ──────────────────────────────────────────────────────────────

def add_exchange(
    session_id    : str,
    caller_message: str,
    agent_reply   : str,
    extracted_info: dict | None = None,
) -> None:
    session = get_session(session_id)
    if session is None:
        return

    session["exchange_count"] += 1

    session["exchanges"].append({
        "exchange_number" : session["exchange_count"],
        "caller_message"  : caller_message,
        "agent_reply"     : agent_reply,
        "timestamp"       : datetime.utcnow().isoformat(),
    })

    # Update extracted fields if Gemini returned them
    if extracted_info:
        for field in ("caller_name", "caller_phone", "caller_email",
                      "caller_need", "interest_level"):
            value = extracted_info.get(field)
            if value and session[field] is None:
                session[field] = value

    _save(session_id, session)


# ── End session ───────────────────────────────────────────────────────────────

def end_session(session_id: str) -> dict | None:
    session = get_session(session_id)
    if session is None:
        return None

    session["is_ended"] = True
    session["end_time"] = datetime.utcnow().isoformat()
    _save(session_id, session)
    return session


# ── Delete ────────────────────────────────────────────────────────────────────

def delete_session(session_id: str) -> None:
    client.delete(session_id)


# ── Internal save ─────────────────────────────────────────────────────────────

def _save(session_id: str, session: dict) -> None:
    client.setex(session_id, SESSION_TTL, json.dumps(session))