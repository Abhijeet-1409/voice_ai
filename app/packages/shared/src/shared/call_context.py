from contextvars import ContextVar


# ── Context variable for stream_sid ─────────────────────────────────────────
stream_sid_var: ContextVar[str] = ContextVar("stream_sid", default="-")