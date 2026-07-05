import base64

from typing import Optional, Type

from bridge.events import ExotelEvent, _EVENT_MAP


def parse_exotel_event(raw: dict) -> ExotelEvent:
    """
    Factory: converts a raw JSON dict from Exotel's WebSocket into a typed event.
    This is the ONE place in the app that touches Exotel's raw wire format —
    everything downstream (gateway.py, tasks.py) works with typed objects only.
    """
    event_type: str = raw.get("event")
    cls: Optional[Type[ExotelEvent]] = _get_exotel_event_class(event_type)
    if cls is None:
        raise ValueError(f"Unknown Exotel event type: {event_type!r}")
    return cls(**_extract_fields(event_type, raw))

def _get_exotel_event_class(event_type: str) -> Optional[Type[ExotelEvent]]:
    """
    Maps an Exotel event type string to the corresponding dataclass.
    """
    return _EVENT_MAP.get(event_type)

def _extract_fields(event_type: str, raw: dict) -> dict:
    """
    Extracts fields from the raw JSON dict and maps them to the corresponding dataclass fields.
    """
    match event_type:

        case "connected":
            return {}

        case "start":
            start_data: dict = raw.get("start", {})
            media_format: dict = start_data.get("media_format", {})

            return {
                "stream_sid": raw.get("stream_sid",""),
                "sequence_number": raw.get("sequence_number", ""),
                "call_sid": start_data.get("call_sid", ""),
                "account_sid": start_data.get("account_sid", ""),
                "from_number": start_data.get("from", ""),
                "to_number": start_data.get("to", ""),
                "custom_parameters": start_data.get("custom_parameters", {}),
                "encoding": media_format.get("encoding", ""),
                "sample_rate": media_format.get("sample_rate", ""),
                "bit_rate": media_format.get("bit_rate", ""),
            }

        case "media":
            media_data: dict = raw.get("media", {})

            return {
                "sequence_number": raw.get("sequence_number", ""),
                "stream_sid": raw.get("stream_sid", ""),
                "chunk": media_data.get("chunk", ""),
                "timestamp": media_data.get("timestamp", ""),
                "payload": base64.b64decode(media_data.get("payload", "")),
            }

        case "dtmf":
            dtmf_data: dict = raw.get("dtmf", {})

            return {
                "sequence_number": raw.get("sequence_number", ""),
                "stream_sid": raw.get("stream_sid", ""),
                "digit": dtmf_data.get("digit", ""),
                "duration": dtmf_data.get("duration",""),
            }

        case "mark":
            mark_data: dict = raw.get("mark", {})

            return {
                "sequence_number": raw.get("sequence_number", ""),
                "stream_sid": raw.get("stream_sid", ""),
                "name": mark_data.get("name", ""),
            }

        case "stop":
            stop_data: dict = raw.get("stop", {})

            return {
                "sequence_number": raw.get("sequence_number", ""),
                "stream_sid": raw.get("stream_sid", ""),
                "call_sid": stop_data.get("call_sid", ""),
                "account_sid": stop_data.get("account_sid", ""),
                "reason": stop_data.get("reason", ""),
            }

        case _:
            raise ValueError(f"Unhandled event type: {event_type}")