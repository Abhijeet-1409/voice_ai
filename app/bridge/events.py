from abc import ABC

from typing import Literal

from dataclasses import dataclass, field


# ============================================================
# INBOUND — raw events received FROM Exotel over WebSocket
# ============================================================


@dataclass
class ExotelEvent(ABC):
    """
    Base class for all events received from Exotel's media stream.
    """
    event: str


@dataclass
class ConnectedEvent(ExotelEvent):
    """
    Event indicating that the media stream has been successfully connected.
    """
    event: Literal["connected"] = "connected"


@dataclass
class StartEvent(ExotelEvent):
    """
    Event indicating that the media stream has started.
    """
    event: Literal["start"] = "start"
    sequence_number: str = ""
    stream_sid: str = ""
    call_sid: str = ""
    account_sid: str = ""
    from_number: str = ""
    to_number: str = ""
    custom_parameters: dict = field(default_factory=dict)
    encoding: str = ""
    sample_rate: str = ""
    bit_rate: str = ""


@dataclass
class MediaEvent(ExotelEvent):
    """
    Event indicating that media data has been received from the media stream.
    """
    event: Literal["media"] = "media"
    sequence_number: str = ""
    stream_sid: str = ""
    chunk: str = ""
    timestamp: str = ""
    payload: bytes = b""


@dataclass
class DTMFEvent(ExotelEvent):
    """
    Event indicating that a DTMF digit has been received.
    """
    event: Literal["dtmf"] = "dtmf"
    sequence_number: str = ""
    stream_sid: str = ""
    digit: str = ""
    duration: str = ""


@dataclass
class MarkEvent(ExotelEvent):
    """
    Event indicating that a mark has been received.
    """
    event: Literal["mark"] = "mark"
    sequence_number: str = ""
    stream_sid: str = ""
    name: str = ""


@dataclass
class StopEvent(ExotelEvent):
    """
    Event indicating that the media stream has been stopped.
    """
    event: Literal["stop"] = "stop"
    sequence_number: str = ""
    stream_sid: str = ""
    call_sid: str = ""
    account_sid: str = ""
    reason: str = ""


_EVENT_MAP = {
    "connected": ConnectedEvent,
    "start": StartEvent,
    "media": MediaEvent,
    "dtmf": DTMFEvent,
    "mark": MarkEvent,
    "stop": StopEvent,
}


# ============================================================
# OUTBOUND — events WE send back to Exotel
# ============================================================


@dataclass
class OutboundEvent(ABC):
    """
    Base class for all events sent back to Exotel's media stream.
    """
    event: str


@dataclass
class OutboundMediaEventPayload():
    """
    Payload for the OutboundMediaEvent.
    """
    payload: str = ""


@dataclass
class OutboundMarkEventPayload():
    """
    Payload for the OutboundMarkEvent.
    """
    name: str = ""

@dataclass
class OutboundMediaEvent(OutboundEvent):
    """
    Event indicating that media data is being sent back to Exotel.
    """
    event: Literal["media"] = "media"
    stream_sid: str = ""
    media: OutboundMediaEventPayload = field(default_factory=OutboundMediaEventPayload)


@dataclass
class OutboundMarkEvent(OutboundEvent):
    """"
    Event indicating that a mark is being sent back to Exotel.
    """
    event: Literal["mark"] = "mark"
    stream_sid: str = ""
    mark: OutboundMarkEventPayload = field(default_factory=OutboundMarkEventPayload)


@dataclass
class OutboundClearEvent(OutboundEvent):
    """
    Event indicating that a clear is being sent back to Exotel.
    """
    event: Literal["clear"] = "clear"
    stream_sid: str = ""