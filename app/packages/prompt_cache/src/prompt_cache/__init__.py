"""
Public interface for the prompt-cache client package.

Concrete transport adapters are exposed through :mod:`adapters`, while
shared client exceptions are exposed through :mod:`exceptions`.
"""

from .base import PromptCacheClient
from . import adapters
from . import schema
from . import exceptions

__all__ = [
    "PromptCacheClient",
    "adapters",
    "schema",
    "exceptions",
]