"""Public exports for shared PostgreSQL infrastructure and ORM models."""

from .base import Base
from .call_log import CallLog, save_call_log
from .transcript import Transcript, save_transcript
from .tool_log import ToolLog, save_tool_log
from .user import User, create_user, get_user
from .knowledge_chunk import KnowledgeChunk
from .database import (
    init_db_sync,
    close_db_sync,
    init_db_async,
    close_db_async,
    DatabaseNotInitializedError,
)
from .ticket import Ticket
from .contact import Contact

# Prompt Cache ORM models. Import the parent cache model before models that
# declare foreign keys to it; import jobs before history/outbox models.
from .prompt_cache import (
    PromptCache,
    save_prompt_cache_sync,
    get_prompt_cache_sync,
    save_prompt_cache_async,
    get_prompt_cache_async,
)
from .prompt_cache_job import PromptCacheJob
from .prompt_cache_job_status_history import PromptCacheJobStatusHistory
from .prompt_cache_lock import PromptCacheLock
from .prompt_cache_outbox import PromptCacheOutbox


__all__ = [
    "Base",
    "CallLog",
    "Transcript",
    "ToolLog",
    "User",
    "Contact",
    "Ticket",
    "KnowledgeChunk",
    "create_user",
    "get_user",
    "save_call_log",
    "save_transcript",
    "save_tool_log",
    # Existing legacy names are retained temporarily while their callers
    # are migrated. The current prompt_cache.py implementations are
    # deprecated compatibility shims and raise LegacyPromptCacheAPIError.
    "PromptCache",
    "save_prompt_cache_sync",
    "get_prompt_cache_sync",
    "save_prompt_cache_async",
    "get_prompt_cache_async",
    # New Prompt Cache ORM models.
    "PromptCacheJob",
    "PromptCacheJobStatusHistory",
    "PromptCacheLock",
    "PromptCacheOutbox",
    # Existing database lifecycle exports.
    "init_db_sync",
    "close_db_sync",
    "init_db_async",
    "close_db_async",
    "DatabaseNotInitializedError",
]
