import uuid
from datetime import datetime

from sqlalchemy import String, DateTime
from sqlalchemy.orm import Mapped, mapped_column

from shared.config import PromptCacheType
from .database import get_sync_sessionmaker, get_async_sessionmaker

from .base import Base


class PromptCache(Base):
    """SQLAlchemy model representing a cached prompt entry and its metadata.

    Used for managing Vertex AI prompt cache references, tracking versioning,
    and handling cache expirations.

    Attributes:
        id: Primary key unique identifier (UUID v4 string).
        prompt_type: Category or type of the cached prompt.
        vertex_cache_name: Resource identifier or reference name of the cached prompt in Vertex AI.
        model: Model identifier associated with the cached prompt.
        location: GCP region/location where the Vertex AI cache resides.
        system_instruction_version: Version tag of the system instruction template used.
        voice_agent_guide_version: Version tag of the voice agent guide template used.
        created_at: UTC timestamp when the cache record was created.
        expires_at: UTC timestamp when the cached prompt expires.
    """

    __tablename__ = "prompt_caches"

    id:                          Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    prompt_type:                 Mapped[PromptCacheType] = mapped_column(String, nullable=False)
    vertex_cache_name:           Mapped[str] = mapped_column(String, nullable=False)
    model:                       Mapped[str] = mapped_column(String, nullable=False)
    location:                    Mapped[str] = mapped_column(String, nullable=False)
    system_instruction_version:  Mapped[int] = mapped_column(String, nullable=False)
    voice_agent_guide_version:   Mapped[int] = mapped_column(String, nullable=False)
    created_at:                  Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expires_at:                  Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


def sync_save_prompt_cache_(
        prompt_type: PromptCacheType, 
        vertex_cache_name: str, 
        model: str, location: str, 
        system_instruction_version: int, 
        voice_agent_guide_version: int,
        created_at: datetime, 
        expires_at: datetime
        ):
    """
    Save a new prompt cache entry to the database.

    Args:
        session: SQLAlchemy session object.
        prompt_type (PromptCacheType): Type of the prompt cache.
        vertex_cache_name (str): Name of the vertex cache.
        model (str): Model associated with the prompt.
        location (str): Location associated with the prompt.
        system_instruction_version (int): Version of the system instruction.
        voice_agent_guide_version (int): Version of the voice agent guide.
        expires_at (datetime): Expiration timestamp for the cache entry.

    Returns:
        PromptCache: The newly created PromptCache object.
    """
    new_cache = PromptCache(
        prompt_type=prompt_type,
        vertex_cache_name=vertex_cache_name,
        model=model,
        location=location,
        system_instruction_version=system_instruction_version,
        voice_agent_guide_version=voice_agent_guide_version,
        created_at=created_at,
        expires_at=expires_at
    )
    
    return new_cache 