import uuid
from datetime import datetime

from sqlalchemy import String, DateTime, Integer
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Mapped, mapped_column

from shared.config import PromptCacheType
from shared.logging_setup import get_logger

from .base import Base
from .database import get_sync_sessionmaker, get_async_sessionmaker


_LOGGER = "infra.postgres.prompt_cache"


class PromptCache(Base):
    """Store metadata for reusable Vertex AI context caches."""

    __tablename__ = "prompt_caches"

    id:                         Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    prompt_type:                Mapped[PromptCacheType] = mapped_column(String, nullable=False)
    vertex_cache_name:          Mapped[str] = mapped_column(String, nullable=False)
    model:                      Mapped[str] = mapped_column(String, nullable=False)
    location:                   Mapped[str] = mapped_column(String, nullable=False)
    system_instruction_version: Mapped[int] = mapped_column(Integer, nullable=False)
    voice_agent_guide_version:  Mapped[int] = mapped_column(Integer, nullable=False)
    created_at:                 Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expires_at:                 Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


def sync_save_prompt_cache(
    prompt_type: PromptCacheType,
    vertex_cache_name: str,
    model: str,
    location: str,
    system_instruction_version: int,
    voice_agent_guide_version: int,
    created_at: datetime,
    expires_at: datetime,
) -> PromptCache:
    """Synchronously persist metadata for a Vertex AI context cache."""

    new_cache = PromptCache(
        prompt_type=prompt_type,
        vertex_cache_name=vertex_cache_name,
        model=model,
        location=location,
        system_instruction_version=system_instruction_version,
        voice_agent_guide_version=voice_agent_guide_version,
        created_at=created_at,
        expires_at=expires_at,
    )

    logger = get_logger(_LOGGER)
    session_factory = get_sync_sessionmaker()

    try:
        with session_factory() as session:
            session.add(new_cache)
            session.commit()

        logger.info(
            "Saved prompt cache metadata | "
            "cache_id=%s | prompt_type=%s | model=%s",
            new_cache.id,
            prompt_type,
            model,
        )

    except SQLAlchemyError:
        logger.exception(
            "Failed to save prompt cache metadata | "
            "prompt_type=%s | model=%s",
            prompt_type,
            model,
        )
        raise

    return new_cache


async def async_save_prompt_cache(
    prompt_type: PromptCacheType,
    vertex_cache_name: str,
    model: str,
    location: str,
    system_instruction_version: int,
    voice_agent_guide_version: int,
    created_at: datetime,
    expires_at: datetime,
) -> PromptCache:
    """Asynchronously persist metadata for a Vertex AI context cache."""

    new_cache = PromptCache(
        prompt_type=prompt_type,
        vertex_cache_name=vertex_cache_name,
        model=model,
        location=location,
        system_instruction_version=system_instruction_version,
        voice_agent_guide_version=voice_agent_guide_version,
        created_at=created_at,
        expires_at=expires_at,
    )

    logger = get_logger(_LOGGER)
    session_factory = get_async_sessionmaker()

    try:
        async with session_factory() as session:
            session.add(new_cache)
            await session.commit()

        logger.info(
            "Saved prompt cache metadata | "
            "cache_id=%s | prompt_type=%s | model=%s",
            new_cache.id,
            prompt_type,
            model,
        )

    except SQLAlchemyError:
        logger.exception(
            "Failed to save prompt cache metadata | "
            "prompt_type=%s | model=%s",
            prompt_type,
            model,
        )
        raise

    return new_cache