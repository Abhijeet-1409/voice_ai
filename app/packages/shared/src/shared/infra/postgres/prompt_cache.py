import uuid
from datetime import datetime

from sqlalchemy import String, DateTime, Integer, select, Result
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Mapped, mapped_column

from shared.config import PromptCacheType
from shared.logging_setup import get_logger

from .base import Base
from .database import get_sync_sessionmaker, get_async_sessionmaker


_LOGGER = "infra.postgres.prompt_cache"


class PromptCache(Base):
    """Store metadata for reusable Vertex AI context caches.

    Attributes:
        id: Unique identifier for the prompt cache metadata record (UUID v4).
        prompt_type: Type of prompt cache, such as inbound or outreach.
        vertex_cache_name: Resource name of the cached content stored in Vertex AI.
        model: Vertex AI model associated with the cached content.
        location: GCP region where the cached content is created and managed.
        system_instruction_version: Version of the system instructions used to create the cache.
        voice_agent_guide_version: Version of the voice agent guide included in the cache.
        created_at: Timestamp when the Vertex AI cache was created.
        expires_at: Timestamp when the Vertex AI cache is scheduled to expire.
    """

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

def save_prompt_cache_sync(
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
            f"Saved prompt cache metadata | "
            f"cache_id={new_cache.id} | prompt_type={prompt_type} | model={model}"
        )

    except SQLAlchemyError:
        logger.exception(
            f"Failed to save prompt cache metadata | "
            f"prompt_type={prompt_type} | model={model}"
        )
        raise

    return new_cache


def get_prompt_cache_sync(
    prompt_type: PromptCacheType,
    model: str,
    location: str,
    system_instruction_version: int,
    voice_agent_guide_version: int,
) -> PromptCache | None:
    """Retrieve the newest prompt cache record matching the supplied configuration."""

    logger = get_logger(_LOGGER)
    session_factory = get_sync_sessionmaker()

    try:
        with session_factory() as session:
            stmt = (
                select(PromptCache)
                .where(
                    PromptCache.prompt_type == prompt_type,
                    PromptCache.model == model,
                    PromptCache.location == location,
                    PromptCache.system_instruction_version
                    == system_instruction_version,
                    PromptCache.voice_agent_guide_version
                    == voice_agent_guide_version,
                )
                .order_by(PromptCache.expires_at.desc())
                .limit(1)
            )
            result: Result[PromptCache] = session.execute(stmt)
            cache = result.scalars().first()
            logger.debug(f"Retrieved matching prompt cache metadata: {cache}")
            return cache

    except SQLAlchemyError:
        logger.exception(
            f"Failed to retrieve prompt cache metadata | "
            f"prompt_type={prompt_type} | model={model} | location={location}"
        )
        raise

async def save_prompt_cache_async(
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
            f"Saved prompt cache metadata | "
            f"cache_id={new_cache.id} | prompt_type={prompt_type} | model={model}"
        )

    except SQLAlchemyError:
        logger.exception(
            f"Failed to save prompt cache metadata | "
            f"prompt_type={prompt_type} | model={model}"
        )
        raise

    return new_cache


async def get_prompt_cache_async(
    prompt_type: PromptCacheType,
    model: str,
    location: str,
    system_instruction_version: int,
    voice_agent_guide_version: int,
) -> PromptCache | None:
    """Asynchronously retrieve the newest matching prompt cache record."""

    logger = get_logger(_LOGGER)
    session_factory = get_async_sessionmaker()

    try:
        async with session_factory() as session:
            stmt = (
                select(PromptCache)
                .where(
                    PromptCache.prompt_type == prompt_type,
                    PromptCache.model == model,
                    PromptCache.location == location,
                    PromptCache.system_instruction_version
                    == system_instruction_version,
                    PromptCache.voice_agent_guide_version
                    == voice_agent_guide_version,
                )
                .order_by(PromptCache.expires_at.desc())
                .limit(1)
            )
            result = await session.execute(stmt)
            cache = result.scalars().first()
            logger.debug(f"Retrieved matching prompt cache metadata: {cache}")
            return cache

    except SQLAlchemyError:
        logger.exception(
            f"Failed to retrieve prompt cache metadata | "
            f"prompt_type={prompt_type} | model={model} | location={location}"
        )
        raise
