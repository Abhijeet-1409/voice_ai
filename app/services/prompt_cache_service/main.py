"""FastAPI application entry point for the Prompt Cache Service."""

from contextlib import asynccontextmanager
from typing import AsyncIterator
from uuid import uuid4

from fastapi import FastAPI, Request
from starlette.responses import Response

from shared.infra.postgres.database import (
    close_db_async,
    init_db_async,
)
from shared.infra.redis.client import (
    close_redis_async,
    ping_redis_async,
)
from shared.logging_setup import get_logger

from .api.exception_handlers import register_exception_handlers
from .api.routes.cache_jobs import router as cache_jobs_router
from .api.routes.cache_lookup import router as cache_lookup_router
from .config.settings import get_prompt_cache_settings


logger = get_logger("prompt_cache_service.api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage shared infrastructure during application startup/shutdown."""

    logger.info("Starting Prompt Cache Service API.")

    db_initialized = False

    try:
        # Initialize the shared asynchronous PostgreSQL infrastructure.
        await init_db_async()
        db_initialized = True

        logger.info("PostgreSQL initialization completed.")

        # Redis is an optimization and broker, not the source of truth.
        # Its unavailability should not prevent API startup.
        try:
            await ping_redis_async()
            logger.info("Redis connection check successful.")
        except Exception:
            logger.warning(
                "Redis connection check failed. "
                "The API will start with Redis in a degraded state.",
                exc_info=True,
            )

        yield

    finally:
        logger.info("Shutting down Prompt Cache Service API.")

        # Close only resources owned by this API process.
        try:
            await close_redis_async()
            logger.info("Redis connection closed.")
        except Exception:
            logger.exception("Failed to close Redis connection.")

        if db_initialized:
            await close_db_async()
            logger.info("PostgreSQL connections closed.")

        logger.info("Prompt Cache Service API shutdown completed.")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    settings = get_prompt_cache_settings()

    app = FastAPI(
        title="Prompt Cache Service",
        lifespan=lifespan,
    )

    # Keep the service settings available to application components.
    app.state.settings = settings

    # Register API exception handlers.
    register_exception_handlers(app)

    # Register API routes under the versioned API prefix.
    app.include_router(
        cache_lookup_router,
        prefix="/api/v1",
    )

    app.include_router(
        cache_jobs_router,
        prefix="/api/v1",
    )

    @app.middleware("http")
    async def add_request_id(
        request: Request,
        call_next,
    ) -> Response:
        """Attach a request ID for tracing and error logging."""

        request_id = uuid4().hex
        request.state.request_id = request_id

        response = await call_next(request)

        response.headers["X-Request-ID"] = request_id

        return response

    logger.info("Prompt Cache Service FastAPI application configured.")

    return app


app = create_app()
