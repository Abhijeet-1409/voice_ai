from functools import cache

from pydantic_settings import BaseSettings


class AppBaseSettings(BaseSettings):
    """
    Base application settings, loaded from environment variables (and a
    .env file, if present).

    Fields:
        REDIS_URL: connection URL for Redis, required by both containers.
        DATABASE_URL: connection URL for the database, required by both containers.
        DB_ECHO: whether to echo raw SQL statements, for debugging. Defaults to False.
        LOG_LEVEL: minimum log level to emit (e.g. "DEBUG", "INFO"). Defaults to "DEBUG".
        LOG_FILE: optional path to a log file; if None, file logging is disabled.
        DATA_FORMAT: strftime format used for log timestamps.
        LOG_FORMAT: format string used to render each log record.

    Unknown/extra environment variables are ignored rather than raising
    a validation error.
    """

    # Storage — both containers need these
    REDIS_URL: str
    DATABASE_URL: str
    DB_ECHO: bool = False

    # Logging — both containers need these
    LOG_LEVEL: str = "DEBUG"
    LOG_FILE: str | None = None
    DATA_FORMAT: str ="%Y-%m-%d %H:%M:%S"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    class Config:
        env_file = ".env"
        extra = "ignore"


@cache
def get_app_settings() -> AppBaseSettings:
    """
    Get the application's settings, instantiating them on first call.

    Instantiating AppBaseSettings (a pydantic_settings.BaseSettings
    subclass) reads and validates configuration from environment
    variables (and a .env file, if configured) at call time — not at
    import time — so this should only be called after the app's
    startup/env-loading has run. Because this function is cached, the
    settings are only resolved once; all later calls return the same
    cached AppBaseSettings instance.

    :return: the application's validated settings instance
    """
    return AppBaseSettings()