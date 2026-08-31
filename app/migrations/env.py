import sys
import os
import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config
from alembic import context


# 1. Add the 'dependencies' folder to the system path so Alembic can
# find the 'shared' package. env.py lives at app/migrations/env.py;
# shared lives at app/dependencies/shared/ — one level up from
# migrations/, then into dependencies/.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../dependencies')))

# 2. Import your settings and models.
from shared.config import get_app_settings
from shared.infra.postgres import Base, CallLog, Transcript, ToolLog, KnowledgeChunk, User, Contact, Ticket

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# 3. Dynamically set the database URL from your app settings
# instead of looking for a hardcoded string in alembic.ini.
settings = get_app_settings()
# DATABASE_URL is async (postgresql+asyncpg://...) — this is used
# directly with Alembic's async engine pattern below, NOT the sync
# engine_from_config/pool.NullPool approach (asyncpg has no sync
# counterpart usable that way).
config.set_main_option("sqlalchemy.url", str(settings.DATABASE_URL))

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# 4. Set target_metadata so Alembic's autogenerate can see your tables
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """
    Shared migration logic, run against a real (sync-facing, via
    run_sync) connection object. Same body as a normal sync env.py's
    run_migrations_online — the async wrapping happens in
    run_async_migrations below.
    """
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """
    Async-native equivalent of run_migrations_online. Creates an
    AsyncEngine (required since DATABASE_URL uses the asyncpg driver),
    opens an async connection, and runs the actual (sync-style)
    migration logic against it via AsyncConnection.run_sync — Alembic's
    migration context itself is synchronous, run_sync bridges the two.
    """
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode, via the async engine."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()