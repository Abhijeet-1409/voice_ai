import sys
import os
import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config
from alembic import context


# 1. Add the project root to the system path so Alembic can find the 'shared' module.
# Assuming this file is at shared/infra/migrations/env.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# 2. Import your settings and models.
from shared.config import get_app_settings
from shared.infra.postgres import Base, CallLog, Transcript, ToolLog, KnowledgeBase, User, Contact, Ticket

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