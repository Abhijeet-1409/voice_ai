FROM python:3.12-slim

# ─────────────────────────────────────────────
# System dependencies
# ─────────────────────────────────────────────

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*


# ─────────────────────────────────────────────
# Non-root user
# ─────────────────────────────────────────────

RUN groupadd -r appgroup \
    && useradd -r -g appgroup -u 1000 appuser


# ─────────────────────────────────────────────
# Application root
# ─────────────────────────────────────────────

WORKDIR /app/shared


# ─────────────────────────────────────────────
# Install migration-specific dependencies
# ─────────────────────────────────────────────

COPY requirements/migrations.txt /tmp/migrations.txt

RUN pip install --no-cache-dir -r /tmp/migrations.txt


# ─────────────────────────────────────────────
# Alembic configuration
# ─────────────────────────────────────────────

COPY app/dependencies/shared/alembic.ini \
     /app/shared/alembic.ini


# ─────────────────────────────────────────────
# Copy root environment file
# ─────────────────────────────────────────────

COPY .env /app/shared/.env


# ─────────────────────────────────────────────
# Copy shared code required by migrations
# ─────────────────────────────────────────────

COPY app/dependencies/shared/__init__.py \
     /app/shared/__init__.py

COPY app/dependencies/shared/call_context.py \
     /app/shared/call_context.py

COPY app/dependencies/shared/config/ \
     /app/shared/config/

COPY app/dependencies/shared/logging_setup/ \
     /app/shared/logging_setup/


# ─────────────────────────────────────────────
# PostgreSQL models / database infrastructure
# ─────────────────────────────────────────────

COPY app/dependencies/shared/infra/__init__.py \
     /app/shared/infra/__init__.py

COPY app/dependencies/shared/infra/postgres/ \
     /app/shared/infra/postgres/


# ─────────────────────────────────────────────
# Alembic migration files
# ─────────────────────────────────────────────

COPY app/dependencies/shared/infra/migrations/ \
     /app/shared/infra/migrations/


# ─────────────────────────────────────────────
# Python import path
# ─────────────────────────────────────────────

ENV PYTHONPATH=/app


# ─────────────────────────────────────────────
# Permissions
# ─────────────────────────────────────────────

RUN chown -R appuser:appgroup /app

USER appuser


# ─────────────────────────────────────────────
# Migration working directory
# ─────────────────────────────────────────────

WORKDIR /app/shared


# ─────────────────────────────────────────────
# Migration entrypoint
# ─────────────────────────────────────────────

CMD ["alembic", "upgrade", "head"]