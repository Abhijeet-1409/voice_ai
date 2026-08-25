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

WORKDIR /app


# ─────────────────────────────────────────────
# Install migration dependencies
# ─────────────────────────────────────────────

COPY requirements/migrations.txt /tmp/migrations.txt

RUN pip install --no-cache-dir -r /tmp/migrations.txt


# ─────────────────────────────────────────────
# Copy shared code
# ─────────────────────────────────────────────

COPY app/dependencies/shared /app/shared/


# ─────────────────────────────────────────────
# Copy Alembic project
# ─────────────────────────────────────────────

COPY app/migrations /app/migrations


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

WORKDIR /app/migrations


# ─────────────────────────────────────────────
# Migration entrypoint
# ─────────────────────────────────────────────

CMD ["alembic", "upgrade", "head"]