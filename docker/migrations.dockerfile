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
# Install shared package from its new location
# ─────────────────────────────────────────────

COPY app/packages/shared /app/packages/shared
RUN pip install --no-cache-dir /app/packages/shared


# ─────────────────────────────────────────────
# Copy Alembic project
# ─────────────────────────────────────────────

COPY app/migrations /app/migrations



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