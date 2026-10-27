# Shared

Shared is a reusable Python package for common configuration, logging, and infrastructure integrations used by the Intelics voice bot services.

## Overview

The package centralizes common functionality so that services can reuse the same modules instead of maintaining separate implementations.

## Package Structure

```text
shared/
├── __init__.py
├── call_context.py
├── config/
│   ├── constants.py
│   └── settings.py
├── infra/
│   ├── calendar/
│   ├── crm/
│   ├── postgres/
│   ├── redis/
│   └── vector_store/
└── logging_setup/
    └── logger.py
```

### Modules

- **`config/`** — Application configuration, settings, and constants.
- **`call_context.py`** — Shared call-context functionality.
- **`logging_setup/`** — Common logging configuration and logger utilities.
- **`infra/calendar/`** — Calendar integration interfaces and adapters.
- **`infra/crm/`** — CRM integration interfaces and adapters.
- **`infra/postgres/`** — PostgreSQL database models and database-related functionality.
- **`infra/redis/`** — Redis client and supporting functionality.
- **`infra/vector_store/`** — Vector-store interfaces and adapters.

## Requirements

- Python 3.10 or later
- `pydantic-settings`
- `redis`
- `sqlalchemy`
- `asyncpg`
- `pgvector`

Dependency versions are defined in `pyproject.toml`.

## Installation

From the application root, install the package in editable mode:

```powershell
pip install -e .\packages\shared
```

Editable installation allows local changes to the package source to be reflected without reinstalling the package.

## Usage

Import modules using the `shared` package namespace:

```python
from shared.config import get_app_settings
from shared.infra.postgres import Base
from shared.logging_setup import get_logger
```

The import paths remain the same when the package uses a `src` layout.

## Package Layout

The package uses the following project layout:

```text
packages/
└── shared/
    ├── pyproject.toml
    ├── README.md
    └── src/
        └── shared/
            ├── __init__.py
            ├── call_context.py
            ├── config/
            ├── infra/
            └── logging_setup/
```

Setuptools discovers the Python package under `src/`, as configured in `pyproject.toml`.

## Maintenance

Keep shared configuration, logging utilities, and reusable infrastructure integrations in this package when they are needed by multiple services.

Service-specific business logic, application entry points, and service lifecycle management should remain within their respective service directories.
