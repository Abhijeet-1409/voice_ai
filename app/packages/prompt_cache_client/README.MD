# Prompt Cache Client

A reusable Python package providing synchronous and asynchronous clients for the Intelics Prompt Cache Service.

The package handles communication with the Prompt Cache Service, allowing applications to reuse existing Gemini prompt caches or request new cache creation without directly managing cache lifecycle operations.

## Features

- **Synchronous client** – Perform prompt-cache operations using `httpx.Client`.
- **Asynchronous client** – Perform non-blocking prompt-cache operations using `httpx.AsyncClient`.
- **Cache lookup** – Check whether a reusable prompt cache already exists.
- **Background cache creation** – Submit cache creation jobs to the Prompt Cache Service.
- **Job polling** – Monitor cache creation until completion or failure.
- **Timeout management** – Configure request timeouts, polling intervals, and maximum polling duration.
- **Structured exceptions** – Translate HTTP, connection, timeout, and service-level errors into package-specific exceptions.
- **Configuration management** – Configure the service through environment variables and Pydantic Settings.

## Architecture

The package follows a client-service architecture.

```text
Application (LiveKit Worker)
          |
          v
  Prompt Cache Client
          |
     +----+----+
     |         |
  Sync Client  Async Client
     |         |
     +----+----+
          |
          v
   Prompt Cache Service
          |
          v
    Cache Management
          |
          v
    Vertex AI / Gemini
```

The Prompt Cache Client is responsible for client-side communication and response handling.

The Prompt Cache Service owns:

- Cache lookup and reuse decisions.
- Cache creation and lifecycle management.
- Background job execution.
- Locking and concurrency management.
- Retry handling.
- Persistent job status management.

The client does not implement Gemini cache creation or service-side retry logic.

## Package Structure

```text
prompt_cache_client/
│
├── pyproject.toml
├── README.md
│
└── src/
    └── prompt_cache_client/
        │
        ├── __init__.py
        ├── base.py
        ├── constants.py
        ├── exceptions.py
        ├── prompt_cache_settings.py
        ├── schema.py
        │
        └── adapters/
            ├── __init__.py
            │
            ├── sync_client/
            │   ├── __init__.py
            │   └── implementation.py
            │
            └── async_client/
                ├── __init__.py
                └── implementation.py
```

### Module Responsibilities

| Module | Responsibility |
|---|---|
| `base.py` | Common client abstractions and interfaces |
| `constants.py` | Job statuses and package constants |
| `exceptions.py` | Package-specific exception hierarchy |
| `schema.py` | Request and response models |
| `prompt_cache_settings.py` | Client configuration |
| `adapters/sync_client` | Synchronous HTTP client implementation |
| `adapters/async_client` | Asynchronous HTTP client implementation |

## Installation

### Local Development

The package uses the shared package for common functionality such as logging and shared configuration or enums.

Install both packages from the project root:

```powershell
pip install -e .\app\packages\shared
pip install -e .\app\packages\prompt_cache_client
```

Editable installation means changes to package source code are reflected without reinstalling it.

### Dependencies

- Python 3.10+
- HTTPX
- Pydantic
- Pydantic Settings
- Shared package

## Configuration

The client configuration is managed through `PromptCacheClientSettings`.

Configure the following environment variables in your `.env` file:

```env
PROMPT_CACHE_BASE_URL=http://localhost:8000

PROMPT_CACHE_LOOKUP_ENDPOINT=/api/v1/caches/lookup

PROMPT_CACHE_CREATE_JOB_ENDPOINT=/api/v1/cache-jobs

PROMPT_CACHE_GET_JOB_ENDPOINT=/api/v1/cache-jobs/{job_id}

PROMPT_CACHE_REQUEST_TIMEOUT=10.0

PROMPT_CACHE_POLL_INTERVAL=2.0

PROMPT_CACHE_POLL_TIMEOUT=120.0
```

### Configuration Reference

| Setting | Default | Description |
|---|---|---|
| `PROMPT_CACHE_BASE_URL` | `http://localhost:8000` | Prompt Cache Service base URL |
| `PROMPT_CACHE_LOOKUP_ENDPOINT` | `/api/v1/caches/lookup` | Cache lookup endpoint |
| `PROMPT_CACHE_CREATE_JOB_ENDPOINT` | `/api/v1/cache-jobs` | Cache creation job endpoint |
| `PROMPT_CACHE_GET_JOB_ENDPOINT` | `/api/v1/cache-jobs/{job_id}` | Job status endpoint |
| `PROMPT_CACHE_REQUEST_TIMEOUT` | `10.0` | Maximum duration of an individual HTTP request, in seconds |
| `PROMPT_CACHE_POLL_INTERVAL` | `2.0` | Interval between job status checks, in seconds |
| `PROMPT_CACHE_POLL_TIMEOUT` | `120.0` | Maximum duration for waiting for a cache creation job, in seconds |

## Usage

### Synchronous Client

Use the synchronous client when the calling code is synchronous.

```python
from prompt_cache_client.adapters.sync_client.implementation import (
    SyncPromptCacheClient,
)

client = SyncPromptCacheClient(
    base_url="http://localhost:8000",
)

result = client.create(
    {
        "prompt_cache_type": "inbound",
        "instruction": "System instructions",
        "voice_agent_guide": "Agent guide",
        "tool_schema": [],
    }
)
```

### Asynchronous Client

Use the asynchronous client when working inside an asynchronous application, such as a LiveKit agent or FastAPI application.

```python
from prompt_cache_client.adapters.async_client.implementation import (
    AsyncPromptCacheClient,
)

client = AsyncPromptCacheClient(
    base_url="http://localhost:8000",
)

result = await client.create(
    {
        "prompt_cache_type": "inbound",
        "instruction": "System instructions",
        "voice_agent_guide": "Agent guide",
        "tool_schema": [],
    }
)
```

Both clients support the same high-level operation and return a `CacheResult` after successfully retrieving or creating a reusable cache.

**Resource lifecycle:** Each client instance is intended for one prompt-cache operation. Calling `create()` automatically closes the underlying HTTP client in a `finally` block. Create a fresh client instance for each operation.

## API Workflow

The client follows this workflow when `create()` is called:

1. Validate the supplied prompt-cache payload.
2. Submit a cache lookup request.
3. If a reusable cache exists, return the existing cache result.
4. If no cache exists, submit a cache creation job.
5. Poll the job status endpoint.
6. Return the result when the job reaches `COMPLETED`.
7. Raise a package-specific exception when the job reaches `FAILED` or the polling timeout is exceeded.
8. Close the HTTP client.

### Cache Lookup

`POST /api/v1/caches/lookup`

A cache hit returns:

```json
{
    "cache": {
        "cache_name": "cachedContents/abc123",
        "reused": true,
        "expires_at": "2026-10-31T22:30:00+00:00"
    }
}
```

A cache miss returns:

```json
{
    "cache": null
}
```

A cache miss is a normal response, not an HTTP error.

### Create Cache Job

`POST /api/v1/cache-jobs`

Returns HTTP `202 Accepted` with a job resource:

```json
{
    "id": "job_123456",
    "status": "pending",
    "result": null,
    "error": null,
    "created_at": "2026-10-31T22:00:00+00:00",
    "updated_at": "2026-10-31T22:00:00+00:00",
    "completed_at": null
}
```

### Retrieve Job Status

`GET /api/v1/cache-jobs/{job_id}`

Possible job statuses:

| Status | Description |
|---|---|
| `PENDING` | Job has been accepted and is waiting for execution |
| `RUNNING` | Cache creation is currently running |
| `RETRYING` | A transient failure occurred and the service is retrying |
| `COMPLETED` | Cache creation completed successfully |
| `FAILED` | Cache creation failed permanently |

A completed job contains its cache result.

A failed job contains a structured error:

```json
{
    "id": "job_123456",
    "status": "failed",
    "result": null,
    "error": {
        "code": "prompt_cache_gemini_api_error",
        "message": "Gemini cache creation failed."
    },
    "created_at": "2026-10-31T22:00:00+00:00",
    "updated_at": "2026-10-31T22:00:30+00:00",
    "completed_at": "2026-10-31T22:00:30+00:00"
}
```

## Exception Handling

The package provides a structured exception hierarchy.

```text
PromptCacheError
│
├── PromptCacheClientError
│   ├── PromptCacheConnectionError
│   ├── PromptCacheTimeoutError
│   └── PromptCacheResponseError
│
└── PromptCacheServiceError
    ├── PromptCacheDatabaseError
    └── PromptCacheGeminiAPIError
```

### Exception Responsibilities

| Exception | Description |
|---|---|
| `PromptCacheError` | Base exception for the package |
| `PromptCacheClientError` | Client-side errors |
| `PromptCacheConnectionError` | Connection failures |
| `PromptCacheTimeoutError` | Request or polling timeout |
| `PromptCacheResponseError` | Invalid or unexpected service response |
| `PromptCacheServiceError` | Service-side operation failures |
| `PromptCacheDatabaseError` | Database-related cache service failure |
| `PromptCacheGeminiAPIError` | Gemini cache creation failure |

Example:

```python
from prompt_cache_client.exceptions import (
    PromptCacheConnectionError,
    PromptCacheTimeoutError,
    PromptCacheGeminiAPIError,
)

try:
    result = await client.create(payload)

except PromptCacheConnectionError as exc:
    logger.error("Prompt cache service connection failed: %s", exc)

except PromptCacheTimeoutError as exc:
    logger.error("Prompt cache operation timed out: %s", exc)

except PromptCacheGeminiAPIError as exc:
    logger.error("Gemini cache creation failed: %s", exc)
```

## Design Principles

The package follows these principles:

- **Single responsibility:** The client handles communication with the service, not cache lifecycle management.
- **Separation of concerns:** HTTP transport, schemas, exceptions, and configuration are kept separate.
- **Sync and async support:** Both execution models expose equivalent high-level functionality.
- **Explicit resource management:** HTTP clients are closed automatically after an operation.
- **Structured error handling:** Service failures and transport failures are represented by meaningful exceptions.
- **Configuration-driven behavior:** Endpoints and timeout values are configurable without modifying implementation code.

## Scope

This package is a client library, not the Prompt Cache Service itself.

It does not:

- Create Gemini caches directly.
- Manage Redis locks.
- Execute background jobs.
- Implement service-side retry policies.
- Manage database persistence.
- Decide cache expiration or invalidation policies.

These responsibilities belong to the Prompt Cache Service.

---

**Distribution:** `prompt-cache-client`
**Python import:** `prompt_cache_client`
**Version:** `0.1.0`
**Python:** `>=3.10`