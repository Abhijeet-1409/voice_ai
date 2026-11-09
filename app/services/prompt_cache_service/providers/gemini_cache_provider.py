
"""Google Gen AI provider for creating cached content."""

from __future__ import annotations

import asyncio
import math
from datetime import datetime, timezone
from typing import Any

import google.genai as genai
from google.genai import types
from google.oauth2 import service_account

from ..config.settings import PromptCacheSettings
from ..schemas.cache import CachePayload
from ..schemas.enums import PromptCacheType
from ..schemas.jobs import CacheCreationResult


class ProviderResponseError(RuntimeError):
    """Raised when Gemini returns an unusable cached-content response."""


class GeminiCacheProvider:
    """Create Gemini cached content using the Google Gen AI SDK."""

    GUIDE_PREFIX = (
        "Use the following voice agent guide as "
        "business and conversation guidance.\n\n"
    )

    DISPLAY_NAME_PREFIXES = {
        PromptCacheType.INBOUND_PROMPT_CACHE: "inbound",
        PromptCacheType.OUTBOUND_PROMPT_CACHE: "outreach",
        PromptCacheType.DEFAULT_PROMPT_CACHE: "default",
    }

    def __init__(
        self,
        settings: PromptCacheSettings,
        client: genai.Client | None = None,
    ) -> None:
        self.settings = settings

        # When a client is injected, its lifecycle belongs to the caller.
        # Otherwise, this provider owns the client it creates.
        self._owns_client = client is None

        self._client = client or self._create_client()

        # Create the asynchronous SDK client lazily. It will be reused
        # on the managed event loop belonging to this worker process.
        self._async_client: Any | None = None

        self._closed = False

    def _create_client(self) -> genai.Client:
        """Initialize the Google Gen AI client using service credentials."""

        credentials = (
            service_account.Credentials.from_service_account_file(
                self.settings.GOOGLE_APPLICATION_CREDENTIALS,
                scopes=[
                    "https://www.googleapis.com/auth/cloud-platform"
                ],
            )
        )

        return genai.Client(
            enterprise=True,
            project=self.settings.PROJECT,
            location=self.settings.LOCATION,
            credentials=credentials,
        )

    def _get_async_client(self) -> Any:
        """Return the reusable asynchronous SDK client."""

        if self._closed:
            raise RuntimeError(
                "GeminiCacheProvider has already been closed."
            )

        if self._async_client is None:
            self._async_client = self._client.aio

        return self._async_client

    async def create_cached_content(
        self,
        *,
        payload: CachePayload,
        timeout_seconds: float,
    ) -> CacheCreationResult:
        """Create cached content and return its normalized result."""

        self._validate_timeout(timeout_seconds)

        # Convert our service payload into the Google Gen AI SDK format.
        config = self._build_cached_content_config(payload)

        # Use the SDK's asynchronous API so the worker event loop is not
        # blocked while waiting for Gemini.
        async_client = self._get_async_client()

        response: types.CachedContent = await asyncio.wait_for(
            async_client.caches.create(
                model=self.settings.GEMINI_MODEL,
                config=config,
            ),
            timeout=float(timeout_seconds),
        )

        return self._map_provider_response(response)

    def _build_cached_content_config(
        self,
        payload: CachePayload,
    ) -> types.CreateCachedContentConfig:
        """Convert the service payload into Gemini cached-content config."""

        # 1. Build system instruction content.
        system_instruction_content = types.Content(
            parts=[
                types.Part.from_text(
                    text=payload.instruction,
                )
            ],
        )

        # 2. Build the business and conversation guide.
        guide_contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=(
                            self.GUIDE_PREFIX
                            + payload.voice_agent_guide
                        ),
                    )
                ],
            )
        ]

        # 3. Convert serialized LiveKit Google tool schemas into
        # Google Gen AI FunctionDeclaration objects.
        google_tools = self._build_google_tools(
            payload.tool_schema
        )

        # 4. Preserve the existing call-type display-name convention.
        display_name_prefix = self.DISPLAY_NAME_PREFIXES.get(
            payload.prompt_cache_type
        )

        if display_name_prefix is None:
            raise ValueError(
                "Unsupported prompt-cache type."
            )

        display_name = (
            f"{display_name_prefix}_voice_agent_cache"
        )

        # 5. Preserve the provider TTL format: seconds as a string.
        ttl = f"{self.settings.CACHED_CONTENT_TTL}s"

        return types.CreateCachedContentConfig(
            display_name=display_name,
            system_instruction=system_instruction_content,
            contents=guide_contents,
            tools=google_tools,
            ttl=ttl,
        )

    @staticmethod
    def _build_google_tools(
        tool_schema: list[dict[str, Any]],
    ) -> list[types.Tool]:
        """Convert serialized Google function declarations into SDK tools.

        The input is generated by LiveKit's:
            ToolContext.parse_function_tools(format="google")

        It is not a collection of LiveKit Tool or Toolset instances.
        """

        if not tool_schema:
            return []

        function_declarations = [
            types.FunctionDeclaration.model_validate(declaration)
            for declaration in tool_schema
        ]

        # LiveKit's Google provider groups the declarations into one
        # Google Gen AI Tool, preserving the declaration order.
        return [
            types.Tool(
                function_declarations=function_declarations,
            )
        ]

    @staticmethod
    def _map_provider_response(
        response: types.CachedContent,
    ) -> CacheCreationResult:
        """Normalize Gemini's response into our service schema."""

        cache_name = response.name
        expires_at = response.expire_time

        if not isinstance(cache_name, str) or not cache_name.strip():
            raise ProviderResponseError(
                "Gemini returned no cached-content resource name."
            )

        if not isinstance(expires_at, datetime):
            raise ProviderResponseError(
                "Gemini returned no valid cache expiration timestamp."
            )

        if (
            expires_at.tzinfo is None
            or expires_at.utcoffset() is None
        ):
            raise ProviderResponseError(
                "Gemini returned a timezone-naive expiration timestamp."
            )

        # Normalize the provider timestamp to UTC.
        expires_at_utc = expires_at.astimezone(timezone.utc)

        return CacheCreationResult(
            cache_name=cache_name,
            expires_at=expires_at_utc,
        )

    @staticmethod
    def _validate_timeout(timeout_seconds: float) -> None:
        """Validate the timeout supplied by JobDeadlineService."""

        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError(
                "timeout_seconds must be a finite positive number."
            )

    async def aclose(self) -> None:
        """Close provider-owned SDK clients during process shutdown.

        The provider does not close an injected client because its
        lifecycle is owned by the component that injected it.
        """

        if not self._owns_client or self._closed:
            return

        try:
            if self._async_client is not None:
                await self._async_client.aclose()
        finally:
            self._client.close()
            self._closed = True
