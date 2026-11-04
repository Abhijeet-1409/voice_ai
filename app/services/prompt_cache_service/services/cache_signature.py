"""Deterministic cache signature generation."""

import hashlib
import json
from typing import Any

from schemas.cache import CachePayload


def build_signature_payload(payload: CachePayload) -> dict[str, Any]:
    """Return only the fields that determine the identity of a cache.

    The order of ``tool_schema`` entries is intentionally preserved.
    Request IDs, timestamps, deadlines, and job metadata are excluded.
    """
    return {
        "prompt_cache_type": payload.prompt_cache_type.value,
        "instruction": payload.instruction,
        "voice_agent_guide": payload.voice_agent_guide,
        "tool_schema": payload.tool_schema,
    }


def canonical_json_bytes(data: dict[str, Any]) -> bytes:
    """Serialize a JSON-compatible payload into deterministic UTF-8 bytes.

    Dictionary keys are sorted recursively by ``json.dumps``. Arrays are not
    sorted because tool declaration order is preserved. NaN and Infinity are
    rejected because they are not valid JSON values.
    """
    canonical_json = json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return canonical_json.encode("utf-8")


def calculate_cache_signature(payload: CachePayload) -> str:
    """Calculate the stable SHA-256 signature for a validated cache payload."""
    signature_payload = build_signature_payload(payload)
    canonical_bytes = canonical_json_bytes(signature_payload)
    return hashlib.sha256(canonical_bytes).hexdigest()


class CacheSignatureService:
    """Thin injectable wrapper around the pure signature functions."""

    def build_signature_payload(
        self,
        payload: CachePayload,
    ) -> dict[str, Any]:
        return build_signature_payload(payload)

    def canonical_json_bytes(self, data: dict[str, Any]) -> bytes:
        return canonical_json_bytes(data)

    def calculate(self, payload: CachePayload) -> str:
        return calculate_cache_signature(payload)
