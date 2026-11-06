"""Minimal OpenRouter client for schema-validated structured outputs."""

import json
import re
from logging import getLogger
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, SecretStr, ValidationError

from ..schemas.extraction import strict_json_schema
from .http import HttpClient, NotFoundError

logger = getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.DOTALL)


class LlmError(Exception):
    """The model failed to return output matching the schema after retrying."""


class LlmConfigError(LlmError):
    """OpenRouter refused the request itself (bad key, no credits, unknown model, no provider that supports the
    requested parameters). Every later call would fail the same way, so callers should stop, not continue."""


class OpenRouterClient:
    """Calls `/chat/completions` with `response_format: json_schema` and validates the reply with Pydantic.

    `provider.require_parameters` makes OpenRouter route only to providers that honor structured outputs, and
    a validation failure is retried once with the error appended so the model can correct itself.
    """

    def __init__(self, http: HttpClient, api_key: SecretStr, base_url: str) -> None:
        self._http = http
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")

    async def structured(
        self, model: str, system: str, user: str, schema: type[T], json_schema: dict[str, Any] | None = None
    ) -> T:
        """`json_schema` overrides the schema sent to the model (e.g. to inject enums from config); the reply is
        still validated against `schema`."""
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        last_error: Exception | None = None
        for attempt in range(2):
            content = await self._complete(model, messages, schema, json_schema)
            try:
                return schema.model_validate_json(FENCE_RE.sub("", content.strip()))
            except ValidationError as exc:
                last_error = exc
                logger.warning("%s returned invalid %s (attempt %d): %s", model, schema.__name__, attempt + 1, exc)
                messages += [
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": f"That JSON failed validation:\n{exc}\nReturn corrected JSON only."},
                ]
        raise LlmError(f"{model} could not produce a valid {schema.__name__}: {last_error}")

    async def _complete(
        self,
        model: str,
        messages: list[dict[str, str]],
        schema: type[BaseModel],
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            # No `temperature`: reasoning models reject it, and with require_parameters an unsupported parameter
            # removes every provider (OpenRouter then answers 404 "No endpoints found").
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "strict": True,
                    "schema": json_schema or strict_json_schema(schema),
                },
            },
            "provider": {"require_parameters": True},
        }
        try:
            response = await self._http.request(
                "POST",
                f"{self._base_url}/chat/completions",
                json=body,
                headers={
                    "Authorization": f"Bearer {self._api_key.get_secret_value()}",
                    "X-Title": "jobhunt",
                },
            )
        except NotFoundError as exc:
            raise LlmConfigError(
                f"OpenRouter has no endpoint for model {model!r} with structured outputs ({exc.body or exc}). "
                "Check the model id, your OpenRouter privacy/provider settings, or pick another EXTRACTION_MODEL."
            ) from exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (400, 401, 402, 403):
                raise LlmConfigError(
                    f"OpenRouter rejected the request ({exc.response.status_code}): {exc.response.text[:500]}"
                ) from exc
            raise
        payload = response.json()
        if "error" in payload:
            raise LlmError(json.dumps(payload["error"])[:500])
        try:
            return str(payload["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise LlmError(f"Unexpected OpenRouter response shape: {str(payload)[:500]}") from exc
