"""Thin client for TypeSafe's System One endpoint.

Uses the shared `HttpClient` (retries 429 and 529 with backoff) instead of the SDK to avoid another dependency
for one POST. A failed call raises; callers must never read a failure as a "no".
"""

import json
from logging import getLogger
from typing import Any

from pydantic import SecretStr, ValidationError

from ..schemas.typesafe import Question, SystemOneRequest, SystemOneResponse
from .http import HttpClient
from .usage import UsageMeter

logger = getLogger(__name__)


class TypeSafeError(Exception):
    """The API call failed or returned something that doesn't match the documented response shape."""


class TypeSafeClient:
    def __init__(
        self, http: HttpClient, api_key: SecretStr, base_url: str, model: str, meter: UsageMeter | None = None
    ) -> None:
        self._http = http
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self.model = model
        self._meter = meter

    def build_request(self, state: dict[str, Any], questions: dict[str, Question]) -> SystemOneRequest:
        return SystemOneRequest(model=self.model, state=state, questions=questions)

    async def system_one(self, request: SystemOneRequest) -> SystemOneResponse:
        payload = request.model_dump(mode="json", exclude_none=True)
        # The exact payload is what you need to rerun or debug a surprising answer.
        logger.debug("TypeSafe request: %s", json.dumps(payload)[:4000])
        response = await self._http.request(
            "POST",
            f"{self._base_url}/v1/systemone",
            json=payload,
            headers={"Authorization": f"Bearer {self._api_key.get_secret_value()}"},
        )
        try:
            parsed = SystemOneResponse.model_validate(response.json())
        except (ValidationError, ValueError) as exc:
            raise TypeSafeError(f"Unexpected System One response: {response.text[:500]}") from exc
        if self._meter is not None:
            # TypeSafe doesn't report cost; usage.py prices Jev from its list price.
            self._meter.record("typesafe", parsed.model, parsed.usage.input_tokens, parsed.usage.output_tokens)
        missing = set(request.questions) - set(parsed.answers)
        if missing:
            raise TypeSafeError(f"System One response is missing answers for: {', '.join(sorted(missing))}")
        return parsed
