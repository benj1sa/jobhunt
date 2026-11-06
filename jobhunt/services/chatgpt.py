"""LLM client that runs on your ChatGPT plan (see chatgpt_auth.py for the sign-in).

Calls the OpenAI Responses API with the plan access token, streaming (as plan requests require) and with
`store: false`. Structured output uses `text.format: json_schema`. Usage is metered as provider "chatgpt" at
$0, since it comes out of the plan's allowance rather than a per-token bill.

Model ids are stored as "chatgpt/<model>" (e.g. "chatgpt/gpt-5.6-luna") so extractions made on the plan are kept
apart from OpenRouter ones, and so cost estimates know they are free.
"""

import asyncio
import json
from datetime import datetime
from logging import getLogger
from typing import Any

import httpx
from pydantic import BaseModel

from ..schemas.extraction import strict_json_schema
from .chatgpt_auth import RESOURCE, ChatGPTAuth
from .llm import LlmClient, LlmConfigError, LlmError
from .usage import UsageMeter

logger = getLogger(__name__)

PREFIX = "chatgpt/"
RATE_LIMIT_RETRIES = 4
PLAN_LIMIT_CODES = ("usage_limit_reached", "insufficient_quota", "usage_not_included", "quota")
USAGE_PAGE = "https://chatgpt.com/settings/usage"


def plan_model(model: str) -> str:
    return model.removeprefix(PREFIX)


def _error_fields(response: httpx.Response) -> dict[str, Any]:
    try:
        body = response.json()
    except ValueError:
        return {}
    error = body.get("error") if isinstance(body, dict) else None
    return error if isinstance(error, dict) else body if isinstance(body, dict) else {}


def plan_limit_message(response: httpx.Response) -> str | None:
    """The stop message when a 429 means the plan's usage limit (not just requests coming too fast)."""
    error = _error_fields(response)
    code = f"{error.get('code', '')} {error.get('type', '')}".lower()
    message = str(error.get("message", ""))
    if not any(marker in code for marker in PLAN_LIMIT_CODES) and "usage limit" not in message.lower():
        return None
    resets = ""
    if isinstance(error.get("resets_at"), (int, float)):
        when = datetime.fromtimestamp(float(error["resets_at"])).strftime("%a %-I:%M %p")
        resets = f" It resets {when}."
    elif isinstance(error.get("resets_in_seconds"), (int, float)):
        resets = f" It resets in about {max(1, round(float(error['resets_in_seconds']) / 60))} min."
    return (
        f"ChatGPT plan usage limit reached ({message or code.strip()}).{resets} Everything finished so far is saved; "
        f"run the same command again after the reset to continue. Usage: {USAGE_PAGE}"
    )


def retry_after(response: httpx.Response, attempt: int) -> float:
    try:
        return min(60.0, max(1.0, float(response.headers.get("retry-after", ""))))
    except ValueError:
        return min(60.0, 5.0 * 2.0**attempt)


class ChatGPTClient(LlmClient):
    name = "chatgpt"

    def __init__(
        self,
        http: httpx.AsyncClient,
        auth: ChatGPTAuth,
        meter: UsageMeter | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        self._http = http
        self._auth = auth
        self._meter = meter
        self._reasoning_effort = reasoning_effort

    async def models(self) -> list[tuple[str, str]]:
        """(id, display name) for the models your plan can use."""
        token = await self._auth.access_token()
        response = await self._http.get(f"{RESOURCE}/models", headers={"Authorization": f"Bearer {token}"})
        if response.status_code >= 400:
            raise LlmConfigError(f"ChatGPT model list failed ({response.status_code}): {response.text[:300]}")
        body = response.json()
        result: list[tuple[str, str]] = []
        for entry in body.get("models") or []:  # plan catalog shape
            if isinstance(entry, dict) and entry.get("slug") and entry.get("visibility", "list") == "list":
                result.append((str(entry["slug"]), str(entry.get("display_name") or entry["slug"])))
        for entry in body.get("data") or []:  # standard API shape
            if isinstance(entry, dict) and entry.get("id"):
                result.append((str(entry["id"]), str(entry["id"])))
        return result

    def _body(
        self, model: str, messages: list[dict[str, str]], schema: type[BaseModel], json_schema: dict[str, Any] | None
    ) -> dict[str, Any]:
        instructions = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        body: dict[str, Any] = {
            "model": plan_model(model),
            "instructions": instructions,
            "input": [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema.__name__,
                    "strict": True,
                    "schema": json_schema or strict_json_schema(schema),
                }
            },
            "store": False,
            "stream": True,
        }
        if self._reasoning_effort:
            body["reasoning"] = {"effort": self._reasoning_effort}
        return body

    async def _complete(
        self,
        model: str,
        messages: list[dict[str, str]],
        schema: type[BaseModel],
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        body = self._body(model, messages, schema, json_schema)
        # A 401 gets one retry with a fresh token; a 429 for going too fast waits and retries. Neither request was
        # admitted, so retrying spends nothing. Admitted requests are never retried, so a flaky connection can't
        # spend the plan allowance twice. A 429 for the plan's usage limit stops the run (LlmConfigError).
        refreshed = False
        for attempt in range(RATE_LIMIT_RETRIES + 1):
            token = await self._auth.access_token(force_refresh=refreshed and attempt > 0)
            async with self._http.stream(
                "POST",
                f"{RESOURCE}/responses",
                json=body,
                headers={"Authorization": f"Bearer {token}", "Accept": "text/event-stream"},
                timeout=httpx.Timeout(120, connect=15),
            ) as response:
                if response.status_code == 401 and not refreshed:
                    await response.aread()
                    refreshed = True
                    continue
                if response.status_code == 429:
                    await response.aread()
                    limit = plan_limit_message(response)
                    if limit or attempt == RATE_LIMIT_RETRIES:
                        raise LlmConfigError(limit or f"ChatGPT kept rate-limiting requests: {response.text[:300]}")
                    wait = retry_after(response, attempt)
                    logger.debug("ChatGPT rate limit, waiting %.0fs", wait)
                    await asyncio.sleep(wait)
                    continue
                if response.status_code >= 400:
                    await response.aread()
                    raise self._http_error(model, response)
                return await self._read_stream(model, response)
        raise LlmConfigError("ChatGPT kept rejecting the access token. Run `jobhunt chatgpt login`.")

    @staticmethod
    def _http_error(model: str, response: httpx.Response) -> LlmError:
        text = response.text[:500]
        status = response.status_code
        if status in (400, 401, 403, 404):
            hint = " Run `jobhunt chatgpt models` to see what your plan offers." if status in (400, 404) else ""
            return LlmConfigError(f"ChatGPT rejected the request for {plan_model(model)!r} ({status}): {text}.{hint}")
        return LlmError(f"ChatGPT error {status}: {text}")

    async def _read_stream(self, model: str, response: httpx.Response) -> str:
        deltas: list[str] = []
        completed: dict[str, Any] | None = None
        failure: str | None = None
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                event = json.loads(data)
            except ValueError:
                continue
            kind = event.get("type")
            if kind == "response.output_text.delta":
                deltas.append(str(event.get("delta", "")))
            elif kind == "response.completed":
                completed = event.get("response") or {}
            elif kind in ("response.failed", "response.incomplete", "error"):
                detail = event.get("error") or (event.get("response") or {}).get("error") or event
                failure = json.dumps(detail)[:500]
        if completed is not None:
            usage = completed.get("usage") or {}
            if self._meter is not None:
                self._meter.record(
                    "chatgpt",
                    model,
                    int(usage.get("input_tokens") or 0),
                    int(usage.get("output_tokens") or 0),
                    0.0,
                )
        if failure or completed is None:
            if failure and any(marker in failure for marker in PLAN_LIMIT_CODES):
                raise LlmConfigError(
                    f"ChatGPT plan usage limit reached: {failure}. Everything finished so far is saved; run the "
                    f"same command again after the reset. Usage: {USAGE_PAGE}"
                )
            raise LlmError(f"ChatGPT response did not complete: {failure or 'stream ended early'}")
        text = "".join(deltas)
        if not text:
            for item in completed.get("output") or []:
                for part in item.get("content") or [] if isinstance(item, dict) else []:
                    if isinstance(part, dict) and part.get("type") == "output_text":
                        text += str(part.get("text", ""))
        return text
