import base64
import json
import os
import stat
import time
import urllib.request
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

import httpx
import pytest
import respx

from jobhunt.services.chatgpt import ChatGPTClient
from jobhunt.services.chatgpt_auth import (
    RESOURCE,
    SCOPES,
    TOKEN_URL,
    ChatGPTAuth,
    ChatGPTAuthError,
    ChatGPTCredentials,
)
from jobhunt.services.fit import FitJudgment
from jobhunt.services.llm import LlmConfigError
from jobhunt.services.usage import UsageMeter, list_price

FIT = {"interest_alignment": "strong", "learning_potential": "weak", "rationale": "x"}


def id_token(**claims: object) -> str:
    def part(data: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(data).encode()).rstrip(b"=").decode()

    return f"{part({'alg': 'RS256'})}.{part(claims)}.sig"


def token_reply(access: str = "at-1", refresh: str = "rt-1", **extra: object) -> dict:
    return {
        "access_token": access,
        "refresh_token": refresh,
        "token_type": "Bearer",
        "expires_in": 3600,
        "scope": " ".join(SCOPES),
        **extra,
    }


def signed_in(path: Path, expires_in: float = 3600) -> ChatGPTAuth:
    auth = ChatGPTAuth(path)
    auth.save(
        ChatGPTCredentials(
            client_id="client-123",
            subject="user-1",
            email="ben@example.com",
            access_token="at-old",
            refresh_token="rt-old",
            expires_at=time.time() + expires_in,
            scopes=SCOPES,
        )
    )
    return auth


@pytest.mark.asyncio
@respx.mock
async def test_login_runs_pkce_flow_and_saves_tokens_privately(tmp_path: Path):
    seen: dict[str, str] = {}

    def browser(url: str) -> None:
        # Stand in for the user: OpenAI redirects back to the local callback with a code and the new client id.
        query = {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}
        seen.update(query)
        callback = f"{query['redirect_uri']}?" + urlencode(
            {"code": "code-1", "state": query["state"], "client_id": "client-123"}
        )
        urllib.request.urlopen(callback, timeout=5).read()

    def token(request: httpx.Request) -> httpx.Response:
        form = parse_qs(request.content.decode())
        assert form["grant_type"] == ["authorization_code"] and form["client_id"] == ["client-123"]
        assert form["resource"] == [RESOURCE] and form["code_verifier"]
        claims = {"iss": "https://auth.openai.com", "aud": "client-123", "sub": "user-1",
                  "email": "ben@example.com", "nonce": seen["nonce"]}  # fmt: skip
        return httpx.Response(200, json=token_reply(id_token=id_token(**claims)))

    respx.post(TOKEN_URL).mock(side_effect=token)
    respx.route(host="127.0.0.1").pass_through()
    path = tmp_path / "cfg" / "chatgpt.json"
    creds = await ChatGPTAuth(path).login(open_browser=browser, timeout=10)

    assert seen["client_id"] == "dynamic_agent_client" and seen["agent_name_hint"] == "jobhunt"
    assert "chatgpt.tokens.use.direct" in seen["scope"] and seen["code_challenge_method"] == "S256"
    assert creds.email == "ben@example.com" and creds.has_plan_scopes
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert ChatGPTAuth(path).load().refresh_token == "rt-1"


@pytest.mark.asyncio
@respx.mock
async def test_access_token_refreshes_and_keeps_the_rotated_refresh_token(tmp_path: Path):
    auth = signed_in(tmp_path / "c.json", expires_in=10)  # inside the refresh margin
    route = respx.post(TOKEN_URL).respond(json=token_reply(access="at-new", refresh="rt-new"))

    assert await auth.access_token() == "at-new"
    assert parse_qs(route.calls[0].request.content.decode())["refresh_token"] == ["rt-old"]
    assert auth.load().refresh_token == "rt-new"
    assert await auth.access_token() == "at-new" and route.call_count == 1  # cached until it nears expiry


@pytest.mark.asyncio
@respx.mock
async def test_revoked_sign_in_stops_with_login_hint_and_clears_tokens(tmp_path: Path):
    auth = signed_in(tmp_path / "c.json", expires_in=0)
    respx.post(TOKEN_URL).respond(400, json={"error": "invalid_grant"})
    with pytest.raises(ChatGPTAuthError, match="jobhunt chatgpt login"):
        await auth.access_token()
    assert not auth.load().signed_in
    with pytest.raises(ChatGPTAuthError, match="Not signed in"):
        await auth.access_token()


def sse(*events: dict) -> str:
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)


@pytest.mark.asyncio
@respx.mock
async def test_client_streams_structured_output_and_meters_it_as_free(tmp_path: Path):
    text = json.dumps(FIT)
    done = {
        "type": "response.completed",
        "response": {"status": "completed", "usage": {"input_tokens": 900, "output_tokens": 40}},
    }
    route = respx.post(f"{RESOURCE}/responses").respond(
        200,
        text=sse(
            {"type": "response.output_text.delta", "delta": text[:10]},
            {"type": "response.output_text.delta", "delta": text[10:]},
            done,
        ),
        headers={"content-type": "text/event-stream"},
    )
    meter = UsageMeter()
    async with httpx.AsyncClient() as http:
        client = ChatGPTClient(http, signed_in(tmp_path / "c.json"), meter, reasoning_effort="low")
        with meter.step("fit"):
            result = await client.structured("chatgpt/gpt-5.6-luna", "be brief", "judge", FitJudgment)

    assert result.rationale == "x"
    body = json.loads(route.calls[0].request.content)
    assert body["model"] == "gpt-5.6-luna" and body["store"] is False and body["stream"] is True
    assert body["reasoning"] == {"effort": "low"}
    assert body["instructions"] == "be brief" and body["text"]["format"]["type"] == "json_schema"
    assert route.calls[0].request.headers["authorization"] == "Bearer at-old"
    (usage,) = meter.take("fit")
    assert (usage.provider, usage.input_tokens, usage.cost, usage.estimated) == ("chatgpt", 900, 0.0, False)
    assert list_price("chatgpt/gpt-5.6-terra") == (0.0, 0.0)


@pytest.mark.asyncio
@respx.mock
async def test_plan_limit_stops_the_run_instead_of_failing_each_job(tmp_path: Path):
    respx.post(f"{RESOURCE}/responses").respond(
        429,
        json={"error": {"code": "usage_limit_reached", "message": "You've hit your limit", "resets_in_seconds": 3600}},
    )
    async with httpx.AsyncClient() as http:
        client = ChatGPTClient(http, signed_in(tmp_path / "c.json"))
        with pytest.raises(LlmConfigError, match="usage limit reached.*resets in about 60 min"):
            await client.structured("chatgpt/gpt-5.6-luna", "s", "u", FitJudgment)


@pytest.mark.asyncio
@respx.mock
async def test_rate_limit_waits_and_retries(tmp_path: Path, monkeypatch):
    import jobhunt.services.chatgpt as chatgpt_module

    waits: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr(chatgpt_module.asyncio, "sleep", fake_sleep)
    route = respx.post(f"{RESOURCE}/responses")
    route.side_effect = [
        httpx.Response(429, headers={"retry-after": "3"}, json={"error": {"code": "rate_limit_exceeded"}}),
        httpx.Response(
            200,
            text=sse(
                {"type": "response.output_text.delta", "delta": json.dumps(FIT)},
                {"type": "response.completed", "response": {"usage": {}}},
            ),
        ),
    ]
    async with httpx.AsyncClient() as http:
        client = ChatGPTClient(http, signed_in(tmp_path / "c.json"))
        assert (await client.structured("chatgpt/m", "s", "u", FitJudgment)).rationale == "x"
    assert waits == [3.0]


@pytest.mark.asyncio
@respx.mock
async def test_expired_token_mid_run_is_refreshed_once(tmp_path: Path):
    route = respx.post(f"{RESOURCE}/responses")
    route.side_effect = [
        httpx.Response(401, json={"error": "expired"}),
        httpx.Response(
            200,
            text=sse(
                {"type": "response.output_text.delta", "delta": json.dumps(FIT)},
                {"type": "response.completed", "response": {"usage": {}}},
            ),
        ),
    ]
    respx.post(TOKEN_URL).respond(json=token_reply(access="at-new", refresh="rt-new"))
    async with httpx.AsyncClient() as http:
        client = ChatGPTClient(http, signed_in(tmp_path / "c.json"))
        assert (await client.structured("chatgpt/m", "s", "u", FitJudgment)).rationale == "x"
    assert route.calls[1].request.headers["authorization"] == "Bearer at-new"


def test_strict_schemas_have_nothing_beside_ref():
    from jobhunt.schemas.extraction import JobRequirements, extraction_json_schema, strict_json_schema
    from jobhunt.services.skill_classify import SkillSuggestions

    def refs_with_siblings(node: object) -> list[dict]:
        if isinstance(node, dict):
            found = [node] if "$ref" in node and len(node) > 1 else []
            return found + [hit for value in node.values() for hit in refs_with_siblings(value)]
        if isinstance(node, list):
            return [hit for item in node for hit in refs_with_siblings(item)]
        return []

    for schema in (
        extraction_json_schema(["testing"]),
        strict_json_schema(SkillSuggestions),
        strict_json_schema(FitJudgment),
    ):
        assert refs_with_siblings(schema) == []
    assert "description" in strict_json_schema(JobRequirements)["$defs"]["RoleFamily"]
