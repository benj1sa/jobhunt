"""Sign in with ChatGPT, so extraction and fit draw on your ChatGPT plan instead of an OpenRouter bill.

This is OpenAI's OAuth flow for open-source tools (the same one LearnHarness uses): a browser sign-in at
auth.openai.com with PKCE, a one-shot callback on 127.0.0.1, and the `chatgpt.tokens.use.direct` scope, which lets
the access token call `https://api.openai.com/v1` against your plan's allowance. Tokens are stored outside the repo
(`CHATGPT_CREDENTIALS_PATH`, default ~/.config/jobhunt/chatgpt.json, mode 0600). Access tokens last about an hour;
the refresh token rotates on every refresh, so each new one is saved before it is used.

There is no API-key fallback: if the sign-in expires or lacks plan permission, calls fail with a message to run
`jobhunt chatgpt login` again.
"""

import asyncio
import base64
import contextlib
import fcntl
import hashlib
import json
import os
import secrets
import threading
import time
import uuid
import webbrowser
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from logging import getLogger
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

from .llm import LlmConfigError

logger = getLogger(__name__)

ISSUER = "https://auth.openai.com"
RESOURCE = "https://api.openai.com/v1"
AUTHORIZE_URL = f"{ISSUER}/api/accounts/authorize"
TOKEN_URL = f"{ISSUER}/api/accounts/oauth/token"
DISCOVERY_URL = f"{ISSUER}/.well-known/openid-configuration"
DYNAMIC_CLIENT = "dynamic_agent_client"
SCOPES = ["openid", "profile", "email", "offline_access", "resource.invoke", "chatgpt.tokens.use.direct"]
PLAN_SCOPES = {"resource.invoke", "chatgpt.tokens.use.direct"}
REFRESH_MARGIN_SECONDS = 60
LOGIN_HINT = "Run `jobhunt chatgpt login` to sign in again."


class ChatGPTAuthError(LlmConfigError):
    """Not signed in, sign-in expired, or the plan permission is missing. Stops the run; never falls back."""


@dataclass
class ChatGPTCredentials:
    host_id: str = field(default_factory=lambda: f"urn:uuid:{uuid.uuid4()}")
    client_id: str | None = None  # issued per install on first sign-in
    subject: str | None = None
    email: str | None = None
    access_token: str | None = None
    refresh_token: str | None = None
    expires_at: float = 0.0
    scopes: list[str] = field(default_factory=list)
    version: int = 1

    @property
    def signed_in(self) -> bool:
        return bool(self.access_token and self.refresh_token and self.client_id)

    @property
    def has_plan_scopes(self) -> bool:
        return PLAN_SCOPES <= set(self.scopes)


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def jwt_claims(token: str) -> dict[str, Any]:
    """Payload of a JWT without signature checks.

    Only used on ID tokens received directly from the token endpoint over TLS, which OpenID Connect allows in
    place of signature validation (Core 1.0, section 3.1.3.7). We check `aud` and `nonce` ourselves.
    """
    try:
        payload = token.split(".")[1]
        claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    except (IndexError, ValueError) as exc:
        raise ChatGPTAuthError("ChatGPT returned an unreadable ID token.") from exc
    if not isinstance(claims, dict):
        raise ChatGPTAuthError("ChatGPT returned an unreadable ID token.")
    return claims


@dataclass
class _Callback:
    params: dict[str, str] | None = None
    event: threading.Event = field(default_factory=threading.Event)


def _callback_server(expected_state: str) -> tuple[HTTPServer, _Callback]:
    """One-shot listener on a random 127.0.0.1 port. Ignores anything that isn't the matching callback."""
    result = _Callback()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            url = urlparse(self.path)
            params = {key: values[0] for key, values in parse_qs(url.query).items() if len(values) == 1}
            if url.path != "/auth/callback" or params.get("state") != expected_state or result.event.is_set():
                self.send_response(404)
                self.end_headers()
                return
            result.params = params
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(b"jobhunt: sign-in received. You can close this tab and go back to the terminal.")
            result.event.set()

        def log_message(self, format: str, *args: Any) -> None:  # silence default stderr logging
            return

    return HTTPServer(("127.0.0.1", 0), Handler), result


class ChatGPTAuth:
    def __init__(self, path: Path, http: httpx.AsyncClient | None = None) -> None:
        self.path = path.expanduser()
        self._http = http
        self._lock = asyncio.Lock()

    # Storage ------------------------------------------------------------------------------------------------------

    def load(self) -> ChatGPTCredentials:
        try:
            data = json.loads(self.path.read_text())
        except FileNotFoundError:
            return ChatGPTCredentials()
        except (OSError, ValueError) as exc:
            raise ChatGPTAuthError(f"Can't read {self.path}: {exc}. Delete it and sign in again.") from exc
        known = set(ChatGPTCredentials.__dataclass_fields__)
        return ChatGPTCredentials(**{key: value for key, value in data.items() if key in known})

    def save(self, creds: ChatGPTCredentials) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        tmp = self.path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as handle:
            json.dump(asdict(creds), handle, indent=2)
        os.replace(tmp, self.path)

    @contextlib.contextmanager
    def _file_lock(self) -> Iterator[None]:
        """Serializes refreshes across processes: a rotated refresh token can only be used once."""
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with open(self.path.with_suffix(".lock"), "w") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    # HTTP ---------------------------------------------------------------------------------------------------------

    @contextlib.asynccontextmanager
    async def _client(self) -> Any:
        if self._http is not None:
            yield self._http
        else:
            async with httpx.AsyncClient(timeout=15, follow_redirects=False) as client:
                yield client

    async def _grant(self, fields: dict[str, str]) -> dict[str, Any]:
        async with self._client() as client:
            try:
                response = await client.post(
                    TOKEN_URL,
                    data={**fields, "resource": RESOURCE},
                    headers={"Accept": "application/json"},
                )
            except httpx.HTTPError as exc:
                raise ChatGPTAuthError(f"Couldn't reach auth.openai.com: {exc}") from exc
        try:
            body = response.json()
        except ValueError:
            body = {}
        if response.status_code >= 400:
            error = body.get("error") if isinstance(body, dict) else None
            code = error if isinstance(error, str) else (error or {}).get("code") if isinstance(error, dict) else None
            if code == "invalid_grant" or response.status_code in (401, 403):
                raise ChatGPTAuthError(
                    f"ChatGPT sign-in is no longer valid ({code or response.status_code}). {LOGIN_HINT}"
                )
            raise ChatGPTAuthError(f"ChatGPT token request failed ({response.status_code}): {response.text[:300]}")
        if not isinstance(body, dict):
            raise ChatGPTAuthError("ChatGPT token endpoint returned an unexpected response.")
        return body

    @staticmethod
    def _apply_tokens(creds: ChatGPTCredentials, body: dict[str, Any]) -> None:
        access = body.get("access_token")
        if not isinstance(access, str) or not access or access.startswith("sk-"):
            raise ChatGPTAuthError("ChatGPT didn't return a plan access token.")
        if str(body.get("token_type", "")).lower() != "bearer":
            raise ChatGPTAuthError("ChatGPT returned an unexpected token type.")
        expires_in = body.get("expires_in")
        if not isinstance(expires_in, (int, float)) or expires_in <= 0:
            raise ChatGPTAuthError("ChatGPT returned a token without an expiry.")
        creds.access_token = access
        creds.refresh_token = body.get("refresh_token") or creds.refresh_token
        if not creds.refresh_token:
            raise ChatGPTAuthError("ChatGPT didn't return a refresh token.")
        creds.expires_at = time.time() + float(expires_in)
        if isinstance(body.get("scope"), str):
            creds.scopes = body["scope"].split()

    # Sign in / out ------------------------------------------------------------------------------------------------

    async def login(
        self, open_browser: Callable[[str], Any] = webbrowser.open, timeout: float = 300
    ) -> ChatGPTCredentials:
        """Browser sign-in. Prints nothing itself; `open_browser` receives the URL (the CLI also prints it)."""
        creds = self.load()
        verifier = _b64url(secrets.token_bytes(32))
        state = _b64url(secrets.token_bytes(32))
        nonce = _b64url(secrets.token_bytes(32))
        server, callback = _callback_server(state)
        redirect_uri = f"http://127.0.0.1:{server.server_address[1]}/auth/callback"
        params = {
            "client_id": creds.client_id or DYNAMIC_CLIENT,
            "ext_agent_host_id": creds.host_id,
            "response_type": "code",
            "redirect_uri": redirect_uri,
            "scope": " ".join(SCOPES),
            "resource": RESOURCE,
            "state": state,
            "nonce": nonce,
            "code_challenge_method": "S256",
            "code_challenge": _b64url(hashlib.sha256(verifier.encode()).digest()),
        }
        if creds.client_id is None:
            params["agent_name_hint"] = "jobhunt"
        if creds.email:
            params["login_hint"] = creds.email
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            await asyncio.to_thread(open_browser, f"{AUTHORIZE_URL}?{urlencode(params)}")
            if not await asyncio.to_thread(callback.event.wait, timeout):
                raise ChatGPTAuthError("ChatGPT sign-in timed out. Run `jobhunt chatgpt login` again.")
        finally:
            server.shutdown()
            server.server_close()

        reply = callback.params or {}
        if "error" in reply:
            raise ChatGPTAuthError(f"ChatGPT sign-in was not completed: {reply.get('error')}.")
        client_id = reply.get("client_id") or creds.client_id
        code = reply.get("code")
        if not client_id or client_id == DYNAMIC_CLIENT or not code:
            raise ChatGPTAuthError("ChatGPT sign-in returned no client registration or code.")
        if creds.client_id and client_id != creds.client_id:
            raise ChatGPTAuthError("ChatGPT sign-in came back for a different registration. Try again.")
        body = await self._grant(
            {
                "grant_type": "authorization_code",
                "client_id": client_id,
                "code": code,
                "code_verifier": verifier,
                "redirect_uri": redirect_uri,
            }
        )
        if not isinstance(body.get("id_token"), str):
            raise ChatGPTAuthError("ChatGPT sign-in returned no ID token.")
        claims = jwt_claims(body["id_token"])
        audience = claims.get("aud")
        audiences = audience if isinstance(audience, list) else [audience]
        if claims.get("iss") != ISSUER or client_id not in audiences or claims.get("nonce") != nonce:
            raise ChatGPTAuthError("ChatGPT sign-in could not be verified (issuer, audience or nonce mismatch).")
        if creds.subject and claims.get("sub") != creds.subject:
            # A different ChatGPT account: start a fresh record for it.
            creds = ChatGPTCredentials(host_id=creds.host_id, client_id=client_id)
        creds.client_id = client_id
        creds.subject = str(claims.get("sub") or "")
        creds.email = claims.get("email") if isinstance(claims.get("email"), str) else creds.email
        self._apply_tokens(creds, body)
        self.save(creds)
        if not creds.has_plan_scopes:
            raise ChatGPTAuthError(
                "Signed in, but plan usage wasn't granted (missing chatgpt.tokens.use.direct). Sign in again and "
                "approve using your ChatGPT plan, and check that your plan is eligible (Plus or Pro)."
            )
        return creds

    async def logout(self) -> bool:
        """Revoke the refresh token (best effort) and forget the tokens. Keeps the client registration."""
        creds = self.load()
        if not creds.refresh_token:
            return False
        try:
            async with self._client() as client:
                config = (await client.get(DISCOVERY_URL)).json()
                endpoint = config.get("revocation_endpoint") if isinstance(config, dict) else None
                if isinstance(endpoint, str) and endpoint.startswith(ISSUER):
                    await client.post(
                        endpoint,
                        data={
                            "token": creds.refresh_token,
                            "token_type_hint": "refresh_token",
                            "client_id": creds.client_id or "",
                        },
                    )
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("Couldn't confirm token revocation: %s", exc)
        creds.access_token = creds.refresh_token = None
        creds.expires_at = 0
        self.save(creds)
        return True

    # Use ----------------------------------------------------------------------------------------------------------

    async def access_token(self, force_refresh: bool = False) -> str:
        """A valid access token, refreshing (and saving the rotated refresh token) when it is about to expire."""
        async with self._lock:
            creds = self.load()
            if not creds.signed_in:
                raise ChatGPTAuthError(f"Not signed in to ChatGPT. {LOGIN_HINT.replace('again', 'first')}")
            if not creds.has_plan_scopes:
                raise ChatGPTAuthError(f"This ChatGPT sign-in has no plan-usage permission. {LOGIN_HINT}")
            if not force_refresh and creds.expires_at > time.time() + REFRESH_MARGIN_SECONDS:
                assert creds.access_token is not None
                return creds.access_token
            with self._file_lock():
                # Another process may have refreshed while we waited for the lock.
                fresh = self.load()
                if fresh.access_token != creds.access_token and fresh.expires_at > time.time() + REFRESH_MARGIN_SECONDS:
                    assert fresh.access_token is not None
                    return fresh.access_token
                assert fresh.client_id is not None and fresh.refresh_token is not None
                try:
                    body = await self._grant(
                        {
                            "grant_type": "refresh_token",
                            "client_id": fresh.client_id,
                            "refresh_token": fresh.refresh_token,
                        }
                    )
                except ChatGPTAuthError as exc:
                    if "no longer valid" in str(exc):
                        fresh.access_token = fresh.refresh_token = None
                        fresh.expires_at = 0
                        self.save(fresh)
                    raise
                self._apply_tokens(fresh, body)
                self.save(fresh)
                assert fresh.access_token is not None
                return fresh.access_token
