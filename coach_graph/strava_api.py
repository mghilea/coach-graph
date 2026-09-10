"""Strava REST API: the OAuth handshake, token refresh, and read-only requests.

Strava's MCP connector only authorizes Anthropic's own clients, so the coach reads
the documented REST API as your own API application instead. Everything here is
GET-only; nothing in this module can modify an account.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

from coach_graph import config

AUTHORIZE_URL = "https://www.strava.com/oauth/authorize"
TOKEN_URL = "https://www.strava.com/oauth/token"
API_BASE = "https://www.strava.com/api/v3"

# activity:read_all covers activities the athlete has marked private.
SCOPES = "read,activity:read_all,profile:read_all"

_SUCCESS_PAGE = b"""<html><body style="font-family:system-ui;padding:3rem">
<h2>Strava connected.</h2><p>You can close this tab and return to your terminal.</p>
</body></html>"""


class StravaAuthError(RuntimeError):
    """Raised when the coach has no usable credentials and cannot get them."""


def load_tokens() -> dict[str, Any] | None:
    if not config.TOKEN_PATH.exists():
        return None
    try:
        return json.loads(config.TOKEN_PATH.read_text())
    except json.JSONDecodeError:
        return None


def save_tokens(tokens: dict[str, Any]) -> None:
    config.TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.TOKEN_PATH.write_text(json.dumps(tokens, indent=2))
    config.TOKEN_PATH.chmod(0o600)


def _require_app() -> tuple[str, str]:
    if not (config.STRAVA_CLIENT_ID and config.STRAVA_CLIENT_SECRET):
        raise StravaAuthError(
            "No Strava API application configured. Create one at "
            "https://www.strava.com/settings/api and set STRAVA_CLIENT_ID and "
            "STRAVA_CLIENT_SECRET in .env."
        )
    return config.STRAVA_CLIENT_ID, config.STRAVA_CLIENT_SECRET


def _capture_code(port: int) -> str:
    captured: dict[str, str] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            captured.update({k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()})
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(_SUCCESS_PAGE)))
            self.end_headers()
            self.wfile.write(_SUCCESS_PAGE)

        def log_message(self, *args) -> None:  # keep the CLI quiet
            pass

    with HTTPServer(("127.0.0.1", port), Handler) as server:
        server.handle_request()

    if "error" in captured:
        raise StravaAuthError(f"Strava authorization failed: {captured['error']}")
    if "code" not in captured:
        raise StravaAuthError("Strava did not return an authorization code.")
    return captured["code"]


def authorize_url() -> str:
    client_id, _ = _require_app()
    return AUTHORIZE_URL + "?" + urlencode(
        {
            "client_id": client_id,
            "redirect_uri": config.OAUTH_CALLBACK_URL,
            "response_type": "code",
            "approval_prompt": "auto",
            # Strava's classic OAuth takes scopes comma-separated, not space-separated.
            "scope": SCOPES,
        }
    )


async def authorize() -> dict[str, Any]:
    """Run the browser handshake once and persist the resulting tokens."""
    client_id, client_secret = _require_app()
    url = authorize_url()
    print(f"\nAuthorize coach-graph with Strava:\n  {url}\n")
    webbrowser.open(url)

    loop = asyncio.get_running_loop()
    code = await loop.run_in_executor(None, _capture_code, config.OAUTH_CALLBACK_PORT)

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            TOKEN_URL,
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "code": code,
                "grant_type": "authorization_code",
            },
        )
    if response.status_code != 200:
        raise StravaAuthError(f"Token exchange failed: {response.status_code} {response.text}")

    tokens = response.json()
    save_tokens(tokens)
    return tokens


async def _refresh(tokens: dict[str, Any]) -> dict[str, Any]:
    client_id, client_secret = _require_app()
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            TOKEN_URL,
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "grant_type": "refresh_token",
                "refresh_token": tokens["refresh_token"],
            },
        )
    if response.status_code != 200:
        raise StravaAuthError(
            f"Token refresh failed: {response.status_code} {response.text}. "
            "Run `uv run coach connect` to re-authorize."
        )
    # A refresh response carries no athlete block; keep what we already had.
    refreshed = {**tokens, **response.json()}
    save_tokens(refreshed)
    return refreshed


async def access_token() -> str:
    """A valid access token, refreshed when it is close to expiry."""
    tokens = load_tokens()
    if not tokens or "access_token" not in tokens:
        raise StravaAuthError("Not connected to Strava. Run `uv run coach connect` first.")
    if tokens.get("expires_at", 0) <= time.time() + 60:
        tokens = await _refresh(tokens)
    return tokens["access_token"]


def _cache_key(path: str, params: dict[str, Any]) -> str:
    blob = json.dumps({"path": path, "params": params}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


def _cache_read(key: str) -> Any | None:
    """A still-fresh response for this key, or None."""
    if config.CACHE_TTL <= 0:
        return None
    try:
        entry = json.loads((config.CACHE_DIR / f"{key}.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if time.time() - entry.get("at", 0) > config.CACHE_TTL:
        return None
    return entry.get("data")


def _cache_write(key: str, data: Any) -> None:
    if config.CACHE_TTL <= 0:
        return
    try:
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        config.CACHE_DIR.chmod(0o700)  # cached training data is as private as the tokens
        path = config.CACHE_DIR / f"{key}.json"
        path.write_text(json.dumps({"at": time.time(), "data": data}))
        path.chmod(0o600)
    except OSError:
        pass  # a cache that cannot be written is a slow path, not a failure


def clear_cache() -> int:
    """Drop every cached response. Returns how many were removed."""
    if not config.CACHE_DIR.exists():
        return 0
    removed = 0
    for entry in config.CACHE_DIR.glob("*.json"):
        entry.unlink(missing_ok=True)
        removed += 1
    return removed


async def get(path: str, **params: Any) -> Any:
    """GET a Strava API path, refreshing credentials as needed.

    Responses are cached on disk for `config.CACHE_TTL` seconds. Within one
    conversation the data agent re-lists activities on every turn to orient itself,
    and those repeats are identical; this serves them without a round trip.
    """
    clean = {k: v for k, v in params.items() if v is not None}
    key = _cache_key(path, clean)
    cached = _cache_read(key)
    if cached is not None:
        return cached

    token = await access_token()
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.get(
            f"{API_BASE}{path}",
            params=clean,
            headers={"Authorization": f"Bearer {token}"},
        )
    if response.status_code == 401:
        raise StravaAuthError("Strava rejected the token. Run `uv run coach connect` to re-authorize.")
    if response.status_code == 429:
        raise RuntimeError("Strava rate limit reached. Wait a few minutes and try again.")
    if response.status_code == 404:
        raise RuntimeError(f"Strava has no record of {path}.")
    response.raise_for_status()

    payload = response.json()
    _cache_write(key, payload)
    return payload
