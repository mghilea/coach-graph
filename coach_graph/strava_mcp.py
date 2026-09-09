"""Connection to Strava's official remote MCP server (https://mcp.strava.com/mcp).

Strava's MCP endpoint authenticates with OAuth rather than an API key, so the
first run opens a browser for consent and caches the resulting tokens on disk.
"""

from __future__ import annotations

import asyncio
import json
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp.client.auth import OAuthClientProvider, TokenStorage
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken
from pydantic import AnyUrl

from coach_graph import config

_SUCCESS_PAGE = b"""<html><body style="font-family:system-ui;padding:3rem">
<h2>Strava connected.</h2><p>You can close this tab and return to your terminal.</p>
</body></html>"""


class FileTokenStorage(TokenStorage):
    """Persists the OAuth client registration and tokens so consent is a one-off."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def _read(self) -> dict:
        if not self._path.exists():
            return {}
        return json.loads(self._path.read_text())

    def _write(self, data: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=2))
        self._path.chmod(0o600)

    async def get_tokens(self) -> OAuthToken | None:
        raw = self._read().get("tokens")
        return OAuthToken.model_validate(raw) if raw else None

    async def set_tokens(self, tokens: OAuthToken) -> None:
        self._write({**self._read(), "tokens": tokens.model_dump(mode="json")})

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        raw = self._read().get("client_info")
        return OAuthClientInformationFull.model_validate(raw) if raw else None

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        self._write({**self._read(), "client_info": client_info.model_dump(mode="json")})


async def _open_browser(authorization_url: str) -> None:
    print(f"\nAuthorize coach-graph with Strava:\n  {authorization_url}\n")
    webbrowser.open(authorization_url)


def _capture_code(port: int) -> tuple[str, str | None]:
    captured: dict[str, str] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            captured.update(
                {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
            )
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
        raise RuntimeError(f"Strava authorization failed: {captured['error']}")
    return captured["code"], captured.get("state")


async def _wait_for_callback() -> tuple[str, str | None]:
    return await asyncio.get_running_loop().run_in_executor(
        None, _capture_code, config.OAUTH_CALLBACK_PORT
    )


def build_auth() -> OAuthClientProvider:
    return OAuthClientProvider(
        server_url=config.STRAVA_MCP_URL,
        client_metadata=OAuthClientMetadata(
            client_name="coach-graph",
            redirect_uris=[AnyUrl(config.OAUTH_CALLBACK_URL)],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
        ),
        storage=FileTokenStorage(config.TOKEN_PATH),
        redirect_handler=_open_browser,
        callback_handler=_wait_for_callback,
    )


def build_client() -> MultiServerMCPClient:
    return MultiServerMCPClient(
        {
            "strava": {
                "transport": "streamable_http",
                "url": config.STRAVA_MCP_URL,
                "auth": build_auth(),
            }
        }
    )


async def load_tools() -> list:
    return await build_client().get_tools()
