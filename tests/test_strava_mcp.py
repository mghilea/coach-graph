from __future__ import annotations

from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

from coach_graph.strava_mcp import FileTokenStorage


async def test_token_storage_round_trips_tokens_and_registration(tmp_path):
    storage = FileTokenStorage(tmp_path / "oauth.json")
    assert await storage.get_tokens() is None
    assert await storage.get_client_info() is None

    await storage.set_tokens(OAuthToken(access_token="abc", token_type="Bearer", refresh_token="ref"))
    await storage.set_client_info(
        OAuthClientInformationFull(client_id="cid", redirect_uris=["http://localhost:8765/callback"])
    )

    tokens = await storage.get_tokens()
    assert tokens.access_token == "abc"
    assert tokens.refresh_token == "ref"
    # client registration must survive writing tokens, or every run re-registers
    assert (await storage.get_client_info()).client_id == "cid"
    assert (tmp_path / "oauth.json").stat().st_mode & 0o077 == 0
