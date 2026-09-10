"""Credential handling for the Strava REST API. No network, no real tokens."""

from __future__ import annotations

import time

import pytest

from coach_graph import config, strava_api


@pytest.fixture(autouse=True)
def isolated_token_path(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "TOKEN_PATH", tmp_path / "strava-tokens.json")
    monkeypatch.setattr(config, "STRAVA_CLIENT_ID", "12345")
    monkeypatch.setattr(config, "STRAVA_CLIENT_SECRET", "secret")


def test_no_tokens_before_connecting():
    assert strava_api.load_tokens() is None


def test_tokens_round_trip_and_are_owner_only():
    strava_api.save_tokens({"access_token": "abc", "refresh_token": "ref", "expires_at": 1})
    assert strava_api.load_tokens()["access_token"] == "abc"
    assert config.TOKEN_PATH.stat().st_mode & 0o077 == 0


def test_corrupt_token_file_reads_as_missing():
    config.TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.TOKEN_PATH.write_text("{not json")
    assert strava_api.load_tokens() is None


def test_authorize_url_uses_comma_separated_scopes():
    # Strava's classic OAuth rejects space-separated scopes.
    url = strava_api.authorize_url()
    assert "scope=read%2Cactivity%3Aread_all%2Cprofile%3Aread_all" in url
    assert "client_id=12345" in url


def test_missing_app_credentials_are_explained(monkeypatch):
    monkeypatch.setattr(config, "STRAVA_CLIENT_ID", "")
    with pytest.raises(strava_api.StravaAuthError, match="settings/api"):
        strava_api.authorize_url()


async def test_access_token_without_connecting_tells_you_to_connect():
    with pytest.raises(strava_api.StravaAuthError, match="coach connect"):
        await strava_api.access_token()


async def test_valid_token_is_used_as_is(monkeypatch):
    strava_api.save_tokens(
        {"access_token": "fresh", "refresh_token": "ref", "expires_at": time.time() + 3600}
    )

    async def fail(_):  # refreshing a valid token would be a wasted round trip
        raise AssertionError("should not refresh a token that is still valid")

    monkeypatch.setattr(strava_api, "_refresh", fail)
    assert await strava_api.access_token() == "fresh"


async def test_expiring_token_is_refreshed(monkeypatch):
    strava_api.save_tokens(
        {"access_token": "stale", "refresh_token": "ref", "expires_at": time.time() + 10}
    )
    called = {}

    async def refresh(tokens):
        called["refresh_token"] = tokens["refresh_token"]
        return {**tokens, "access_token": "renewed"}

    monkeypatch.setattr(strava_api, "_refresh", refresh)
    assert await strava_api.access_token() == "renewed"
    assert called["refresh_token"] == "ref"


# --- disk-backed response cache -------------------------------------------------
# Each MCP tool call is a fresh subprocess, so the cache only helps if it survives
# on disk; these exercise it directly rather than through a live request.


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "CACHE_TTL", 300)


def test_cache_miss_then_hit():
    key = strava_api._cache_key("/athlete/activities", {"after": 1})
    assert strava_api._cache_read(key) is None
    strava_api._cache_write(key, {"count": 3})
    assert strava_api._cache_read(key) == {"count": 3}


def test_cache_entry_expires(monkeypatch):
    key = strava_api._cache_key("/athlete", {})
    strava_api._cache_write(key, {"id": 1})
    monkeypatch.setattr(config, "CACHE_TTL", 0.0001)
    time.sleep(0.01)
    assert strava_api._cache_read(key) is None


def test_cache_can_be_disabled(monkeypatch):
    monkeypatch.setattr(config, "CACHE_TTL", 0)
    key = strava_api._cache_key("/athlete", {})
    strava_api._cache_write(key, {"id": 1})
    assert strava_api._cache_read(key) is None


def test_different_params_do_not_collide():
    a = strava_api._cache_key("/athlete/activities", {"after": 1})
    b = strava_api._cache_key("/athlete/activities", {"after": 2})
    assert a != b


def test_param_order_does_not_change_the_key():
    a = strava_api._cache_key("/x", {"after": 1, "per_page": 50})
    b = strava_api._cache_key("/x", {"per_page": 50, "after": 1})
    assert a == b


def test_corrupt_cache_entry_reads_as_a_miss():
    key = strava_api._cache_key("/athlete", {})
    strava_api._cache_write(key, {"id": 1})
    (config.CACHE_DIR / f"{key}.json").write_text("{truncated")
    assert strava_api._cache_read(key) is None


def test_cached_training_data_is_owner_only():
    key = strava_api._cache_key("/athlete", {})
    strava_api._cache_write(key, {"id": 1})
    assert (config.CACHE_DIR / f"{key}.json").stat().st_mode & 0o077 == 0
    assert config.CACHE_DIR.stat().st_mode & 0o077 == 0


def test_clear_cache_removes_entries():
    for i in range(3):
        strava_api._cache_write(strava_api._cache_key(f"/a{i}", {}), {"i": i})
    assert strava_api.clear_cache() == 3
    assert strava_api.clear_cache() == 0
