"""The Strava issuer-mismatch workaround must stay narrow: same origin only."""

from __future__ import annotations

import pytest
from mcp.client.auth import oauth2
from mcp.client.auth.exceptions import OAuthFlowError
from mcp.shared.auth import OAuthMetadata

from coach_graph.strava_mcp import allow_same_origin_issuer_mismatch


@pytest.fixture(autouse=True)
def restore_validator():
    original = oauth2.validate_metadata_issuer
    yield
    oauth2.validate_metadata_issuer = original


def metadata(issuer: str) -> OAuthMetadata:
    return OAuthMetadata(
        issuer=issuer,
        authorization_endpoint="https://www.strava.com/oauth/authorize",
        token_endpoint="https://www.strava.com/oauth/token",
    )


def test_strava_mismatch_is_tolerated():
    allow_same_origin_issuer_mismatch()

    # exactly what mcp.strava.com serves today
    oauth2.validate_metadata_issuer(metadata("https://www.strava.com/"), "https://www.strava.com/mcp-issuer")


def test_matching_issuer_still_passes():
    allow_same_origin_issuer_mismatch()

    oauth2.validate_metadata_issuer(metadata("https://www.strava.com/"), "https://www.strava.com/")


@pytest.mark.parametrize(
    "declared",
    [
        "https://evil.example.com/",  # different host
        "http://www.strava.com/",  # downgraded scheme
        "https://www.strava.com.evil.example/",  # lookalike host
    ],
)
def test_cross_origin_mismatch_still_rejected(declared):
    allow_same_origin_issuer_mismatch()

    with pytest.raises(OAuthFlowError):
        oauth2.validate_metadata_issuer(metadata(declared), "https://www.strava.com/mcp-issuer")


def test_patch_is_idempotent():
    allow_same_origin_issuer_mismatch()
    once = oauth2.validate_metadata_issuer
    allow_same_origin_issuer_mismatch()

    assert oauth2.validate_metadata_issuer is once, "repeated calls must not nest wrappers"
