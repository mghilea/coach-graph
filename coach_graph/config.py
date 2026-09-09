"""Runtime configuration, read once from the environment."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

STRAVA_MCP_URL = os.getenv("STRAVA_MCP_URL", "https://mcp.strava.com/mcp")
OAUTH_CALLBACK_PORT = int(os.getenv("COACH_OAUTH_PORT", "8765"))
OAUTH_CALLBACK_URL = f"http://localhost:{OAUTH_CALLBACK_PORT}/callback"

DATA_DIR = Path(os.getenv("COACH_DATA_DIR", PROJECT_ROOT / "data"))
STATE_DIR = Path(os.getenv("COACH_STATE_DIR", Path.home() / ".config" / "coach-graph"))
PROFILE_PATH = DATA_DIR / "profile.yaml"
PLANS_DIR = DATA_DIR / "plans"
TOKEN_PATH = STATE_DIR / "strava-oauth.json"
CHECKPOINT_PATH = STATE_DIR / "conversations.sqlite"

COACH_MODEL = os.getenv("COACH_MODEL", "claude-sonnet-5")
ROUTER_MODEL = os.getenv("COACH_ROUTER_MODEL", "claude-haiku-4-5-20251001")


def chat_model(model: str, **kwargs):
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model=model, **kwargs)
