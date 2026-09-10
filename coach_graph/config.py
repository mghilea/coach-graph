"""Runtime configuration, read once from the environment."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

STRAVA_CLIENT_ID = os.getenv("STRAVA_CLIENT_ID", "")
STRAVA_CLIENT_SECRET = os.getenv("STRAVA_CLIENT_SECRET", "")
OAUTH_CALLBACK_PORT = int(os.getenv("COACH_OAUTH_PORT", "8765"))
OAUTH_CALLBACK_URL = f"http://localhost:{OAUTH_CALLBACK_PORT}/callback"

DATA_DIR = Path(os.getenv("COACH_DATA_DIR", PROJECT_ROOT / "data"))
STATE_DIR = Path(os.getenv("COACH_STATE_DIR", Path.home() / ".config" / "coach-graph"))
PROFILE_PATH = DATA_DIR / "profile.yaml"
PLANS_DIR = DATA_DIR / "plans"
CURRENT_PLAN_PATH = PLANS_DIR / "current.json"
OBSERVATIONS_PATH = DATA_DIR / "observations.yaml"
TOKEN_PATH = STATE_DIR / "strava-tokens.json"
CACHE_DIR = STATE_DIR / "cache"
# Seconds a Strava response stays reusable. Each MCP tool call is a fresh
# subprocess, so this cache has to live on disk to survive at all. 0 disables it.
CACHE_TTL = int(os.getenv("COACH_CACHE_TTL", "300"))
CHECKPOINT_PATH = STATE_DIR / "conversations.sqlite"

COACH_MODEL = os.getenv("COACH_MODEL", "claude-sonnet-5")
ROUTER_MODEL = os.getenv("COACH_ROUTER_MODEL", "claude-haiku-4-5-20251001")
# The data agent calls tools and writes up the figures they return; it does no coaching
# reasoning, and that work dominated turn latency on the coach's model.
DATA_MODEL = os.getenv("COACH_DATA_MODEL", ROUTER_MODEL)


def chat_model(model: str, **kwargs):
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model=model, **kwargs)
