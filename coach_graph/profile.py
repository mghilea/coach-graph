"""Athlete context that Strava does not store: goals, constraints, preferences."""

from __future__ import annotations

import yaml

from coach_graph import config

TEMPLATE = """\
# Everything here is fed to your coach. Edit freely; free text is fine.
name: null
goals:
  - Describe what you are training for, with dates if you have them.
constraints:
  - Injuries, time available per week, equipment, anything limiting.
preferences:
  - How you like to be coached, sessions you enjoy or hate.
personal_bests: {}
notes: null
"""


def load() -> dict:
    if not config.PROFILE_PATH.exists():
        return {}
    return yaml.safe_load(config.PROFILE_PATH.read_text()) or {}


def ensure_exists() -> bool:
    """Writes the starter template on first run. True if it created the file."""
    if config.PROFILE_PATH.exists():
        return False
    config.PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.PROFILE_PATH.write_text(TEMPLATE)
    return True


def as_prompt_block() -> str:
    data = load()
    if not data:
        return "No athlete profile on file yet — ask about goals before assuming any."
    return yaml.safe_dump(data, sort_keys=False).strip()
