"""Athlete context that Strava does not store: goals, constraints, preferences.

Maintained by the `profiler` node rather than by hand — it reads durable facts out of
the conversation and merges them in. Merges only ever add: a hand-written line is never
deleted by an agent, because the athlete's own words about their body and goals outrank
anything inferred from a chat turn.

Deliberately not here: personal bests and anything else derivable from training data.
Those are pulled from Strava into `observations.py`. The one exception is the `observed`
block, which the habits tool owns outright and refreshes wholesale.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import yaml

from coach_graph import config

HEADER = """\
# Maintained by your coach from your conversations and your Strava history.
# Edit anything here freely — your own words are never overwritten or removed.
"""

TEMPLATE = HEADER + """\
name: null
goals:
  - Describe what you are training for, with dates if you have them.
constraints:
  - Injuries, time available per week, equipment, anything limiting.
preferences:
  - How you like to be coached, sessions you enjoy or hate.
notes: null
"""

LIST_FIELDS = ("goals", "constraints", "preferences")

# The starter lines above are prompts to the reader, not facts. They make way for the
# first real entry instead of sitting above it forever.
PLACEHOLDERS = {
    "describe what you are training for, with dates if you have them.",
    "injuries, time available per week, equipment, anything limiting.",
    "how you like to be coached, sessions you enjoy or hate.",
}


def load() -> dict:
    if not config.PROFILE_PATH.exists():
        return {}
    try:
        return yaml.safe_load(config.PROFILE_PATH.read_text()) or {}
    except yaml.YAMLError:
        return {}


def save(data: dict) -> None:
    """Write the profile back, keeping the header — yaml.safe_dump drops comments."""
    config.PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    config.PROFILE_PATH.write_text(HEADER + body)


def ensure_exists() -> bool:
    """Writes the starter template on first run. True if it created the file."""
    if config.PROFILE_PATH.exists():
        return False
    config.PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.PROFILE_PATH.write_text(TEMPLATE)
    return True


def _is_placeholder(entry: Any) -> bool:
    return str(entry).strip().lower() in PLACEHOLDERS


def update(changes: dict[str, Any]) -> dict[str, Any]:
    """Merge newly learned facts in. Returns only what actually changed.

    Additive by design: entries already present stay, near-duplicates are dropped, and
    a name already on file is not replaced.
    """
    data = load()
    changed: dict[str, Any] = {}

    name = (changes.get("name") or "").strip()
    if name and not data.get("name"):
        data["name"] = name
        changed["name"] = name

    for field in LIST_FIELDS:
        # `if str(v).strip()` would let a None through as the literal string "None".
        incoming = [
            str(v).strip()
            for v in (changes.get(field) or [])
            if v is not None and str(v).strip()
        ]
        if not incoming:
            continue
        existing = [e for e in (data.get(field) or []) if not _is_placeholder(e)]
        seen = {str(e).strip().lower() for e in existing}
        fresh = [v for v in incoming if v.lower() not in seen]
        if fresh:
            data[field] = existing + fresh
            changed[field] = fresh

    if changed:
        data["updated"] = date.today().isoformat()
        save(data)
    return changed


def set_observed(observed: dict[str, Any]) -> None:
    """Replace the machine-owned habits block. Never touches the stated fields."""
    data = load()
    data["observed"] = {**observed, "refreshed": date.today().isoformat()}
    save(data)


def as_prompt_block() -> str:
    data = load()
    if not data:
        return "No athlete profile on file yet — ask about goals before assuming any."
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True).strip()
