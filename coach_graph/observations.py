"""What the coach has learned that outlives a single conversation.

Two kinds of thing live here. Personal bests are pulled from Strava rather than typed
in by hand — the athlete should not be maintaining a table the data already contains.
Notes are durable factual observations an agent noticed once and would otherwise
forget when the thread ends.

Both are snapshots. Agents are told to treat them as context, never as a substitute
for looking at fresh data.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import yaml

from coach_graph import config

# Which activities have already been mined for best efforts. Bounded so the file cannot
# grow without limit; an id that ages out is simply rescanned once.
SCANNED_LIMIT = 400


def load() -> dict[str, Any]:
    if not config.OBSERVATIONS_PATH.exists():
        return {}
    try:
        return yaml.safe_load(config.OBSERVATIONS_PATH.read_text()) or {}
    except yaml.YAMLError:
        return {}


def save(data: dict[str, Any]) -> None:
    config.OBSERVATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.OBSERVATIONS_PATH.write_text(yaml.safe_dump(data, sort_keys=False))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def set_personal_bests(bests: dict[str, Any], scanned_ids: list[int] | None = None,
                       replace: bool = False) -> None:
    """Store bests, and record which activities they were mined from.

    Scanned ids accumulate by default, so a caller passing only what it just looked at
    cannot silently discard the history and cause a full rescan. `replace` is the
    explicit reset, for rebuilding from scratch.
    """
    data = load()
    history = [] if replace else scanned()
    merged = list(dict.fromkeys([*history, *(scanned_ids or [])]))[-SCANNED_LIMIT:]
    data["personal_bests"] = {"refreshed": _now(), "bests": bests, "scanned_ids": merged}
    save(data)


def scanned() -> list[int]:
    """Activity ids already mined for best efforts.

    Bests stored before ids were tracked still name the activity each came from, so
    those count as scanned rather than triggering a full rescan. Runs that were scanned
    but set no best are not represented and get looked at once more, which is harmless.
    """
    block = load().get("personal_bests") or {}
    if block.get("scanned_ids"):
        return block["scanned_ids"]
    return list(dict.fromkeys(
        best["activity_id"] for best in (block.get("bests") or {}).values()
        if isinstance(best, dict) and best.get("activity_id")
    ))


def personal_bests() -> dict[str, Any]:
    return (load().get("personal_bests") or {}).get("bests") or {}


def unscanned(activity_ids: list[int]) -> list[int]:
    """Which of these activities have never been mined for best efforts.

    This is the invalidation signal: a new run means the bests may be out of date, and
    no run means they cannot be, whatever the calendar says. A clock would refresh
    after a fortnight of rest and miss a personal best set the morning after a refresh.
    """
    already = set(scanned())
    return [i for i in activity_ids if i not in already]


def add_note(text: str, source: str = "coach") -> dict[str, Any]:
    """Append a durable observation. Identical text is not recorded twice."""
    data = load()
    notes = data.setdefault("notes", [])
    if any(note.get("text", "").strip().lower() == text.strip().lower() for note in notes):
        return {"recorded": False, "reason": "already noted"}
    notes.append({"date": datetime.now(timezone.utc).date().isoformat(),
                  "source": source, "text": text.strip()})
    save(data)
    return {"recorded": True, "note_count": len(notes)}


def as_prompt_block() -> str:
    data = load()
    if not data:
        return "No observations on file yet."

    lines = []
    bests = personal_bests()
    if bests:
        lines.append("Personal bests (from Strava):")
        for distance, best in bests.items():
            parts = [f"  {distance}: {best.get('time')}"]
            if best.get("pace"):
                parts.append(f"({best['pace']})")
            if best.get("date"):
                parts.append(f"on {best['date']}")
            lines.append(" ".join(parts))

    notes = data.get("notes") or []
    if notes:
        lines.append("Observations:")
        for note in notes[-15:]:
            lines.append(f"  [{note.get('date')}] {note.get('text')}")

    return "\n".join(lines) if lines else "No observations on file yet."
