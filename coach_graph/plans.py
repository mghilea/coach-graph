"""The active plan, and how it compares to what actually happened.

The planner writes concrete target_* fields at the moment it prescribes a session,
so adherence is a comparison rather than a re-reading of prose. Nothing here is
stored: a verdict computed from a saved snapshot goes stale the moment an activity
is uploaded, so it is derived on demand from the plan plus fresh Strava data.
"""

from __future__ import annotations

import json
from typing import Any

from coach_graph import config

# How far short a session may fall and still count as met.
DISTANCE_TOLERANCE = 0.9
DURATION_TOLERANCE = 0.9
PACE_TOLERANCE = 1.05  # 5% slower than target still counts

# Sports that satisfy the same prescription.
FAMILIES = [
    {"Run", "TrailRun", "VirtualRun"},
    {"Ride", "VirtualRide", "MountainBikeRide", "GravelRide", "EBikeRide"},
    {"Swim"},
    {"WeightTraining", "Workout", "Crossfit"},
]


def load_current() -> dict[str, Any] | None:
    """The active plan, or None when nothing has been prescribed yet."""
    if not config.CURRENT_PLAN_PATH.exists():
        return None
    try:
        return json.loads(config.CURRENT_PLAN_PATH.read_text())
    except json.JSONDecodeError:
        return None


def pace_to_seconds(pace: str | None) -> float | None:
    """'9:08/mi' -> 548.0. None for anything unparseable."""
    if not pace:
        return None
    head = str(pace).split("/")[0].strip()
    parts = head.split(":")
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        if len(parts) == 1:
            return float(parts[0]) * 60
    except ValueError:
        return None
    return None


def same_sport(planned: str | None, actual: str | None) -> bool:
    if not planned or not actual:
        return False
    if planned.lower() == actual.lower():
        return True
    return any({planned, actual} <= family for family in FAMILIES)


def _shortfall(session: dict[str, Any], activity: dict[str, Any]) -> list[str]:
    """Which of the session's stated targets the activity failed to meet."""
    missed = []

    target_distance = session.get("target_distance_mi")
    actual_distance = activity.get("distance_mi")
    if target_distance and actual_distance is not None:
        if actual_distance < target_distance * DISTANCE_TOLERANCE:
            missed.append(f"distance {actual_distance} mi vs {target_distance} mi planned")

    target_minutes = session.get("target_duration_min")
    actual_seconds = activity.get("moving_time_s")
    if target_minutes and actual_seconds is not None:
        if actual_seconds < target_minutes * 60 * DURATION_TOLERANCE:
            missed.append(f"duration {round(actual_seconds / 60)} min vs {target_minutes} min planned")

    target_pace = pace_to_seconds(session.get("target_pace"))
    actual_pace = pace_to_seconds(activity.get("average_pace"))
    if target_pace and actual_pace:
        if actual_pace > target_pace * PACE_TOLERANCE:
            missed.append(f"pace {activity.get('average_pace')} vs {session.get('target_pace')} planned")

    cap = session.get("target_hr_max")
    actual_hr = activity.get("average_heartrate")
    if cap and actual_hr and actual_hr > cap:
        missed.append(f"average HR {actual_hr} over the {cap} bpm cap")

    return missed


def compare(sessions: list[dict[str, Any]], activities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Match each prescribed session to what was actually done that day."""
    results = []
    unmatched = list(activities)

    for session in sessions:
        if session.get("sport", "").lower() == "rest":
            results.append({**_planned(session), "verdict": "rest day"})
            continue

        match = next(
            (
                a
                for a in unmatched
                if (a.get("start_local") or "")[:10] == session.get("date")
                and same_sport(session.get("sport"), a.get("sport"))
            ),
            None,
        )
        if match is None:
            results.append({**_planned(session), "verdict": "missed", "actual": None})
            continue

        unmatched.remove(match)
        missed = _shortfall(session, match)
        results.append(
            {
                **_planned(session),
                "verdict": "met" if not missed else "short",
                "shortfall": missed or None,
                "actual": {
                    "activity_id": match.get("id"),
                    "name": match.get("name"),
                    "distance_mi": match.get("distance_mi"),
                    "moving_time": match.get("moving_time"),
                    "average_pace": match.get("average_pace"),
                    "average_heartrate": match.get("average_heartrate"),
                },
            }
        )

    return results


def _planned(session: dict[str, Any]) -> dict[str, Any]:
    planned = {
        "date": session.get("date"),
        "title": session.get("title"),
        "sport": session.get("sport"),
        "target_distance_mi": session.get("target_distance_mi"),
        "target_duration_min": session.get("target_duration_min"),
        "target_pace": session.get("target_pace"),
        "target_hr_max": session.get("target_hr_max"),
    }
    return {"planned": {k: v for k, v in planned.items() if v is not None}}


def summarise(results: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in results:
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
    scored = [r for r in results if r["verdict"] in ("met", "short", "missed")]
    return {
        "sessions": len(results),
        "met": counts.get("met", 0),
        "short": counts.get("short", 0),
        "missed": counts.get("missed", 0),
        "adherence_pct": round(100 * counts.get("met", 0) / len(scored)) if scored else None,
    }


def active_window() -> str | None:
    """The span the active plan covers, for telling the router a plan exists at all."""
    plan = load_current()
    dates = sorted(s["date"] for s in (plan or {}).get("sessions") or [] if s.get("date"))
    return f"{dates[0]} to {dates[-1]}" if dates else None


def as_prompt_block() -> str:
    """The active plan, for the coach and planner prompts."""
    plan = load_current()
    if not plan:
        return "No training plan is active."
    lines = [f"Active plan: {plan.get('title')}"]
    for session in plan.get("sessions") or []:
        targets = ", ".join(
            str(v) for v in (
                f"{session['target_distance_mi']} mi" if session.get("target_distance_mi") else None,
                f"{session['target_duration_min']} min" if session.get("target_duration_min") else None,
                session.get("target_pace"),
                f"HR<{session['target_hr_max']}" if session.get("target_hr_max") else None,
            ) if v
        )
        lines.append(f"  {session.get('date')} {session.get('sport')}: {session.get('title')}"
                     + (f" ({targets})" if targets else ""))
    return "\n".join(lines)


def active_window() -> str | None:
    """The span the active plan covers, for telling the router a plan exists at all."""
    plan = load_current()
    dates = sorted(s["date"] for s in (plan or {}).get("sessions") or [] if s.get("date"))
    return f"{dates[0]} to {dates[-1]}" if dates else None


def as_prompt_block() -> str:
    """The active plan, for the coach and planner prompts."""
    plan = load_current()
    if not plan:
        return "No training plan is active."
    lines = [f"Active plan: {plan.get('title')}"]
    for session in plan.get("sessions") or []:
        targets = ", ".join(
            str(v) for v in (
                f"{session['target_distance_mi']} mi" if session.get("target_distance_mi") else None,
                f"{session['target_duration_min']} min" if session.get("target_duration_min") else None,
                session.get("target_pace"),
                f"HR<{session['target_hr_max']}" if session.get("target_hr_max") else None,
            ) if v
        )
        lines.append(f"  {session.get('date')} {session.get('sport')}: {session.get('title')}"
                     + (f" ({targets})" if targets else ""))
    return "\n".join(lines)


# How much past schedule current.json keeps. Adherence only ever looks back weeks, and
# the dated archives are the real history.
KEEP_DAYS = 90


def merge(existing: dict[str, Any] | None, incoming: dict[str, Any]) -> dict[str, Any]:
    """Fold a newly prescribed plan into the active one, session by date.

    A session for a date the plan already covers replaces that day; every other day
    survives. This is what stops "what should I run tomorrow?" from throwing away a
    nine-week build — the planner answers with one session, and only that day moves.

    Deliberately per-date rather than per-span: a plan that names Tuesday and Thursday
    is not saying Wednesday is cancelled. To clear a day the planner prescribes Rest
    for it, which is a session like any other.
    """
    from datetime import date, timedelta

    by_date: dict[str, dict[str, Any]] = {}
    for session in (existing or {}).get("sessions") or []:
        if session.get("date"):
            by_date[session["date"]] = session
    for session in incoming.get("sessions") or []:
        if session.get("date"):
            by_date[session["date"]] = session

    cutoff = (date.today() - timedelta(days=KEEP_DAYS)).isoformat()
    sessions = [s for d, s in sorted(by_date.items()) if d >= cutoff]

    # Weekly focuses merge like sessions: a new one replaces that week, others stand.
    weeks_by_start: dict[str, dict[str, Any]] = {}
    for week in [*((existing or {}).get("weeks") or []), *(incoming.get("weeks") or [])]:
        if week.get("starting"):
            weeks_by_start[week["starting"]] = week
    weeks = [w for d, w in sorted(weeks_by_start.items()) if d >= cutoff]

    # The newest prose describes the block as it now stands. Prose carries a default so
    # that a missing field cannot fail a plan, which means an omission arrives as "" —
    # keep what the plan already said rather than leaving the athlete with no summary.
    previous = existing or {}
    return {
        "title": incoming.get("title") or previous.get("title") or "Training block",
        "rationale": incoming.get("rationale") or previous.get("rationale", ""),
        "sessions": sessions,
        "weeks": weeks,
        "watch_for": incoming.get("watch_for") or previous.get("watch_for", ""),
        "beyond": incoming.get("beyond") or previous.get("beyond", ""),
    }
