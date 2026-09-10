"""A local MCP server exposing read-only Strava data as tools.

Strava's hosted connector is closed to third-party clients, so the coach runs its
own MCP server over stdio and that server talks to the documented REST API. The
`strava_data` agent still speaks MCP; only the server behind the tools changed.

Units are imperial throughout — miles, minutes per mile, mph, feet — except where a
sport has its own universal convention (swimming in min/100m, rowing in min/500m).

Payloads are trimmed and streams are summarised rather than returned per-second —
this data lands in a sub-agent's context, so raw dumps are the thing to avoid.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from statistics import fmean
from typing import Any

from mcp.server.fastmcp import FastMCP

from coach_graph import observations, plans, profile, strava_api

mcp = FastMCP("strava", log_level="WARNING")

STREAM_KEYS = "time,heartrate,watts,velocity_smooth,altitude,cadence"

METRES_PER_MILE = 1609.344

# Distance over time only means something for sports that cover ground, and each is
# read in its own unit. Anything not listed here — squash, pickleball, weight training —
# gets no pace at all: a 0.43 mi squash match reduces to a three-digit pace, which is not
# a slow session but a meaningless one, and the agent is told to flag high HR for pace.
PACE_PER_MILE = {"Run", "TrailRun", "VirtualRun", "Walk", "Hike", "Snowshoe"}
PACE_PER_100M = {"Swim"}
PACE_PER_500M = {"Rowing", "Canoeing", "Kayaking", "StandUpPaddling"}
SPEED_MPH = {"Ride", "VirtualRide", "MountainBikeRide", "GravelRide", "EBikeRide",
             "EMountainBikeRide", "Handcycle", "Velomobile"}
RUN_SPORTS = {"Run", "TrailRun", "VirtualRun"}


def _miles(metres: float | None) -> float | None:
    return round(metres / METRES_PER_MILE, 2) if metres else None


def _feet(metres: float | None) -> int | None:
    return round(metres * 3.28084) if metres else None


def _mmss(seconds: float) -> str:
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


def _pace(speed_mps: float | None, sport: str | None = None) -> str | None:
    """Pace in the unit this sport is actually read in, or None if it has no pace.

    `sport` of None means "assume a foot sport" — used where Strava gives a speed
    without repeating the sport, as in per-mile splits.
    """
    if not speed_mps:
        return None
    if sport is None or sport in PACE_PER_MILE:
        return f"{_mmss(METRES_PER_MILE / speed_mps)}/mi"
    if sport in PACE_PER_100M:
        return f"{_mmss(100 / speed_mps)}/100m"
    if sport in PACE_PER_500M:
        return f"{_mmss(500 / speed_mps)}/500m"
    return None


def _speed_mph(speed_mps: float | None, sport: str | None = None) -> float | None:
    """Cyclists read mph, not minutes per mile."""
    if not speed_mps or sport not in SPEED_MPH:
        return None
    return round(speed_mps * 2.236936, 1)


def _duration(seconds: int | None) -> str | None:
    if not seconds:
        return None
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def _summarise_activity(activity: dict[str, Any]) -> dict[str, Any]:
    sport = activity.get("sport_type") or activity.get("type")
    speed = activity.get("average_speed")
    fields = {
        "id": activity.get("id"),
        "name": activity.get("name"),
        "sport": sport,
        "start_local": activity.get("start_date_local"),
        "distance_mi": _miles(activity.get("distance")),
        "moving_time": _duration(activity.get("moving_time")),
        # raw seconds so adherence can compare against a planned duration
        "moving_time_s": activity.get("moving_time"),
        "elapsed_time": _duration(activity.get("elapsed_time")),
        "elevation_gain_ft": _feet(activity.get("total_elevation_gain")),
        "average_pace": _pace(speed, sport),
        "average_speed_mph": _speed_mph(speed, sport),
        "average_heartrate": activity.get("average_heartrate"),
        "max_heartrate": activity.get("max_heartrate"),
        "average_watts": activity.get("average_watts"),
        "weighted_average_watts": activity.get("weighted_average_watts"),
        "relative_effort": activity.get("suffer_score"),
        "is_race": activity.get("workout_type") == 1,
    }
    return {k: v for k, v in fields.items() if v is not None}


@mcp.tool()
async def list_activities(days: int = 28, limit: int = 50) -> dict[str, Any]:
    """List the athlete's recent activities, most recent first.

    Use this for any question about training volume, consistency, or trends.
    `days` is how far back to look; `limit` caps how many come back.
    Distances are in miles.
    """
    after = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())
    raw = await strava_api.get("/athlete/activities", after=after, per_page=min(limit, 200))
    activities = [_summarise_activity(a) for a in raw]

    total = round(sum(a.get("distance_mi") or 0 for a in activities), 1)
    return {
        "window_days": days,
        "count": len(activities),
        "total_distance_mi": total,
        "activities": activities,
    }


@mcp.tool()
async def get_activity(activity_id: int) -> dict[str, Any]:
    """Full detail for one activity, including per-mile splits and gear.

    Call this after `list_activities` when a specific session needs a closer look.
    """
    raw = await strava_api.get(f"/activities/{activity_id}")
    sport = raw.get("sport_type") or raw.get("type")
    detail = _summarise_activity(raw)
    detail["description"] = raw.get("description")
    detail["calories"] = raw.get("calories")
    detail["device"] = raw.get("device_name")
    # splits_standard is Strava's per-mile breakdown; splits_metric is per-kilometre.
    detail["mile_splits"] = [
        {
            "mile": split.get("split"),
            "pace": _pace(split.get("average_speed"), sport),
            "speed_mph": _speed_mph(split.get("average_speed"), sport),
            "heartrate": split.get("average_heartrate"),
            "elevation_delta_ft": _feet(split.get("elevation_difference")),
        }
        for split in (raw.get("splits_standard") or [])
    ]
    return {k: v for k, v in detail.items() if v is not None}


@mcp.tool()
async def get_activity_streams(activity_id: int) -> dict[str, Any]:
    """Within-activity time series for one activity, summarised.

    Returns per-stream min/mean/max, a first-half vs second-half comparison, and a
    ~40-point downsample. Use this for questions about drift, fade, pacing
    discipline, or how effort evolved during a session.
    """
    raw = await strava_api.get(
        f"/activities/{activity_id}/streams", keys=STREAM_KEYS, key_by_type="true"
    )

    summary: dict[str, Any] = {"activity_id": activity_id}
    for name, stream in raw.items():
        data = [v for v in (stream or {}).get("data", []) if isinstance(v, (int, float))]
        if not data or name == "time":
            continue
        half = len(data) // 2
        summary[name] = {
            "min": round(min(data), 1),
            "mean": round(fmean(data), 1),
            "max": round(max(data), 1),
            "first_half_mean": round(fmean(data[:half]), 1) if half else None,
            "second_half_mean": round(fmean(data[half:]), 1) if half else None,
            "samples": len(data),
            # A coarse trace is enough to see shape; the raw series would be thousands of points.
            "downsampled": [round(v, 1) for v in data[:: max(1, len(data) // 40)]][:40],
        }
    return summary


@mcp.tool()
async def get_activity_laps(activity_id: int) -> list[dict[str, Any]]:
    """Lap or interval breakdown for one activity.

    The most direct way to read a structured workout: rep pace, duration, and heart rate.
    """
    # Laps carry no sport of their own, so the parent activity supplies the unit.
    activity = await strava_api.get(f"/activities/{activity_id}")
    sport = activity.get("sport_type") or activity.get("type")
    raw = await strava_api.get(f"/activities/{activity_id}/laps")
    return [
        {
            "lap": lap.get("lap_index"),
            "name": lap.get("name"),
            "distance_mi": _miles(lap.get("distance")),
            "moving_time": _duration(lap.get("moving_time")),
            "pace": _pace(lap.get("average_speed"), sport),
            "speed_mph": _speed_mph(lap.get("average_speed"), sport),
            "average_heartrate": lap.get("average_heartrate"),
            "max_heartrate": lap.get("max_heartrate"),
            "average_watts": lap.get("average_watts"),
        }
        for lap in raw
    ]


@mcp.tool()
async def get_athlete() -> dict[str, Any]:
    """The athlete's profile: name, sex, weight, and FTP where set."""
    raw = await strava_api.get("/athlete")
    fields = {
        "id": raw.get("id"),
        "firstname": raw.get("firstname"),
        "lastname": raw.get("lastname"),
        "sex": raw.get("sex"),
        "weight_lb": round(raw["weight"] * 2.20462, 1) if raw.get("weight") else None,
        "ftp": raw.get("ftp"),
        "city": raw.get("city"),
        "country": raw.get("country"),
    }
    return {k: v for k, v in fields.items() if v is not None}


@mcp.tool()
async def get_athlete_zones() -> dict[str, Any]:
    """The athlete's configured heart-rate and power zones.

    Needed to judge whether a session actually landed in its intended zone.
    """
    return await strava_api.get("/athlete/zones")


@mcp.tool()
async def get_athlete_stats() -> dict[str, Any]:
    """Lifetime and year-to-date totals for runs, rides, and swims."""
    athlete = await strava_api.get("/athlete")
    raw = await strava_api.get(f"/athletes/{athlete['id']}/stats")

    def totals(block: dict[str, Any] | None) -> dict[str, Any] | None:
        if not block:
            return None
        return {
            "count": block.get("count"),
            "distance_mi": _miles(block.get("distance")),
            "moving_time": _duration(block.get("moving_time")),
            "elevation_gain_ft": _feet(block.get("elevation_gain")),
        }

    return {
        key: totals(raw.get(key))
        for key in ("recent_run_totals", "ytd_run_totals", "all_run_totals",
                    "recent_ride_totals", "ytd_ride_totals", "all_ride_totals")
        if raw.get(key)
    }


@mcp.tool()
async def check_plan_adherence(days: int = 14) -> dict[str, Any]:
    """Compare the active training plan against what the athlete actually did.

    Answers "did I stick to the plan?" with real numbers: for each prescribed session
    it reports the target, the matching activity, and whether the target was met,
    fallen short of, or missed entirely. Sessions dated in the future are listed as
    upcoming rather than judged. Use this whenever the question touches the plan,
    progress toward a goal, or consistency.
    """
    plan = plans.load_current()
    if not plan:
        return {"plan": None, "message": "No plan has been prescribed yet."}

    today = date.today().isoformat()
    window_start = (date.today() - timedelta(days=days)).isoformat()

    sessions = plan.get("sessions") or []
    due = [s for s in sessions if window_start <= (s.get("date") or "") <= today]
    upcoming = [s for s in sessions if (s.get("date") or "") > today]

    activities = (await list_activities(days=days + 1, limit=200))["activities"]
    results = plans.compare(due, activities)

    return {
        "plan_title": plan.get("title"),
        "window_days": days,
        "summary": plans.summarise(results),
        "sessions": results,
        "upcoming": [
            {"date": s.get("date"), "title": s.get("title"), "sport": s.get("sport")}
            for s in upcoming
        ],
    }


@mcp.tool()
async def get_personal_bests(refresh: bool = False, days: int = 365,
                             max_activities: int = 25) -> dict[str, Any]:
    """The athlete's best times per distance, from Strava's own best efforts.

    Kept current by activity rather than by clock: only runs never mined before are
    fetched, and their efforts are merged into what is already stored. Costs one call
    when nothing new has been run, so it is cheap to call whenever bests are relevant.
    Pass `refresh` to rebuild from scratch and ignore what is stored.
    """
    after = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())
    raw = await strava_api.get("/athlete/activities", after=after, per_page=200)
    runs = [a for a in raw if (a.get("sport_type") or a.get("type")) in RUN_SPORTS]

    if refresh:
        pending, best = runs, {}
    else:
        new_ids = set(observations.unscanned([a["id"] for a in runs if a.get("id")]))
        pending = [a for a in runs if a.get("id") in new_ids]
        best = dict(observations.personal_bests())
        if not pending:
            return {"source": "stored", "new_runs": 0, "bests": best}

    scanned_ids: list[int] = []
    for activity in pending[:max_activities]:
        detail = await strava_api.get(f"/activities/{activity['id']}")
        scanned_ids.append(activity["id"])
        for effort in detail.get("best_efforts") or []:
            name, seconds = effort.get("name"), effort.get("elapsed_time")
            distance = effort.get("distance")
            if not name or not seconds:
                continue
            if name not in best or seconds < best[name]["seconds"]:
                best[name] = {
                    "time": _duration(seconds),
                    "seconds": seconds,
                    "pace": _pace(distance / seconds) if distance else None,
                    "date": (effort.get("start_date_local") or "")[:10],
                    "activity_id": activity.get("id"),
                }

    observations.set_personal_bests(best, scanned_ids=scanned_ids, replace=refresh)
    return {"source": "strava", "new_runs": len(scanned_ids), "bests": best}


@mcp.tool()
async def get_training_habits(days: int = 90) -> dict[str, Any]:
    """When and how much the athlete actually trains, and store it on their profile.

    Derived arithmetically from activity timestamps rather than asked about or inferred:
    which days of the week they train, what time of day, how many sessions a week, and
    typical and longest run distances. Use it when the question turns on routine,
    consistency, or fitting sessions into their week.
    """
    after = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())
    raw = await strava_api.get("/athlete/activities", after=after, per_page=200)
    if not raw:
        return {"window_days": days, "sessions": 0, "message": "No activities in this window."}

    weekdays: dict[str, int] = {}
    parts_of_day: dict[str, int] = {}
    sports: dict[str, int] = {}
    run_distances: list[float] = []

    for activity in raw:
        sport = activity.get("sport_type") or activity.get("type") or "Unknown"
        sports[sport] = sports.get(sport, 0) + 1

        # Strava stamps start_date_local with a Z it does not mean; the clock is local.
        stamp = (activity.get("start_date_local") or "").rstrip("Z")
        try:
            when = datetime.fromisoformat(stamp)
        except ValueError:
            continue
        weekdays[when.strftime("%a")] = weekdays.get(when.strftime("%a"), 0) + 1
        bucket = "morning" if when.hour < 11 else "midday" if when.hour < 16 else "evening"
        parts_of_day[bucket] = parts_of_day.get(bucket, 0) + 1

        if sport in RUN_SPORTS and activity.get("distance"):
            run_distances.append(activity["distance"] / METRES_PER_MILE)

    weeks = max(days / 7, 1)
    order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    # A day counts as habitual once it carries at least a third of the busiest day.
    threshold = max(weekdays.values()) / 3 if weekdays else 0

    habits = {
        "window_days": days,
        "sessions": len(raw),
        "sessions_per_week": round(len(raw) / weeks, 1),
        "usual_days": [d for d in order if weekdays.get(d, 0) >= threshold],
        "sessions_by_day": {d: weekdays[d] for d in order if d in weekdays},
        "usual_time_of_day": max(parts_of_day, key=parts_of_day.get) if parts_of_day else None,
        "sessions_by_time_of_day": parts_of_day,
        "sports": dict(sorted(sports.items(), key=lambda kv: -kv[1])),
    }
    if run_distances:
        habits["runs_per_week"] = round(len(run_distances) / weeks, 1)
        habits["run_miles_per_week"] = round(sum(run_distances) / weeks, 1)
        habits["typical_run_mi"] = round(fmean(run_distances), 1)
        habits["longest_run_mi"] = round(max(run_distances), 1)

    profile.set_observed(habits)
    return habits


@mcp.tool()
async def record_observation(text: str) -> dict[str, Any]:
    """Record one durable factual observation about the athlete for future conversations.

    For patterns worth remembering that no single activity shows — "heart rate drifts
    after 90 minutes", "runs faster on Sunday mornings than Wednesday evenings".
    Record facts you can point to data for, not training advice, and not one-off
    details already visible in the activity list.
    """
    return observations.add_note(text, source="strava_data")


if __name__ == "__main__":
    mcp.run()
