"""The local MCP server's tools: units, trimming, and stream summarising."""

from __future__ import annotations

import pytest

from coach_graph import strava_server

MPS_8MIN_MILE = 1609.344 / 480


@pytest.fixture
def fake_api(monkeypatch):
    """Route strava_api.get to canned payloads instead of the network."""
    responses: dict[str, object] = {}

    async def get(path, **params):
        if path not in responses:
            raise AssertionError(f"unexpected request: {path}")
        return responses[path]

    monkeypatch.setattr(strava_server.strava_api, "get", get)
    return responses


def test_pace_is_minutes_per_mile():
    assert strava_server._pace(MPS_8MIN_MILE) == "8:00/mi"
    assert strava_server._pace(None) is None


def test_distances_are_miles_and_climb_is_feet():
    assert strava_server._miles(1609.344) == 1.0
    assert strava_server._feet(100) == 328


# Distance over time is only meaningful for sports that cover ground, and each sport is
# read in its own unit. The data agent is told to flag unusually high heart rate for
# pace, so a bogus pace on a court sport invents a flag that isn't there.


@pytest.mark.parametrize(
    "sport,speed,expected",
    [
        ("Run", MPS_8MIN_MILE, "8:00/mi"),
        ("Walk", 0.835, "32:07/mi"),
        ("Swim", 1.041, "1:36/100m"),
        ("Rowing", 3.5, "2:22/500m"),
        ("Squash", 0.09, None),
        ("Pickleball", 0.514, None),
        ("WeightTraining", 0.0, None),
        ("Ride", 8.3, None),
    ],
)
def test_pace_uses_the_unit_each_sport_is_read_in(sport, speed, expected):
    assert strava_server._pace(speed, sport) == expected


def test_rides_report_mph_not_pace():
    assert strava_server._speed_mph(8.3, "Ride") == 18.6
    assert strava_server._speed_mph(MPS_8MIN_MILE, "Run") is None


def test_court_sports_carry_no_speed_metric_at_all():
    summary = strava_server._summarise_activity(
        {"id": 1, "name": "Evening Squash", "sport_type": "Squash",
         "distance": 430, "moving_time": 4834, "average_speed": 0.09,
         "average_heartrate": 148.4}
    )
    assert "average_pace" not in summary
    assert "average_speed_mph" not in summary
    # the session itself still counts — only the misleading figure is dropped
    assert summary["average_heartrate"] == 148.4
    assert summary["moving_time"] == "1:20:34"


def test_swim_pace_is_per_hundred_metres():
    summary = strava_server._summarise_activity(
        {"id": 2, "sport_type": "Swim", "distance": 1180,
         "moving_time": 1129, "average_speed": 1.041}
    )
    assert summary["average_pace"] == "1:36/100m"


def test_duration_grows_an_hours_field_only_when_needed():
    assert strava_server._duration(1830) == "30:30"
    assert strava_server._duration(3661) == "1:01:01"


def test_activity_summary_drops_empty_fields():
    summary = strava_server._summarise_activity(
        {"id": 1, "name": "Long run", "sport_type": "Run", "distance": 21097.5,
         "moving_time": 6300, "average_speed": MPS_8MIN_MILE,
         "average_heartrate": 152, "average_watts": None}
    )
    assert summary["distance_mi"] == 13.11
    assert summary["average_pace"] == "8:00/mi"
    assert summary["average_heartrate"] == 152
    assert "average_watts" not in summary  # absent, not None


def test_summary_keeps_raw_seconds_for_adherence():
    # plans.compare needs the unformatted duration to test a planned minute target
    summary = strava_server._summarise_activity(
        {"id": 1, "sport_type": "Run", "distance": 8046, "moving_time": 2400}
    )
    assert summary["moving_time_s"] == 2400
    assert summary["moving_time"] == "40:00"


async def test_list_activities_totals_the_window(fake_api):
    fake_api["/athlete/activities"] = [
        {"id": 1, "name": "Easy", "sport_type": "Run", "distance": 8046, "moving_time": 2400},
        {"id": 2, "name": "Long", "sport_type": "Run", "distance": 16093, "moving_time": 6000},
    ]
    result = await strava_server.list_activities(days=14)
    assert result["count"] == 2
    assert result["total_distance_mi"] == 15.0
    assert result["window_days"] == 14


async def test_streams_are_summarised_not_dumped(fake_api):
    # A real long run is thousands of samples; the coach must never receive them raw.
    drifting = list(range(140, 180)) * 100
    fake_api["/activities/9/streams"] = {
        "time": {"data": list(range(len(drifting)))},
        "heartrate": {"data": drifting},
    }
    result = await strava_server.get_activity_streams(activity_id=9)

    hr = result["heartrate"]
    assert hr["samples"] == 4000
    assert len(hr["downsampled"]) <= 40
    assert hr["min"] == 140 and hr["max"] == 179
    assert "data" not in hr
    assert "time" not in result  # the time axis carries no coaching signal


async def test_streams_report_first_and_second_half_for_drift(fake_api):
    fake_api["/activities/9/streams"] = {"heartrate": {"data": [140] * 50 + [170] * 50}}
    hr = (await strava_server.get_activity_streams(activity_id=9))["heartrate"]
    assert hr["first_half_mean"] == 140
    assert hr["second_half_mean"] == 170


async def test_laps_expose_rep_pace(fake_api):
    # laps carry no sport of their own, so the parent activity supplies the unit
    fake_api["/activities/9"] = {"id": 9, "sport_type": "Run"}
    fake_api["/activities/9/laps"] = [
        {"lap_index": 1, "name": "Rep 1", "distance": 1609.344, "moving_time": 420,
         "average_speed": 1609.344 / 420, "average_heartrate": 168}
    ]
    laps = await strava_server.get_activity_laps(activity_id=9)
    assert laps[0]["pace"] == "7:00/mi"
    assert laps[0]["distance_mi"] == 1.0


async def test_swim_laps_are_read_per_hundred_metres(fake_api):
    fake_api["/activities/7"] = {"id": 7, "sport_type": "Swim"}
    fake_api["/activities/7/laps"] = [
        {"lap_index": 1, "distance": 100, "moving_time": 96, "average_speed": 1.041}
    ]
    laps = await strava_server.get_activity_laps(activity_id=7)
    assert laps[0]["pace"] == "1:36/100m"


async def test_every_tool_is_registered():
    names = {tool.name for tool in await strava_server.mcp.list_tools()}
    assert names == {
        "list_activities", "get_activity", "get_activity_streams",
        "get_activity_laps", "get_athlete", "get_athlete_zones", "get_athlete_stats",
        "check_plan_adherence", "get_personal_bests", "get_training_habits",
        "record_observation",
    }


# --- habits, derived rather than asked about ------------------------------------


@pytest.fixture
def isolated_profile(monkeypatch, tmp_path):
    from coach_graph import config
    monkeypatch.setattr(config, "PROFILE_PATH", tmp_path / "profile.yaml")


def _run(day: str, hour: int, miles: float = 5.0):
    return {"id": abs(hash((day, hour))) % 10**6, "sport_type": "Run",
            "start_date_local": f"{day}T{hour:02d}:00:00Z",
            "distance": miles * 1609.344, "moving_time": 2400}


async def test_habits_find_the_usual_days_and_time(fake_api, isolated_profile):
    from coach_graph import profile
    # Mondays and Wednesdays, always early
    fake_api["/athlete/activities"] = [
        _run("2026-09-07", 7), _run("2026-09-09", 7), _run("2026-08-31", 6),
        _run("2026-09-02", 6), _run("2026-08-24", 7),
    ]
    habits = await strava_server.get_training_habits(days=28)

    assert habits["usual_time_of_day"] == "morning"
    assert set(habits["usual_days"]) == {"Mon", "Wed"}
    assert habits["sessions"] == 5
    assert habits["typical_run_mi"] == 5.0
    # the tool is what keeps the profile's observed block current
    assert profile.load()["observed"]["usual_days"] == habits["usual_days"]


async def test_habits_report_weekly_running_volume(fake_api, isolated_profile):
    fake_api["/athlete/activities"] = [_run("2026-09-07", 7, 6.0), _run("2026-09-09", 18, 4.0)]
    habits = await strava_server.get_training_habits(days=7)
    assert habits["run_miles_per_week"] == 10.0
    assert habits["longest_run_mi"] == 6.0


async def test_habits_survive_an_empty_window(fake_api, isolated_profile):
    fake_api["/athlete/activities"] = []
    habits = await strava_server.get_training_habits(days=28)
    assert habits["sessions"] == 0
    assert "message" in habits


async def test_habits_skip_activities_with_unreadable_timestamps(fake_api, isolated_profile):
    fake_api["/athlete/activities"] = [
        _run("2026-09-07", 7),
        {"id": 2, "sport_type": "Run", "start_date_local": "not-a-date", "distance": 8046},
    ]
    habits = await strava_server.get_training_habits(days=28)
    assert habits["sessions"] == 2          # both counted as sessions
    assert habits["sessions_by_day"] == {"Mon": 1}   # only one had a usable clock


# --- personal bests are kept current by activity, not by clock ----------------------


@pytest.fixture
def isolated_observations(monkeypatch, tmp_path):
    from coach_graph import config
    monkeypatch.setattr(config, "OBSERVATIONS_PATH", tmp_path / "observations.yaml")


def _pb_run(activity_id: int, name: str, seconds: int, day: str = "2026-09-01"):
    return {"id": activity_id, "sport_type": "Run", "start_date_local": f"{day}T07:00:00Z",
            "distance": 8046.0, "moving_time": seconds,
            "best_efforts": [{"name": name, "elapsed_time": seconds, "distance": 5000,
                              "start_date_local": f"{day}T07:00:00Z"}]}


async def test_first_call_mines_every_run(fake_api, isolated_observations):
    fake_api["/athlete/activities"] = [_pb_run(1, "5K", 1680)]
    fake_api["/activities/1"] = _pb_run(1, "5K", 1680)

    result = await strava_server.get_personal_bests()
    assert result["source"] == "strava" and result["new_runs"] == 1
    assert result["bests"]["5K"]["time"] == "28:00"


async def test_a_second_call_with_no_new_runs_costs_one_request(fake_api, isolated_observations):
    fake_api["/athlete/activities"] = [_pb_run(1, "5K", 1680)]
    fake_api["/activities/1"] = _pb_run(1, "5K", 1680)
    await strava_server.get_personal_bests()

    # the detail endpoint is now absent: requesting it again would raise
    del fake_api["/activities/1"]
    result = await strava_server.get_personal_bests()

    assert result["source"] == "stored" and result["new_runs"] == 0
    assert result["bests"]["5K"]["time"] == "28:00"


async def test_a_new_faster_run_updates_the_best(fake_api, isolated_observations):
    fake_api["/athlete/activities"] = [_pb_run(1, "5K", 1680)]
    fake_api["/activities/1"] = _pb_run(1, "5K", 1680)
    await strava_server.get_personal_bests()

    fake_api["/athlete/activities"] = [_pb_run(1, "5K", 1680), _pb_run(2, "5K", 1620, "2026-09-08")]
    fake_api["/activities/2"] = _pb_run(2, "5K", 1620, "2026-09-08")
    result = await strava_server.get_personal_bests()

    assert result["new_runs"] == 1          # only the unseen run was fetched
    assert result["bests"]["5K"]["time"] == "27:00"
    assert result["bests"]["5K"]["activity_id"] == 2


async def test_a_new_slower_run_leaves_the_best_alone(fake_api, isolated_observations):
    fake_api["/athlete/activities"] = [_pb_run(1, "5K", 1680)]
    fake_api["/activities/1"] = _pb_run(1, "5K", 1680)
    await strava_server.get_personal_bests()

    fake_api["/athlete/activities"] = [_pb_run(1, "5K", 1680), _pb_run(2, "5K", 1900, "2026-09-08")]
    fake_api["/activities/2"] = _pb_run(2, "5K", 1900, "2026-09-08")
    result = await strava_server.get_personal_bests()

    assert result["bests"]["5K"]["time"] == "28:00"
    assert result["bests"]["5K"]["activity_id"] == 1


async def test_refresh_rebuilds_from_scratch(fake_api, isolated_observations):
    fake_api["/athlete/activities"] = [_pb_run(1, "5K", 1680)]
    fake_api["/activities/1"] = _pb_run(1, "5K", 1680)
    await strava_server.get_personal_bests()

    # a best that is no longer supported by any activity must not survive a rebuild
    fake_api["/athlete/activities"] = [_pb_run(2, "5K", 1900, "2026-09-08")]
    fake_api["/activities/2"] = _pb_run(2, "5K", 1900, "2026-09-08")
    result = await strava_server.get_personal_bests(refresh=True)

    assert result["bests"]["5K"]["time"] == "31:40"
