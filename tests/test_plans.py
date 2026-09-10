"""Plan-versus-actual comparison. The point is that targets are captured as numbers
at prescription time, so adherence is a diff rather than a re-reading of prose."""

from __future__ import annotations

import json

import pytest

from coach_graph import config, plans

MPS_8MIN_MILE = 1609.344 / 480


@pytest.fixture(autouse=True)
def isolated_plan(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PLANS_DIR", tmp_path / "plans")
    monkeypatch.setattr(config, "CURRENT_PLAN_PATH", tmp_path / "plans" / "current.json")


def write_plan(sessions, title="Test block"):
    config.PLANS_DIR.mkdir(parents=True, exist_ok=True)
    config.CURRENT_PLAN_PATH.write_text(
        json.dumps({"title": title, "rationale": "", "sessions": sessions, "watch_for": ""})
    )


def activity(**kw):
    base = {"id": 1, "name": "Morning Run", "sport": "Run",
            "start_local": "2026-09-10T07:00:00Z", "distance_mi": 6.0,
            "moving_time_s": 2880, "moving_time": "48:00", "average_pace": "8:00/mi"}
    return {**base, **kw}


def test_no_plan_reads_as_none():
    assert plans.load_current() is None


def test_corrupt_plan_reads_as_none():
    config.PLANS_DIR.mkdir(parents=True, exist_ok=True)
    config.CURRENT_PLAN_PATH.write_text("{truncated")
    assert plans.load_current() is None


@pytest.mark.parametrize(
    "text,seconds",
    [("9:08/mi", 548.0), ("5:41/km", 341.0), ("8:00", 480.0), (None, None), ("easy", None)],
)
def test_pace_parsing(text, seconds):
    assert plans.pace_to_seconds(text) == seconds


def test_sport_families_match_but_unrelated_sports_do_not():
    assert plans.same_sport("Run", "TrailRun")
    assert plans.same_sport("Ride", "VirtualRide")
    assert not plans.same_sport("Run", "Swim")
    assert not plans.same_sport("Run", None)


def test_session_met_when_targets_are_hit():
    session = {"date": "2026-09-10", "title": "Easy", "sport": "Run",
               "target_distance_mi": 6.0, "target_pace": "8:30/mi"}
    result = plans.compare([session], [activity()])[0]
    assert result["verdict"] == "met"
    assert result["actual"]["distance_mi"] == 6.0


def test_session_short_when_distance_falls_away():
    session = {"date": "2026-09-10", "title": "Long", "sport": "Run",
               "target_distance_mi": 12.0}
    result = plans.compare([session], [activity()])[0]
    assert result["verdict"] == "short"
    assert "distance 6.0 mi vs 12.0 mi planned" in result["shortfall"]


def test_session_short_when_pace_is_slower_than_target():
    session = {"date": "2026-09-10", "title": "Tempo", "sport": "Run",
               "target_pace": "7:00/mi"}
    result = plans.compare([session], [activity()])[0]
    assert result["verdict"] == "short"
    assert any("pace" in reason for reason in result["shortfall"])


def test_small_shortfall_still_counts_as_met():
    # 5.7 of 6.0 miles is the session, not a failure
    session = {"date": "2026-09-10", "title": "Easy", "sport": "Run",
               "target_distance_mi": 6.0}
    result = plans.compare([session], [activity(distance_mi=5.7)])[0]
    assert result["verdict"] == "met"


def test_heart_rate_cap_is_enforced():
    session = {"date": "2026-09-10", "title": "Recovery", "sport": "Run",
               "target_hr_max": 135}
    result = plans.compare([session], [activity(average_heartrate=151)])[0]
    assert result["verdict"] == "short"
    assert "average HR 151 over the 135 bpm cap" in result["shortfall"]


def test_missed_when_nothing_happened_that_day():
    session = {"date": "2026-09-11", "title": "Long", "sport": "Run",
               "target_distance_mi": 12.0}
    result = plans.compare([session], [activity()])[0]
    assert result["verdict"] == "missed"
    assert result["actual"] is None


def test_wrong_sport_does_not_satisfy_a_session():
    session = {"date": "2026-09-10", "title": "Long run", "sport": "Run",
               "target_distance_mi": 6.0}
    result = plans.compare([session], [activity(sport="Swim")])[0]
    assert result["verdict"] == "missed"


def test_one_activity_cannot_satisfy_two_sessions():
    sessions = [
        {"date": "2026-09-10", "title": "AM", "sport": "Run", "target_distance_mi": 6.0},
        {"date": "2026-09-10", "title": "PM", "sport": "Run", "target_distance_mi": 6.0},
    ]
    results = plans.compare(sessions, [activity()])
    assert [r["verdict"] for r in results] == ["met", "missed"]


def test_rest_days_are_not_scored():
    result = plans.compare([{"date": "2026-09-10", "title": "Rest", "sport": "Rest"}], [])[0]
    assert result["verdict"] == "rest day"


def test_summary_counts_and_scores():
    results = [{"verdict": "met"}, {"verdict": "met"}, {"verdict": "short"},
               {"verdict": "missed"}, {"verdict": "rest day"}]
    summary = plans.summarise(results)
    assert summary["met"] == 2 and summary["short"] == 1 and summary["missed"] == 1
    # rest days do not count against adherence
    assert summary["adherence_pct"] == 50


def test_prompt_block_lists_targets():
    write_plan([{"date": "2026-09-14", "title": "Tempo", "sport": "Run",
                 "target_distance_mi": 6.0, "target_pace": "7:45/mi"}])
    block = plans.as_prompt_block()
    assert "2026-09-14" in block and "6.0 mi" in block and "7:45/mi" in block
