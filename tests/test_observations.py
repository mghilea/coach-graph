"""Durable observations: bests pulled from Strava, plus what the agents noticed."""

from __future__ import annotations

import pytest

from coach_graph import config, observations


@pytest.fixture(autouse=True)
def isolated_observations(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "OBSERVATIONS_PATH", tmp_path / "observations.yaml")


def test_empty_before_anything_is_recorded():
    assert observations.load() == {}
    assert observations.personal_bests() == {}
    assert "No observations" in observations.as_prompt_block()


def test_corrupt_file_reads_as_empty():
    config.OBSERVATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.OBSERVATIONS_PATH.write_text("[unclosed")
    assert observations.load() == {}


def test_personal_bests_round_trip():
    observations.set_personal_bests({"5k": {"time": "24:31", "pace": "7:53/mi",
                                            "date": "2026-08-29"}})
    assert observations.personal_bests()["5k"]["time"] == "24:31"
    assert "5k: 24:31" in observations.as_prompt_block()


# Bests are invalidated by activity, not by a clock. A clock refreshes after a fortnight
# of rest and misses a best set the morning after a refresh; a new run is the only thing
# that can actually change them.


def test_everything_is_unscanned_before_anything_is_fetched():
    assert observations.scanned() == []
    assert observations.unscanned([1, 2, 3]) == [1, 2, 3]


def test_scanned_activities_are_not_rescanned():
    observations.set_personal_bests({"5K": {"time": "24:31"}}, scanned_ids=[1, 2])
    assert observations.unscanned([1, 2, 3]) == [3]


def test_no_new_runs_means_nothing_to_do():
    observations.set_personal_bests({"5K": {"time": "24:31"}}, scanned_ids=[1, 2])
    assert observations.unscanned([1, 2]) == []


def test_scanned_ids_accumulate_across_calls():
    observations.set_personal_bests({}, scanned_ids=[1, 2])
    observations.set_personal_bests({}, scanned_ids=[2, 3])
    assert observations.scanned() == [1, 2, 3]  # deduplicated, order kept


def test_scanned_ids_stay_bounded():
    observations.set_personal_bests({}, scanned_ids=list(range(observations.SCANNED_LIMIT + 50)))
    kept = observations.scanned()
    assert len(kept) == observations.SCANNED_LIMIT
    assert kept[-1] == observations.SCANNED_LIMIT + 49  # the newest survive


def test_bests_stored_before_ids_were_tracked_count_as_scanned():
    # written by an earlier version: bests, but no scanned_ids
    observations.save({"personal_bests": {"refreshed": "2026-09-09T00:00:00+00:00",
                                          "bests": {"5K": {"time": "24:31", "activity_id": 77}}}})
    assert observations.scanned() == [77]
    assert observations.unscanned([77, 88]) == [88]


def test_omitting_scanned_ids_preserves_the_ones_on_file():
    observations.set_personal_bests({}, scanned_ids=[1, 2])
    observations.set_personal_bests({"5K": {"time": "24:00"}})
    assert observations.scanned() == [1, 2]


def test_replace_resets_the_history_for_a_rebuild():
    observations.set_personal_bests({}, scanned_ids=[1, 2])
    observations.set_personal_bests({}, scanned_ids=[3], replace=True)
    assert observations.scanned() == [3]


def test_notes_accumulate():
    observations.add_note("Heart rate drifts past 90 minutes.")
    observations.add_note("Runs faster on Sunday mornings.")
    assert len(observations.load()["notes"]) == 2
    assert "drifts past 90 minutes" in observations.as_prompt_block()


def test_the_same_note_is_not_recorded_twice():
    first = observations.add_note("Heart rate drifts past 90 minutes.")
    second = observations.add_note("  heart rate DRIFTS past 90 minutes.  ")
    assert first["recorded"] is True
    assert second["recorded"] is False
    assert len(observations.load()["notes"]) == 1


def test_recording_a_note_leaves_bests_intact():
    observations.set_personal_bests({"5k": {"time": "24:31"}})
    observations.add_note("Prefers morning sessions.")
    assert observations.personal_bests()["5k"]["time"] == "24:31"
    assert len(observations.load()["notes"]) == 1


def test_prompt_block_shows_only_the_recent_notes():
    for i in range(20):
        observations.add_note(f"Observation number {i}.")
    block = observations.as_prompt_block()
    assert "Observation number 19." in block
    assert "Observation number 0." not in block
