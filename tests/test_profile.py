"""The profile is agent-maintained, so the merge rules are what protect the athlete's
own words from being rewritten by an inference drawn from one chat turn."""

from __future__ import annotations

import pytest
import yaml

from coach_graph import config, profile


@pytest.fixture(autouse=True)
def isolated_profile(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PROFILE_PATH", tmp_path / "profile.yaml")


def test_template_is_written_once():
    assert profile.ensure_exists() is True
    assert profile.ensure_exists() is False


def test_corrupt_profile_reads_as_empty():
    config.PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.PROFILE_PATH.write_text("goals: [unclosed")
    assert profile.load() == {}


def test_first_real_goal_replaces_the_placeholder():
    profile.ensure_exists()
    changed = profile.update({"goals": ["Half marathon under 2:00 on 2027-04-18."]})
    assert changed["goals"] == ["Half marathon under 2:00 on 2027-04-18."]
    # the starter line was a prompt to the reader, not a fact
    assert profile.load()["goals"] == ["Half marathon under 2:00 on 2027-04-18."]


def test_updates_add_without_removing():
    profile.update({"goals": ["Half marathon under 2:00."]})
    profile.update({"goals": ["Run a 10k in October."]})
    assert profile.load()["goals"] == ["Half marathon under 2:00.", "Run a 10k in October."]


def test_hand_written_entries_survive_an_agent_update():
    profile.save({"goals": ["My own words, typed by me."]})
    profile.update({"goals": ["Something the agent inferred."]})
    assert "My own words, typed by me." in profile.load()["goals"]


def test_the_same_fact_is_not_added_twice():
    profile.update({"constraints": ["Left knee sore since 2026-08-20."]})
    changed = profile.update({"constraints": ["  left knee SORE since 2026-08-20.  "]})
    assert changed == {}
    assert len(profile.load()["constraints"]) == 1


def test_a_name_already_on_file_is_not_replaced():
    profile.save({"name": "Raluca"})
    assert profile.update({"name": "Someone Else"}) == {}
    assert profile.load()["name"] == "Raluca"


def test_a_missing_name_is_filled_in():
    profile.ensure_exists()
    assert profile.update({"name": "Raluca"})["name"] == "Raluca"


def test_empty_update_writes_nothing():
    profile.ensure_exists()
    before = config.PROFILE_PATH.read_text()
    assert profile.update({"goals": [], "constraints": [None, "  "]}) == {}
    assert config.PROFILE_PATH.read_text() == before


def test_observed_block_is_replaced_wholesale_and_leaves_stated_fields_alone():
    profile.update({"goals": ["Half marathon under 2:00."]})
    profile.set_observed({"usual_days": ["Tue", "Thu"], "run_miles_per_week": 12.4})
    profile.set_observed({"usual_days": ["Mon", "Wed", "Sat"], "run_miles_per_week": 18.1})

    data = profile.load()
    assert data["observed"]["usual_days"] == ["Mon", "Wed", "Sat"]
    assert data["observed"]["run_miles_per_week"] == 18.1
    assert data["goals"] == ["Half marathon under 2:00."]
    assert "refreshed" in data["observed"]


def test_the_header_survives_an_agent_write():
    # it is the only place the file says the athlete may edit it freely
    profile.ensure_exists()
    profile.update({"goals": ["Half marathon under 2:00."]})
    assert config.PROFILE_PATH.read_text().startswith("# Maintained by your coach")
    assert "Edit anything here freely" in config.PROFILE_PATH.read_text()


def test_profile_stays_valid_yaml_after_agent_writes():
    profile.update({"goals": ["Sub-2:00 half: pace is 9:09/mi — mind the colon."]})
    reloaded = yaml.safe_load(config.PROFILE_PATH.read_text())
    assert reloaded["goals"] == ["Sub-2:00 half: pace is 9:09/mi — mind the colon."]
