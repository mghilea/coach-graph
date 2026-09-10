"""The profiler writes to the profile, so what it declines to write matters most."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from coach_graph import config, profile
from coach_graph.agents import profiler as profiler_module
from coach_graph.agents.profiler import ProfileUpdate, build_profiler_node


@pytest.fixture(autouse=True)
def isolated_profile(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PROFILE_PATH", tmp_path / "profile.yaml")


def stub_model(monkeypatch, value):
    class Structured:
        async def ainvoke(self, messages, **kwargs):
            if isinstance(value, Exception):
                raise value
            return value

    class Model:
        def with_structured_output(self, schema):
            return Structured()

    monkeypatch.setattr(profiler_module.config, "chat_model", lambda *a, **kw: Model())


def turn():
    return {"messages": [HumanMessage("I want a sub-2 half"), AIMessage("Noted.")]}


async def test_an_empty_extraction_leaves_the_profile_untouched(monkeypatch):
    profile.ensure_exists()
    before = config.PROFILE_PATH.read_text()
    stub_model(monkeypatch, ProfileUpdate())

    assert await build_profiler_node()(turn()) == {}
    assert config.PROFILE_PATH.read_text() == before


async def test_stated_facts_are_merged_in(monkeypatch):
    profile.ensure_exists()
    stub_model(monkeypatch, ProfileUpdate(
        goals=["Half marathon under 2:00."],
        preferences=["Prefers morning sessions."],
    ))

    result = await build_profiler_node()(turn())

    assert result["profile_changes"]["goals"] == ["Half marathon under 2:00."]
    saved = profile.load()
    assert saved["goals"] == ["Half marathon under 2:00."]
    assert saved["preferences"] == ["Prefers morning sessions."]


async def test_a_fact_already_on_file_is_not_reported_as_a_change(monkeypatch):
    profile.save({"goals": ["Half marathon under 2:00."]})
    stub_model(monkeypatch, ProfileUpdate(goals=["Half marathon under 2:00."]))

    assert await build_profiler_node()(turn()) == {}


async def test_a_failed_extraction_never_fails_the_turn(monkeypatch):
    profile.ensure_exists()
    stub_model(monkeypatch, RuntimeError("model unavailable"))

    # the athlete already has their answer by this point; losing the update is acceptable
    assert await build_profiler_node()(turn()) == {}
