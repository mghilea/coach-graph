"""Graph wiring tests with stubbed models — no network, no credentials."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from coach_graph import config, graph as graph_module
from coach_graph.agents import coach as coach_module
from coach_graph.agents import planner as planner_module
from coach_graph.agents import supervisor as supervisor_module
from coach_graph.agents.planner import Plan, Session, render
from coach_graph.agents.supervisor import Route

class StubStructured:
    def __init__(self, value):
        self.value = value

    async def ainvoke(self, messages, **kwargs):
        return self.value


class StubModel:
    def __init__(self, reply="stub reply", structured=None):
        self.reply = reply
        self.structured = structured

    def with_structured_output(self, schema):
        return StubStructured(self.structured)

    async def ainvoke(self, messages, **kwargs):
        return AIMessage(self.reply)


@pytest.fixture
def stub_models(monkeypatch):
    """Routes each agent's model through a stub keyed by the model name it asks for."""
    plan = Plan(
        title="Week of 14 Sep",
        rationale="Hold volume, add one quality session.",
        sessions=[Session(day="Mon", title="Easy run", sport="Run", details="40min Z2", purpose="Aerobic base")],
    )
    state = {"route": Route(needs_data=False, specialist="coach", data_request="")}

    def fake_chat_model(model, **kwargs):
        if model == config.ROUTER_MODEL:
            return StubModel(structured=state["route"])
        return StubModel(reply="Coach says: nice work.", structured=plan)

    for module in (supervisor_module, coach_module, planner_module):
        monkeypatch.setattr(module.config, "chat_model", fake_chat_model)
    return state


@pytest.fixture
def spy_strava(monkeypatch):
    calls = []

    def build(tools):
        async def fetch(graph_state):
            calls.append(graph_state.get("data_request"))
            return {"strava_findings": "Ran 42km over 5 sessions last week."}

        return fetch

    monkeypatch.setattr(graph_module, "build_strava_node", build)
    return calls


async def run(user: str) -> dict:
    compiled = graph_module.build_graph(tools=[])
    return await compiled.ainvoke({"messages": [HumanMessage(user)]})


async def test_question_skips_strava_and_reaches_coach(stub_models, spy_strava):
    result = await run("What does a threshold session do?")

    assert spy_strava == []
    assert result["messages"][-1].text == "Coach says: nice work."


async def test_data_question_routes_through_strava_to_coach(stub_models, spy_strava):
    stub_models["route"] = Route(needs_data=True, specialist="coach", data_request="Last 4 weeks of runs")

    result = await run("How was my week?")

    assert spy_strava == ["Last 4 weeks of runs"]
    assert result["messages"][-1].text == "Coach says: nice work."


async def test_plan_request_routes_through_strava_to_planner(stub_models, spy_strava, tmp_path, monkeypatch):
    monkeypatch.setattr(planner_module.config, "PLANS_DIR", tmp_path / "plans")
    stub_models["route"] = Route(needs_data=True, specialist="planner", data_request="Recent load")

    result = await run("Plan my week.")

    assert spy_strava == ["Recent load"]
    assert "Week of 14 Sep" in result["messages"][-1].text
    assert list((tmp_path / "plans").glob("*-plan.md")), "plan should be saved to disk"


async def test_findings_reset_after_a_turn(stub_models, spy_strava):
    stub_models["route"] = Route(needs_data=True, specialist="coach", data_request="Last week")

    result = await run("How was my week?")

    assert result["strava_findings"] == ""


def test_render_includes_sessions_and_watch_for():
    plan = Plan(
        title="Base week",
        rationale="Build aerobic volume.",
        sessions=[Session(day="Tue", title="Intervals", sport="Run", details="6x800m", purpose="VO2max")],
        watch_for="Calf tightness",
    )

    markdown = render(plan)

    assert "## Base week" in markdown
    assert "6x800m" in markdown
    assert "Calf tightness" in markdown
