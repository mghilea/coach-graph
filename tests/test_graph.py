"""Graph wiring tests with stubbed models — no network, no credentials."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from coach_graph import config, graph as graph_module
from coach_graph.agents import coach as coach_module
from coach_graph.agents import planner as planner_module
from coach_graph.agents import profiler as profiler_module
from coach_graph.agents import supervisor as supervisor_module
from coach_graph.agents.planner import Plan, Session, Week, render
from coach_graph.agents.profiler import ProfileUpdate
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
        # The supervisor and the profiler share the router model, so the stub has to
        # answer by requested schema rather than by model name.
        if isinstance(self.structured, dict):
            return StubStructured(self.structured.get(schema))
        return StubStructured(self.structured)

    async def ainvoke(self, messages, **kwargs):
        return AIMessage(self.reply)


@pytest.fixture
def stub_models(monkeypatch):
    """Routes each agent's model through a stub keyed by the model name it asks for."""
    plan = Plan(
        title="Week of 14 Sep",
        rationale="Hold volume, add one quality session.",
        sessions=[Session(date="2026-09-14", title="Easy run", sport="Run", target_duration_min=40,
                    details="40min Z2", purpose="Aerobic base")],
    )
    state = {
        "route": Route(needs_data=False, specialist="coach", data_request=""),
        "profile_update": ProfileUpdate(),
    }

    def fake_chat_model(model, **kwargs):
        if model == config.ROUTER_MODEL:
            return StubModel(structured={Route: state["route"],
                                         ProfileUpdate: state["profile_update"]})
        return StubModel(reply="Coach says: nice work.", structured=plan)

    for module in (supervisor_module, coach_module, planner_module, profiler_module):
        monkeypatch.setattr(module.config, "chat_model", fake_chat_model)
    return state


@pytest.fixture(autouse=True)
def isolated_files(monkeypatch, tmp_path):
    """Nothing in these tests should touch the developer's own profile or plans."""
    monkeypatch.setattr(config, "PROFILE_PATH", tmp_path / "profile.yaml")
    monkeypatch.setattr(config, "PLANS_DIR", tmp_path / "plans")
    monkeypatch.setattr(config, "CURRENT_PLAN_PATH", tmp_path / "plans" / "current.json")
    monkeypatch.setattr(config, "OBSERVATIONS_PATH", tmp_path / "observations.yaml")


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
        sessions=[Session(date="2026-09-15", title="Intervals", sport="Run", target_distance_mi=3.0,
                    target_pace="7:00/mi", details="6x800m", purpose="VO2max")],
        watch_for="Calf tightness",
    )

    markdown = render(plan)

    assert "## Base week" in markdown
    assert "6x800m" in markdown
    assert "Calf tightness" in markdown


# --- a failing turn must never end the conversation ---------------------------------
# Regression: the planner's structured output came back without its required `title`,
# pydantic raised, and the ValidationError propagated out of graph.astream and killed
# the CLI. The plan was lost, and so was the session.


class FailingStructured:
    """Fails a set number of times, then succeeds."""

    def __init__(self, failures, value=None):
        self.remaining = failures
        self.value = value
        self.calls = 0

    async def ainvoke(self, messages, **kwargs):
        self.calls += 1
        if self.remaining > 0:
            self.remaining -= 1
            raise ValueError("1 validation error for Plan\ntitle\n  Field required")
        return self.value


def test_a_plan_is_usable_with_only_its_sessions():
    # the fields that failed validation in the wild are cosmetic
    plan = Plan(sessions=[Session(date="2026-09-14", sport="Run")])
    assert plan.title and plan.sessions[0].title
    assert render(plan)


async def test_planner_retries_once_before_giving_up(monkeypatch, stub_models):
    good = Plan(title="Recovered", rationale="", sessions=[
        Session(date="2026-09-14", title="Easy", sport="Run", details="", purpose="")])
    flaky = FailingStructured(failures=1, value=good)
    monkeypatch.setattr(planner_module.config, "chat_model",
                        lambda *a, **kw: type("M", (), {"with_structured_output": lambda s, schema: flaky})())

    result = await planner_module.build_planner_node()({"messages": [HumanMessage("plan me a week")]})

    assert flaky.calls == 2
    assert "Recovered" in result["messages"][-1].text


async def test_planner_answers_instead_of_raising_when_both_attempts_fail(monkeypatch, stub_models):
    hopeless = FailingStructured(failures=99)
    monkeypatch.setattr(planner_module.config, "chat_model",
                        lambda *a, **kw: type("M", (), {"with_structured_output": lambda s, schema: hopeless})())

    result = await planner_module.build_planner_node()({"messages": [HumanMessage("plan me a week")]})

    assert hopeless.calls == 2
    assert "couldn't put that plan together" in result["messages"][-1].text
    assert result["strava_findings"] == ""


async def test_a_failing_coach_still_returns_a_message(monkeypatch, stub_models):
    class Boom:
        async def ainvoke(self, messages, **kwargs):
            raise RuntimeError("overloaded")

    monkeypatch.setattr(coach_module.config, "chat_model", lambda *a, **kw: Boom())

    result = await coach_module.build_coach_node()({"messages": [HumanMessage("how am I doing?")]})

    assert "Ask again" in result["messages"][-1].text


async def test_a_failing_planner_does_not_overwrite_the_active_plan(monkeypatch, stub_models):
    from coach_graph import config as cfg
    cfg.PLANS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.CURRENT_PLAN_PATH.write_text('{"title": "Still here", "sessions": []}')

    hopeless = FailingStructured(failures=99)
    monkeypatch.setattr(planner_module.config, "chat_model",
                        lambda *a, **kw: type("M", (), {"with_structured_output": lambda s, schema: hopeless})())

    await planner_module.build_planner_node()({"messages": [HumanMessage("plan me a week")]})

    assert "Still here" in cfg.CURRENT_PLAN_PATH.read_text()


# --- the CLI must survive a node that returns nothing -------------------------------
# Regression: LangGraph streams an empty-dict return as None. The profiler returns
# nothing on any turn it learns no new facts — the common case — so `update.get(...)`
# raised AttributeError after the answer had already been printed.


async def test_turn_survives_a_node_that_returns_nothing():
    from coach_graph import cli

    class FakeGraph:
        async def astream(self, request, options, stream_mode):
            yield {"supervisor": {"needs_data": False, "specialist": "coach"}}
            yield {"coach": {"messages": [AIMessage("Here is your answer.")]}}
            yield {"profiler": None}          # the shape that broke it

    await cli._turn(FakeGraph(), "how am I doing?", "t")


async def test_turn_reports_profile_changes_when_there_are_some():
    from coach_graph import cli

    class FakeGraph:
        async def astream(self, request, options, stream_mode):
            yield {"coach": {"messages": [AIMessage("Answer.")]}}
            yield {"profiler": {"profile_changes": {"goals": ["Sub-2:00 half."]}}}

    await cli._turn(FakeGraph(), "i want a sub 2 half", "t")


async def test_the_latest_plan_replaces_the_active_one(monkeypatch, tmp_path):
    from coach_graph import config as cfg, plans as plans_module
    monkeypatch.setattr(cfg, "PLANS_DIR", tmp_path / "plans")
    monkeypatch.setattr(cfg, "CURRENT_PLAN_PATH", tmp_path / "plans" / "current.json")

    planner_module.save(Plan(title="Block", rationale="First.", sessions=[
        Session(date="2026-09-14", title="Squash", sport="Squash"),
        Session(date="2026-09-20", title="Long run", sport="Run", target_distance_mi=6.0)]))

    saved, markdown = planner_module.save(Plan(title="Block", rationale="Second.", sessions=[
        Session(date="2026-09-20", title="Shortened long run", sport="Run", target_distance_mi=4.0)]))

    # nothing is carried over: the plan is exactly what was just prescribed
    assert [s.date for s in saved.sessions] == ["2026-09-20"]
    assert "Squash" not in markdown
    assert plans_module.load_current()["rationale"] == "Second."


def test_the_outlook_sits_below_the_sessions():
    plan = Plan(title="Block", rationale="What the block builds.",
                beyond="Long runs toward 10 mi through October.",
                sessions=[Session(date="2026-09-14", title="Squash", sport="Squash")])
    markdown = render(plan)
    assert markdown.index("What the block builds.") < markdown.index("Squash")
    assert markdown.index("Squash") < markdown.index("Beyond this block:")


def test_no_changelog_reaches_the_plan():
    # a plan is read before a session, not to review its own edit history
    plan = Plan(title="Block", rationale="Summary.",
                sessions=[Session(date="2026-09-14", title="Squash", sport="Squash")])
    assert "Changed in this revision" not in render(plan)
    assert "changes" not in Plan.model_fields


# --- the markdown is what actually gets read before a session -----------------------
# Regression: target_distance_mi, target_pace, target_duration_min and target_hr_max
# existed only in the JSON, so the plan the athlete reads did not say how far to run.


def _session(**kw):
    return Session(**{"date": "2026-09-13", "title": "Long run", "sport": "Run", **kw})


def test_every_target_reaches_the_markdown():
    plan = Plan(title="Block", sessions=[_session(
        target_distance_mi=5.0, target_duration_min=75, target_pace="9:09/mi", target_hr_max=165)])
    markdown = render(plan)
    assert "5.0 mi" in markdown
    assert "75 min" in markdown
    assert "9:09/mi" in markdown
    assert "HR <165" in markdown


def test_a_session_with_no_targets_gets_no_target_line():
    markdown = render(Plan(title="Block", sessions=[
        Session(date="2026-09-14", title="Squash", sport="Squash", details="Monday game.")]))
    assert "· " not in markdown
    assert "Monday game." in markdown


def test_the_weekday_is_shown_because_plans_are_read_by_week():
    markdown = render(Plan(title="Block", sessions=[_session(target_distance_mi=5.0)]))
    assert "2026-09-13 (Sun)" in markdown


def test_an_unparseable_date_still_renders():
    markdown = render(Plan(title="Block", sessions=[
        Session(date="sometime next week", title="Easy run", sport="Run")]))
    assert "sometime next week" in markdown


def test_empty_prose_fields_do_not_leave_dangling_bullets():
    markdown = render(Plan(title="Block", sessions=[_session(target_distance_mi=5.0)]))
    assert "- _Why:_\n" not in markdown
    assert "\n- \n" not in markdown


# --- week by week -------------------------------------------------------------------
# The focus of each week is authored; its mileage and counts are arithmetic over the
# sessions, so the summary cannot contradict the plan printed below it.


def _week_plan(**kw):
    return Plan(**{
        "title": "Block",
        "weeks": [Week(starting="2026-09-14", focus="Rebuild frequency.")],
        "sessions": [
            Session(date="2026-09-14", title="Squash", sport="Squash"),
            Session(date="2026-09-16", title="Easy", sport="Run", target_distance_mi=3.5),
            Session(date="2026-09-19", title="Rest", sport="Rest"),
            Session(date="2026-09-20", title="Long", sport="Run", target_distance_mi=6.0),
        ], **kw})


def test_weekly_numbers_are_counted_not_asserted():
    week = planner_module.week_summaries(_week_plan())[0]
    assert week["miles"] == 9.5
    assert week["longest"] == 6.0
    assert week["sessions"] == 3          # the rest day is not a session


def test_sessions_group_by_their_monday():
    plan = Plan(title="B", sessions=[
        Session(date="2026-09-20", title="Sun", sport="Run", target_distance_mi=6.0),
        Session(date="2026-09-21", title="Mon", sport="Run", target_distance_mi=3.0)])
    weeks = planner_module.week_summaries(plan)
    # consecutive days, but a week boundary runs between them
    assert [w["starting"] for w in weeks] == ["2026-09-14", "2026-09-21"]


def test_a_focus_dated_mid_week_still_finds_its_week():
    plan = _week_plan(weeks=[Week(starting="2026-09-17", focus="Given as a Thursday.")])
    assert planner_module.week_summaries(plan)[0]["focus"] == "Given as a Thursday."


def test_a_week_with_no_focus_still_reports_its_numbers():
    plan = _week_plan(weeks=[])
    week = planner_module.week_summaries(plan)[0]
    assert week["focus"] == "" and week["miles"] == 9.5


def test_the_section_sits_between_the_summary_and_the_sessions():
    plan = Plan(title="B", rationale="Summary.",
                weeks=[Week(starting="2026-09-14", focus="Week one."),
                       Week(starting="2026-09-21", focus="Week two.")],
                sessions=[Session(date="2026-09-16", title="Easy", sport="Run"),
                          Session(date="2026-09-23", title="Easy", sport="Run")])
    markdown = render(plan)
    assert markdown.index("Summary.") < markdown.index("### Week by week")
    assert markdown.index("### Week by week") < markdown.index("2026-09-16")


def test_a_single_week_block_skips_the_section():
    # a summary of one week, directly above that week, is just noise
    assert "### Week by week" not in render(_week_plan())


def test_undated_sessions_do_not_create_a_phantom_week():
    plan = Plan(title="B", sessions=[Session(date="whenever", title="Easy", sport="Run")])
    assert planner_module.week_summaries(plan) == []


def test_time_based_sessions_are_not_lost_from_the_weekly_total():
    # a mileage total alone made a five-session week look like a six-mile week
    plan = Plan(title="B", sessions=[
        Session(date="2026-09-16", title="Easy", sport="Run", target_duration_min=30),
        Session(date="2026-09-17", title="Easy", sport="Run", target_duration_min=32),
        Session(date="2026-09-20", title="Long", sport="Run", target_distance_mi=6.0),
        # a second week, so the section renders at all
        Session(date="2026-09-23", title="Easy", sport="Run", target_duration_min=30)])
    first = planner_module.week_summaries(plan)[0]
    assert first["miles"] == 6.0
    assert first["minutes"] == 62
    assert "6.0 mi" in render(plan) and "1h 2m other" in render(plan)


@pytest.mark.parametrize("minutes,expected", [(45, "45m"), (60, "1h"), (62, "1h 2m"), (150, "2h 30m")])
def test_weekly_time_reads_in_hours(minutes, expected):
    assert planner_module._hours(minutes) == expected
