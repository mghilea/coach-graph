"""Planner agent: prescribes future sessions as a structured, saved plan."""

from __future__ import annotations

from datetime import date

from langchain_core.messages import AIMessage, SystemMessage
from pydantic import BaseModel, Field

from coach_graph import config, profile
from coach_graph.state import CoachState

PLANNER_PROMPT = """You prescribe training for this athlete. Today is {today}.

Athlete profile:
{profile}

{findings}

Build the plan the athlete asked for — it may be one session or several weeks.

- Progress load from what they have actually been doing. A jump in volume or \
intensity beyond roughly 10% per week needs a reason you state out loud.
- Every session needs a purpose the athlete can understand, and enough detail to \
execute it: duration or distance, intensity (pace, power, heart-rate zone, or RPE), \
and structure for intervals.
- Respect the constraints in the profile — time available, injuries, equipment.
- Include recovery. Rest days are prescriptions too.
- In `rationale`, explain the shape of the block in a few sentences: what it is \
building, and what you are deliberately holding back.
- If the training data is too thin to prescribe safely, say so in `rationale` and \
keep the plan conservative."""

NO_DATA = "No fresh Strava data for this turn; rely on the conversation and profile."


class Session(BaseModel):
    day: str = Field(description="Date or day label, e.g. 'Mon 14 Sep' or 'Day 1'")
    title: str = Field(description="Short name, e.g. 'Threshold intervals'")
    sport: str = Field(description="Run, ride, swim, strength, rest, etc.")
    details: str = Field(description="Distance/duration, intensity, full structure")
    purpose: str = Field(description="One line: why this session exists")


class Plan(BaseModel):
    title: str = Field(description="Name of the block, e.g. 'Week of 14 Sep — base'")
    rationale: str = Field(description="A few sentences on the shape of the block")
    sessions: list[Session]
    watch_for: str = Field("", description="Signals that should change the plan")


def render(plan: Plan) -> str:
    lines = [f"## {plan.title}", "", plan.rationale, ""]
    for session in plan.sessions:
        lines += [
            f"**{session.day} — {session.title}** ({session.sport})",
            f"- {session.details}",
            f"- _Why:_ {session.purpose}",
            "",
        ]
    if plan.watch_for:
        lines += [f"**Watch for:** {plan.watch_for}"]
    return "\n".join(lines).strip()


def save(plan: Plan, markdown: str) -> None:
    config.PLANS_DIR.mkdir(parents=True, exist_ok=True)
    (config.PLANS_DIR / f"{date.today().isoformat()}-plan.md").write_text(markdown + "\n")


def build_planner_node():
    model = config.chat_model(config.COACH_MODEL).with_structured_output(Plan)

    async def planner(state: CoachState) -> dict:
        findings = state.get("strava_findings")
        prompt = PLANNER_PROMPT.format(
            today=date.today().isoformat(),
            profile=profile.as_prompt_block(),
            findings=f"Recent training:\n{findings}" if findings else NO_DATA,
        )
        plan: Plan = await model.ainvoke([SystemMessage(prompt), *state["messages"]])
        markdown = render(plan)
        save(plan, markdown)
        return {"messages": [AIMessage(markdown)], "strava_findings": ""}

    return planner
