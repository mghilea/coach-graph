"""Planner agent: prescribes future sessions as a structured, saved plan."""

from __future__ import annotations

import logging
from datetime import date, timedelta

from langchain_core.messages import AIMessage, SystemMessage
from pydantic import BaseModel, Field

from coach_graph import config, observations, plans, profile
from coach_graph.state import CoachState

PLANNER_PROMPT = """You prescribe training for this athlete. Today is {today}.

Athlete profile:
{profile}

{observations}

{plan}

{findings}

Build the plan the athlete asked for — it may be one session or several weeks.

- Prescribe at most the next three weeks as dated sessions. When the goal is further \
out than that, put how the remaining blocks should progress in `beyond` rather than \
listing every session; the plan is revisited as the race approaches.
- Progress load from what they have actually been doing. A jump in volume or \
intensity beyond roughly 10% per week needs a reason you state out loud.
- Every session needs a purpose the athlete can understand, and enough detail to \
execute it: duration or distance, intensity (pace, power, heart-rate zone, or RPE), \
and structure for intervals.
- Respect the constraints in the profile — time available, injuries, equipment.
- What you return REPLACES the active plan entirely. When one is already active, \
continue it rather than starting over, and restate the whole block even if you are \
only changing one day — every session you leave out is dropped. If the athlete asks \
about a single day, answer with that day inside the block it belongs to.
- Include recovery. Rest days are prescriptions too.
- Fuel the sessions that need it. Under an hour, water is usually enough; from around \
75 minutes carbohydrate starts to matter (roughly 30-60 g an hour, about one gel every \
25-40 minutes), and long or hot sessions need sodium as well as fluid. Say what to take \
and when, and use long runs to rehearse race-day fuelling rather than trying it new on \
the day. Leave `fueling` null when a session genuinely does not need it, and send \
anything touching a medical or dietary condition to a professional.
- Give every session a real ISO date and fill the target_* fields with concrete \
numbers — those are checked against what the athlete actually did. Distances are in \
MILES and paces in minutes per MILE ("9:08/mi"). Ranges and caveats go in `details`.
- Prescribe every run, ride, walk and swim by DISTANCE, not by time: set \
`target_distance_mi` and leave `target_duration_min` null. The athlete wants to know \
how far to go, and weekly mileage only adds up if every session carries a distance. \
Use `target_duration_min` only where distance is meaningless — strength, mobility, \
court sports. If an easy run is really about time on feet, still convert it to a \
distance at their current easy pace and say so in `details`.
- Always give the block a `title` naming what it is and when, e.g. "Princeton build — \
weeks of 14 and 21 Sep".
- Always write `rationale`, in **60 words or fewer** — short sentences, not three long \
ones. It says what this block builds and the one judgement shaping it: the thing a \
reader could not work out from the sessions themselves. Do not walk through the week \
day by day, list the constraints, or recap fuelling — all of that is visible below it, \
and this is read before every session. Never describe the change you were just asked \
for: there is nowhere in the plan for a changelog, and a reader months from now needs \
the block described, not its edit history.
- Give one `weeks` entry per week the sessions cover, keyed by that week's Monday, \
saying in a few words what the week is for. Do not put mileage or session counts in \
the focus — those are counted from the sessions themselves.
- `beyond` is 30 words or fewer.
- If the training data is too thin to prescribe safely, say so in `rationale` and \
keep the plan conservative."""

logger = logging.getLogger(__name__)

NO_DATA = "No fresh Strava data for this turn; rely on the conversation and profile."


class Session(BaseModel):
    """One prescribed session.

    The target_* fields are what adherence is later measured against, so they must be
    concrete numbers rather than the ranges and hedges that belong in `details`. Leave
    one null when the session genuinely does not constrain it — an easy run capped by
    heart rate has no target pace.
    """

    date: str = Field(description="ISO date this session falls on, e.g. '2026-09-14'")
    title: str = Field("Session", description="Short name, e.g. 'Threshold intervals'")
    sport: str = Field(description="Run, Ride, Swim, WeightTraining, Rest, etc.")
    target_distance_mi: float | None = Field(
        None, description="Planned distance in MILES. Set this for every run, ride, "
        "walk and swim — weekly mileage only adds up if each one carries a distance.",
    )
    target_duration_min: int | None = Field(
        None, description="Planned duration in minutes. Only for sessions where "
        "distance is meaningless: strength, mobility, court sports.",
    )
    target_pace: str | None = Field(
        None, description="Target pace in minutes per MILE, e.g. '9:08/mi'"
    )
    target_hr_max: int | None = Field(
        None, description="Upper heart-rate bound in bpm, if the session caps effort"
    )
    details: str = Field("", description="Distance/duration, intensity, full structure")
    purpose: str = Field("", description="One line: why this session exists")
    fueling: str | None = Field(
        None,
        description="What to take before, during and after — carbohydrate per hour, "
        "gels, electrolytes, fluid — for sessions long or hard enough to need it. "
        "Null for sessions that do not.",
    )


class Week(BaseModel):
    """What one week of the block is for. Its numbers are computed, not stated here."""

    starting: str = Field(description="ISO date of the Monday this week begins, e.g. '2026-09-14'")
    focus: str = Field(description="One short line on what this week is for. 12 words or fewer.")


class Plan(BaseModel):
    """Only `sessions` is genuinely required.

    The prose fields carry defaults because a model that omits a heading should still
    produce a usable plan — a missing title is a cosmetic gap, and raising on it loses
    the whole block and, before this was handled, the conversation with it.
    """

    sessions: list[Session]
    title: str = Field("Training block", description="Name of the block, e.g. 'Week of 14 Sep — base'")
    weeks: list[Week] = Field(
        default_factory=list,
        description="One entry per week the sessions cover, saying what that week is for.",
    )
    rationale: str = Field("", description="A few sentences on the shape of the block")
    watch_for: str = Field("", description="Signals that should change the plan")
    beyond: str = Field(
        "", description="One or two sentences on how the blocks after this one should "
        "progress toward the goal. Empty when the plan already reaches it.",
    )


def targets(session: Session) -> str:
    """The numbers to hit, on one line. Empty when the session prescribes none.

    These live in the structured fields so adherence can check them, but the markdown
    is what gets read before a session — leaving them out of it meant the distance and
    the heart-rate cap were only ever visible in the JSON.
    """
    parts = []
    if session.target_distance_mi:
        parts.append(f"{session.target_distance_mi} mi")
    if session.target_duration_min:
        parts.append(f"{session.target_duration_min} min")
    if session.target_pace:
        parts.append(str(session.target_pace))
    if session.target_hr_max:
        parts.append(f"HR <{session.target_hr_max}")
    return " · ".join(parts)


def _hours(minutes: int) -> str:
    """90 -> '1h 30m'. Time-based sessions read in hours once a week adds up."""
    hours, mins = divmod(int(minutes), 60)
    if hours and mins:
        return f"{hours}h {mins}m"
    return f"{hours}h" if hours else f"{mins}m"


def _monday(iso_date: str) -> str | None:
    """The Monday of the week this date falls in, so weeks group the way a plan reads."""
    try:
        day = date.fromisoformat(iso_date)
    except (ValueError, TypeError):
        return None
    return (day - timedelta(days=day.weekday())).isoformat()


def week_summaries(plan: Plan) -> list[dict]:
    """One row per week the sessions cover: the focus as written, the numbers as counted.

    Mileage and session counts are arithmetic over the sessions rather than anything the
    model asserts, so the summary cannot disagree with the plan below it.
    """
    focus_by_week = {}
    for week in plan.weeks:
        key = _monday(week.starting)
        if key:
            focus_by_week[key] = week.focus

    grouped: dict[str, list[Session]] = {}
    for session in plan.sessions:
        key = _monday(session.date)
        if key:
            grouped.setdefault(key, []).append(session)

    rows = []
    for key in sorted(grouped):
        sessions = grouped[key]
        distances = [s.target_distance_mi for s in sessions if s.target_distance_mi]
        # Easy runs are often prescribed in minutes, so a mileage total alone reads as
        # though a five-session week were six miles long. Both are reported.
        minutes = [s.target_duration_min for s in sessions if s.target_duration_min]
        rows.append({
            "starting": key,
            "focus": focus_by_week.get(key, ""),
            "miles": round(sum(distances), 1) if distances else 0.0,
            "minutes": sum(minutes) if minutes else 0,
            "sessions": sum(1 for s in sessions if (s.sport or "").lower() != "rest"),
            "longest": max(distances) if distances else 0.0,
        })
    return rows


def _weekday(iso_date: str) -> str:
    """'2026-09-13' -> 'Sun'. A training plan is read a week at a time."""
    try:
        return date.fromisoformat(iso_date).strftime("%a")
    except (ValueError, TypeError):
        return ""


def render(plan: Plan) -> str:
    lines = [f"## {plan.title}", "", plan.rationale, ""]

    weeks = week_summaries(plan)
    if len(weeks) > 1:
        lines.append("### Week by week")
        lines.append("")
        for week in weeks:
            label = date.fromisoformat(week["starting"]).strftime("%a %-d %b")
            counted = [f"{week['miles']} mi" if week["miles"] else "",
                       f"{week['sessions']} sessions",
                       f"long run {week['longest']} mi" if week["longest"] else "",
                       # With runs set by distance, remaining time is strength and court work.
                       f"{_hours(week['minutes'])} other" if week["minutes"] else ""]
            lines.append(f"**Week of {label}** — " + ", ".join(p for p in counted if p))
            if week["focus"]:
                lines.append(f"{week['focus']}")
            lines.append("")

    for session in plan.sessions:
        day = _weekday(session.date)
        heading = f"{session.date}{f' ({day})' if day else ''} — {session.title}"
        lines.append(f"**{heading}** ({session.sport})")

        hit = targets(session)
        if hit:
            lines.append(f"- **{hit}**")
        if session.details:
            lines.append(f"- {session.details}")
        if session.purpose:
            lines.append(f"- _Why:_ {session.purpose}")
        if session.fueling:
            lines.append(f"- _Fuel:_ {session.fueling}")
        lines.append("")
    if plan.beyond:
        lines += [f"**Beyond this block:** {plan.beyond}", ""]
    if plan.watch_for:
        lines += [f"**Watch for:** {plan.watch_for}"]
    return "\n".join(lines).strip()


def save(plan: Plan) -> tuple[Plan, str]:
    """Write the plan out. The latest one replaces whatever was active before.

    No folding into the previous plan: a prescribed block is the plan, whole. Which is
    why the planner is told to restate the block even when it is changing a single day —
    anything it leaves out is gone. Same-day writes overwrite, so a day's archive is
    where that day's editing finished.
    """
    markdown = render(plan)
    payload = plan.model_dump_json(indent=2) + "\n"

    config.PLANS_DIR.mkdir(parents=True, exist_ok=True)
    (config.PLANS_DIR / f"{date.today().isoformat()}-plan.md").write_text(markdown + "\n")
    (config.PLANS_DIR / f"{date.today().isoformat()}-plan.json").write_text(payload)
    config.CURRENT_PLAN_PATH.write_text(payload)
    return plan, markdown


def build_planner_node():
    model = config.chat_model(config.COACH_MODEL).with_structured_output(Plan)

    async def planner(state: CoachState) -> dict:
        findings = state.get("strava_findings")
        prompt = PLANNER_PROMPT.format(
            today=date.today().isoformat(),
            profile=profile.as_prompt_block(),
            observations=observations.as_prompt_block(),
            plan=plans.as_prompt_block(),
            findings=f"Recent training:\n{findings}" if findings else NO_DATA,
        )
        messages = [SystemMessage(prompt), *state["messages"]]

        # Structured output occasionally comes back missing a field. That is worth one
        # more attempt, and never worth ending the conversation over.
        plan: Plan | None = None
        for attempt in range(2):
            try:
                plan = await model.ainvoke(messages)
                break
            except Exception:
                logger.warning("planner attempt %d failed to produce a valid plan", attempt + 1,
                               exc_info=True)

        if plan is None:
            return {
                "messages": [AIMessage(
                    "I couldn't put that plan together just now. Ask again and I'll retry — "
                    "narrowing it to the next week or two usually helps."
                )],
                "strava_findings": "",
            }

        _, markdown = save(plan)
        return {"messages": [AIMessage(markdown)], "strava_findings": ""}

    return planner
