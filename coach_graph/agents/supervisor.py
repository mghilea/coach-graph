"""Routing step: decides whether to pull Strava data, and who answers."""

from __future__ import annotations

from datetime import date

from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field

from coach_graph import config, plans
from coach_graph.state import CoachState, Specialist

ROUTER_PROMPT = """You route turns in a personal endurance-coaching assistant. Today is {today}.

Whoever answers already has, without any lookup: the athlete's profile (goals, \
constraints, preferences), stored observations and personal bests, and the full active \
training plan. {plan}

Decide two things about the athlete's latest message:

1. needs_data — whether answering it requires *fresh* Strava data: what the athlete \
actually did, their recent load, trends, or heart-rate and power streams. \
Set false for anything answerable from the plan or profile already in context — what \
is scheduled next, what this week looks like, how far a session is, what the goal is — \
and for follow-ups answerable from the conversation, chit-chat, and general training \
theory. Reading the plan is not fetching data.
2. specialist — "planner" only when new or changed sessions need prescribing: build me \
a plan, adjust my week, what should I do tomorrow. "coach" for everything else, \
including simply reading out what the plan already says ("what is my next workout"), \
analysis of past work, questions, feedback, encouragement.

When needs_data is true, write data_request as a short instruction to a data \
assistant, naming the time window and metrics needed. Otherwise leave it empty."""


class Route(BaseModel):
    needs_data: bool = Field(description="Fetch fresh Strava data for this turn?")
    specialist: Specialist = Field(description="Who should answer: coach or planner")
    data_request: str = Field("", description="What to fetch, if anything")


def build_supervisor():
    model = config.chat_model(config.ROUTER_MODEL).with_structured_output(Route)

    async def supervise(state: CoachState) -> dict:
        window = plans.active_window()
        prompt = SystemMessage(ROUTER_PROMPT.format(
            today=date.today().isoformat(),
            plan=f"The active plan covers {window}." if window
            else "No training plan is active yet.",
        ))
        route: Route = await model.ainvoke([prompt, *state["messages"][-6:]])
        return {
            "needs_data": route.needs_data,
            "specialist": route.specialist,
            "data_request": route.data_request,
        }

    return supervise
