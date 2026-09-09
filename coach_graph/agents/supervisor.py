"""Routing step: decides whether to pull Strava data, and who answers."""

from __future__ import annotations

from datetime import date

from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field

from coach_graph import config
from coach_graph.state import CoachState, Specialist

ROUTER_PROMPT = """You route turns in a personal endurance-coaching assistant. Today is {today}.

Decide two things about the athlete's latest message:

1. needs_data — whether answering it requires *fresh* Strava data (activities, \
training load, fitness trends, heart-rate or power streams, segments, gear). \
Set false for follow-ups answerable from what is already in the conversation, and \
for chit-chat or general training-theory questions.
2. specialist — "planner" when they want future sessions prescribed (a plan, a \
workout suggestion, what to do tomorrow, how to structure a week). "coach" for \
everything else: analysis of past work, questions, feedback, encouragement.

When needs_data is true, write data_request as a short instruction to a data \
assistant, naming the time window and metrics needed. Otherwise leave it empty."""


class Route(BaseModel):
    needs_data: bool = Field(description="Fetch fresh Strava data for this turn?")
    specialist: Specialist = Field(description="Who should answer: coach or planner")
    data_request: str = Field("", description="What to fetch, if anything")


def build_supervisor():
    model = config.chat_model(config.ROUTER_MODEL).with_structured_output(Route)

    async def supervise(state: CoachState) -> dict:
        prompt = SystemMessage(ROUTER_PROMPT.format(today=date.today().isoformat()))
        route: Route = await model.ainvoke([prompt, *state["messages"][-6:]])
        return {
            "needs_data": route.needs_data,
            "specialist": route.specialist,
            "data_request": route.data_request,
        }

    return supervise
