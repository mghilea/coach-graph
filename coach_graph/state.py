"""Shared state passed between the supervisor and the specialist agents."""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

Specialist = Literal["coach", "planner"]


class CoachState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    specialist: Specialist
    needs_data: bool
    data_request: str
    strava_findings: str
