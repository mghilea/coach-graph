"""Assembles the multi-agent graph.

    supervisor ─┬─► strava_data ─┬─► coach ──► END
                │                └─► planner ─► END
                └────────────────────┘
"""

from __future__ import annotations

from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from coach_graph.agents.coach import build_coach_node
from coach_graph.agents.planner import build_planner_node
from coach_graph.agents.strava import build_strava_node
from coach_graph.agents.supervisor import build_supervisor
from coach_graph.state import CoachState


def _after_supervisor(state: CoachState) -> str:
    return "strava_data" if state.get("needs_data") else state.get("specialist", "coach")


def _after_data(state: CoachState) -> str:
    return state.get("specialist", "coach")


def build_graph(
    tools: list[BaseTool], checkpointer: BaseCheckpointSaver | None = None
):
    graph = StateGraph(CoachState)
    graph.add_node("supervisor", build_supervisor())
    graph.add_node("strava_data", build_strava_node(tools))
    graph.add_node("coach", build_coach_node())
    graph.add_node("planner", build_planner_node())

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", _after_supervisor, ["strava_data", "coach", "planner"])
    graph.add_conditional_edges("strava_data", _after_data, ["coach", "planner"])
    graph.add_edge("coach", END)
    graph.add_edge("planner", END)

    return graph.compile(checkpointer=checkpointer)
