"""Data agent: the only node holding Strava MCP tools.

It runs as an isolated sub-agent so that raw tool payloads (activity lists,
per-second streams) never enter the main conversation. Only its written summary
is handed back to the coach or planner.
"""

from __future__ import annotations

from datetime import date

from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent

from coach_graph import config
from coach_graph.state import CoachState

DATA_PROMPT = """You retrieve and summarise Strava data for a coaching assistant. \
Today is {today}.

Call the Strava tools you need, then write a compact briefing for the coach:

- Lead with the numbers that answer the request: volume, duration, pace or power, \
heart rate, elevation, training load, trend direction.
- Cover the requested window, and note the comparison with the preceding period \
when it is informative.
- Flag anything a coach should notice: a spike or drop in load, a missed week, \
unusually high heart rate for pace, a long gap between sessions.
- Use bullet points and real figures. Never invent a number that no tool returned; \
say plainly if the data is missing or the athlete has no activities in range.
- Do not give training advice. Report only."""


def build_strava_node(tools: list[BaseTool]):
    agent = create_react_agent(
        config.chat_model(config.COACH_MODEL),
        tools,
        prompt=DATA_PROMPT.format(today=date.today().isoformat()),
        name="strava_data_agent",
    )

    async def fetch(state: CoachState) -> dict:
        request = state.get("data_request") or "Summarise the last 4 weeks of training."
        try:
            result = await agent.ainvoke({"messages": [HumanMessage(request)]})
            findings = result["messages"][-1].text
        except Exception as exc:  # surfaced to the coach, not fatal to the turn
            findings = f"Strava lookup failed ({exc}). Answer without fresh data and say so."
        return {"strava_findings": findings}

    return fetch
