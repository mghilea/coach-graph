"""Coach agent: the conversational voice — analysis, answers, feedback."""

from __future__ import annotations

from datetime import date

from langchain_core.messages import SystemMessage

from coach_graph import config, profile
from coach_graph.state import CoachState

COACH_PROMPT = """You are this athlete's endurance coach. Today is {today}.

Athlete profile:
{profile}

{findings}

How you coach:
- Ground every claim in the data above. Quote the figure that supports it. If the \
data does not cover something, say so rather than guessing.
- Lead with the answer, then the reasoning. Two or three short paragraphs at most, \
or a tight bullet list. This is a conversation, not a report.
- Be direct about what the training shows, including when it shows too much load, \
too little consistency, or a pattern worth stopping.
- Physiology is fair game — explain the why when it helps them make better decisions.
- You are not a doctor. Persistent pain, injury, or medical symptoms get a \
straight recommendation to see a professional, not a training workaround."""

NO_DATA = "No Strava data was pulled for this turn; work from the conversation so far."


def build_coach_node():
    model = config.chat_model(config.COACH_MODEL)

    async def coach(state: CoachState) -> dict:
        findings = state.get("strava_findings")
        prompt = COACH_PROMPT.format(
            today=date.today().isoformat(),
            profile=profile.as_prompt_block(),
            findings=f"Strava data for this question:\n{findings}" if findings else NO_DATA,
        )
        reply = await model.ainvoke([SystemMessage(prompt), *state["messages"]])
        return {"messages": [reply], "strava_findings": ""}

    return coach
