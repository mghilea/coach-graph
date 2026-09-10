"""Profiler: keeps the athlete profile current from what they tell the coach.

Runs after the reply is produced, on the cheap model, so it costs nothing the athlete
waits for. It extracts only durable facts that were actually stated — goals, injuries,
scheduling preferences — and never infers from training data, which the habits tool
derives directly and more reliably.
"""

from __future__ import annotations

from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field

from coach_graph import config, profile
from coach_graph.state import CoachState

PROFILER_PROMPT = """You maintain the athlete profile for a coaching assistant.

Profile on file:
{profile}

Read the conversation and extract only durable facts the athlete stated about \
themselves that are missing from the profile:

- goals — races, target times, dates, what they are training for
- constraints — injuries, illness, time available, equipment, travel
- preferences — which days they train, what time of day, sessions they like or avoid, \
how they want to be coached

Rules:
- Only what they actually said. Never infer from training data, and never guess at a \
motive or a reason they did not give.
- Durable facts only. "I'm tired today" is not a constraint; "my knee has hurt for \
three weeks" is.
- Do not restate anything already on file, even in different words.
- Write each as one short standalone sentence that will still make sense months later, \
with dates made absolute. Today is {today}.
- Return empty lists for anything the conversation did not add. Most turns add nothing, \
and empty lists are the right answer then — but a goal, injury or scheduling preference \
the athlete states outright always belongs on the profile."""


class ProfileUpdate(BaseModel):
    """Empty lists mean the turn added nothing.

    An earlier version carried a `nothing_new` flag defaulting to True, and the model
    set it inconsistently on turns that plainly did contain a goal — the absence of
    entries is the same signal with nothing to get wrong.
    """

    name: str | None = Field(None, description="The athlete's name, only if newly stated")
    goals: list[str] = Field(default_factory=list, description="Newly stated goals")
    constraints: list[str] = Field(default_factory=list, description="Newly stated constraints")
    preferences: list[str] = Field(default_factory=list, description="Newly stated preferences")


def build_profiler_node():
    model = config.chat_model(config.ROUTER_MODEL).with_structured_output(ProfileUpdate)

    async def profiler(state: CoachState) -> dict:
        from datetime import date

        prompt = PROFILER_PROMPT.format(
            profile=profile.as_prompt_block(), today=date.today().isoformat()
        )
        try:
            update: ProfileUpdate = await model.ainvoke(
                [SystemMessage(prompt), *state["messages"][-8:]]
            )
        except Exception:
            return {}  # a profile that failed to update must not fail the turn

        changed = profile.update(update.model_dump())
        return {"profile_changes": changed} if changed else {}

    return profiler
