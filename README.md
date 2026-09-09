# coach-graph

A locally-run multi-agent endurance coach. LangGraph orchestrates the agents, and
your training data comes from Strava's official remote MCP connector
(`https://mcp.strava.com/mcp`) — read-only, OAuth, no API keys to manage.

## The graph

```
                 ┌──────────────┐
   your turn ──► │  supervisor  │  picks: fresh data? coach or planner?
                 └──────┬───────┘
             ┌──────────┴──────────┐
             ▼                     │
      ┌─────────────┐              │
      │ strava_data │  MCP tools   │
      └──────┬──────┘              │
             └──────────┬──────────┘
                 ┌──────┴──────┐
                 ▼             ▼
             ┌───────┐   ┌─────────┐
             │ coach │   │ planner │
             └───┬───┘   └────┬────┘
                 └─────┬──────┘
                       ▼  reply + sqlite checkpoint
```

- **supervisor** (`agents/supervisor.py`) — a cheap model classifying each turn:
  does this need fresh Strava data, and should the coach or the planner answer?
- **strava_data** (`agents/strava.py`) — the only node holding MCP tools. It runs as
  an isolated ReAct sub-agent and returns a written briefing, so raw activity JSON
  and per-second streams never enter the main conversation.
- **coach** (`agents/coach.py`) — analysis, answers, feedback, grounded in that
  briefing plus your profile.
- **planner** (`agents/planner.py`) — prescribes sessions as a structured `Plan`,
  rendered to markdown and saved under `data/plans/`.

State is checkpointed to SQLite, so conversations survive restarts.

## Setup

Requires Python 3.11+, [uv](https://docs.astral.sh/uv/), and **an active Strava
subscription** — the MCP connector is a subscriber feature.

```bash
uv sync
cp .env.example .env      # add your ANTHROPIC_API_KEY
uv run coach connect      # opens a browser for Strava OAuth, lists the tools
```

`connect` registers the client, stores tokens at `~/.config/coach-graph/strava-oauth.json`
(mode 600), and refreshes them automatically thereafter. To re-authorize, delete
that file and run it again.

## Use

```bash
uv run coach                              # chat
uv run coach chat --thread easter-block   # a separate conversation
```

In-chat: `/new` starts a fresh conversation, `/profile` prints your profile, `/quit` exits.

```
you › how did my last two weeks compare?
you › what should I run tomorrow?
you › why does my heart rate drift on long runs?
```

## Your profile

`data/profile.yaml` (created on first run, gitignored) holds what Strava doesn't:
goals and race dates, injuries, hours available, coaching preferences. Both the
coach and the planner read it every turn — it's the difference between generic
advice and coaching. Free text is fine.

## Tests

```bash
uv run pytest
```

Model calls and the MCP transport are stubbed, so the suite needs no credentials
and no network. The live OAuth handshake against `mcp.strava.com` is the one path
tests can't cover — `coach connect` is the check for it.

## Notes

- The Strava connector is **read-only**: the coach can't upload or edit activities.
- Swapping models is a one-line change in `config.py::chat_model`; everything else
  is provider-agnostic LangChain.
