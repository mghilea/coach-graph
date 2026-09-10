# coach-graph

A locally-run multi-agent endurance coach. LangGraph orchestrates the agents, and
your training data arrives over MCP — from a local server this project runs itself
(`coach_graph/strava_server.py`), backed by Strava's REST API. Read-only throughout.

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
  and per-second streams never enter the main conversation. Its tools come from
  `strava_server.py` over an MCP stdio session: activities, laps, streams, zones, totals.
- **coach** (`agents/coach.py`) — analysis, answers, feedback, grounded in that
  briefing plus your profile.
- **planner** (`agents/planner.py`) — prescribes sessions as a structured `Plan`.
  Each session carries concrete targets (date, distance, pace, HR cap) alongside its
  prose and its fuelling, saved as markdown to read and JSON to check against.
- **profiler** (`agents/profiler.py`) — runs on the cheap model after the answer is
  already on screen, and merges anything durable you said into your profile.

State is checkpointed to SQLite, so conversations survive restarts.

## Did I actually do it?

The planner writes machine-readable targets at the moment it prescribes a session, so
adherence is a comparison rather than a re-reading of prose. `check_plan_adherence`
matches each prescribed session to what you actually did that day:

```
 ✓ 2026-09-05  Long run        planned 9.0mi 11:30/mi
     actual  9.01mi 1:39:47 11:04/mi
 ✗ 2026-09-07  Easy shakeout   planned 3.0mi
     ↳ no matching activity
 ~ 2026-09-08  Tempo           planned 5.0mi 9:00/mi
     actual  4.71mi 46:57 9:58/mi
     ↳ pace 9:58/mi vs 9:00/mi planned
```

Nothing is stored: a verdict saved from a snapshot goes stale the moment an activity is
uploaded, so it is computed on demand from the plan plus fresh Strava data.

## What the coach remembers

`data/observations.yaml` holds what outlives a conversation — personal bests, pulled
from Strava's own best efforts rather than typed in by hand, and durable observations
an agent noticed once (`record_observation`). Both the coach and the planner read it
every turn, and both are told to treat it as a snapshot rather than a substitute for
looking at fresh data.

Bests are kept current **by activity, not by a clock**. Each run is mined once, its id
recorded, and only runs never seen before are fetched — so a call costs one request
when nothing new has been run, and one extra per new run when something has. A timer
would refresh after a fortnight of rest and still miss a personal best set the morning
after it last ran.

## Setup

Requires Python 3.11+, [uv](https://docs.astral.sh/uv/), and a **Strava API
application** — create one at [strava.com/settings/api](https://www.strava.com/settings/api)
(any name and category), and set its **Authorization Callback Domain** to `localhost`.
Creation is instant and free. No Strava subscription is needed.

```bash
uv sync
cp .env.example .env      # add ANTHROPIC_API_KEY, STRAVA_CLIENT_ID, STRAVA_CLIENT_SECRET
uv run coach connect      # opens a browser for Strava OAuth, lists the MCP tools
```

`connect` runs the OAuth handshake, stores tokens at `~/.config/coach-graph/strava-tokens.json`
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

## Your profile, which you don't maintain

`data/profile.yaml` holds what Strava doesn't: goals and race dates, injuries, hours
available, coaching preferences. Both the coach and the planner read it every turn —
it's the difference between generic advice and coaching.

You don't fill it in. Say something durable and the **profiler** picks it up:

```
you › i want to train for a half marathon with a sub 2 time
coach › ...
profile updated (goals +1, preferences +2) · /profile to see it
```

Merges only ever add. A line you typed yourself is never rewritten or removed by an
agent, because your own words about your body and your goals outrank anything inferred
from one chat turn. The starter placeholders are the exception — they step aside for
the first real entry.

The `observed:` block is the one part the system owns outright: `get_training_habits`
derives which days you train, what time of day, and your weekly mileage straight from
activity timestamps, and rewrites it wholesale. Personal bests aren't here either —
they come from Strava into `data/observations.yaml`. Nothing derivable from your
training data is something you should be typing.

## Tests

```bash
uv run pytest
```

Model calls and the MCP transport are stubbed, so the suite needs no credentials
and no network. The live OAuth handshake against `mcp.strava.com` is the one path
tests can't cover — `coach connect` is the check for it.

## Notes

- Everything is **read-only**: the coach can't upload or edit activities. `strava_api.py`
  issues nothing but GETs.
- **Why a local MCP server rather than Strava's hosted one.** `https://mcp.strava.com/mcp`
  authorizes only Anthropic's own clients. It refuses RFC 7591 dynamic registration outright
  (`invalid_client_metadata` for every payload, including an empty one), and a self-registered
  application gets resolved and then rejected at the authorization step itself:
  `{"resource":"MCP Authorize","field":"client_id","code":"invalid"}` — as against
  `{"resource":"Application",...}` for an app that simply doesn't exist. Two different layers,
  so the app is real and the allowlist is what turns it away. Strava says they are "looking to
  support other AI clients in the future"; until then the tools are served locally. The agents
  are indifferent — they speak MCP either way, so switching back is a change to
  `strava_mcp.build_client()` alone.
- Strava's classic OAuth wants **comma-separated** scopes, unlike the space-separated OAuth
  standard. `strava_api.SCOPES` gets this right; a space-separated list fails at authorize.
- Streams are summarised in `strava_server.py`, never passed through raw — a long run is
  thousands of samples, and that payload lands in a sub-agent's context.
- **Units are imperial** — miles, min/mile, mph, feet — except where a sport has its own
  universal convention (swimming in min/100m, rowing in min/500m). Sports that cover no
  ground get no pace at all: a 0.27 mi squash match reduces to a three-digit pace, and
  the data agent is told to flag unusually high heart rate *for pace*, so a bogus figure
  would invent a flag that isn't there.
- Strava responses are cached on disk for `COACH_CACHE_TTL` seconds (default 300). Each
  MCP tool call is a fresh subprocess, so an in-memory cache would not survive.
- The profiler runs *after* the reply is printed, so its model call costs no visible
  wait, and a failure there is swallowed — you already have your answer by then.
- Plans carry fuelling for sessions long or hard enough to need it (carbohydrate per
  hour, gels, electrolytes), and defer to a professional on anything medical.
- Swapping models is a one-line change in `config.py::chat_model`; everything else
  is provider-agnostic LangChain.
