"""Terminal chat with the coach."""

from __future__ import annotations

import argparse
import asyncio
import uuid

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from rich.console import Console
from rich.markdown import Markdown
from rich.rule import Rule

from coach_graph import config, profile, strava_api, strava_mcp, trace
from coach_graph.graph import build_graph

console = Console()
HELP = ("[dim]/new starts a fresh conversation · /profile shows your profile · "
        "/trace toggles the activity trace · /quit exits[/dim]")


async def _connect() -> list:
    if strava_api.load_tokens() is None:
        console.print(
            "[yellow]Not connected to Strava.[/yellow] Run [cyan]uv run coach connect[/cyan] first."
        )
        raise SystemExit(1)
    try:
        with console.status("[dim]Starting the Strava MCP server…[/dim]"):
            tools = await strava_mcp.load_tools()
    except Exception as exc:
        console.print(f"[red]Could not start the Strava MCP server[/red]\n[dim]{exc!r}[/dim]")
        raise SystemExit(1)
    console.print(f"[green]Strava MCP ready[/green] [dim]({len(tools)} tools)[/dim]")
    return tools


async def cmd_connect() -> None:
    try:
        tokens = await strava_api.authorize()
    except strava_api.StravaAuthError as exc:
        console.print(f"[red]{exc}[/red]")
        raise SystemExit(1)

    athlete = tokens.get("athlete") or {}
    name = " ".join(filter(None, [athlete.get("firstname"), athlete.get("lastname")]))
    console.print(f"[green]Connected to Strava[/green]" + (f" [dim]as {name}[/dim]" if name else ""))
    console.print(f"[dim]Tokens stored at {config.TOKEN_PATH}[/dim]")

    tools = await _connect()
    for tool in sorted(tools, key=lambda t: t.name):
        console.print(f"  [cyan]{tool.name}[/cyan] — {(tool.description or '').strip().splitlines()[0][:100]}")


async def cmd_chat(thread: str, tracing: bool = False) -> None:
    if profile.ensure_exists():
        console.print(f"[yellow]Created {config.PROFILE_PATH}[/yellow] — fill it in for better coaching.")

    tools = await _connect()
    config.CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    async with AsyncSqliteSaver.from_conn_string(str(config.CHECKPOINT_PATH)) as saver:
        graph = build_graph(tools, checkpointer=saver)
        console.print(Rule("[bold]coach-graph[/bold]"))
        console.print(HELP)

        while True:
            try:
                user = console.input("\n[bold cyan]you ›[/bold cyan] ").strip()
            except (EOFError, KeyboardInterrupt):
                return
            if not user:
                continue
            if user in {"/quit", "/exit"}:
                return
            if user == "/new":
                thread = uuid.uuid4().hex[:12]
                console.print(f"[dim]Started a new conversation ({thread}).[/dim]")
                continue
            if user == "/trace":
                tracing = not tracing
                console.print(f"[dim]trace {'on' if tracing else 'off'}[/dim]")
                continue
            if user == "/profile":
                console.print(Markdown(f"```yaml\n{profile.as_prompt_block()}\n```"))
                console.print(f"[dim]{config.PROFILE_PATH}[/dim]")
                continue

            try:
                await _turn(graph, user, thread, tracing)
            except (EOFError, KeyboardInterrupt):
                return
            except Exception as exc:
                # The conversation is checkpointed and still good; keep the session alive.
                console.print(f"\n[red]That turn failed.[/red] [dim]{exc!r}[/dim]")
                console.print("[dim]Your conversation is intact — try again, or /new to start over.[/dim]")


def _trace_line(label: str, body: str) -> None:
    console.print(f"[dim]  {label:<7}[/dim] {body}")


async def _turn(graph, user: str, thread: str, tracing: bool = False) -> None:
    request = {"messages": [HumanMessage(user)]}
    options: dict = {"configurable": {"thread_id": thread}}
    changes: dict = {}
    before = trace.snapshot() if tracing else None
    tracer = None

    if tracing:
        def show(call: dict) -> None:
            cost = f"[dim]{call['seconds']:.1f}s · {call['chars']:,} chars[/dim]"
            if call.get("error"):
                cost = f"[red]failed[/red] [dim]{call['error'][:60]}[/dim]"
            _trace_line("tool", f"[cyan]{call['name']}[/cyan] [dim]{call['args'][:60]}[/dim] {cost}")

        tracer = trace.ToolTracer(on_call=show)
        options["callbacks"] = [tracer]

    with console.status("[dim]thinking…[/dim]") as status:
        async for chunk in graph.astream(request, options, stream_mode="updates"):
            for node, update in chunk.items():
                # A node that returns nothing arrives as None, not an empty dict —
                # which is exactly what the profiler does on a turn it learns nothing from.
                update = update or {}
                if node == "supervisor":
                    needs = update.get("needs_data")
                    status.update("[dim]consulting Strava…[/dim]" if needs else "[dim]thinking…[/dim]")
                    if tracing:
                        status.stop()
                        _trace_line("route", f"[cyan]{update.get('specialist', 'coach')}[/cyan]"
                                             + (" [dim]· fresh data[/dim]" if needs else ""))
                        if update.get("data_request"):
                            _trace_line("ask", f"[dim]{update['data_request'][:78]}[/dim]")
                elif node == "strava_data":
                    status.update("[dim]reading your training…[/dim]")
                    if tracing:
                        _trace_line("brief", f"[dim]{len(update.get('strava_findings') or '')} chars to the coach[/dim]")
                elif node == "profiler":
                    changes = update.get("profile_changes") or {}
                else:
                    if update.get("messages"):
                        # Print before the profiler runs, so its call costs no visible wait.
                        status.stop()
                        console.print("\n[bold green]coach ›[/bold green]")
                        console.print(Markdown(update["messages"][-1].text))

    if changes:
        learned = ", ".join(
            f"{field} +{len(value)}" if isinstance(value, list) else field
            for field, value in changes.items()
        )
        console.print(f"[dim]profile updated ({learned}) · /profile to see it[/dim]")

    if tracing:
        written = trace.changes(before)
        if written["files"]:
            _trace_line("wrote", ", ".join(f"[cyan]{f}[/cyan]" for f in written["files"]))
        if written["cached"]:
            _trace_line("cached", f"[dim]{len(written['cached'])} Strava responses[/dim]")


def main() -> None:
    parser = argparse.ArgumentParser(prog="coach", description="Your LangGraph Strava coach")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("connect", help="Authorize Strava and list available MCP tools")
    chat = sub.add_parser("chat", help="Chat with your coach (default)")
    chat.add_argument("--thread", default="main", help="Conversation to resume")
    chat.add_argument("--trace", action="store_true",
                      help="Show routing, tool calls and file writes as they happen")

    args = parser.parse_args()
    if args.command == "connect":
        asyncio.run(cmd_connect())
    else:
        asyncio.run(cmd_chat(getattr(args, "thread", "main"), getattr(args, "trace", False)))


if __name__ == "__main__":
    main()
