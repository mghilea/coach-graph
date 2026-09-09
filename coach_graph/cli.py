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

from coach_graph import config, profile, strava_mcp
from coach_graph.graph import build_graph

console = Console()
HELP = "[dim]/new starts a fresh conversation · /profile shows your profile · /quit exits[/dim]"


async def _connect() -> list:
    try:
        with console.status("[dim]Connecting to Strava MCP…[/dim]"):
            tools = await strava_mcp.load_tools()
    except Exception as exc:
        while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
            exc = exc.exceptions[0]
        console.print(f"[red]Could not reach {config.STRAVA_MCP_URL}[/red]\n[dim]{exc!r}[/dim]")
        console.print(
            "\nThe connector needs an active Strava subscription. To re-authorize, "
            f"delete [cyan]{config.TOKEN_PATH}[/cyan] and run [cyan]coach connect[/cyan] again."
        )
        raise SystemExit(1)
    console.print(f"[green]Connected to Strava[/green] [dim]({len(tools)} tools)[/dim]")
    return tools


async def cmd_connect() -> None:
    tools = await _connect()
    for tool in sorted(tools, key=lambda t: t.name):
        console.print(f"  [cyan]{tool.name}[/cyan] — {(tool.description or '').strip()[:100]}")


async def cmd_chat(thread: str) -> None:
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
            if user == "/profile":
                console.print(Markdown(f"```yaml\n{profile.as_prompt_block()}\n```"))
                console.print(f"[dim]{config.PROFILE_PATH}[/dim]")
                continue

            await _turn(graph, user, thread)


async def _turn(graph, user: str, thread: str) -> None:
    request = {"messages": [HumanMessage(user)]}
    options = {"configurable": {"thread_id": thread}}
    reply = None

    with console.status("[dim]thinking…[/dim]") as status:
        async for chunk in graph.astream(request, options, stream_mode="updates"):
            for node, update in chunk.items():
                if node == "supervisor":
                    status.update("[dim]consulting Strava…[/dim]" if update.get("needs_data") else "[dim]thinking…[/dim]")
                elif node == "strava_data":
                    status.update("[dim]reading your training…[/dim]")
                elif update.get("messages"):
                    reply = update["messages"][-1]

    if reply is not None:
        console.print(f"\n[bold green]coach ›[/bold green]")
        console.print(Markdown(reply.text))


def main() -> None:
    parser = argparse.ArgumentParser(prog="coach", description="Your LangGraph Strava coach")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("connect", help="Authorize Strava and list available MCP tools")
    chat = sub.add_parser("chat", help="Chat with your coach (default)")
    chat.add_argument("--thread", default="main", help="Conversation to resume")

    args = parser.parse_args()
    if args.command == "connect":
        asyncio.run(cmd_connect())
    else:
        asyncio.run(cmd_chat(getattr(args, "thread", "main")))


if __name__ == "__main__":
    main()
