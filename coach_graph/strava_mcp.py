"""Connection to the coach's own Strava MCP server.

Strava's hosted connector (`https://mcp.strava.com/mcp`) authorizes only Anthropic's
clients: it resolves a self-registered application and then rejects it at the MCP
authorization step (`{"resource":"MCP Authorize","field":"client_id","code":"invalid"}`).
So the coach runs `coach_graph.strava_server` as a local stdio MCP server, backed by
Strava's documented REST API and your own API application.

The `strava_data` agent is unchanged by this: it still receives MCP tools over an MCP
session. Only the server on the other end is local.
"""

from __future__ import annotations

import os
import sys

from langchain_mcp_adapters.client import MultiServerMCPClient

from coach_graph import config


def build_client() -> MultiServerMCPClient:
    return MultiServerMCPClient(
        {
            "strava": {
                "transport": "stdio",
                "command": sys.executable,
                "args": ["-m", "coach_graph.strava_server"],
                # Run from the project root so the server's own load_dotenv() finds .env
                # no matter where `coach` was invoked, and inherit the shell's environment
                # so exported credentials work too.
                "cwd": str(config.PROJECT_ROOT),
                "env": dict(os.environ),
            }
        }
    )


async def load_tools() -> list:
    return await build_client().get_tools()
