"""Turn-level tracing: what the agents actually did, while they do it.

Two mechanisms, because the work happens in two places. Tool calls and routing are
observed in this process through LangChain callbacks and the graph's own update
stream. File writes are not: the tools that write run inside the MCP server
subprocess, so their writes are found by comparing the data directory before and
after the turn — which catches them wherever they came from.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from langchain_core.callbacks import AsyncCallbackHandler

from coach_graph import config

# Churn that says nothing about what the agents decided.
IGNORED = {"conversations.sqlite", "conversations.sqlite-wal", "conversations.sqlite-shm"}


class ToolTracer(AsyncCallbackHandler):
    """Records every tool call the agents make, with its arguments and cost."""

    def __init__(self, on_call=None) -> None:
        self.started: dict[Any, tuple[float, str, str]] = {}
        self.calls: list[dict[str, Any]] = []
        # Called as each tool finishes, so a trace reads live rather than in a lump.
        self.on_call = on_call

    def _record(self, call: dict[str, Any]) -> None:
        self.calls.append(call)
        if self.on_call:
            self.on_call(call)

    async def on_tool_start(self, serialized, input_str, *, run_id, **kwargs) -> None:
        self.started[run_id] = (time.perf_counter(), serialized.get("name", "?"), str(input_str))

    async def on_tool_end(self, output, *, run_id, **kwargs) -> None:
        if run_id not in self.started:
            return
        began, name, args = self.started.pop(run_id)
        text = getattr(output, "content", output)
        if isinstance(text, list) and text and isinstance(text[0], dict):
            text = text[0].get("text", "")
        self._record({
            "name": name,
            "args": args,
            "seconds": time.perf_counter() - began,
            "chars": len(str(text)),
        })

    async def on_tool_error(self, error, *, run_id, **kwargs) -> None:
        if run_id not in self.started:
            return
        began, name, args = self.started.pop(run_id)
        self._record({
            "name": name, "args": args, "seconds": time.perf_counter() - began,
            "chars": 0, "error": repr(error),
        })

    def reset(self) -> None:
        self.started.clear()
        self.calls.clear()


def _watched() -> list[Path]:
    return [config.DATA_DIR, config.STATE_DIR]


def snapshot() -> dict[Path, tuple[float, int]]:
    """Modification time and size of every file the coach might write."""
    seen: dict[Path, tuple[float, int]] = {}
    for root in _watched():
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.name in IGNORED:
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            seen[path] = (stat.st_mtime, stat.st_size)
    return seen


def changes(before: dict[Path, tuple[float, int]]) -> dict[str, list[str]]:
    """What was written since `before`, split into files and cached responses.

    Cache entries are counted rather than listed — a dozen hashed filenames is noise,
    but the fact that Strava was read at all is not.
    """
    after = snapshot()
    written, cached = [], []
    for path, stamp in after.items():
        if before.get(path) == stamp:
            continue
        target = cached if config.CACHE_DIR in path.parents else written
        target.append(_display(path))
    return {"files": sorted(written), "cached": sorted(cached)}


def _display(path: Path) -> str:
    """A path the reader can place: relative to the project, or ~-prefixed."""
    for root, prefix in ((config.PROJECT_ROOT, ""), (Path.home(), "~/")):
        try:
            return prefix + str(path.relative_to(root))
        except ValueError:
            continue
    return str(path)
