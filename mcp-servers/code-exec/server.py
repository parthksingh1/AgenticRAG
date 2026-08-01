"""code-exec-mcp — sandboxed Python for analysis of retrieved tabular data.

This is the highest-risk server in the system, so its design is defence in depth
rather than one mechanism. The code it runs is written by a model, which may in
turn be repeating text from a document an attacker uploaded, so it is treated as
hostile input at every layer:

1. **RestrictedPython compiles it**, not :func:`compile`. Dunder attribute
   access, imports and the statements that reach interpreter internals are
   rejected at compile time.
2. **The namespace is an allowlist.** ``__builtins__`` is replaced entirely.
3. **It runs in a subprocess that can be killed**, which is the only mechanism
   here that actually stops a runaway loop — a thread cannot be cancelled, so a
   thread-based timeout leaks a spinning core forever.
4. **An address-space limit** where the platform provides one, so an allocation
   bomb kills one worker rather than the server.
5. **No network, no filesystem, no environment.**

The honest position is that RestrictedPython is a hardening layer, not a security
boundary. In production this server runs in its own container with no network
egress and a read-only root filesystem; see the Dockerfile and docs/SECURITY.md.
The layers here shrink the blast radius, and the container is what contains it.

Results are never cached: the tool is not deterministic and may be given
different data on each call.

Example:
    >>> import asyncio
    >>> asyncio.run(execute("result = sum([1, 2, 3])")).content["result"]
    6
"""

from __future__ import annotations

import asyncio
from typing import Any

from mcp_common.server import ToolResult, build_app, build_table, require, tool
from sandbox import MAX_OUTPUT_CHARS, run_sandboxed

VERSION = "1.0.0"

#: Wall-clock ceiling. A model writing an accidental infinite loop is routine.
TIMEOUT_SECONDS = 5.0

MAX_CODE_LENGTH = 20_000


async def execute(code: str, data: Any = None) -> ToolResult:
    """Run a snippet in the sandbox and return its ``result`` and stdout.

    Args:
        code: Python source. Assign to ``result`` to return a value.
        data: Bound in the namespace as ``data``, typically rows from a
            retrieved table.

    Example:
        >>> import asyncio
        >>> outcome = asyncio.run(execute("result = max(data)", data=[3, 1, 2]))
        >>> outcome.content["result"]
        3
    """
    if len(code) > MAX_CODE_LENGTH:
        return ToolResult.failure(f"code too long ({len(code)} chars, limit {MAX_CODE_LENGTH})")

    # The subprocess spawn and join are blocking, so they run off the event loop;
    # the timeout itself is enforced inside `run_sandboxed`, which can terminate.
    outcome = await asyncio.to_thread(run_sandboxed, code, data, timeout_seconds=TIMEOUT_SECONDS)

    if "error" in outcome:
        return ToolResult.failure(outcome["error"], metadata={"stdout": outcome.get("stdout", "")})

    printed = outcome.get("stdout", "")
    truncated = len(printed) > MAX_OUTPUT_CHARS
    if truncated:
        printed = printed[:MAX_OUTPUT_CHARS] + "\n...[output truncated]"

    return ToolResult.success(
        {"result": outcome.get("result"), "stdout": printed, "truncated": truncated},
        summary=_summarise(outcome.get("result"), printed),
    )


def _summarise(result: Any, printed: str) -> str:
    """One-line description for the thinking panel.

    Example:
        >>> _summarise(42, "")
        'result: 42'
        >>> _summarise(None, "hello")
        'printed 5 characters'
        >>> _summarise(None, "")
        'executed with no result'
    """
    if result is not None:
        return f"result: {str(result)[:120]}"
    if printed:
        return f"printed {len(printed)} characters"
    return "executed with no result"


@tool(
    "run_python",
    "Execute a short Python snippet in a sandbox to analyse data. Assign the "
    "value you want back to a variable named `result`. The `data` argument is "
    "available as a variable called `data`. Only `math` and `statistics` are "
    "available, and there is no network or filesystem access.",
    {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python source; assign your answer to `result`.",
            },
            "data": {"description": "Optional data bound as the `data` variable."},
        },
        "required": ["code"],
    },
    deterministic=False,
    read_only=True,
)
async def run_python_handler(arguments: dict[str, Any], _tenant_id: str) -> ToolResult:
    """Execute a snippet in the sandbox."""
    return await execute(require(arguments, "code"), arguments.get("data"))


TOOLS = build_table(run_python_handler)

app = build_app(
    name="code-exec",
    version=VERSION,
    tools=TOOLS,
    description=__doc__ or "",
)
