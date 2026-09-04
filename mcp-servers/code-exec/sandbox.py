"""Launches and supervises the sandbox worker.

**Why a separate process.** The obvious implementation runs the code in a thread
with ``asyncio.wait_for``. That does not work: a Python thread cannot be
cancelled, so ``while True: pass`` keeps a core spinning for the life of the
server even after the timeout "fires". The server appears to recover while
leaking a CPU permanently, and a handful of such calls exhausts the machine.
Killing a process is the only mechanism here that actually stops running code.

**Why an explicit subprocess rather than multiprocessing.** ``spawn`` re-imports
the parent's ``__main__`` in the child, so a ``multiprocessing``-based sandbox
only works when whatever launched the server happens to be import-safe — it dies
under pytest, and under any runner with a non-trivial entry point. Launching
``worker.py`` directly has no such coupling, and the JSON pipe means nothing is
unpickled from a process that is about to execute untrusted code.

Example:
    >>> run_sandboxed("result = 1 + 1", None, timeout_seconds=10)["result"]
    2
"""

from __future__ import annotations

import json

# Every subprocess call below uses a fixed argument vector and shell=False.
import subprocess  # nosec B404
import sys
from pathlib import Path
from typing import Any

WORKER = Path(__file__).parent / "worker.py"

MAX_OUTPUT_CHARS = 20_000


def run_sandboxed(code: str, data: Any, *, timeout_seconds: float) -> dict[str, Any]:
    """Run a snippet in a killable subprocess.

    Returns:
        A dict with either ``result`` and ``stdout``, or ``error``.

    Example:
        >>> run_sandboxed("result = sum(data)", [1, 2, 3], timeout_seconds=10)["result"]
        6
        >>> "error" in run_sandboxed("import os", None, timeout_seconds=10)
        True
    """
    job = json.dumps({"code": code, "data": data}, default=str)

    try:
        process = subprocess.Popen(  # noqa: S603 # nosec B603
            # -I isolates the worker: PYTHONPATH and the user site directory are
            # ignored, so nothing can inject a module into the process that is
            # about to run untrusted code. -S is deliberately *not* used, since
            # it would also remove site-packages and with it RestrictedPython.
            [sys.executable, "-I", str(WORKER)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(WORKER.parent),
        )
    except OSError as exc:
        return {"error": f"could not start the sandbox worker: {exc}"}

    try:
        stdout, stderr = process.communicate(job, timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        # The line that makes the timeout real. A thread-based implementation
        # cannot do this, which is why the work happens in a process.
        process.kill()
        process.communicate()
        return {
            "error": (
                f"execution exceeded {timeout_seconds}s and was terminated; "
                "check for an unbounded loop"
            )
        }

    if not stdout.strip():
        detail = (stderr or "").strip().splitlines()[-1:] or ["no output"]
        return {"error": f"the sandbox worker exited without a result: {detail[0][:200]}"}

    try:
        return dict(json.loads(stdout))
    except json.JSONDecodeError:
        return {"error": f"the sandbox worker returned malformed output: {stdout[:200]}"}
