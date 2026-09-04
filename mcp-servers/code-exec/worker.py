"""Sandbox worker process: reads a job on stdin, writes the outcome on stdout.

Run as ``python worker.py``. It is a script rather than a
``multiprocessing`` target on purpose.

``multiprocessing`` with the ``spawn`` start method re-imports the parent's
``__main__`` in the child, which means the sandbox only works when whatever
launched the server happens to be import-safe. Under pytest, or under any
runner with a non-trivial entry point, the child re-runs that entry point and
dies. An explicit worker script has no such coupling: the child imports this
file and nothing else, identically on every platform and under every runner.

The pipe carries JSON, so nothing is unpickled from a process that is about to
execute untrusted code.

Protocol:
    stdin  {"code": str, "data": any}
    stdout {"result": any, "stdout": str} | {"error": str, "stdout": str}
"""

from __future__ import annotations

import contextlib
import json
import math
import statistics
import sys
from typing import Any

#: Address-space ceiling, where the platform provides one.
MEMORY_LIMIT_BYTES = 512 * 1024 * 1024

MAX_OUTPUT_CHARS = 20_000

#: The complete set of names the sandbox can see. An allowlist rather than a
#: denylist, because a denylist is unwinnable: there is always one more path
#: from an allowed object back to ``type`` and from there to everything.
SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs, "all": all, "any": any, "bool": bool, "bytes": bytes,
    "dict": dict, "divmod": divmod, "enumerate": enumerate, "filter": filter,
    "float": float, "format": format, "frozenset": frozenset, "hash": hash,
    "hex": hex, "int": int, "isinstance": isinstance, "issubclass": issubclass,
    "len": len, "list": list, "map": map, "max": max, "min": min, "oct": oct,
    "ord": ord, "chr": chr, "pow": pow, "range": range, "repr": repr,
    "reversed": reversed, "round": round, "set": set, "slice": slice,
    "sorted": sorted, "str": str, "sum": sum, "tuple": tuple, "zip": zip,
    "True": True, "False": False, "None": None,
    "ValueError": ValueError, "TypeError": TypeError, "KeyError": KeyError,
    "IndexError": IndexError, "ZeroDivisionError": ZeroDivisionError,
    "StopIteration": StopIteration, "Exception": Exception,
}  # fmt: skip

#: Modules exposed as pre-bound names. Import is disabled, so these are the only
#: libraries reachable and there is no path to anything else.
SAFE_MODULES: dict[str, Any] = {"math": math, "statistics": statistics}


class Printer:
    """Collects ``print`` output.

    RestrictedPython rewrites ``print`` into calls on this object rather than the
    builtin, which captures output without handing the sandbox a real file.
    """

    def __init__(self) -> None:
        """Create an empty collector."""
        self.parts: list[str] = []
        self.length = 0

    def write(self, text: str) -> None:
        """Append text until the cap is reached."""
        if self.length < MAX_OUTPUT_CHARS * 2:
            self.parts.append(text)
            self.length += len(text)

    def _call_print(self, *args: Any, **kwargs: Any) -> None:
        """Handle a rewritten ``print`` call."""
        separator = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        self.write(separator.join(str(a) for a in args) + end)

    def value(self) -> str:
        """The captured output."""
        return "".join(self.parts)


def apply_limits() -> None:
    """Apply resource limits where the platform supports them.

    POSIX only. On Windows the container's memory limit is the boundary, which
    is recorded here rather than silently assumed.
    """
    try:
        import resource
    except ImportError:  # pragma: no cover - Windows
        return
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
    # No child processes: forbids fork even if something reached it. Not
    # settable on every platform, and the container still constrains it there.
    with contextlib.suppress(ValueError, OSError):
        resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))


def serialisable(value: Any) -> Any:
    """Coerce a value into something JSON can carry.

    Anything exotic becomes its ``repr``: failing to serialise a value the code
    computed successfully would be an unhelpful way to lose it.

    Example:
        >>> serialisable({1, 2})
        [1, 2]
    """
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, list | tuple):
        return [serialisable(v) for v in value]
    if isinstance(value, set):
        return sorted((serialisable(v) for v in value), key=repr)
    if isinstance(value, dict):
        return {str(k): serialisable(v) for k, v in value.items()}
    return repr(value)[:2000]


def run(code: str, data: Any) -> dict[str, Any]:
    """Compile and execute a snippet, returning the outcome as a plain dict."""
    try:
        from RestrictedPython import compile_restricted
        from RestrictedPython.Eval import default_guarded_getitem, default_guarded_getiter
        from RestrictedPython.Guards import (
            guarded_iter_unpack_sequence,
            guarded_unpack_sequence,
            safer_getattr,
        )
    except ImportError:
        return {"error": "sandbox unavailable: RestrictedPython is not installed"}

    try:
        bytecode = compile_restricted(code, filename="<agent>", mode="exec")
    except SyntaxError as exc:
        return {"error": f"rejected by the sandbox compiler: {exc}"}

    if bytecode is None:
        return {"error": "rejected by the sandbox compiler"}

    printer = Printer()

    def guarded_write(target: Any) -> Any:
        """Allow writes only to containers created inside the sandbox."""
        if isinstance(target, list | dict | set):
            return target
        msg = f"writing to {type(target).__name__} is not allowed in the sandbox"
        raise TypeError(msg)

    namespace: dict[str, Any] = {
        "__builtins__": dict(SAFE_BUILTINS),
        "_getattr_": safer_getattr,
        "_getitem_": default_guarded_getitem,
        "_getiter_": default_guarded_getiter,
        "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
        "_unpack_sequence_": guarded_unpack_sequence,
        "_write_": guarded_write,
        "_print_": lambda *_a, **_k: printer,
        **SAFE_MODULES,
        "data": data,
        "result": None,
    }

    try:
        exec(bytecode, namespace)  # noqa: S102 # nosec B102
    except BaseException as exc:  # noqa: BLE001 - including MemoryError and SystemExit
        return {"error": f"{type(exc).__name__}: {exc}", "stdout": printer.value()}

    return {"result": serialisable(namespace.get("result")), "stdout": printer.value()}


def main() -> None:
    """Read one job from stdin, execute it, write the outcome to stdout."""
    # A limit we could not set is not a reason to refuse the call.
    with contextlib.suppress(ValueError, OSError):
        apply_limits()

    try:
        job = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError as exc:
        sys.stdout.write(json.dumps({"error": f"malformed job: {exc}"}))
        return

    outcome = run(str(job.get("code", "")), job.get("data"))
    sys.stdout.write(json.dumps(outcome, default=str))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
