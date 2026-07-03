# ADR 0006 — A killable subprocess for code execution

**Status:** accepted · **Date:** 2026-08-31

## Context

The code-execution MCP server runs code an LLM wrote, which may have been
influenced by a document a user uploaded. Two things must hold: it cannot reach
anything it should not, and it cannot run forever.

The first implementation used RestrictedPython on a thread with a timeout. It
satisfied neither.

## Decision

RestrictedPython in a **separate process**, launched as `python -I worker.py`,
communicating over a JSON pipe on stdin and stdout, killed by the parent on
timeout.

`-I` isolates the interpreter from the environment and the user site directory.
`-S` was tried and dropped: it also removes `site-packages`, which is where
RestrictedPython lives, so the worker could not import the very thing that makes
it safe.

## Consequences

**Good.** The timeout is real. `while True: pass` is terminated — verified by a
test that would previously hang the whole suite. Process isolation sits under the
AST restrictions, so a RestrictedPython escape still lands inside a process with
no network and a hard wall-clock limit.

**Bad.** Process startup costs roughly 40ms per call. Fine for a tool invoked
occasionally; it would not be for one on the hot path.

**Rejected: a thread with a timeout.** Python cannot interrupt a thread that is
not cooperating. The timeout fired, the thread kept spinning a core, and the
supervising process had to be killed from outside — which is exactly how this was
discovered.

**Rejected: `multiprocessing`.** The spawn start method re-imports `__main__`,
which under pytest is pytest itself. An explicit worker script has no such
ambiguity about what it is.

**Rejected: a container per call.** The right answer at scale, and far too much
operational weight for a tool used a few times a day. Revisit if the sandbox ever
runs untrusted code from more than one tenant concurrently.

## Verification

`mcp-servers/tests/test_servers.py` — eight documented escape attempts (imports,
dunder traversal, file access, network, subprocess, resource exhaustion, infinite
loops, memory), all blocked, plus a timeout test that terminates a genuine
infinite loop.
