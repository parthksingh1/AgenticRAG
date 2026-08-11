"""The `agrag` command-line client.

    agrag ask "What is the carry-over limit for annual leave?"
    agrag upload handbook.pdf
    agrag docs
    agrag search "deploy freeze" --top-k 3
    agrag eval --set golden

Designed to be pipeable. Every command takes `--json`, and when stdout is not a
terminal the output is plain text with no colour codes — a CLI that writes ANSI
escapes into a file somebody is grepping is a CLI people stop using in scripts.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path
from typing import Any

CONFIG_PATH = Path.home() / ".config" / "agrag" / "config.json"

#: Written 0o600. It holds an API key, and a world-readable credential file on a
#: shared machine is a credential anyone on that machine has.
CONFIG_MODE = 0o600


def main(argv: list[str] | None = None) -> int:
    """Dispatch a subcommand."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "command", None):
        parser.print_help()
        return 2

    handlers = {
        "ask": cmd_ask,
        "search": cmd_search,
        "upload": cmd_upload,
        "docs": cmd_docs,
        "eval": cmd_eval,
        "config": cmd_config,
    }

    try:
        return handlers[args.command](args)
    except KeyboardInterrupt:
        # A user pressing Ctrl-C is not an error worth a traceback.
        print("\ninterrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - a CLI reports, it does not traceback
        print(f"error: {exc}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Example:
        >>> build_parser().parse_args(["ask", "hello"]).command
        'ask'
    """
    parser = argparse.ArgumentParser(prog="agrag", description=__doc__)
    parser.add_argument("--api-url", default=None, help="Override the API base URL.")
    parser.add_argument("--api-key", default=None, help="Override the API key.")
    parser.add_argument("--json", action="store_true", help="Emit JSON.")

    sub = parser.add_subparsers(dest="command")

    ask = sub.add_parser("ask", help="Ask a question.")
    ask.add_argument("question", nargs="+")
    ask.add_argument("--stream", action="store_true", help="Stream tokens as they arrive.")
    ask.add_argument("--model", default=None)
    ask.add_argument("--no-citations", action="store_true")

    search = sub.add_parser("search", help="Retrieve without generating an answer.")
    search.add_argument("query", nargs="+")
    search.add_argument("--top-k", type=int, default=5)

    upload = sub.add_parser("upload", help="Upload and index a file.")
    upload.add_argument("paths", nargs="+")
    upload.add_argument("--title", default=None)

    sub.add_parser("docs", help="List indexed documents.")

    evaluate = sub.add_parser("eval", help="Run an eval set.")
    evaluate.add_argument("--set", dest="set_name", default="golden")
    evaluate.add_argument("--offline", action="store_true")
    evaluate.add_argument("--gate", action="store_true")

    config = sub.add_parser("config", help="Read or write the stored credentials.")
    config.add_argument("--set-key", default=None)
    config.add_argument("--set-url", default=None)
    config.add_argument("--show", action="store_true")

    return parser


def cmd_ask(args: argparse.Namespace) -> int:
    """Ask a question."""
    question = " ".join(args.question)

    if args.stream and not args.json:
        # Streaming and --json are incompatible: a JSON document cannot be
        # emitted incrementally and still be valid JSON at every point.
        return _stream(args, question)

    answer = _client(args).ask(question, model=args.model)

    if args.json:
        print(json.dumps(_answer_dict(answer), indent=2))
        return 0

    print(answer.content)

    if answer.citations and not args.no_citations:
        print()
        for citation in answer.citations:
            page = f" p.{citation.page_number}" if citation.page_number else ""
            print(_dim(f"  [{citation.index}] {citation.document_title}{page}"))

    if answer.refused:
        print(_dim("\n  (the corpus does not cover this)"))

    print(
        _dim(
            f"\n  {answer.model} · {answer.latency_ms / 1000:.1f}s · "
            f"${answer.cost_usd:.4f} · {answer.completion_tokens} tokens"
        )
    )
    # A refusal is a valid answer, not a failure, so the exit code stays 0.
    return 0


def _stream(args: argparse.Namespace, question: str) -> int:
    """Stream an answer to the terminal."""
    import asyncio

    from agrag import AsyncAgRag

    async def run() -> None:
        async with AsyncAgRag(_key(args), base_url=_url(args)) as client:
            async for token in client.stream(question):
                sys.stdout.write(token)
                sys.stdout.flush()
        sys.stdout.write("\n")

    asyncio.run(run())
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    """Retrieve without generating."""
    hits = _client(args).search(" ".join(args.query), top_k=args.top_k)

    if args.json:
        print(json.dumps(hits, indent=2))
        return 0

    if not hits:
        print("no results")
        return 0

    for i, hit in enumerate(hits, start=1):
        score = _dim(f"{hit['score']:.4f}")
        print(f"{i}. {hit['document_title']}  {score}")
        snippet = " ".join(str(hit["content"]).split())[:220]
        print(_dim(f"   {snippet}…\n"))
    return 0


def cmd_upload(args: argparse.Namespace) -> int:
    """Upload one or more files."""
    client = _client(args)
    results = []
    failed = False

    for raw in args.paths:
        path = Path(raw)
        if not path.is_file():
            print(f"error: {path} is not a file", file=sys.stderr)
            failed = True
            continue
        document = client.upload(str(path), title=args.title)
        results.append({"id": document.id, "title": document.title, "status": document.status})
        if not args.json:
            print(f"uploaded {path.name} -> {document.id} ({document.status})")

    if args.json:
        print(json.dumps(results, indent=2))

    # Non-zero if any file failed, so `agrag upload *.pdf && next-step` behaves.
    return 1 if failed else 0


def cmd_docs(args: argparse.Namespace) -> int:
    """List indexed documents."""
    documents = _client(args).documents()

    if args.json:
        print(
            json.dumps(
                [
                    {"id": d.id, "title": d.title, "status": d.status, "chunks": d.chunk_count}
                    for d in documents
                ],
                indent=2,
            )
        )
        return 0

    if not documents:
        print("no documents indexed")
        return 0

    width = max(len(d.title) for d in documents)
    for document in documents:
        note = f"  {document.error_message}" if document.error_message else ""
        print(f"{document.title:<{width}}  {document.status:<10} {document.chunk_count:>5}{note}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Run an eval set by delegating to the harness.

    Shelling out rather than importing: the harness needs the repository on the
    path, and a CLI installed from PyPI has no repository. This works when run
    from a checkout and says so clearly when it is not.
    """
    import subprocess

    command = [sys.executable, "-m", "evals.run", "--set", args.set_name]
    if args.offline:
        command.append("--offline")
    if args.gate:
        command.append("--gate")

    try:
        return subprocess.run(command, check=False).returncode  # noqa: S603 - fixed argv
    except FileNotFoundError:
        print("error: the eval harness needs a repository checkout", file=sys.stderr)
        return 1


def cmd_config(args: argparse.Namespace) -> int:
    """Read or write the stored credentials."""
    config = _load_config()

    if args.set_key:
        config["api_key"] = args.set_key
    if args.set_url:
        config["api_url"] = args.set_url

    if args.set_key or args.set_url:
        _save_config(config)
        print(f"saved to {CONFIG_PATH}")
        return 0

    if args.show or not config:
        key = config.get("api_key", "")
        # Never print the whole key. A terminal is scrollback, a screen share
        # and often a log.
        masked = f"{key[:12]}…{key[-4:]}" if len(key) > 20 else ("(set)" if key else "(not set)")
        print(f"api_url: {config.get('api_url', 'http://localhost:8000')}")
        print(f"api_key: {masked}")
        print(f"\nconfig:  {CONFIG_PATH}")
    return 0


def _client(args: argparse.Namespace) -> Any:
    """Build an SDK client from flags, environment, then config file."""
    from agrag import AgRag

    return AgRag(_key(args), base_url=_url(args))


def _key(args: argparse.Namespace) -> str:
    """Resolve the API key.

    Precedence is flag, environment, config file — most explicit first, so a
    one-off override does not require editing a file.

    Raises:
        SystemExit: with an actionable message when no key is available.
    """
    key = args.api_key or os.getenv("AGRAG_API_KEY") or _load_config().get("api_key")
    if not key:
        print(
            "error: no API key.\n"
            "  agrag config --set-key agr_...\n"
            "  or export AGRAG_API_KEY=agr_...",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return str(key)


def _url(args: argparse.Namespace) -> str:
    """Resolve the API base URL, defaulting to a local stack."""
    return str(
        args.api_url
        or os.getenv("AGRAG_API_URL")
        or _load_config().get("api_url")
        or "http://localhost:8000"
    )


def _load_config() -> dict[str, Any]:
    """Read the config file, tolerating its absence or corruption.

    A malformed config should not make every command fail with a JSON error; the
    flags and the environment still work.
    """
    if not CONFIG_PATH.exists():
        return {}
    try:
        return dict(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    except (ValueError, OSError):
        return {}


def _save_config(config: dict[str, Any]) -> None:
    """Write the config file with owner-only permissions."""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    # Windows does not implement POSIX modes. The file still lands in the
    # user's profile, which is the protection available there.
    with contextlib.suppress(OSError):
        CONFIG_PATH.chmod(CONFIG_MODE)


def _answer_dict(answer: Any) -> dict[str, Any]:
    """Render an answer for `--json`."""
    return {
        "content": answer.content,
        "refused": answer.refused,
        "model": answer.model,
        "citations": [
            {
                "index": c.index,
                "document_id": c.document_id,
                "document_title": c.document_title,
                "page_number": c.page_number,
                "snippet": c.snippet,
            }
            for c in answer.citations
        ],
        "cost_usd": answer.cost_usd,
        "latency_ms": answer.latency_ms,
        "tokens": {"prompt": answer.prompt_tokens, "completion": answer.completion_tokens},
    }


def _dim(text: str) -> str:
    """Dim text, but only when stdout is a terminal.

    Escape codes written into a file somebody is grepping are why people stop
    using a CLI in scripts.

    Example:
        >>> "x" in _dim("x")
        True
    """
    if not sys.stdout.isatty() or os.getenv("NO_COLOR"):
        return text
    return f"\033[2m{text}\033[0m"


if __name__ == "__main__":
    raise SystemExit(main())
