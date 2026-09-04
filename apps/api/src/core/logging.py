"""Structured logging.

One JSON object per line, every line carrying ``request_id``, ``tenant_id``,
``user_id`` and (when tracing is active) ``trace_id`` / ``span_id`` so a log line
can always be joined back to its Jaeger trace and Langfuse observation.

Standard-library loggers (uvicorn, sqlalchemy, celery) are intercepted and
re-emitted through the same sink, so there is exactly one log format in
production.

Example:
    >>> from src.core.logging import configure_logging, get_logger
    >>> configure_logging(level="INFO", json_output=False)
    >>> get_logger(__name__).info("ingestion finished", chunks=42)  # doctest: +SKIP
"""

from __future__ import annotations

import logging
import sys
import types
from typing import TYPE_CHECKING, Any

from loguru import logger

from src.core.context import current_context

if TYPE_CHECKING:  # pragma: no cover
    from loguru import Logger, Record

# Fields that must never reach a log sink, whatever a caller passes.
_REDACTED_KEYS = frozenset(
    {
        "password",
        "secret",
        "token",
        "api_key",
        "apikey",
        "authorization",
        "cookie",
        "set-cookie",
        "secret_key",
        "access_key",
        "private_key",
        "refresh_token",
    }
)
_REDACTED = "[redacted]"


class _InterceptHandler(logging.Handler):
    """Route standard-library log records into loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        """Forward one stdlib record, preserving level and exception info."""
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Walk out of the logging machinery so the reported source is the caller.
        # Annotated Optional up front: currentframe() is typed to return
        # FrameType, but f_back is FrameType | None, and the loop below can
        # walk off the top of the stack.
        frame: types.FrameType | None
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def _redact(value: Any) -> Any:
    """Recursively blank out values whose key looks like a credential."""
    if isinstance(value, dict):
        return {
            k: (_REDACTED if k.lower() in _REDACTED_KEYS else _redact(v)) for k, v in value.items()
        }
    if isinstance(value, list | tuple):
        return type(value)(_redact(item) for item in value)
    return value


def _enrich(record: Record) -> None:
    """Attach ambient request context and OTel ids to every record."""
    context = current_context()
    record["extra"].update(context.as_log_fields())
    record["extra"] = _redact(record["extra"])

    trace_id, span_id = _current_trace_ids()
    if trace_id:
        record["extra"]["trace_id"] = trace_id
        record["extra"]["span_id"] = span_id


def _current_trace_ids() -> tuple[str | None, str | None]:
    """Return the active W3C trace and span ids, if OpenTelemetry is installed."""
    try:
        from opentelemetry import trace
    except ImportError:  # pragma: no cover - otel is a hard dependency in practice
        return None, None

    span = trace.get_current_span()
    ctx = span.get_span_context()
    if not ctx.is_valid:
        return None, None
    return format(ctx.trace_id, "032x"), format(ctx.span_id, "016x")


def configure_logging(*, level: str = "INFO", json_output: bool = True) -> None:
    """Install the single global log sink.

    Args:
        level: Minimum level to emit.
        json_output: Emit newline-delimited JSON (production) rather than the
            colourised human format (local development).

    Example:
        >>> configure_logging(level="DEBUG", json_output=False)
    """
    logger.remove()
    logger.configure(patcher=_enrich)

    if json_output:
        logger.add(
            sys.stdout,
            level=level,
            serialize=True,
            backtrace=False,
            diagnose=False,  # never render local variables: they may hold secrets
            enqueue=True,
        )
    else:
        logger.add(
            sys.stdout,
            level=level,
            colorize=True,
            backtrace=True,
            diagnose=False,
            format=(
                "<green>{time:HH:mm:ss.SSS}</green> "
                "<level>{level: <8}</level> "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan> "
                "<level>{message}</level> "
                "<dim>{extra}</dim>"
            ),
        )

    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)
    for noisy in (
        "uvicorn",
        "uvicorn.access",
        "uvicorn.error",
        "sqlalchemy.engine",
        "celery",
        "httpx",
        "opensearch",
        "neo4j",
    ):
        std_logger = logging.getLogger(noisy)
        std_logger.handlers = [_InterceptHandler()]
        std_logger.propagate = False


def get_logger(name: str | None = None) -> Logger:
    """Return a logger bound to a module name.

    Example:
        >>> log = get_logger("src.retrieval.hybrid")
        >>> log.bind(strategy="rrf").debug("fused results", n=20)  # doctest: +SKIP
    """
    return logger.bind(logger_name=name) if name else logger


__all__ = ["configure_logging", "get_logger", "logger"]
