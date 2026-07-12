"""OpenTelemetry tracing and Langfuse.

Two systems because they answer different questions. OTel answers "where did the
3 seconds go" across services; Langfuse answers "what exactly did the model see
and produce". Running only one leaves a real gap: a trace with no prompt cannot
explain a bad answer, and a prompt with no trace cannot explain a slow one.

They are joined by the trace id. Every Langfuse observation carries the W3C
trace id as a tag, so a slow span in Jaeger links to the exact generation in
Langfuse and back.

Tracing is optional throughout. A missing collector must not break a request, so
every function here degrades to a no-op — and says so once at startup rather
than warning per request.

Example:
    >>> from src.observability.tracing import span_attributes
    >>> span_attributes(tenant_id="t", model="m")["agrag.tenant_id"]
    't'
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from src.core.config import Settings
from src.core.logging import get_logger

log = get_logger(__name__)

_tracer: Any | None = None
_langfuse: Any | None = None
_enabled = False


def setup_tracing(settings: Settings, app: Any | None = None) -> bool:
    """Configure OTel exporters and instrumentation.

    Returns:
        Whether tracing was successfully enabled. A failure is logged once and
        the process continues untraced rather than refusing to start.
    """
    global _tracer, _enabled

    if not settings.otel_traces_enabled:
        log.info("tracing disabled by configuration")
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create(
            {
                "service.name": settings.otel_service_name,
                "service.version": "0.1.0",
                "deployment.environment": settings.app_env.value,
            }
        )
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporter(endpoint=settings.otel_exporter_otlp_endpoint, insecure=True)
            )
        )
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer("agrag")
        _enabled = True
    except Exception as exc:  # noqa: BLE001 - never fail startup over telemetry
        log.warning("tracing unavailable; continuing untraced", reason=str(exc))
        return False

    _instrument(app)
    log.info("tracing enabled", endpoint=settings.otel_exporter_otlp_endpoint)
    return True


def _instrument(app: Any | None) -> None:
    """Attach auto-instrumentation to the libraries that carry latency.

    Each is attempted independently: a missing optional instrumentation package
    should cost that library's spans, not all of them.
    """
    attempts = (
        ("fastapi", _instrument_fastapi, app),
        ("sqlalchemy", _instrument_sqlalchemy, None),
        ("redis", _instrument_redis, None),
        ("httpx", _instrument_httpx, None),
    )
    for name, install, argument in attempts:
        try:
            install(argument)
        except Exception as exc:  # noqa: BLE001 - one library's spans, not all
            log.debug("instrumentation unavailable", library=name, reason=str(exc))


def _instrument_fastapi(app: Any) -> None:
    """Instrument the FastAPI application."""
    if app is None:
        return
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    FastAPIInstrumentor.instrument_app(app, excluded_urls="healthz,readyz,metrics")


def _instrument_sqlalchemy(_: Any) -> None:
    """Instrument SQLAlchemy."""
    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

    from src.core.db import get_engine

    SQLAlchemyInstrumentor().instrument(engine=get_engine().sync_engine)


def _instrument_redis(_: Any) -> None:
    """Instrument Redis."""
    from opentelemetry.instrumentation.redis import RedisInstrumentor

    RedisInstrumentor().instrument()


def _instrument_httpx(_: Any) -> None:
    """Instrument outbound HTTP, which covers every provider and MCP call."""
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

    HTTPXClientInstrumentor().instrument()


def span_attributes(**values: Any) -> dict[str, Any]:
    """Namespace attributes and drop the empty ones.

    The ``agrag.`` prefix keeps custom attributes distinguishable from OTel's
    semantic conventions, which the collector's tenant processor relies on.

    Example:
        >>> span_attributes(tenant_id="t", model=None)
        {'agrag.tenant_id': 't'}
    """
    return {f"agrag.{k}": v for k, v in values.items() if v is not None}


@contextmanager
def traced(name: str, **attributes: Any) -> Iterator[Any]:
    """Open a span, or do nothing when tracing is unavailable.

    Yields the span so callers can add attributes discovered mid-operation, and
    yields ``None`` when disabled — so call sites must tolerate that rather than
    assuming a span exists.

    Example:
        >>> with traced("test.operation", tenant_id="t") as span:
        ...     pass
    """
    if _tracer is None:
        yield None
        return

    with _tracer.start_as_current_span(name) as span:
        for key, value in span_attributes(**attributes).items():
            span.set_attribute(key, value)
        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            from opentelemetry.trace import Status, StatusCode

            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise


def current_trace_id() -> str | None:
    """The active W3C trace id, or None.

    Example:
        >>> current_trace_id() is None or isinstance(current_trace_id(), str)
        True
    """
    try:
        from opentelemetry import trace
    except ImportError:  # pragma: no cover - otel is a hard dependency
        return None

    context = trace.get_current_span().get_span_context()
    return format(context.trace_id, "032x") if context.is_valid else None


def setup_langfuse(settings: Settings) -> Any | None:
    """Configure the Langfuse client, or return None when it is not set up.

    Returns None rather than raising when keys are absent: a local stack without
    Langfuse credentials should still serve chat.
    """
    global _langfuse

    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        log.info("Langfuse not configured; LLM traces will not be recorded")
        return None

    try:
        from langfuse import Langfuse

        _langfuse = Langfuse(
            public_key=settings.langfuse_public_key.get_secret_value(),
            secret_key=settings.langfuse_secret_key.get_secret_value(),
            host=settings.langfuse_host,
        )
    except Exception as exc:  # noqa: BLE001 - observability must not block startup
        log.warning("Langfuse unavailable", reason=str(exc))
        return None

    log.info("Langfuse enabled", host=settings.langfuse_host)
    return _langfuse


def langfuse_trace_url(trace_id: str | None, settings: Settings) -> str | None:
    """Deep link to a trace in Langfuse.

    Stored on the message so a support conversation or a failing eval case links
    straight to what the model actually saw.

    Example:
        >>> from src.core.config import Settings
        >>> url = langfuse_trace_url("abc123", Settings(langfuse_host="http://lf:3000"))
        >>> url.endswith("/trace/abc123")
        True
    """
    if not trace_id:
        return None
    return f"{settings.langfuse_host.rstrip('/')}/trace/{trace_id}"


async def shutdown_tracing() -> None:
    """Flush pending spans and Langfuse events on shutdown.

    Without this the last few seconds of traces are lost on every deploy, which
    is precisely the window most worth having.
    """
    global _enabled

    if _langfuse is not None:
        try:
            _langfuse.flush()
        except Exception as exc:  # noqa: BLE001 - shutdown must not fail
            log.warning("langfuse flush failed", reason=str(exc))

    if _enabled:
        try:
            from opentelemetry import trace

            provider = trace.get_tracer_provider()
            if hasattr(provider, "shutdown"):
                provider.shutdown()
        except Exception as exc:  # noqa: BLE001 - shutdown must not fail
            log.warning("tracer shutdown failed", reason=str(exc))
    _enabled = False
