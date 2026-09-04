"""HTTP middleware.

Ordering matters and is the main thing to get right here. Middleware wraps
inside-out, so the list in :func:`install_middleware` reads outermost-first:

1. **Errors** — outermost, so it catches anything the others raise and always
   produces a structured body rather than an HTML traceback page.
2. **Context** — binds the request id, tenant and trace id before anything else
   runs, so every log line and database query inside is scoped.
3. **Metrics** — times the request; inside context so its labels have a tenant.
4. **Rate limiting** — innermost, because a throttled request should still have
   produced a log line and a metric.

The tenant is bound here rather than in a route dependency because
:mod:`src.core.db` reads it to scope queries, and a dependency runs too late for
any query made during dependency resolution itself.

Example:
    >>> from src.api.middleware import ErrorMiddleware
    >>> ErrorMiddleware.__name__
    'ErrorMiddleware'
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.core.config import Settings
from src.core.context import _request_id, _tenant_id, _user_id, new_request_id
from src.core.errors import AgRagError, AuthenticationError, RateLimitedError
from src.core.logging import get_logger

log = get_logger(__name__)

Handler = Callable[[Request], Awaitable[Response]]

#: Paths that must answer without authentication or rate limiting, or the
#: orchestrator cannot tell a busy service from a dead one.
UNGUARDED_PATHS = frozenset({"/healthz", "/readyz", "/metrics", "/docs", "/openapi.json", "/redoc"})


class ErrorMiddleware(BaseHTTPMiddleware):
    """Converts exceptions into RFC-9457 problem responses."""

    async def dispatch(self, request: Request, call_next: Handler) -> Response:
        """Catch application and unexpected errors."""
        try:
            return await call_next(request)
        except AgRagError as exc:
            if exc.status_code >= 500:
                log.error("request failed", code=exc.code, detail=str(exc))
            else:
                log.info("request rejected", code=exc.code, detail=str(exc))

            headers = {}
            if isinstance(exc, RateLimitedError):
                headers["Retry-After"] = str(max(int(exc.retry_after_seconds), 1))

            return JSONResponse(
                status_code=exc.status_code,
                content=exc.to_problem(instance=str(request.url.path)),
                media_type="application/problem+json",
                headers=headers,
            )
        except Exception as exc:
            # Deliberately opaque to the caller: an internal message could carry
            # a connection string or a row of another tenant's data. The detail
            # goes to the log, where the trace id ties it back to this request.
            log.exception("unhandled error", error=str(exc))
            return JSONResponse(
                status_code=500,
                content={
                    "type": "https://docs.agrag.dev/errors/internal_error",
                    "title": "internal_error",
                    "status": 500,
                    "detail": "An unexpected error occurred.",
                    "instance": str(request.url.path),
                },
                media_type="application/problem+json",
            )


class ContextMiddleware(BaseHTTPMiddleware):
    """Binds the request id, tenant and user to the ambient context."""

    async def dispatch(self, request: Request, call_next: Handler) -> Response:
        """Bind context for the duration of the request."""
        request_id = request.headers.get("x-request-id") or new_request_id()
        tokens = [
            (_request_id, _request_id.set(request_id)),
            (_tenant_id, _tenant_id.set(None)),
            (_user_id, _user_id.set(None)),
        ]
        request.state.request_id = request_id

        try:
            response = await call_next(request)
        finally:
            for var, token in reversed(tokens):
                var.reset(token)

        response.headers["X-Request-Id"] = request_id
        return response


class TenantMiddleware(BaseHTTPMiddleware):
    """Resolves the caller and binds their tenant before any query runs.

    Authentication happens here rather than in a dependency because the database
    session reads the tenant from the ambient context; a dependency would bind it
    after any query made during dependency resolution had already run unscoped.
    """

    def __init__(self, app: FastAPI, *, settings: Settings) -> None:
        """Store settings for the dev-mode check."""
        super().__init__(app)
        self._settings = settings

    async def dispatch(self, request: Request, call_next: Handler) -> Response:
        """Authenticate and bind the tenant, or pass through unguarded paths."""
        if request.url.path in UNGUARDED_PATHS or request.method == "OPTIONS":
            return await call_next(request)

        from src.api.auth import resolve_principal
        from src.core.db import session_scope

        authorization = request.headers.get("authorization")
        dev_tenant = request.headers.get("x-dev-tenant")
        if not authorization and not dev_tenant:
            # Decided before opening a session: a request with no credential at
            # all should answer 401, not 500 because the database happened to be
            # unreachable, and an unauthenticated probe should cost no query.
            raise AuthenticationError("Provide a session token or an API key.")

        try:
            async with session_scope() as session:
                principal = await resolve_principal(
                    request,
                    session,
                    self._settings,
                    authorization=authorization,
                    dev_tenant=dev_tenant,
                )
        except AgRagError:
            raise
        except Exception as exc:
            log.error("authentication failed unexpectedly", reason=str(exc))
            raise

        request.state.principal = principal
        tenant_token = _tenant_id.set(principal.tenant_id)
        user_token = _user_id.set(principal.user_id)
        try:
            return await call_next(request)
        finally:
            _tenant_id.reset(tenant_token)
            _user_id.reset(user_token)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Records request counts and latency."""

    async def dispatch(self, request: Request, call_next: Handler) -> Response:
        """Time the request and record the outcome."""
        from src.observability.metrics import record_http_request

        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
        finally:
            duration_ms = (time.perf_counter() - started) * 1000
            # The route template, not the path: labelling by path turns every
            # document id into its own time series and blows up cardinality.
            route = request.scope.get("route")
            template = getattr(route, "path", request.url.path)
            record_http_request(
                method=request.method,
                path=template,
                status=status,
                duration_ms=duration_ms,
            )

        response.headers["X-Response-Time-Ms"] = f"{duration_ms:.1f}"
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Applies the tenant's token bucket to write and chat endpoints.

    Reads are not limited here: they are cheap, and a limit that throttles a
    dashboard refresh is a support ticket rather than a protection.
    """

    #: Endpoint prefixes that consume rate-limit budget, and their cost in tokens.
    COSTS: dict[str, float] = {  # noqa: RUF012 - configuration table
        "/v1/chat": 1.0,
        "/api/chat": 1.0,
        "/api/documents/upload": 2.0,
        "/api/search": 0.5,
    }

    async def dispatch(self, request: Request, call_next: Handler) -> Response:
        """Consume budget before serving a metered endpoint."""
        cost = next(
            (c for prefix, c in self.COSTS.items() if request.url.path.startswith(prefix)), 0.0
        )
        if cost <= 0:
            return await call_next(request)

        bucket = getattr(request.app.state, "rate_limiter", None)
        principal = getattr(request.state, "principal", None)
        if bucket is None or principal is None:
            return await call_next(request)

        decision = await bucket.consume(f"{principal.tenant_id}:{request.url.path}", cost)
        decision.raise_if_limited()
        return await call_next(request)


def install_middleware(app: FastAPI, *, settings: Settings) -> None:
    """Install the middleware stack in the right order.

    Starlette applies middleware in reverse registration order, so the last one
    added is outermost. They are registered here innermost-first, and the
    docstring at the top of this module describes the resulting order.
    """
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(MetricsMiddleware)
    # Starlette's add_middleware() stub types the factory as taking only
    # `app`; it does not model a subclass constructor with extra keyword
    # arguments, which is the documented way to configure a
    # BaseHTTPMiddleware subclass. Runtime behaviour is correct — Starlette
    # partially applies the extra kwargs before calling the class — so this
    # is a stub gap, not a real type error.
    app.add_middleware(TenantMiddleware, settings=settings)  # type: ignore[arg-type]
    app.add_middleware(ContextMiddleware)
    app.add_middleware(ErrorMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-Id", "X-Response-Time-Ms", "Retry-After"],
    )
