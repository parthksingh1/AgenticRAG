"""Request-scoped ambient context.

Holds the identifiers that must appear on every log line, span and database
query without being threaded through every function signature: request id,
tenant id, user id and the active trace id.

The tenant id in particular is load-bearing: :mod:`src.core.db` reads it to
apply the mandatory row-level tenant filter, so a code path that forgets to set
it fails closed rather than returning another tenant's rows.

Example:
    >>> from src.core.context import request_context, current_tenant_id
    >>> with request_context(tenant_id="t_acme", request_id="req_1"):
    ...     current_tenant_id()
    't_acme'
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field

from src.core.errors import AuthenticationError

_request_id: ContextVar[str | None] = ContextVar("agrag_request_id", default=None)
_tenant_id: ContextVar[str | None] = ContextVar("agrag_tenant_id", default=None)
_user_id: ContextVar[str | None] = ContextVar("agrag_user_id", default=None)
_conversation_id: ContextVar[str | None] = ContextVar("agrag_conversation_id", default=None)


@dataclass(frozen=True, slots=True)
class RequestContext:
    """Immutable snapshot of the ambient identifiers for one request."""

    request_id: str
    tenant_id: str | None = None
    user_id: str | None = None
    conversation_id: str | None = None
    extra: dict[str, str] = field(default_factory=dict)

    def as_log_fields(self) -> dict[str, str]:
        """Return the non-null identifiers as flat string fields for logging."""
        fields = {
            "request_id": self.request_id,
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
        }
        return {k: v for k, v in fields.items() if v is not None} | self.extra


def new_request_id() -> str:
    """Generate a fresh request identifier.

    Example:
        >>> new_request_id().startswith("req_")
        True
    """
    return f"req_{uuid.uuid4().hex[:16]}"


def current_request_id() -> str | None:
    """Return the active request id, or None outside a request."""
    return _request_id.get()


def current_tenant_id() -> str | None:
    """Return the active tenant id, or None if no tenant is bound."""
    return _tenant_id.get()


def require_tenant_id() -> str:
    """Return the active tenant id, raising if none is bound.

    Used by the data layer so that an unscoped query is a hard error instead of
    a silent cross-tenant read.

    Raises:
        AuthenticationError: when no tenant is bound to the current context.
    """
    tenant_id = _tenant_id.get()
    if tenant_id is None:
        msg = "No tenant bound to the current context; refusing to run an unscoped query."
        raise AuthenticationError(msg)
    return tenant_id


def current_user_id() -> str | None:
    """Return the active user id, or None for machine-to-machine calls."""
    return _user_id.get()


def current_conversation_id() -> str | None:
    """Return the active conversation id, if the request is a chat turn."""
    return _conversation_id.get()


def current_context() -> RequestContext:
    """Snapshot the ambient identifiers.

    Example:
        >>> with request_context(request_id="req_x"):
        ...     current_context().request_id
        'req_x'
    """
    return RequestContext(
        request_id=_request_id.get() or "req_unbound",
        tenant_id=_tenant_id.get(),
        user_id=_user_id.get(),
        conversation_id=_conversation_id.get(),
    )


@contextmanager
def request_context(
    *,
    request_id: str | None = None,
    tenant_id: str | None = None,
    user_id: str | None = None,
    conversation_id: str | None = None,
) -> Iterator[RequestContext]:
    """Bind ambient identifiers for the duration of the block.

    Values left as ``None`` inherit whatever is already bound, so nested blocks
    can narrow the context (e.g. adding a conversation id) without clearing it.

    Example:
        >>> with request_context(tenant_id="t_1"):
        ...     with request_context(conversation_id="c_1"):
        ...         (current_tenant_id(), current_conversation_id())
        ('t_1', 'c_1')
    """
    tokens: list[tuple[ContextVar[str | None], Token[str | None]]] = [
        (_request_id, _request_id.set(request_id or _request_id.get() or new_request_id())),
        (_tenant_id, _tenant_id.set(tenant_id if tenant_id is not None else _tenant_id.get())),
        (_user_id, _user_id.set(user_id if user_id is not None else _user_id.get())),
        (
            _conversation_id,
            _conversation_id.set(
                conversation_id if conversation_id is not None else _conversation_id.get()
            ),
        ),
    ]
    try:
        yield current_context()
    finally:
        for var, token in reversed(tokens):
            var.reset(token)
