"""Custom exception hierarchy.

Every error the application raises deliberately derives from :class:`AgRagError`,
which carries the HTTP status, a stable machine-readable ``code`` and a safe
public ``message``. Middleware converts these into RFC-9457 problem responses;
anything that is *not* an ``AgRagError`` becomes a generic 500 so internal detail
never leaks to a caller.

Example:
    >>> err = TenantIsolationError(tenant_id="t_1", resource="document:abc")
    >>> err.status_code, err.code
    (403, 'tenant_isolation_violation')
"""

from __future__ import annotations

from typing import Any


class AgRagError(Exception):
    """Base class for all application errors.

    Attributes:
        status_code: HTTP status the middleware should return.
        code: Stable, machine-readable identifier clients can branch on.
        message: Human-readable text safe to show to the caller.
        details: Extra structured context (never include secrets or PII).
    """

    status_code: int = 500
    code: str = "internal_error"
    message: str = "An unexpected error occurred."

    def __init__(
        self,
        message: str | None = None,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Create an error, optionally overriding the class-level message."""
        self.message = message or self.message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        """Render message *and* details, for logs and tracebacks.

        ``message`` alone is often a generic public string ("The model provider
        is unavailable."), which makes a traceback useless for debugging. The
        API response is built from :meth:`to_problem`, which is free to be
        terser, so enriching ``__str__`` costs nothing externally.

        Example:
            >>> str(ProviderError(provider="openai", reason="429 rate limited"))
            'The model provider is unavailable. (provider=openai, reason=429 rate limited)'
        """
        if not self.details:
            return self.message
        rendered = ", ".join(f"{k}={v}" for k, v in self.details.items())
        return f"{self.message} ({rendered})"

    def to_problem(self, *, instance: str | None = None) -> dict[str, Any]:
        """Render as an RFC-9457 problem+json body.

        Example:
            >>> NotFoundError().to_problem()["status"]
            404
        """
        problem: dict[str, Any] = {
            "type": f"https://docs.agrag.dev/errors/{self.code}",
            "title": self.code,
            "status": self.status_code,
            "detail": self.message,
        }
        if instance:
            problem["instance"] = instance
        if self.details:
            problem["details"] = self.details
        return problem


# ---------------------------------------------------------------------------
# 4xx: the caller's problem
# ---------------------------------------------------------------------------


class ValidationFailedError(AgRagError):
    """Request body or parameters failed domain validation."""

    status_code = 422
    code = "validation_failed"
    message = "The request was not valid."


class AuthenticationError(AgRagError):
    """No credentials, or credentials that could not be verified."""

    status_code = 401
    code = "unauthenticated"
    message = "Authentication required."


class AuthorizationError(AgRagError):
    """Authenticated, but lacking the scope for this operation."""

    status_code = 403
    code = "forbidden"
    message = "You do not have permission to perform this action."


class TenantIsolationError(AgRagError):
    """A query attempted to reach data belonging to another tenant.

    This is a bug or an attack, never routine. It is logged at ERROR with the
    offending resource and feeds the security alert rules.
    """

    status_code = 403
    code = "tenant_isolation_violation"
    message = "cross-tenant access denied"

    def __init__(self, *, tenant_id: str, resource: str) -> None:
        """Record which tenant tried to reach which resource."""
        super().__init__(details={"tenant_id": tenant_id, "resource": resource})


class NotFoundError(AgRagError):
    """The requested resource does not exist, or is invisible to this tenant."""

    status_code = 404
    code = "not_found"
    message = "Resource not found."


class ConflictError(AgRagError):
    """The operation conflicts with current state (duplicate, version clash)."""

    status_code = 409
    code = "conflict"
    message = "The request conflicts with the current state of the resource."


class PayloadTooLargeError(AgRagError):
    """Upload exceeds the configured size ceiling."""

    status_code = 413
    code = "payload_too_large"
    message = "The uploaded payload is too large."


class UnsupportedMediaTypeError(AgRagError):
    """No parser is registered for this file type."""

    status_code = 415
    code = "unsupported_media_type"
    message = "This file type is not supported."


class RateLimitedError(AgRagError):
    """Tenant exceeded its token-bucket allowance for this endpoint."""

    status_code = 429
    code = "rate_limited"
    message = "Rate limit exceeded. Please retry later."

    def __init__(self, *, retry_after_seconds: float) -> None:
        """Carry the Retry-After hint the middleware puts on the response."""
        super().__init__(details={"retry_after_seconds": round(retry_after_seconds, 2)})
        self.retry_after_seconds = retry_after_seconds


# ---------------------------------------------------------------------------
# Domain-specific
# ---------------------------------------------------------------------------


class GuardrailViolationError(AgRagError):
    """Input or output was blocked by a guardrail.

    The ``kind`` identifies which guardrail fired so the frontend can render an
    appropriate message and the admin dashboard can aggregate by failure mode.
    """

    status_code = 400
    code = "guardrail_violation"
    message = "The request was blocked by a safety guardrail."

    def __init__(self, *, kind: str, reason: str, score: float | None = None) -> None:
        """Record which guardrail fired and why."""
        details: dict[str, Any] = {"kind": kind, "reason": reason}
        if score is not None:
            details["score"] = round(score, 4)
        super().__init__(details=details)
        self.kind = kind


class BudgetExceededError(AgRagError):
    """Tenant hit its token or cost ceiling."""

    status_code = 402
    code = "budget_exceeded"
    message = "Token or cost budget exhausted for this tenant."

    def __init__(self, *, limit: float, used: float, window: str) -> None:
        """Record the budget window that was exhausted."""
        super().__init__(details={"limit": limit, "used": used, "window": window})


class IngestionError(AgRagError):
    """A document could not be parsed, chunked or indexed."""

    status_code = 422
    code = "ingestion_failed"
    message = "The document could not be ingested."


class RetrievalError(AgRagError):
    """A retrieval backend failed in a way the caller should know about."""

    status_code = 503
    code = "retrieval_unavailable"
    message = "The retrieval backend is temporarily unavailable."


class ToolExecutionError(AgRagError):
    """An MCP tool call failed or timed out."""

    status_code = 502
    code = "tool_execution_failed"
    message = "A tool call failed."

    def __init__(self, *, tool: str, reason: str) -> None:
        """Record which tool failed."""
        super().__init__(details={"tool": tool, "reason": reason})


class ProviderError(AgRagError):
    """An upstream LLM provider failed after retries and fallbacks."""

    status_code = 502
    code = "provider_unavailable"
    message = "The model provider is unavailable."

    def __init__(self, *, provider: str, reason: str) -> None:
        """Record which provider failed."""
        super().__init__(details={"provider": provider, "reason": reason})


class AgentBudgetExhaustedError(AgRagError):
    """The agent hit its per-turn iteration, tool-call or token ceiling.

    Surfaced to the user as a partial answer rather than an HTTP failure, so the
    status code is deliberately 200.
    """

    status_code = 200
    code = "agent_budget_exhausted"
    message = "The agent reached its budget for this turn."


class ConfigurationError(AgRagError):
    """The deployment is misconfigured. Raised at startup, not per-request."""

    status_code = 500
    code = "misconfigured"
    message = "The service is misconfigured."
