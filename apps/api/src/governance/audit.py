"""The audit log.

Every action that changes configuration, grants access, exports data or deletes
it is recorded with who did it, from where, and what the value was before and
after. The before/after pair is the part that matters: "someone changed the
guardrail config" is not an answer to an incident, and "someone set
injection_threshold from 0.8 to 0.0 at 02:14" is.

Failures are recorded too. A denied attempt to mint an admin API key is more
interesting than a successful one.

Writing an audit entry never fails the action it describes — an audit log that
can take the system down gets disabled — but it does log loudly, because a gap
in the log is itself an incident.

Example:
    >>> redact_secrets({"name": "k", "secret": "abc"})
    {'name': 'k', 'secret': '***'}
"""

from __future__ import annotations

from typing import Any

from src.core.logging import get_logger

log = get_logger(__name__)

#: Actions worth recording. A free-text action field drifts into inconsistency
#: within a month and makes the log unfilterable; this list is the vocabulary.
ACTIONS = frozenset(
    {
        "tenant.config.updated",
        "tenant.created",
        "api_key.created",
        "api_key.revoked",
        "user.invited",
        "user.role_changed",
        "user.removed",
        "document.deleted",
        "document.exported",
        "conversation.deleted",
        "prompt.promoted",
        "experiment.created",
        "experiment.promoted",
        "webhook.created",
        "webhook.deleted",
        "gdpr.export",
        "gdpr.erasure",
        "eval.run_triggered",
        "auth.denied",
    }
)

#: Keys whose values never enter the log. Substring matched, so ``api_key_hash``
#: and ``clerk_secret_key`` are both caught.
SECRET_KEY_MARKERS = ("secret", "password", "token", "key_hash", "api_key", "credential")

REDACTED = "***"


async def record(
    *,
    action: str,
    tenant_id: str | None = None,
    actor_user_id: str | None = None,
    actor_api_key_id: str | None = None,
    actor_ip: str | None = None,
    resource_type: str | None = None,
    resource_id: str | None = None,
    before: dict[str, Any] | None = None,
    after: dict[str, Any] | None = None,
    success: bool = True,
) -> None:
    """Write one audit entry.

    Args:
        action: One of :data:`ACTIONS`. An unknown action is still recorded — a
            typo must not lose the entry — but is logged as a warning so it gets
            fixed rather than silently accepted forever.
        tenant_id: The tenant the action touched; defaults to the request's.
        actor_user_id: Who did it; defaults to the request's user.
        actor_api_key_id: The API key used, when the actor was a machine.
        actor_ip: The client address, for correlating an incident to a source.
        resource_type: What kind of thing changed ("tenant", "api_key").
        resource_id: Which one.
        before: The prior state, secrets redacted.
        after: The new state, secrets redacted.
        success: False for denied or failed attempts, which are recorded too.

    Example:
        >>> await record(action="tenant.created")  # doctest: +SKIP
    """
    from src.core.context import current_request_id, current_tenant_id, current_user_id
    from src.core.db import session_scope
    from src.models.telemetry import AuditLog
    from src.observability.tracing import current_trace_id

    if action not in ACTIONS:
        log.warning("audit entry used an unrecognised action", action=action)

    tenant = tenant_id or current_tenant_id()
    if tenant is None:
        log.error("dropping an audit entry with no tenant", action=action)
        return

    try:
        async with session_scope() as session:
            session.add(
                AuditLog(
                    tenant_id=tenant,
                    actor_user_id=actor_user_id or current_user_id(),
                    actor_api_key_id=actor_api_key_id,
                    actor_ip=actor_ip,
                    action=action,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    before=redact_secrets(before) if before else None,
                    after=redact_secrets(after) if after else None,
                    success=success,
                    trace_id=current_trace_id(),
                )
            )
    except Exception as exc:  # noqa: BLE001 - auditing must not fail the action
        log.error(
            "could not write an audit entry; the action still happened",
            action=action,
            resource_id=resource_id,
            request_id=current_request_id(),
            reason=str(exc),
        )


def redact_secrets(payload: dict[str, Any]) -> dict[str, Any]:
    """Strip secret-shaped values from an audit payload.

    The audit log is read by more people than the database is, and an API key
    hash sitting in a diff that an admin can page through is a credential leak
    with a paper trail.

    Example:
        >>> redact_secrets({"model_policy": {"clerk_secret_key": "sk"}})
        {'model_policy': {'clerk_secret_key': '***'}}
        >>> redact_secrets({"scopes": ["read"]})
        {'scopes': ['read']}
    """
    out: dict[str, Any] = {}
    for key, value in payload.items():
        if any(marker in key.lower() for marker in SECRET_KEY_MARKERS):
            out[key] = REDACTED
        elif isinstance(value, dict):
            out[key] = redact_secrets(value)
        else:
            out[key] = value
    return out


def diff(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Reduce two states to the fields that actually changed.

    Storing whole objects makes the log unreadable at review time — the reader
    has to diff two blobs by eye to find the one field that moved.

    Example:
        >>> diff({"a": 1, "b": 2}, {"a": 1, "b": 3})
        {'b': {'from': 2, 'to': 3}}
        >>> diff({"a": 1}, {"a": 1, "c": 9})
        {'c': {'from': None, 'to': 9}}
    """
    changed: dict[str, Any] = {}
    for key in set(before) | set(after):
        old = before.get(key)
        new = after.get(key)
        if old != new:
            changed[key] = {"from": old, "to": new}
    return changed
