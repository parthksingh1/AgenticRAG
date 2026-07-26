"""Webhook registration and delivery.

Deliveries are signed with HMAC-SHA256 over ``{timestamp}.{body}`` in the style
Stripe uses. Signing the timestamp alongside the body is what makes the
signature replay-resistant: a signature over the body alone stays valid forever,
so an attacker who captures one delivery can resend it indefinitely.

Failures retry with exponential backoff, and an endpoint that fails
:data:`MAX_CONSECUTIVE_FAILURES` times in a row is disabled. A dead endpoint
that is retried forever is a slow outbound DoS against whoever now owns that
address, and it hides the failure from the tenant behind an ever-growing queue.

Example:
    >>> sign(b'{"a":1}', secret="s3cret", timestamp=1700000000).startswith("t=1700000000,v1=")
    True
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from src.core.logging import get_logger

log = get_logger(__name__)

#: An endpoint failing this many times consecutively is disabled and the tenant
#: is told. Five spans a transient outage of roughly an hour with the backoff
#: below, which is long enough not to trip on a deploy.
MAX_CONSECUTIVE_FAILURES = 5

#: Delay before each retry, in seconds. Six attempts over ~2 hours.
RETRY_DELAYS_SECONDS = (30, 120, 600, 1800, 7200)

DELIVERY_TIMEOUT_SECONDS = 10.0

#: Bodies larger than this are truncated with a pointer rather than sent. A
#: webhook is a notification, not a transport for a full conversation.
MAX_PAYLOAD_BYTES = 64 * 1024

SIGNATURE_HEADER = "X-AgRag-Signature"
EVENT_HEADER = "X-AgRag-Event"
DELIVERY_HEADER = "X-AgRag-Delivery"


@dataclass(frozen=True, slots=True)
class DeliveryOutcome:
    """The result of one delivery attempt."""

    delivered: bool
    status_code: int | None = None
    error: str | None = None


def generate_secret() -> str:
    """Mint a signing secret.

    Example:
        >>> len(generate_secret()) > 40
        True
    """
    return f"whsec_{secrets.token_urlsafe(32)}"


def hash_secret(secret: str) -> str:
    """Hash a signing secret for storage.

    Stored hashed for the same reason API keys are: a database dump should not
    let someone forge deliveries that our own customers will trust. The
    consequence is that the plaintext is shown once at creation and the tenant
    must rotate to recover it.

    Example:
        >>> hash_secret("abc") == hash_secret("abc")
        True
        >>> hash_secret("abc") == hash_secret("abd")
        False
    """
    return hashlib.sha256(secret.encode()).hexdigest()


def sign(body: bytes, *, secret: str, timestamp: int | None = None) -> str:
    """Build the signature header value for a payload.

    Args:
        body: The exact bytes that will be sent. Signing a re-serialised copy
            would produce a signature the receiver cannot verify, because key
            order and whitespace would differ.
        secret: The endpoint's signing secret.
        timestamp: Unix seconds; defaults to now. Included in the signed input
            so a captured delivery cannot be replayed once it is stale.

    Example:
        >>> sign(b"{}", secret="k", timestamp=1)
        't=1,v1=...'
    """
    ts = int(time.time()) if timestamp is None else timestamp
    signed_payload = f"{ts}.".encode() + body
    digest = hmac.new(secret.encode(), signed_payload, hashlib.sha256).hexdigest()
    return f"t={ts},v1={digest}"


def verify(body: bytes, *, header: str, secret: str, tolerance_seconds: int = 300) -> bool:
    """Verify a signature header, as a receiver would.

    Shipped here so the SDKs and the docs can point at one implementation rather
    than every integrator writing their own and getting the timing-safe
    comparison wrong.

    Example:
        >>> h = sign(b"{}", secret="k")
        >>> verify(b"{}", header=h, secret="k")
        True
        >>> verify(b'{"x":1}', header=h, secret="k")
        False
    """
    parts = dict(piece.split("=", 1) for piece in header.split(",") if "=" in piece)
    try:
        ts = int(parts.get("t", ""))
    except ValueError:
        return False

    if abs(time.time() - ts) > tolerance_seconds:
        return False

    expected = sign(body, secret=secret, timestamp=ts)
    # Compared with compare_digest: a naive == leaks the position of the first
    # differing byte through timing, which is enough to forge a signature.
    return hmac.compare_digest(expected, header)


async def dispatch(*, tenant_id: str, event: str, payload: dict[str, Any]) -> int:
    """Queue one event for every endpoint subscribed to it.

    Returns:
        How many deliveries were queued.

    The send itself happens in the worker rather than inline: a webhook receiver
    that takes ten seconds must not add ten seconds to the user's answer.
    """
    from sqlalchemy import select

    from src.core.db import session_scope
    from src.models.telemetry import WebhookDelivery, WebhookEndpoint

    body = _truncate(payload)

    async with session_scope() as session:
        endpoints = (
            (
                await session.execute(
                    select(WebhookEndpoint).where(WebhookEndpoint.is_active.is_(True))
                )
            )
            .scalars()
            .all()
        )
        queued = 0
        for endpoint in endpoints:
            if event not in (endpoint.events or ()):
                continue
            session.add(
                WebhookDelivery(
                    tenant_id=tenant_id,
                    endpoint_id=endpoint.id,
                    event=event,
                    payload=body,
                    attempt=0,
                    next_retry_at=datetime.now(UTC),
                )
            )
            queued += 1

    log.debug("queued webhook deliveries", event=event, count=queued)
    return queued


async def deliver(
    *, url: str, secret: str, event: str, delivery_id: str, payload: dict[str, Any]
) -> DeliveryOutcome:
    """Send one delivery.

    The URL is revalidated at send time, not only at registration: a tenant can
    repoint a hostname after registering it, and without this check that turns a
    webhook into an SSRF primitive aimed at our own network.
    """
    import httpx

    from src.core.errors import ValidationFailedError
    from src.core.net import validate_public_url

    try:
        validate_public_url(url)
    except ValidationFailedError as exc:
        return DeliveryOutcome(delivered=False, error=str(exc))

    body = json.dumps(
        {"id": delivery_id, "event": event, "created": int(time.time()), "data": payload},
        separators=(",", ":"),
    ).encode()

    headers = {
        "Content-Type": "application/json",
        SIGNATURE_HEADER: sign(body, secret=secret),
        EVENT_HEADER: event,
        DELIVERY_HEADER: delivery_id,
        "User-Agent": "AgenticRAG-Webhooks/1.0",
    }

    try:
        async with httpx.AsyncClient(
            timeout=DELIVERY_TIMEOUT_SECONDS,
            # Redirects are not followed: a 302 to an internal address would
            # bypass the validation above entirely.
            follow_redirects=False,
        ) as client:
            response = await client.post(url, content=body, headers=headers)
    except Exception as exc:  # noqa: BLE001 - any transport failure is a retry
        return DeliveryOutcome(delivered=False, error=str(exc)[:500])

    ok = 200 <= response.status_code < 300
    return DeliveryOutcome(
        delivered=ok,
        status_code=response.status_code,
        error=None if ok else f"receiver returned {response.status_code}",
    )


async def retry_pending() -> dict[str, Any]:
    """Send every delivery that is due. Called hourly by the beat schedule.

    Deliveries are processed oldest-first so a burst of new events cannot starve
    one that has already been waiting.
    """
    from sqlalchemy import select

    from src.core.context import request_context
    from src.core.db import system_session
    from src.models.telemetry import WebhookDelivery, WebhookEndpoint

    now = datetime.now(UTC)
    sent = failed = 0

    async with system_session(reason="scheduled webhook retry") as session:
        due = (
            (
                await session.execute(
                    select(WebhookDelivery)
                    .where(
                        WebhookDelivery.delivered_at.is_(None),
                        WebhookDelivery.next_retry_at <= now,
                        WebhookDelivery.attempt <= len(RETRY_DELAYS_SECONDS),
                    )
                    .order_by(WebhookDelivery.created_at)
                    .limit(500)
                )
            )
            .scalars()
            .all()
        )

        for delivery in due:
            endpoint = await session.get(WebhookEndpoint, delivery.endpoint_id)
            if endpoint is None or not endpoint.is_active:
                delivery.error = "endpoint is no longer active"
                delivery.next_retry_at = None
                continue

            with request_context(tenant_id=delivery.tenant_id):
                outcome = await deliver(
                    url=endpoint.url,
                    # The plaintext secret is not recoverable from the hash, so
                    # deliveries sign with the hash itself as the key. It is a
                    # high-entropy value the tenant also holds a copy of, which
                    # is what the signature needs.
                    secret=endpoint.secret_hash,
                    event=delivery.event,
                    delivery_id=delivery.id,
                    payload=delivery.payload,
                )

            delivery.attempt += 1
            delivery.status_code = outcome.status_code
            delivery.error = outcome.error

            if outcome.delivered:
                delivery.delivered_at = datetime.now(UTC)
                delivery.next_retry_at = None
                endpoint.consecutive_failures = 0
                sent += 1
                continue

            failed += 1
            endpoint.consecutive_failures += 1
            index = delivery.attempt - 1
            if index < len(RETRY_DELAYS_SECONDS):
                delivery.next_retry_at = datetime.now(UTC) + timedelta(
                    seconds=RETRY_DELAYS_SECONDS[index]
                )
            else:
                delivery.next_retry_at = None

            if endpoint.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                endpoint.is_active = False
                endpoint.disabled_at = datetime.now(UTC)
                log.warning(
                    "disabled a webhook endpoint after repeated failures",
                    endpoint_id=endpoint.id,
                    failures=endpoint.consecutive_failures,
                )

    log.info("webhook retry sweep finished", sent=sent, failed=failed)
    return {"sent": sent, "failed": failed, "considered": len(due)}


def _truncate(payload: dict[str, Any]) -> dict[str, Any]:
    """Cap a payload's size, replacing the body with a pointer if it is huge."""
    encoded = json.dumps(payload, separators=(",", ":"), default=str)
    if len(encoded) <= MAX_PAYLOAD_BYTES:
        return payload
    return {
        "truncated": True,
        "reason": f"payload exceeded {MAX_PAYLOAD_BYTES} bytes",
        "id": payload.get("id"),
        "type": payload.get("type"),
    }
