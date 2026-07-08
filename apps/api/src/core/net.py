"""Outbound-request safety.

Anywhere the system fetches a URL a user supplied — a webhook endpoint, a page
to ingest — it is making a request from inside the network perimeter on behalf
of someone outside it. That is server-side request forgery, and the interesting
targets are not on the public internet: the cloud metadata endpoint at
``169.254.169.254`` hands out credentials, and internal services usually trust
anything that can reach them.

The defence here is to resolve the hostname and check every address it resolves
to, rather than to pattern-match the URL. A name like ``evil.example.com`` can
resolve to ``127.0.0.1``, and ``http://0x7f.1/`` is loopback written in a way no
denylist of strings will catch.

DNS rebinding — where the name resolves to a public address during validation
and a private one when the request is actually made — is not fully solvable
without pinning the connection to the validated address. Callers that need that
guarantee should pass the resolved address to the transport; :func:`safe_targets`
returns them for exactly that reason.

Example:
    >>> validate_public_url("https://example.com/hook").host
    'example.com'
    >>> validate_public_url("http://169.254.169.254/latest/meta-data/")
    Traceback (most recent call last):
    ...
    src.core.errors.ValidationFailedError: ...
"""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urlparse

from src.core.errors import ValidationFailedError

#: Only these schemes are ever fetched. ``file://`` would read the container's
#: filesystem and ``gopher://`` is a classic protocol-smuggling vector.
ALLOWED_SCHEMES = frozenset({"http", "https"})

#: Ports that are almost never a legitimate webhook target but are common
#: internal services. Blocking them is not sufficient on its own — the address
#: checks are what actually matter — but it removes the obvious cases.
BLOCKED_PORTS = frozenset({22, 23, 25, 445, 3306, 5432, 6379, 9200, 11211, 27017})

MAX_URL_LENGTH = 2048


@dataclass(frozen=True, slots=True)
class SafeTarget:
    """A URL that resolved only to public addresses."""

    url: str
    host: str
    port: int
    addresses: tuple[str, ...]


def validate_public_url(url: str, *, resolve: bool = True) -> SafeTarget:
    """Validate that a URL points somewhere it is safe to send a request.

    Args:
        url: The URL to check.
        resolve: Whether to resolve the hostname. Off only in tests, where DNS
            would be both slow and unreliable.

    Returns:
        The validated target, including the addresses it resolved to.

    Raises:
        ValidationFailedError: when the URL is malformed, uses a scheme or port
            that is not allowed, or resolves to a non-public address.

    Example:
        >>> validate_public_url("ftp://example.com")
        Traceback (most recent call last):
        ...
        src.core.errors.ValidationFailedError: ...
    """
    if len(url) > MAX_URL_LENGTH:
        msg = f"the URL is longer than {MAX_URL_LENGTH} characters"
        raise ValidationFailedError(msg)

    parsed = urlparse(url.strip())
    if parsed.scheme not in ALLOWED_SCHEMES:
        msg = f"{parsed.scheme or 'that'} is not an allowed scheme; use http or https"
        raise ValidationFailedError(msg)
    if not parsed.hostname:
        msg = "the URL has no host"
        raise ValidationFailedError(msg)

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if port in BLOCKED_PORTS:
        msg = f"port {port} is not an allowed destination"
        raise ValidationFailedError(msg)

    host = parsed.hostname
    addresses = _resolve(host) if resolve else ()
    for address in addresses:
        _reject_if_private(address, host=host)

    return SafeTarget(url=url, host=host, port=port, addresses=addresses)


def _resolve(host: str) -> tuple[str, ...]:
    """Resolve a hostname to every address it points at.

    A literal address short-circuits: ``getaddrinfo`` would accept it anyway, but
    resolving it costs a syscall and can be slow when DNS is unhealthy.

    Raises:
        ValidationFailedError: when the name does not resolve. A webhook to a
            name that does not exist is a configuration error worth reporting at
            registration rather than discovering on the first delivery.
    """
    try:
        return (str(ipaddress.ip_address(host)),)
    except ValueError:
        pass

    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        msg = f"{host} does not resolve"
        raise ValidationFailedError(msg) from exc

    return tuple({str(info[4][0]) for info in infos})


def _reject_if_private(address: str, *, host: str) -> None:
    """Reject an address that is not globally routable.

    ``is_global`` covers loopback, link-local (which is where the cloud metadata
    endpoint lives), the RFC1918 ranges, multicast and the reserved blocks in one
    check, and stays correct for IPv6 — including the IPv4-mapped forms that a
    hand-rolled range check reliably misses.

    Raises:
        ValidationFailedError: when the address is not public.
    """
    parsed = ipaddress.ip_address(address)
    mapped = getattr(parsed, "ipv4_mapped", None)
    if mapped is not None:
        parsed = mapped

    if not parsed.is_global:
        msg = f"{host} resolves to {address}, which is not a public address"
        raise ValidationFailedError(msg)


def safe_targets(url: str) -> tuple[str, ...]:
    """Return the public addresses a URL resolves to.

    Example:
        >>> safe_targets("https://example.com") != ()
        True
    """
    return validate_public_url(url).addresses
