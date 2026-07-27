"""Authentication and authorisation.

Two credential types reach the same place — a :class:`Principal` carrying a
tenant and a set of scopes — so every route authorises the same way regardless
of how the caller arrived:

* **A Clerk JWT** for browser sessions. Verified against Clerk's JWKS with the
  signature, issuer, audience and expiry all checked. The organisation claim
  becomes the tenant.
* **An API key** for programmatic access, including the OpenAI-compatible
  endpoint. Only the SHA-256 hash is stored, so a database leak does not yield
  working credentials.

The dev-mode escape hatch exists because requiring a live Clerk tenant to run
``docker compose up`` would defeat the five-minute quickstart. It is refused
outright when ``APP_ENV`` is a deployed environment, so it cannot be left on by
accident — a config mistake that would otherwise let anyone assume any tenant.

Example:
    >>> Principal(tenant_id="t", scopes=frozenset({"read"})).has_scope("read")
    True
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import Depends, Header, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import Settings, get_settings
from src.core.context import current_tenant_id
from src.core.errors import AuthenticationError, AuthorizationError, ConfigurationError
from src.core.logging import get_logger
from src.models.tenant import ApiKey, ApiKeyScope, Tenant

log = get_logger(__name__)

#: Prefix on every issued key, so a leaked secret is recognisable in a log or a
#: paste and can be revoked without guessing what it is.
API_KEY_PREFIX = "agr_"
API_KEY_BYTES = 32


@dataclass(frozen=True, slots=True)
class Principal:
    """Who is making a request, and what they may do."""

    tenant_id: str
    scopes: frozenset[str]
    user_id: str | None = None
    api_key_id: str | None = None
    email: str | None = None
    is_admin: bool = False
    #: "clerk", "api_key" or "dev". Recorded on audit entries so an action can be
    #: attributed to a session or a key.
    method: str = "unknown"

    def has_scope(self, scope: str | ApiKeyScope) -> bool:
        """Whether this principal holds a scope; admin implies all.

        Example:
            >>> Principal(tenant_id="t", scopes=frozenset({"admin"})).has_scope("write")
            True
            >>> Principal(tenant_id="t", scopes=frozenset({"read"})).has_scope("write")
            False
        """
        wanted = scope.value if isinstance(scope, ApiKeyScope) else scope
        return ApiKeyScope.ADMIN.value in self.scopes or wanted in self.scopes

    def require(self, scope: str | ApiKeyScope) -> None:
        """Raise unless the principal holds a scope.

        Raises:
            AuthorizationError: with the missing scope named, so a client can
                see what it needs rather than guessing.
        """
        if not self.has_scope(scope):
            wanted = scope.value if isinstance(scope, ApiKeyScope) else scope
            raise AuthorizationError(f"this credential lacks the {wanted!r} scope")


def generate_api_key() -> tuple[str, str, str]:
    """Mint a key, returning ``(secret, hash, prefix)``.

    The secret is shown once and never stored. The prefix is stored so the UI can
    show ``agr_a1b2...`` without being able to reconstruct the key.

    Example:
        >>> secret, digest, prefix = generate_api_key()
        >>> secret.startswith("agr_") and len(digest) == 64
        True
        >>> secret.startswith(prefix)
        True
    """
    secret = f"{API_KEY_PREFIX}{secrets.token_urlsafe(API_KEY_BYTES)}"
    return secret, hash_api_key(secret), secret[:12]


def hash_api_key(secret: str) -> str:
    """SHA-256 of a key.

    A fast hash rather than bcrypt on purpose: the key is 256 bits of entropy
    from a CSPRNG, so there is nothing to brute-force, and a slow hash on every
    request would add latency to no benefit. This reasoning does *not* transfer
    to passwords.

    Example:
        >>> len(hash_api_key("agr_example"))
        64
    """
    return hashlib.sha256(secret.encode()).hexdigest()


async def principal_from_api_key(secret: str, session: AsyncSession) -> Principal:
    """Resolve an API key to a principal.

    Raises:
        AuthenticationError: for an unknown, revoked or expired key. The message
            is identical in all three cases, so probing cannot distinguish
            "wrong key" from "revoked key".
    """
    from src.core.db import SKIP_TENANT_GUARD

    digest = hash_api_key(secret)
    # The lookup runs before a tenant is known, so it must bypass the guard —
    # which is why the query matches on the hash and nothing else.
    result = await session.execute(
        select(ApiKey).where(ApiKey.key_hash == digest),
        execution_options={SKIP_TENANT_GUARD: True},
    )
    key = result.scalar_one_or_none()

    now = datetime.now(UTC)
    invalid = (
        key is None
        or key.revoked_at is not None
        or (key.expires_at is not None and key.expires_at < now)
    )
    if invalid or key is None:
        log.warning("api key rejected", prefix=secret[:12])
        raise AuthenticationError("Invalid or expired API key.")

    key.last_used_at = now
    return Principal(
        tenant_id=key.tenant_id,
        scopes=frozenset(key.scopes or []),
        api_key_id=key.id,
        is_admin=ApiKeyScope.ADMIN.value in (key.scopes or []),
        method="api_key",
    )


class ClerkVerifier:
    """Verifies Clerk-issued JWTs against the published JWKS.

    Keys are cached in-process: fetching JWKS on every request adds a round trip
    to Clerk to each API call, and a Clerk outage would then take authentication
    down with it.
    """

    def __init__(self, *, jwks_url: str | None, issuer: str | None) -> None:
        """Configure the verifier."""
        self._jwks_url = jwks_url
        self._issuer = issuer
        self._client: Any | None = None

    async def verify(self, token: str) -> dict[str, Any]:
        """Verify a token and return its claims.

        Raises:
            AuthenticationError: for any signature, issuer or expiry failure.
            ConfigurationError: when Clerk is not configured at all, which is a
                deployment problem rather than a caller problem.
        """
        if not self._jwks_url:
            msg = "Clerk is not configured (set CLERK_JWKS_URL)"
            raise ConfigurationError(msg)

        import jwt
        from jwt import PyJWKClient

        if self._client is None:
            self._client = PyJWKClient(self._jwks_url, cache_keys=True)

        try:
            signing_key = self._client.get_signing_key_from_jwt(token)
            return dict(
                jwt.decode(
                    token,
                    signing_key.key,
                    algorithms=["RS256"],
                    issuer=self._issuer,
                    options={"require": ["exp", "iat"], "verify_aud": False},
                )
            )
        except Exception as exc:
            log.warning("clerk token rejected", reason=str(exc))
            raise AuthenticationError("Invalid or expired session token.") from exc


def principal_from_claims(claims: dict[str, Any]) -> Principal:
    """Map verified Clerk claims onto a principal.

    Raises:
        AuthenticationError: when the token carries no organisation. A user
            without one has no workspace, and defaulting to any tenant would be
            a cross-tenant leak.

    Example:
        >>> principal_from_claims({"org_id": "org_1", "sub": "u_1", "org_role": "admin"}).is_admin
        True
    """
    organisation = claims.get("org_id") or claims.get("organization_id")
    if not organisation:
        raise AuthenticationError(
            "This session is not associated with a workspace. Select or create one to continue."
        )

    role = str(claims.get("org_role") or claims.get("role") or "member").lower()
    is_admin = role in ("admin", "owner", "org:admin")
    scopes = {ApiKeyScope.READ.value, ApiKeyScope.WRITE.value}
    if is_admin:
        scopes.add(ApiKeyScope.ADMIN.value)

    return Principal(
        tenant_id=str(organisation),
        scopes=frozenset(scopes),
        user_id=str(claims.get("sub") or "") or None,
        email=claims.get("email"),
        is_admin=is_admin,
        method="clerk",
    )


async def resolve_principal(
    request: Request,
    session: AsyncSession,
    settings: Settings,
    *,
    authorization: str | None,
    dev_tenant: str | None,
) -> Principal:
    """Work out who is calling, from whichever credential they presented.

    Raises:
        AuthenticationError: when no usable credential is present.
    """
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
        if token.startswith(API_KEY_PREFIX):
            return await principal_from_api_key(token, session)

        verifier: ClerkVerifier = request.app.state.clerk
        return principal_from_claims(await verifier.verify(token))

    if dev_tenant:
        if settings.is_production or not settings.auth_dev_mode:
            # Refusing loudly rather than ignoring the header: a deployment that
            # accidentally left dev mode on would otherwise let any caller assume
            # any tenant, silently.
            log.error("dev auth header presented in a deployed environment")
            raise AuthenticationError("Development authentication is disabled here.")
        return Principal(
            tenant_id=dev_tenant,
            scopes=frozenset(
                {ApiKeyScope.READ.value, ApiKeyScope.WRITE.value, ApiKeyScope.ADMIN.value}
            ),
            user_id="usr_dev",
            is_admin=True,
            method="dev",
        )

    raise AuthenticationError("Provide a session token or an API key.")


async def get_principal(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
    x_dev_tenant: Annotated[str | None, Header()] = None,
    settings: Annotated[Settings, Depends(get_settings)] = None,  # type: ignore[assignment]
) -> Principal:
    """FastAPI dependency resolving the caller.

    The tenant is bound to the ambient context by the middleware before this
    runs for the database session's benefit; this returns the richer principal
    for authorisation decisions.
    """
    from src.core.db import session_scope

    async with session_scope() as session:
        principal = await resolve_principal(
            request,
            session,
            settings or get_settings(),
            authorization=authorization,
            dev_tenant=x_dev_tenant,
        )

    request.state.principal = principal
    return principal


def require_scope(scope: ApiKeyScope):  # noqa: ANN201 - returns a FastAPI dependency
    """Build a dependency that enforces a scope.

    Example:
        >>> dependency = require_scope(ApiKeyScope.ADMIN)
        >>> callable(dependency)
        True
    """

    async def check(principal: Annotated[Principal, Depends(get_principal)]) -> Principal:
        """Verify the caller holds the required scope."""
        principal.require(scope)
        return principal

    return check


def require_admin():  # noqa: ANN201 - returns a FastAPI dependency
    """Build a dependency that admits only workspace admins."""

    async def check(principal: Annotated[Principal, Depends(get_principal)]) -> Principal:
        """Verify the caller is an admin."""
        if not principal.is_admin:
            raise AuthorizationError("This action requires workspace administrator access.")
        return principal

    return check


async def load_tenant(session: AsyncSession, tenant_id: str) -> Tenant:
    """Load a tenant row.

    Raises:
        AuthenticationError: when the tenant does not exist, which means a valid
            credential is pointing at a deleted workspace.
    """
    from src.core.db import SKIP_TENANT_GUARD

    result = await session.execute(
        select(Tenant).where(Tenant.id == tenant_id),
        execution_options={SKIP_TENANT_GUARD: True},
    )
    tenant = result.scalar_one_or_none()
    if tenant is None or tenant.deleted_at is not None:
        raise AuthenticationError("This workspace no longer exists.")
    return tenant


def active_tenant() -> str:
    """The tenant bound to the current request.

    Raises:
        AuthenticationError: when called outside an authenticated request.
    """
    tenant_id = current_tenant_id()
    if tenant_id is None:
        raise AuthenticationError("No workspace is bound to this request.")
    return tenant_id
