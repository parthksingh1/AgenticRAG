"""Tenant, user and API-key models.

A tenant is a Clerk organisation. It owns every other row in the system and
carries its own retrieval configuration, guardrail policy, model policy and
budget, so two tenants on the same deployment can run materially different
pipelines.

Example:
    >>> from src.models.tenant import Tenant, RetrievalStrategy
    >>> Tenant.__tablename__
    'tenants'
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base, JSONColumn, SoftDeleteMixin, TenantScoped, TimestampMixin, new_id


class TenantPlan(StrEnum):
    """Commercial tier. Gates the premium retrieval strategies."""

    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"
    DEMO = "demo"


class RetrievalStrategy(StrEnum):
    """Retrieval pipelines a tenant may enable.

    These compose: a tenant can run ``HYBRID`` fusion with ``HYDE`` rewriting and
    ``CORRECTIVE`` evaluation simultaneously. ``COLBERT`` is gated to paid plans
    because late interaction is materially more expensive to serve.
    """

    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"
    COLBERT = "colbert"
    HYDE = "hyde"
    MULTI_QUERY = "multi_query"
    CORRECTIVE = "corrective"
    SELF_RAG = "self_rag"
    GRAPH = "graph"
    ADAPTIVE = "adaptive"
    TEMPORAL = "temporal"


class ChunkingStrategy(StrEnum):
    """How documents are split before embedding."""

    LAYOUT_AWARE = "layout_aware"
    SEMANTIC = "semantic"
    LATE = "late"
    FIXED = "fixed"


class ApiKeyScope(StrEnum):
    """Coarse permissions attached to a tenant API key."""

    READ = "read"
    WRITE = "write"
    ADMIN = "admin"


class Tenant(Base, TimestampMixin, SoftDeleteMixin):
    """A customer organisation. The root of every ownership chain."""

    __tablename__ = "tenants"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("ten"))
    clerk_org_id: Mapped[str | None] = mapped_column(String(128), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    slug: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    plan: Mapped[TenantPlan] = mapped_column(String(20), default=TenantPlan.FREE, nullable=False)

    # ── Pipeline configuration ───────────────────────────────────────────────
    embedding_model: Mapped[str] = mapped_column(
        String(200), default="BAAI/bge-large-en-v1.5", nullable=False
    )
    chunking_strategy: Mapped[ChunkingStrategy] = mapped_column(
        String(32), default=ChunkingStrategy.LAYOUT_AWARE, nullable=False
    )
    enabled_strategies: Mapped[list[str]] = mapped_column(
        JSONColumn,
        default=lambda: [RetrievalStrategy.HYBRID.value],
        nullable=False,
        comment="Retrieval strategies this tenant runs; see RetrievalStrategy.",
    )
    contextual_retrieval_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    graph_extraction_enabled: Mapped[bool] = mapped_column(Boolean, default=False)

    # ── Behaviour ────────────────────────────────────────────────────────────
    custom_instructions: Mapped[str | None] = mapped_column(Text)
    response_template: Mapped[str | None] = mapped_column(
        Text, comment="Jinja2 template applied by the formatter node."
    )
    guardrail_config: Mapped[dict[str, Any]] = mapped_column(
        JSONColumn,
        default=dict,
        nullable=False,
        comment="Per-guardrail thresholds and block/redact mode. See GuardrailPolicy.",
    )
    model_policy: Mapped[dict[str, Any]] = mapped_column(
        JSONColumn,
        default=dict,
        nullable=False,
        comment="Allowed models, default model and routing preferences.",
    )

    # ── Budgets ──────────────────────────────────────────────────────────────
    daily_token_budget: Mapped[int] = mapped_column(BigInteger, default=2_000_000, nullable=False)
    monthly_cost_cap_usd: Mapped[float | None] = mapped_column(Numeric(12, 4))

    # ── Governance ───────────────────────────────────────────────────────────
    data_region: Mapped[str] = mapped_column(
        String(32), default="us-east-1", nullable=False, comment="Residency pin for storage."
    )
    retention_days: Mapped[int | None] = mapped_column(
        comment="Auto-purge conversations older than this. Null means keep forever."
    )

    users: Mapped[list[User]] = relationship(back_populates="tenant", cascade="all, delete-orphan")
    api_keys: Mapped[list[ApiKey]] = relationship(
        back_populates="tenant", cascade="all, delete-orphan"
    )

    def has_strategy(self, strategy: RetrievalStrategy) -> bool:
        """Return whether this tenant has a retrieval strategy enabled.

        Example:
            >>> t = Tenant(name="Acme", slug="acme", enabled_strategies=["hybrid"])
            >>> t.has_strategy(RetrievalStrategy.HYBRID)
            True
            >>> t.has_strategy(RetrievalStrategy.COLBERT)
            False
        """
        return strategy.value in (self.enabled_strategies or [])


class User(Base, TenantScoped, TimestampMixin, SoftDeleteMixin):
    """A person inside a tenant, mirrored from Clerk on first sign-in."""

    __tablename__ = "users"
    __table_args__ = (
        UniqueConstraint("tenant_id", "clerk_user_id", name="uq_users_tenant_id_clerk_user_id"),
        Index("ix_users_tenant_id_email", "tenant_id", "email"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("usr"))
    tenant_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    clerk_user_id: Mapped[str] = mapped_column(String(128), nullable=False)
    email: Mapped[str] = mapped_column(String(320), nullable=False)
    display_name: Mapped[str | None] = mapped_column(String(200))
    role: Mapped[str] = mapped_column(String(32), default="member", nullable=False)
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    tenant: Mapped[Tenant] = relationship(back_populates="users")

    @property
    def is_admin(self) -> bool:
        """True when the user may reach the /admin surface."""
        return self.role in ("admin", "owner")


class ApiKey(Base, TenantScoped, TimestampMixin):
    """A hashed API key granting programmatic access to one tenant.

    Only the SHA-256 hash is stored. The plaintext is shown exactly once at
    creation time; ``prefix`` exists so the UI can display ``agr_live_a1b2...``
    without being able to reconstruct the secret.
    """

    __tablename__ = "api_keys"
    __table_args__ = (Index("ix_api_keys_key_hash", "key_hash", unique=True),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("key"))
    tenant_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    key_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    prefix: Mapped[str] = mapped_column(String(16), nullable=False)
    scopes: Mapped[list[str]] = mapped_column(
        JSONColumn, default=lambda: [ApiKeyScope.READ.value], nullable=False
    )
    created_by_user_id: Mapped[str | None] = mapped_column(String(64))
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    tenant: Mapped[Tenant] = relationship(back_populates="api_keys")

    def has_scope(self, scope: ApiKeyScope) -> bool:
        """Return whether the key carries a scope (admin implies all).

        Example:
            >>> ApiKey(scopes=["admin"]).has_scope(ApiKeyScope.WRITE)
            True
        """
        scopes = self.scopes or []
        return ApiKeyScope.ADMIN.value in scopes or scope.value in scopes
