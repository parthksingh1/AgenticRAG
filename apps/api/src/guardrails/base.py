"""Guardrail protocol, decisions and the pipeline that runs them.

Design commitments, each of which is a decision that could reasonably have gone
the other way:

* **A guardrail returns a decision, it does not raise.** Raising makes it
  impossible to run several guardrails and see everything that fired, which is
  exactly what the admin failure explorer needs. The *pipeline* raises, once,
  after collecting every verdict.
* **Detector failure is not an ALLOW.** If a classifier times out, treating the
  input as clean means an outage silently disables safety. Failures produce a
  FLAG with the error recorded, and the tenant's policy decides whether that
  escalates to a block.
* **Redaction beats blocking where it can.** Blocking a message because it
  contains an email address is a bad experience for the ordinary case; redacting
  it and continuing preserves the user's intent while protecting the data.
* **Guardrails run concurrently.** They are independent, and running four
  detectors serially in front of every turn is latency the user pays for nothing.

Example:
    >>> GuardrailResult.allow(kind=GuardrailKind.PII, detector="presidio").blocked
    False
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.core.errors import GuardrailViolationError
from src.core.logging import get_logger
from src.models.telemetry import GuardrailDecision, GuardrailKind, GuardrailStage

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class GuardrailResult:
    """One guardrail's verdict on one piece of text."""

    kind: GuardrailKind
    decision: GuardrailDecision
    detector: str
    score: float | None = None
    threshold: float | None = None
    reason: str | None = None
    #: Structured findings. Never the offending text for PII — only entity types,
    #: so the audit trail does not itself become a store of personal data.
    evidence: dict[str, Any] = field(default_factory=dict)
    #: Set when the guardrail rewrote the text (redaction).
    transformed_text: str | None = None
    latency_ms: int = 0

    @classmethod
    def allow(cls, *, kind: GuardrailKind, detector: str, **extra: Any) -> GuardrailResult:
        """A clean verdict."""
        return cls(kind=kind, decision=GuardrailDecision.ALLOW, detector=detector, **extra)

    @classmethod
    def block(
        cls, *, kind: GuardrailKind, detector: str, reason: str, **extra: Any
    ) -> GuardrailResult:
        """A verdict that stops the turn."""
        return cls(
            kind=kind,
            decision=GuardrailDecision.BLOCK,
            detector=detector,
            reason=reason,
            **extra,
        )

    @classmethod
    def flag(
        cls, *, kind: GuardrailKind, detector: str, reason: str, **extra: Any
    ) -> GuardrailResult:
        """A verdict that records a concern without stopping the turn."""
        return cls(
            kind=kind,
            decision=GuardrailDecision.FLAG,
            detector=detector,
            reason=reason,
            **extra,
        )

    @classmethod
    def redact(
        cls, *, kind: GuardrailKind, detector: str, text: str, reason: str, **extra: Any
    ) -> GuardrailResult:
        """A verdict that rewrote the text and allows the turn to continue."""
        return cls(
            kind=kind,
            decision=GuardrailDecision.REDACT,
            detector=detector,
            reason=reason,
            transformed_text=text,
            **extra,
        )

    @property
    def blocked(self) -> bool:
        """True when this verdict stops the turn."""
        return self.decision is GuardrailDecision.BLOCK


class GuardrailPolicy(BaseModel):
    """Per-tenant guardrail configuration.

    Every threshold is tenant-configurable because the right setting genuinely
    differs: a healthcare tenant wants PII blocked outright, a support-desk
    tenant wants it redacted and the conversation to continue.
    """

    model_config = ConfigDict(frozen=True)

    injection_enabled: bool = True
    injection_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    #: Run the LLM judge only when the cheap detectors are uncertain. Always
    #: running it triples the cost of the input stage for no measurable gain.
    injection_llm_judge: bool = True

    pii_enabled: bool = True
    #: "redact" or "block".
    pii_mode: str = "redact"
    pii_entities: tuple[str, ...] = (
        "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "US_SSN",
        "IBAN_CODE", "IP_ADDRESS", "PERSON",
    )  # fmt: skip
    pii_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    toxicity_enabled: bool = True
    toxicity_threshold: float = Field(default=0.8, ge=0.0, le=1.0)

    off_topic_enabled: bool = True
    #: Minimum best retrieval score for a query to count as in scope.
    off_topic_min_score: float = Field(default=0.25, ge=0.0, le=1.0)

    moderation_enabled: bool = True

    groundedness_enabled: bool = True
    groundedness_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    #: Drop unsupported citations rather than blocking the whole answer.
    citation_drop_unsupported: bool = True

    #: When a detector errors, treat the FLAG as a block. Off by default: a
    #: classifier outage should not take chat down for every tenant.
    fail_closed: bool = False

    def mode_for_pii(self) -> GuardrailDecision:
        """Decision to apply when PII is found.

        Example:
            >>> GuardrailPolicy(pii_mode="block").mode_for_pii().value
            'block'
            >>> GuardrailPolicy().mode_for_pii().value
            'redact'
        """
        return GuardrailDecision.BLOCK if self.pii_mode == "block" else GuardrailDecision.REDACT


class Guardrail(ABC):
    """Inspects text and returns a verdict."""

    kind: GuardrailKind
    stage: GuardrailStage
    detector: str

    @abstractmethod
    async def check(self, text: str, *, context: GuardrailContext) -> GuardrailResult:
        """Inspect ``text`` and return a verdict. Must not raise."""

    async def aclose(self) -> None:
        """Release resources. Overridden where there are any."""
        return


@dataclass(slots=True)
class GuardrailContext:
    """Everything a guardrail may need beyond the text itself."""

    tenant_id: str
    policy: GuardrailPolicy
    user_id: str | None = None
    conversation_id: str | None = None
    message_id: str | None = None
    #: Retrieved context, needed by groundedness and citation checks.
    retrieved_chunks: Sequence[Any] = ()
    #: Best retrieval score, used by the off-topic check.
    top_retrieval_score: float | None = None
    #: The user's question, needed when checking an answer.
    query: str | None = None


@dataclass(slots=True)
class PipelineOutcome:
    """The combined result of running a stage's guardrails."""

    text: str
    results: tuple[GuardrailResult, ...]
    stage: GuardrailStage

    @property
    def blocked(self) -> bool:
        """True when any guardrail blocked."""
        return any(r.decision is GuardrailDecision.BLOCK for r in self.results)

    @property
    def blocking_result(self) -> GuardrailResult | None:
        """The first blocking verdict, if any."""
        return next((r for r in self.results if r.decision is GuardrailDecision.BLOCK), None)

    @property
    def was_modified(self) -> bool:
        """True when any guardrail rewrote the text."""
        return any(r.decision is GuardrailDecision.REDACT for r in self.results)

    @property
    def flags(self) -> tuple[str, ...]:
        """Kinds that fired at all, for the message's ``guardrail_flags`` column."""
        return tuple(
            sorted(
                {r.kind.value for r in self.results if r.decision is not GuardrailDecision.ALLOW}
            )
        )

    def raise_if_blocked(self) -> None:
        """Raise the standard violation error when the stage blocked.

        Raises:
            GuardrailViolationError: carrying the kind, so the frontend can show
                an appropriate message and the dashboard can aggregate.
        """
        blocking = self.blocking_result
        if blocking is None:
            return
        raise GuardrailViolationError(
            kind=blocking.kind.value,
            reason=blocking.reason or "blocked by policy",
            score=blocking.score,
        )


class GuardrailPipeline:
    """Runs a stage's guardrails concurrently and combines their verdicts."""

    def __init__(self, guardrails: Sequence[Guardrail], *, stage: GuardrailStage) -> None:
        """Create a pipeline for one stage."""
        self._guardrails = list(guardrails)
        self._stage = stage

    async def run(self, text: str, *, context: GuardrailContext) -> PipelineOutcome:
        """Run every guardrail and apply any redactions in order.

        Redactions are applied sequentially after all checks complete, so two
        guardrails redacting the same text cannot race and lose one another's
        edits.
        """
        if not self._guardrails:
            return PipelineOutcome(text=text, results=(), stage=self._stage)

        started = time.perf_counter()
        results = await asyncio.gather(*(self._run_one(g, text, context) for g in self._guardrails))

        current = text
        for result in results:
            if result.decision is GuardrailDecision.REDACT and result.transformed_text is not None:
                current = result.transformed_text

        outcome = PipelineOutcome(text=current, results=tuple(results), stage=self._stage)
        if outcome.flags:
            log.info(
                "guardrails fired",
                stage=self._stage.value,
                flags=outcome.flags,
                blocked=outcome.blocked,
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
        return outcome

    async def _run_one(
        self, guardrail: Guardrail, text: str, context: GuardrailContext
    ) -> GuardrailResult:
        """Run one guardrail, converting an unexpected failure into a verdict.

        A detector that crashes must not be read as "the input is fine". It
        produces a FLAG, which the tenant's ``fail_closed`` setting can escalate.
        """
        started = time.perf_counter()
        try:
            result = await guardrail.check(text, context=context)
        except Exception as exc:  # noqa: BLE001 - a crashing detector is not an ALLOW
            log.error(
                "guardrail raised; treating as a flag, not a pass",
                detector=guardrail.detector,
                kind=guardrail.kind.value,
                reason=str(exc),
            )
            decision = (
                GuardrailDecision.BLOCK if context.policy.fail_closed else GuardrailDecision.FLAG
            )
            return GuardrailResult(
                kind=guardrail.kind,
                decision=decision,
                detector=guardrail.detector,
                reason=f"detector error: {exc}",
                evidence={"error": True},
                latency_ms=int((time.perf_counter() - started) * 1000),
            )

        elapsed = int((time.perf_counter() - started) * 1000)
        return result if result.latency_ms else _with_latency(result, elapsed)

    async def aclose(self) -> None:
        """Close every guardrail."""
        await asyncio.gather(*(g.aclose() for g in self._guardrails))


def _with_latency(result: GuardrailResult, latency_ms: int) -> GuardrailResult:
    """Return a copy of a result carrying its measured latency."""
    return GuardrailResult(
        kind=result.kind,
        decision=result.decision,
        detector=result.detector,
        score=result.score,
        threshold=result.threshold,
        reason=result.reason,
        evidence=result.evidence,
        transformed_text=result.transformed_text,
        latency_ms=latency_ms,
    )
