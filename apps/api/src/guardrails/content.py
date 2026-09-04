"""Toxicity, moderation and off-topic guardrails.

Three checks that share a shape but differ in what they protect:

* **Toxicity** screens user input before it reaches a provider, because sending
  abusive content to a third-party API is both a policy problem and a way to get
  an account suspended.
* **Moderation** screens generated output. It is a separate check because a
  clean question can still produce a problematic answer.
* **Off-topic** detection uses the retrieval score rather than a classifier. If
  nothing in the tenant's corpus is remotely similar to the question, the
  honest answer is "I do not have information about that" — and answering anyway
  from unrelated context is the most common way a RAG system embarrasses itself.

Off-topic deliberately *flags* rather than blocks by default. Retrieval scores
are noisy, and refusing a legitimate question because its phrasing scored badly
is a worse failure than answering with an appropriate caveat.

Example:
    >>> OffTopicGuardrail().kind.value
    'off_topic'
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.core.logging import get_logger
from src.guardrails.base import Guardrail, GuardrailContext, GuardrailResult
from src.models.telemetry import GuardrailKind, GuardrailStage

if TYPE_CHECKING:
    import httpx

log = get_logger(__name__)


class ToxicityGuardrail(Guardrail):
    """Screens text with a toxicity classifier."""

    kind = GuardrailKind.TOXICITY
    detector = "toxic-bert"

    def __init__(
        self,
        model_name: str = "unitary/toxic-bert",
        *,
        device: str | None = None,
        stage: GuardrailStage = GuardrailStage.INPUT,
    ) -> None:
        """Configure the guardrail without loading the model."""
        self.model_name = model_name
        self.stage = stage
        self._device = device
        self._pipeline: Any | None = None

    def _load(self) -> Any:
        """Load the classifier on first use."""
        if self._pipeline is None:
            from transformers import pipeline

            log.info("loading toxicity classifier", model=self.model_name)
            self._pipeline = pipeline(
                "text-classification", model=self.model_name, device=self._device, truncation=True
            )
        return self._pipeline

    async def check(self, text: str, *, context: GuardrailContext) -> GuardrailResult:
        """Score toxicity and block above the tenant's threshold."""
        import asyncio

        policy = context.policy
        if not policy.toxicity_enabled:
            return GuardrailResult.allow(kind=self.kind, detector="disabled")

        try:
            classifier = self._load()
            output = await asyncio.to_thread(classifier, text[:2000])
        except Exception as exc:  # noqa: BLE001 - a missing model is not a pass
            log.warning("toxicity classifier unavailable", reason=str(exc))
            return GuardrailResult.flag(
                kind=self.kind,
                detector=self.detector,
                reason="toxicity classifier unavailable",
                evidence={"error": True},
            )

        score = _toxic_probability(output)
        if score >= policy.toxicity_threshold:
            return GuardrailResult.block(
                kind=self.kind,
                detector=self.detector,
                reason="content flagged as toxic",
                score=score,
                threshold=policy.toxicity_threshold,
            )
        return GuardrailResult.allow(
            kind=self.kind, detector=self.detector, score=score, threshold=policy.toxicity_threshold
        )


class ModerationGuardrail(Guardrail):
    """Screens generated output with the OpenAI Moderation API."""

    kind = GuardrailKind.MODERATION
    stage = GuardrailStage.OUTPUT
    detector = "openai-moderation"

    def __init__(self, *, api_key: str | None = None, timeout: float = 8.0) -> None:
        """Create the guardrail. Without an API key it is a no-op."""
        self._api_key = api_key
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def check(self, text: str, *, context: GuardrailContext) -> GuardrailResult:
        """Call the moderation endpoint and block on a flagged result."""
        if not context.policy.moderation_enabled or not self._api_key:
            return GuardrailResult.allow(kind=self.kind, detector="disabled")

        import httpx

        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)

        try:
            response = await self._client.post(
                "https://api.openai.com/v1/moderations",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={"model": "omni-moderation-latest", "input": text[:8000]},
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:  # noqa: BLE001 - an outage is not a pass
            log.warning("moderation API unavailable", reason=str(exc))
            return GuardrailResult.flag(
                kind=self.kind,
                detector=self.detector,
                reason="moderation API unavailable",
                evidence={"error": True},
            )

        result = (payload.get("results") or [{}])[0]
        if result.get("flagged"):
            categories = sorted(k for k, v in (result.get("categories") or {}).items() if v)
            return GuardrailResult.block(
                kind=self.kind,
                detector=self.detector,
                reason=f"output flagged: {', '.join(categories) or 'unspecified'}",
                evidence={"categories": categories},
            )
        return GuardrailResult.allow(kind=self.kind, detector=self.detector)

    async def aclose(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None


class OffTopicGuardrail(Guardrail):
    """Flags questions the tenant's corpus cannot answer.

    Uses the retrieval score instead of a topic classifier, which is both cheaper
    and more accurate for the actual question being asked: not "is this about the
    right subject" but "do we have anything relevant".
    """

    kind = GuardrailKind.OFF_TOPIC
    stage = GuardrailStage.INPUT
    detector = "retrieval-score"

    def __init__(self, *, block: bool = False) -> None:
        """Create the guardrail.

        Args:
            block: Whether to block rather than flag. Off by default because
                retrieval scores are noisy and refusing a legitimate question is
                worse than answering it with a caveat.
        """
        self._block = block

    async def check(self, text: str, *, context: GuardrailContext) -> GuardrailResult:
        """Compare the best retrieval score against the tenant's floor."""
        policy = context.policy
        if not policy.off_topic_enabled:
            return GuardrailResult.allow(kind=self.kind, detector="disabled")

        score = context.top_retrieval_score
        if score is None:
            # Retrieval has not run yet; nothing to judge on.
            return GuardrailResult.allow(kind=self.kind, detector="not_evaluated")

        if score >= policy.off_topic_min_score:
            return GuardrailResult.allow(
                kind=self.kind,
                detector=self.detector,
                score=score,
                threshold=policy.off_topic_min_score,
            )

        reason = "no sufficiently relevant documents in this tenant's corpus"
        maker = GuardrailResult.block if self._block else GuardrailResult.flag
        return maker(
            kind=self.kind,
            detector=self.detector,
            reason=reason,
            score=score,
            threshold=policy.off_topic_min_score,
        )


class ScriptedGuardrail(Guardrail):
    """Returns a fixed verdict. Used to drive pipeline tests deterministically."""

    def __init__(
        self,
        result: GuardrailResult,
        *,
        kind: GuardrailKind = GuardrailKind.TOXICITY,
        stage: GuardrailStage = GuardrailStage.INPUT,
        raises: bool = False,
    ) -> None:
        """Configure the scripted verdict."""
        self.kind = kind
        self.stage = stage
        self.detector = "scripted"
        self._result = result
        self._raises = raises

    async def check(self, text: str, *, context: GuardrailContext) -> GuardrailResult:
        """Return the scripted verdict, or raise to exercise failure handling."""
        if self._raises:
            msg = "scripted detector failure"
            raise RuntimeError(msg)
        return self._result


def _toxic_probability(output: Any) -> float:
    """Extract the toxic-class probability from a classifier response.

    Handles both label shapes the model family emits, and treats an unrecognised
    response as non-toxic with zero confidence rather than guessing.

    Example:
        >>> _toxic_probability([{"label": "toxic", "score": 0.91}])
        0.91
        >>> _toxic_probability([{"label": "non-toxic", "score": 0.99}])
        0.0
        >>> _toxic_probability([])
        0.0
    """
    if not output:
        return 0.0
    rows = output[0] if isinstance(output[0], list) else output
    for row in rows:
        label = str(row.get("label", "")).lower()
        if label in ("toxic", "label_1", "hate", "offensive"):
            return float(row.get("score", 0.0))
    return 0.0
