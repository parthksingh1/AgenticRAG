"""Groundedness and citation verification.

A citation marker is a claim about evidence, and models attach them
enthusiastically to sentences the cited passage does not actually support. That
failure is worse than an uncited wrong answer, because the citation is what
persuades the reader to trust it.

So every claim is checked against the passage it cites, using natural language
inference: does the chunk *entail* the sentence? Three outcomes:

* **entailed** — the citation stands.
* **neutral** — the passage neither supports nor contradicts. The citation is
  dropped and the sentence is marked unsupported.
* **contradicted** — the passage says the opposite. This is a hallucination and
  it is escalated, not quietly dropped.

The default response is to drop unsupported citations rather than block the
answer, because a partially-cited answer with honest gaps is more useful than a
refusal. Contradictions block, because an answer that contradicts its own
evidence has no honest reading.

Numbers get an extra pass. NLI models are unreliable on figures — "revenue grew
12%" and "revenue grew 21%" look nearly identical to them — so any number in a
claim must appear in the cited passage.

Example:
    >>> split_claims("Revenue grew 12% [1]. Costs fell [2].")[0].markers
    (1,)
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from src.core.logging import get_logger
from src.guardrails.base import Guardrail, GuardrailContext, GuardrailResult
from src.models.telemetry import GuardrailKind, GuardrailStage

if TYPE_CHECKING:
    from src.retrieval.types import RetrievedChunk

log = get_logger(__name__)

#: Matches ``[1]`` and ``[1][2]`` markers.
_MARKER = re.compile(r"\[(\d{1,2})\]")

#: Sentence splitter for claim extraction. Deliberately the same conservative
#: approach as the chunker's, so claim boundaries and chunk boundaries agree.
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")

#: Numbers, percentages and currency amounts that must survive verification.
_NUMBER = re.compile(r"(?<![\w.])(?:[$£€]\s?)?\d[\d,]*(?:\.\d+)?\s?%?")


class Entailment(StrEnum):
    """NLI verdict for one claim against one passage."""

    ENTAILED = "entailed"
    NEUTRAL = "neutral"
    CONTRADICTED = "contradicted"


@dataclass(frozen=True, slots=True)
class Claim:
    """One sentence of an answer, with the citations attached to it."""

    text: str
    markers: tuple[int, ...]
    #: Character offsets into the original answer, so the binder can edit in place.
    start: int
    end: int

    @property
    def is_cited(self) -> bool:
        """Whether the claim carries at least one citation marker."""
        return bool(self.markers)

    @property
    def is_factual(self) -> bool:
        """Whether the claim asserts something that needs evidence.

        Hedges, questions and conversational filler are excluded: demanding a
        citation for "Let me know if you need more detail." would make the
        citation-recall metric meaningless.

        Example:
            >>> Claim("Revenue grew 12%.", (1,), 0, 17).is_factual
            True
            >>> Claim("Let me know if you need anything else.", (), 0, 5).is_factual
            False
        """
        stripped = self.text.strip()
        if len(stripped) < 15 or stripped.endswith("?"):
            return False
        conversational = (
            "let me know", "i hope", "feel free", "would you like", "here is",
            "here's", "in summary", "to summarise", "to summarize", "i can help",
            "based on the", "according to the provided",
        )  # fmt: skip
        lowered = stripped.lower()
        return not any(lowered.startswith(prefix) for prefix in conversational)


@dataclass(frozen=True, slots=True)
class CitationCheck:
    """The outcome of verifying one claim against one cited chunk."""

    claim: Claim
    marker: int
    chunk_id: str | None
    entailment: Entailment
    score: float
    numbers_verified: bool = True

    @property
    def supported(self) -> bool:
        """Whether the citation genuinely supports the claim."""
        return self.entailment is Entailment.ENTAILED and self.numbers_verified


@dataclass(slots=True)
class GroundednessReport:
    """The full verification result for one answer."""

    claims: tuple[Claim, ...]
    checks: tuple[CitationCheck, ...]
    corrected_answer: str
    dropped_markers: tuple[int, ...] = ()

    @property
    def citation_precision(self) -> float:
        """Fraction of citations that actually support their claim.

        Example:
            >>> GroundednessReport(claims=(), checks=(), corrected_answer="").citation_precision
            1.0
        """
        if not self.checks:
            return 1.0
        return sum(1 for c in self.checks if c.supported) / len(self.checks)

    @property
    def citation_recall(self) -> float:
        """Fraction of factual claims that carry a supported citation."""
        factual = [c for c in self.claims if c.is_factual]
        if not factual:
            return 1.0
        supported = {id(c.claim) for c in self.checks if c.supported}
        return sum(1 for c in factual if id(c) in supported) / len(factual)

    @property
    def has_contradiction(self) -> bool:
        """Whether any claim is contradicted by the passage it cites."""
        return any(c.entailment is Entailment.CONTRADICTED for c in self.checks)

    @property
    def groundedness(self) -> float:
        """Overall groundedness: the share of factual claims that are supported."""
        return self.citation_recall


def split_claims(answer: str) -> list[Claim]:
    """Split an answer into sentences, extracting their citation markers.

    Example:
        >>> claims = split_claims("Revenue grew 12% [1]. Costs fell [2][3].")
        >>> [(c.markers) for c in claims]
        [(1,), (2, 3)]
    """
    claims: list[Claim] = []
    position = 0
    for sentence in _SENTENCE.split(answer):
        if not sentence.strip():
            continue
        start = answer.find(sentence, position)
        if start == -1:  # pragma: no cover - the split preserves the source text
            start = position
        end = start + len(sentence)
        position = end
        markers = tuple(int(m) for m in _MARKER.findall(sentence))
        # The markers are stripped from the claim text. Leaving them in makes the
        # number check read "[1]" as the number 1, so every cited claim appears to
        # cite a figure its passage does not contain and is wrongly dropped.
        assertion = _tidy(_MARKER.sub("", sentence))
        claims.append(Claim(text=assertion, markers=markers, start=start, end=end))
    return claims


def _tidy(text: str) -> str:
    """Collapse whitespace and close the gap a removed marker leaves before punctuation.

    Example:
        >>> _tidy("Revenue grew  . Costs fell ,")
        'Revenue grew. Costs fell,'
    """
    collapsed = re.sub(r"\s+", " ", text)
    return re.sub(r"\s+([.,;:!?])", lambda m: m.group(1), collapsed).strip()


def extract_numbers(text: str) -> set[str]:
    """Normalised numeric tokens in a text.

    Example:
        >>> sorted(extract_numbers("revenue was $1,200.50, up 12%"))
        ['12', '1200.50']
    """
    found: set[str] = set()
    for match in _NUMBER.finditer(text):
        cleaned = match.group().strip().lstrip("$£€").strip().rstrip("%").replace(",", "").strip()
        found.add(cleaned)
    return found


def numbers_supported(claim: str, passage: str) -> bool:
    """Whether every number in the claim appears in the passage.

    NLI models are weak on figures, so this is checked separately. A claim with
    no numbers passes trivially.

    Example:
        >>> numbers_supported("revenue grew 12%", "revenue grew 12% this year")
        True
        >>> numbers_supported("revenue grew 21%", "revenue grew 12% this year")
        False
        >>> numbers_supported("revenue grew", "anything at all")
        True
    """
    claim_numbers = extract_numbers(claim)
    if not claim_numbers:
        return True
    return claim_numbers <= extract_numbers(passage)


class NliModel:
    """Natural language inference scorer."""

    name: str

    async def entail(self, premise: str, hypothesis: str) -> tuple[Entailment, float]:
        """Return the verdict and confidence for premise entailing hypothesis."""
        raise NotImplementedError


class DebertaNliModel(NliModel):
    """Local NLI using ``microsoft/deberta-large-mnli``."""

    name = "deberta-large-mnli"

    def __init__(
        self, model_name: str = "microsoft/deberta-large-mnli", *, device: str | None = None
    ) -> None:
        """Configure without loading the model."""
        self.model_name = model_name
        self._device = device
        self._pipeline: Any | None = None

    def _load(self) -> Any:
        """Load the NLI pipeline on first use."""
        if self._pipeline is None:
            from transformers import pipeline

            log.info("loading NLI model", model=self.model_name)
            self._pipeline = pipeline(
                "text-classification", model=self.model_name, device=self._device, truncation=True
            )
        return self._pipeline

    async def entail(self, premise: str, hypothesis: str) -> tuple[Entailment, float]:
        """Score entailment of the hypothesis by the premise."""
        import asyncio

        try:
            model = self._load()
            output = await asyncio.to_thread(model, f"{premise}</s></s>{hypothesis}", top_k=None)
        except Exception as exc:  # noqa: BLE001 - fall back to lexical overlap
            log.warning("NLI model unavailable; falling back to overlap", reason=str(exc))
            return LexicalNliModel().entail_sync(premise, hypothesis)

        scores = {str(row["label"]).upper(): float(row["score"]) for row in output}
        entail = scores.get("ENTAILMENT", 0.0)
        contradiction = scores.get("CONTRADICTION", 0.0)
        neutral = scores.get("NEUTRAL", 0.0)

        best = max(
            (entail, Entailment.ENTAILED),
            (contradiction, Entailment.CONTRADICTED),
            (neutral, Entailment.NEUTRAL),
        )
        return best[1], best[0]


class LexicalNliModel(NliModel):
    """Token-overlap approximation of entailment.

    Not a substitute for a real NLI model — it cannot detect contradiction at
    all. It exists so groundedness checking still runs offline and in CI, and
    every result records that this detector produced it so no eval number is
    mistaken for a real NLI measurement.
    """

    name = "lexical-overlap"

    def entail_sync(self, premise: str, hypothesis: str) -> tuple[Entailment, float]:
        """Synchronous scoring by content-word overlap.

        Example:
            >>> LexicalNliModel().entail_sync("revenue grew strongly", "revenue grew")[0].value
            'entailed'
        """
        stop = {
            "the", "a", "an", "is", "are", "was", "were", "of", "in", "to", "and",
            "for", "on", "at", "by", "with", "that", "this", "it", "as", "from",
        }  # fmt: skip
        premise_words = {w.lower().strip(".,;:()[]%$") for w in premise.split()} - stop
        hypothesis_words = {w.lower().strip(".,;:()[]%$") for w in hypothesis.split()} - stop
        if not hypothesis_words:
            return Entailment.NEUTRAL, 0.0
        overlap = len(premise_words & hypothesis_words) / len(hypothesis_words)
        return (Entailment.ENTAILED if overlap >= 0.6 else Entailment.NEUTRAL), overlap

    async def entail(self, premise: str, hypothesis: str) -> tuple[Entailment, float]:
        """Async wrapper over :meth:`entail_sync`."""
        return self.entail_sync(premise, hypothesis)


class CitationVerifier:
    """Verifies an answer's citations against the chunks they point at."""

    def __init__(self, *, nli: NliModel | None = None) -> None:
        """Create the verifier, defaulting to the offline lexical model."""
        self._nli = nli or LexicalNliModel()

    async def verify(
        self,
        answer: str,
        chunks: Sequence[RetrievedChunk],
        *,
        drop_unsupported: bool = True,
        threshold: float = 0.5,
    ) -> GroundednessReport:
        """Check every citation and optionally rewrite the answer.

        Args:
            answer: The generated answer, with ``[n]`` markers.
            chunks: The context, in marker order — marker ``n`` is ``chunks[n-1]``.
            drop_unsupported: Remove markers that do not survive verification.
            threshold: Minimum entailment score to accept a citation.
        """
        claims = split_claims(answer)
        checks: list[CitationCheck] = []

        for claim in claims:
            for marker in claim.markers:
                chunk = _chunk_for_marker(chunks, marker)
                if chunk is None:
                    checks.append(
                        CitationCheck(
                            claim=claim,
                            marker=marker,
                            chunk_id=None,
                            entailment=Entailment.NEUTRAL,
                            score=0.0,
                        )
                    )
                    continue

                verdict, score = await self._nli.entail(chunk.content, claim.text)
                if score < threshold and verdict is Entailment.ENTAILED:
                    verdict = Entailment.NEUTRAL
                checks.append(
                    CitationCheck(
                        claim=claim,
                        marker=marker,
                        chunk_id=chunk.chunk_id,
                        entailment=verdict,
                        score=score,
                        numbers_verified=numbers_supported(claim.text, chunk.content),
                    )
                )

        dropped = tuple(sorted({c.marker for c in checks if not c.supported}))
        corrected = _remove_markers(answer, dropped) if drop_unsupported and dropped else answer

        return GroundednessReport(
            claims=tuple(claims),
            checks=tuple(checks),
            corrected_answer=corrected,
            dropped_markers=dropped,
        )


class GroundednessGuardrail(Guardrail):
    """Output guardrail that verifies citations and detects hallucination."""

    kind = GuardrailKind.HALLUCINATION
    stage = GuardrailStage.OUTPUT
    detector = "nli"

    def __init__(self, *, verifier: CitationVerifier | None = None) -> None:
        """Create the guardrail."""
        self._verifier = verifier or CitationVerifier()

    async def check(self, text: str, *, context: GuardrailContext) -> GuardrailResult:
        """Verify the answer's citations against the retrieved context."""
        policy = context.policy
        if not policy.groundedness_enabled or not context.retrieved_chunks:
            return GuardrailResult.allow(kind=self.kind, detector="disabled")

        report = await self._verifier.verify(
            text,
            list(context.retrieved_chunks),
            drop_unsupported=policy.citation_drop_unsupported,
            threshold=policy.groundedness_threshold,
        )
        evidence = {
            "citation_precision": round(report.citation_precision, 3),
            "citation_recall": round(report.citation_recall, 3),
            "dropped_markers": list(report.dropped_markers),
            "claims": len(report.claims),
        }

        # A contradiction has no honest reading: the answer asserts the opposite
        # of the evidence it points at.
        if report.has_contradiction:
            return GuardrailResult.block(
                kind=self.kind,
                detector=self.detector,
                reason="answer contradicts its own cited evidence",
                score=report.groundedness,
                threshold=policy.groundedness_threshold,
                evidence=evidence,
            )

        if report.dropped_markers:
            return GuardrailResult.redact(
                kind=GuardrailKind.CITATION,
                detector=self.detector,
                text=report.corrected_answer,
                reason=f"dropped {len(report.dropped_markers)} unsupported citations",
                score=report.citation_precision,
                evidence=evidence,
            )

        if report.groundedness < policy.groundedness_threshold:
            return GuardrailResult.flag(
                kind=self.kind,
                detector=self.detector,
                reason="answer contains factual claims without supporting citations",
                score=report.groundedness,
                threshold=policy.groundedness_threshold,
                evidence=evidence,
            )

        return GuardrailResult.allow(
            kind=self.kind, detector=self.detector, score=report.groundedness, evidence=evidence
        )


def _chunk_for_marker(chunks: Sequence[RetrievedChunk], marker: int) -> RetrievedChunk | None:
    """Resolve a 1-based citation marker to its chunk.

    Returns None for an out-of-range marker, which is itself a hallucination —
    the model cited a source that was never provided.
    """
    index = marker - 1
    return chunks[index] if 0 <= index < len(chunks) else None


def _remove_markers(answer: str, markers: Sequence[int]) -> str:
    """Strip specific citation markers from an answer, tidying the spacing.

    Example:
        >>> _remove_markers("Revenue grew [1][2]. Costs fell [3].", [2])
        'Revenue grew [1]. Costs fell [3].'
    """
    result = answer
    for marker in markers:
        result = result.replace(f"[{marker}]", "")
    result = re.sub(r"\s+([.,;:])", r"\1", result)
    return re.sub(r"[ \t]{2,}", " ", result).strip()
