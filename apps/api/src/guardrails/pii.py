"""PII detection and redaction.

Presidio does the work when it is available. A regex detector backs it up,
because a PII guardrail that silently does nothing when a dependency is missing
is worse than no guardrail at all — it creates the belief that data is being
protected while it is not. The fallback is narrower but it is real, and every
result records which detector produced it.

Redaction is the default response rather than blocking. A user asking "can you
resend the invoice to jane@acme.com?" has a legitimate question; refusing it
teaches them to work around the system, whereas redacting the address answers
them while keeping the data out of the model and the logs.

Redaction is applied to **both directions**: on input so PII never reaches a
third-party model, and on output so a model that memorised or retrieved PII
cannot hand it back.

Example:
    >>> RegexPiiDetector().detect("mail me at jane@acme.com")[0].entity_type
    'EMAIL_ADDRESS'
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from src.core.logging import get_logger
from src.guardrails.base import Guardrail, GuardrailContext, GuardrailResult
from src.models.telemetry import GuardrailDecision, GuardrailKind, GuardrailStage

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class PiiSpan:
    """One detected piece of personal data."""

    entity_type: str
    start: int
    end: int
    score: float

    def placeholder(self) -> str:
        """The token that replaces this span.

        Example:
            >>> PiiSpan("EMAIL_ADDRESS", 0, 5, 1.0).placeholder()
            '[EMAIL_ADDRESS]'
        """
        return f"[{self.entity_type}]"


#: Patterns for the fallback detector. Deliberately conservative: a false
#: positive redacts something harmless, which is a much cheaper mistake than a
#: false negative leaking a card number, but redacting every number would make
#: the product useless.
_PATTERNS: tuple[tuple[str, str], ...] = (
    ("EMAIL_ADDRESS", r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    ("US_SSN", r"\b\d{3}-\d{2}-\d{4}\b"),
    ("CREDIT_CARD", r"\b(?:\d[ \-]?){13,19}\b"),
    (
        "PHONE_NUMBER",
        r"(?<!\d)(?:\+\d{1,3}[ \-]?)?(?:\(\d{2,4}\)[ \-]?)?"
        r"\d{3}[ \-]\d{3,4}[ \-]?\d{0,4}(?!\d)",
    ),
    ("IBAN_CODE", r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"),
    ("IP_ADDRESS", r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    ("API_KEY", r"\b(?:sk|pk|rk)[-_](?:live|test|prod)?[-_]?[A-Za-z0-9]{16,}\b"),
    ("AWS_ACCESS_KEY", r"\bAKIA[0-9A-Z]{16}\b"),
)  # fmt: skip

_COMPILED = tuple((name, re.compile(pattern)) for name, pattern in _PATTERNS)


class PiiDetector:
    """Finds PII spans in text."""

    name: str

    def detect(
        self, text: str, *, entities: Sequence[str] = (), threshold: float = 0.5
    ) -> list[PiiSpan]:
        """Return detected spans, sorted by position."""
        raise NotImplementedError


class RegexPiiDetector(PiiDetector):
    """Pattern-based detector used when Presidio is unavailable.

    Catches structured identifiers (emails, cards, keys) reliably. It cannot
    catch names, which is precisely what Presidio's NER is for — so a deployment
    running on this fallback is recorded as such in every guardrail event rather
    than appearing equivalent.
    """

    name = "regex"

    def detect(
        self, text: str, *, entities: Sequence[str] = (), threshold: float = 0.5
    ) -> list[PiiSpan]:
        """Find PII spans by pattern.

        Example:
            >>> spans = RegexPiiDetector().detect("card 4111 1111 1111 1111 now")
            >>> spans[0].entity_type
            'CREDIT_CARD'
        """
        wanted = set(entities) if entities else None
        spans: list[PiiSpan] = []

        for entity_type, pattern in _COMPILED:
            if wanted is not None and entity_type not in wanted:
                continue
            for match in pattern.finditer(text):
                if entity_type == "CREDIT_CARD" and not _luhn_valid(match.group()):
                    continue
                spans.append(
                    PiiSpan(
                        entity_type=entity_type, start=match.start(), end=match.end(), score=0.9
                    )
                )

        return _resolve_overlaps(spans)


class PresidioPiiDetector(PiiDetector):
    """Microsoft Presidio detector, including NER for names and locations."""

    name = "presidio"

    def __init__(self, *, language: str = "en") -> None:
        """Configure the detector without loading the NLP engine."""
        self._language = language
        self._analyzer: Any | None = None

    def _load(self) -> Any:
        """Load the analyzer on first use."""
        if self._analyzer is None:
            from presidio_analyzer import AnalyzerEngine

            log.info("loading Presidio analyzer")
            self._analyzer = AnalyzerEngine()
        return self._analyzer

    def detect(
        self, text: str, *, entities: Sequence[str] = (), threshold: float = 0.5
    ) -> list[PiiSpan]:
        """Analyse the text with Presidio."""
        analyzer = self._load()
        results = analyzer.analyze(
            text=text,
            language=self._language,
            entities=list(entities) or None,
            score_threshold=threshold,
        )
        return _resolve_overlaps(
            [
                PiiSpan(entity_type=r.entity_type, start=r.start, end=r.end, score=float(r.score))
                for r in results
            ]
        )


def redact(text: str, spans: Sequence[PiiSpan]) -> str:
    """Replace detected spans with typed placeholders.

    Replacement runs back to front so earlier offsets stay valid — replacing
    forwards shifts every subsequent span and corrupts the output.

    Example:
        >>> spans = RegexPiiDetector().detect("write to jane@acme.com today")
        >>> redact("write to jane@acme.com today", spans)
        'write to [EMAIL_ADDRESS] today'
    """
    result = text
    for span in sorted(spans, key=lambda s: s.start, reverse=True):
        result = result[: span.start] + span.placeholder() + result[span.end :]
    return result


class PiiGuardrail(Guardrail):
    """Detects PII and either redacts or blocks, per tenant policy."""

    kind = GuardrailKind.PII

    def __init__(
        self,
        *,
        detector: PiiDetector | None = None,
        stage: GuardrailStage = GuardrailStage.INPUT,
    ) -> None:
        """Create the guardrail, preferring Presidio when it imports."""
        self._detector = detector or _best_available_detector()
        self.stage = stage
        self.detector = self._detector.name

    async def check(self, text: str, *, context: GuardrailContext) -> GuardrailResult:
        """Detect PII and apply the tenant's configured response."""
        policy = context.policy
        if not policy.pii_enabled:
            return GuardrailResult.allow(kind=self.kind, detector="disabled")

        spans = self._detector.detect(
            text, entities=policy.pii_entities, threshold=policy.pii_threshold
        )
        if not spans:
            return GuardrailResult.allow(kind=self.kind, detector=self.detector)

        # Record only the entity types and counts. Storing the matched text would
        # turn the audit trail into the PII store it exists to protect against.
        evidence = {
            "entity_types": sorted({s.entity_type for s in spans}),
            "count": len(spans),
            "detector": self._detector.name,
        }

        if policy.mode_for_pii() is GuardrailDecision.BLOCK:
            return GuardrailResult.block(
                kind=self.kind,
                detector=self.detector,
                reason=f"message contains {len(spans)} PII entities",
                score=max(s.score for s in spans),
                evidence=evidence,
            )

        return GuardrailResult.redact(
            kind=self.kind,
            detector=self.detector,
            text=redact(text, spans),
            reason=f"redacted {len(spans)} PII entities",
            score=max(s.score for s in spans),
            evidence=evidence,
        )


def _best_available_detector() -> PiiDetector:
    """Return Presidio if importable, otherwise the regex fallback."""
    try:
        import presidio_analyzer  # noqa: F401
    except ImportError:
        log.warning(
            "Presidio unavailable; PII detection falls back to patterns and will not catch names"
        )
        return RegexPiiDetector()
    return PresidioPiiDetector()


def _resolve_overlaps(spans: list[PiiSpan]) -> list[PiiSpan]:
    """Drop spans contained within a higher-scoring span.

    Without this, an email inside a longer match produces nested placeholders and
    corrupted output.

    Example:
        >>> a = PiiSpan("PHONE_NUMBER", 0, 12, 0.9)
        >>> b = PiiSpan("CREDIT_CARD", 2, 8, 0.5)
        >>> [s.entity_type for s in _resolve_overlaps([a, b])]
        ['PHONE_NUMBER']
    """
    ordered = sorted(spans, key=lambda s: (s.start, -(s.end - s.start), -s.score))
    kept: list[PiiSpan] = []
    for span in ordered:
        if any(span.start < k.end and span.end > k.start for k in kept):
            continue
        kept.append(span)
    return sorted(kept, key=lambda s: s.start)


def _luhn_valid(candidate: str) -> bool:
    """Whether a digit string passes the Luhn checksum.

    Without this every 16-digit order number is redacted as a credit card, which
    makes the product unusable for anyone in logistics or finance.

    Example:
        >>> _luhn_valid("4111 1111 1111 1111")
        True
        >>> _luhn_valid("1234 5678 9012 3456")
        False
    """
    digits = [int(c) for c in candidate if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for index, digit in enumerate(reversed(digits)):
        if index % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0
