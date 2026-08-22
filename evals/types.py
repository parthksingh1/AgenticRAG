"""Eval case and result types.

The shapes here are deliberately explicit rather than dictionaries. An eval
harness whose cases are untyped dicts fails at the worst possible moment: months
later, when a typo in a field name has been silently scoring every case as
passing and the baseline is built on it.

A case declares what a good answer looks like from several angles — the expected
answer, the sources that must be cited, substrings that must appear, and whether
refusing is the right behaviour. No single one of those is sufficient. Checking
only string overlap rewards an answer that copies the right words from the wrong
document; checking only citations rewards an answer that cites correctly and says
something else.

Example:
    >>> case = EvalCase(id="g001", query="What is the refund window?", intent="factual")
    >>> case.expects_refusal
    False
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class Intent(StrEnum):
    """What kind of question a case asks.

    The golden set is stratified by these, because a suite that is 80% simple
    factual lookups reports an average that hides the fact that multi-hop
    questions never worked.
    """

    FACTUAL = "factual"
    MULTI_HOP = "multi_hop"
    COMPARATIVE = "comparative"
    SUMMARISATION = "summarisation"
    ANALYTICAL = "analytical"
    TOOL_USE = "tool_use"
    UNANSWERABLE = "unanswerable"
    CONVERSATIONAL = "conversational"


class Difficulty(StrEnum):
    """How hard a case is expected to be."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class AttackKind(StrEnum):
    """What an adversarial case is attempting."""

    PROMPT_INJECTION = "prompt_injection"
    JAILBREAK = "jailbreak"
    PII_EXTRACTION = "pii_extraction"
    DATA_EXFILTRATION = "data_exfiltration"
    TOOL_ABUSE = "tool_abuse"
    HALLUCINATION_BAIT = "hallucination_bait"
    TENANT_ESCAPE = "tenant_escape"


@dataclass(frozen=True, slots=True)
class EvalCase:
    """One graded question.

    Attributes:
        id: Stable identifier. Never reused, because baselines and regression
            history are keyed on it.
        query: The question as a user would ask it.
        intent: What kind of question this is, for stratified reporting.
        difficulty: Expected difficulty.
        expected_answer: A reference answer, used by the judges as context rather
            than as an exact target. Answers are prose; string equality is not a
            grading strategy.
        expected_sources: Document or chunk ids that a correct answer must cite.
            The basis of citation precision and recall.
        must_include: Substrings that must appear. Reserved for facts with one
            correct form — a number, a date, a proper noun.
        must_not_include: Substrings that must not appear. Usually the plausible
            wrong answer, which is what a hallucinating model reaches for.
        expects_refusal: Whether refusing is the correct behaviour. An
            unanswerable question answered confidently is a failure even when the
            answer sounds good.
        attack: Set on adversarial cases.
        tags: Free-form labels for slicing a report.
    """

    id: str
    query: str
    intent: Intent | str = Intent.FACTUAL
    difficulty: Difficulty | str = Difficulty.MEDIUM
    expected_answer: str | None = None
    expected_sources: tuple[str, ...] = ()
    must_include: tuple[str, ...] = ()
    must_not_include: tuple[str, ...] = ()
    expects_refusal: bool = False
    attack: AttackKind | str | None = None
    conversation: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    notes: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> EvalCase:
        """Build a case from its JSONL form.

        Raises:
            ValueError: when a required field is missing. A case with no id or no
                query cannot be graded or tracked, and skipping it silently would
                shrink the suite without anyone noticing.

        Example:
            >>> EvalCase.from_dict({"id": "g1", "query": "hi"}).intent
            'factual'
        """
        for required in ("id", "query"):
            if not raw.get(required):
                msg = f"eval case is missing {required!r}: {raw}"
                raise ValueError(msg)

        return cls(
            id=str(raw["id"]),
            query=str(raw["query"]),
            intent=str(raw.get("intent", Intent.FACTUAL)),
            difficulty=str(raw.get("difficulty", Difficulty.MEDIUM)),
            expected_answer=raw.get("expected_answer"),
            expected_sources=tuple(raw.get("expected_sources") or ()),
            must_include=tuple(raw.get("must_include") or ()),
            must_not_include=tuple(raw.get("must_not_include") or ()),
            expects_refusal=bool(raw.get("expects_refusal", False)),
            attack=raw.get("attack"),
            conversation=tuple(raw.get("conversation") or ()),
            tags=tuple(raw.get("tags") or ()),
            notes=raw.get("notes"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Render the case back to its JSONL form, omitting empty fields."""
        out: dict[str, Any] = {
            "id": self.id,
            "query": self.query,
            "intent": str(self.intent),
        }
        for key, value in (
            ("difficulty", str(self.difficulty)),
            ("expected_answer", self.expected_answer),
            ("expected_sources", list(self.expected_sources)),
            ("must_include", list(self.must_include)),
            ("must_not_include", list(self.must_not_include)),
            ("expects_refusal", self.expects_refusal or None),
            ("attack", str(self.attack) if self.attack else None),
            ("conversation", list(self.conversation)),
            ("tags", list(self.tags)),
            ("notes", self.notes),
        ):
            if value:
                out[key] = value
        return out


@dataclass(slots=True)
class CaseResult:
    """What happened when one case was run."""

    case: EvalCase
    answer: str = ""
    citations: tuple[str, ...] = ()
    retrieved_sources: tuple[str, ...] = ()
    refused: bool = False
    metrics: dict[str, float] = field(default_factory=dict)
    judge_scores: dict[str, Any] = field(default_factory=dict)
    calibrated_score: float | None = None
    judges_disagreed: bool = False
    passed: bool = False
    failure_mode: str | None = None
    latency_ms: int = 0
    cost_usd: float = 0.0
    trace_id: str | None = None
    error: str | None = None

    @property
    def intent(self) -> str:
        """The case's intent, for grouping.

        Example:
            >>> CaseResult(case=EvalCase(id="g1", query="q")).intent
            'factual'
        """
        return str(self.case.intent)

    def to_dict(self) -> dict[str, Any]:
        """Render for the JSON report and the database row."""
        return {
            "case_id": self.case.id,
            "intent": self.intent,
            "difficulty": str(self.case.difficulty),
            "query": self.case.query,
            "expected_answer": self.case.expected_answer,
            "actual_answer": self.answer,
            "expected_sources": list(self.case.expected_sources),
            "retrieved_sources": list(self.retrieved_sources),
            "citations": list(self.citations),
            "refused": self.refused,
            "metrics": self.metrics,
            "judge_scores": self.judge_scores,
            "calibrated_score": self.calibrated_score,
            "judges_disagreed": self.judges_disagreed,
            "passed": self.passed,
            "failure_mode": self.failure_mode,
            "latency_ms": self.latency_ms,
            "cost_usd": round(self.cost_usd, 6),
            "trace_id": self.trace_id,
            "error": self.error,
        }


@dataclass(slots=True)
class RunResult:
    """The outcome of one eval run over one set."""

    set_name: str
    set_version: str
    model: str
    results: list[CaseResult] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    metrics_by_intent: dict[str, dict[str, float]] = field(default_factory=dict)
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    git_sha: str | None = None
    prompt_version: str | None = None
    total_cost_usd: float = 0.0

    @property
    def passed_count(self) -> int:
        """How many cases passed."""
        return sum(1 for r in self.results if r.passed)

    @property
    def failed_count(self) -> int:
        """How many cases failed."""
        return len(self.results) - self.passed_count

    @property
    def pass_rate(self) -> float:
        """Fraction of cases that passed.

        Example:
            >>> RunResult(set_name="golden", set_version="v1", model="m").pass_rate
            0.0
        """
        if not self.results:
            return 0.0
        return round(self.passed_count / len(self.results), 4)

    @property
    def duration_seconds(self) -> float:
        """Wall-clock duration of the run."""
        end = self.finished_at or datetime.now(UTC)
        return round((end - self.started_at).total_seconds(), 2)

    def to_dict(self) -> dict[str, Any]:
        """Render the whole run for the JSON report."""
        return {
            "set_name": self.set_name,
            "set_version": self.set_version,
            "model": self.model,
            "git_sha": self.git_sha,
            "prompt_version": self.prompt_version,
            "case_count": len(self.results),
            "passed": self.passed_count,
            "failed": self.failed_count,
            "pass_rate": self.pass_rate,
            "metrics": self.metrics,
            "metrics_by_intent": self.metrics_by_intent,
            "total_cost_usd": round(self.total_cost_usd, 4),
            "duration_seconds": self.duration_seconds,
            "started_at": self.started_at.isoformat(),
            "finished_at": (self.finished_at or datetime.now(UTC)).isoformat(),
            "results": [r.to_dict() for r in self.results],
        }
