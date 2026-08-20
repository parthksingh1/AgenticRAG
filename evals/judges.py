"""The LLM-as-judge harness.

Some things cannot be measured without a model reading the answer — whether it is
grounded in the retrieved context, whether it actually answers the question asked
— and those are exactly the things that matter most. The problem is that a judge
is another language model, with its own biases, and a single judge's score is an
opinion dressed as a measurement.

Three mechanisms turn it back into something usable:

**Two judges from different providers.** A judge from the same family as the
generator scores its own family's phrasing generously. Two independent judges make
that bias visible instead of invisible.

**Disagreement is surfaced, not averaged away.** When the judges differ by more
than :data:`DISAGREEMENT_THRESHOLD` the case is flagged for a human. Those flagged
cases are the calibration set, which is what makes the whole apparatus honest —
the judges are checked against people, periodically, with the result written down.

**Scores are weighted by measured calibration.** A judge with a high expected
calibration error counts for a fraction of a vote. See
:mod:`src.services.calibration` for the maths.

A judge that fails — rate limit, outage, unparseable output — is dropped from that
case rather than scored as zero. A provider blip is not evidence that an answer is
bad, and treating it as such would make the suite's numbers move with the weather.

Example:
    >>> parse_verdict('{"score": 0.8, "reasoning": "grounded"}')["score"]
    0.8
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

#: Judges differing by more than this on a 0-1 scale go to a human. Chosen so
#: that "both roughly agree it is good" does not queue work, while a genuine
#: split does.
DISAGREEMENT_THRESHOLD = 0.25

#: What each judge is asked to score. Named here rather than in the prompt so a
#: dimension cannot be added to the prompt and silently ignored by the parser.
DIMENSIONS = ("groundedness", "relevance", "completeness", "citation_quality")

#: A judge's own temperature. Zero: a judge that gives a different score to the
#: same answer on two runs makes every comparison between runs meaningless.
JUDGE_TEMPERATURE = 0.0

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "score": {"type": "number", "minimum": 0, "maximum": 1},
        "groundedness": {"type": "number", "minimum": 0, "maximum": 1},
        "relevance": {"type": "number", "minimum": 0, "maximum": 1},
        "completeness": {"type": "number", "minimum": 0, "maximum": 1},
        "citation_quality": {"type": "number", "minimum": 0, "maximum": 1},
        "reasoning": {"type": "string"},
    },
    "required": ["score", "reasoning"],
}


@dataclass(frozen=True, slots=True)
class Verdict:
    """One judge's assessment of one answer."""

    judge: str
    score: float
    dimensions: dict[str, float]
    reasoning: str

    def to_dict(self) -> dict[str, Any]:
        """Render for storage in ``eval_case_results.judge_scores``."""
        return {
            "score": self.score,
            "dimensions": self.dimensions,
            # Truncated: the reasoning is for a human reading one case, not a
            # field anyone aggregates, and full reasoning across 340 cases makes
            # the JSON report unopenable.
            "reasoning": self.reasoning[:1000],
        }


@dataclass(frozen=True, slots=True)
class Panel:
    """The combined verdict of every judge that answered."""

    verdicts: tuple[Verdict, ...]
    combined_score: float
    disagreed: bool
    spread: float

    @property
    def dimension_means(self) -> dict[str, float]:
        """Mean of each dimension across the judges that scored it.

        Example:
            >>> Panel((), 0.0, False, 0.0).dimension_means
            {}
        """
        out: dict[str, float] = {}
        for dimension in DIMENSIONS:
            values = [v.dimensions[dimension] for v in self.verdicts if dimension in v.dimensions]
            if values:
                out[dimension] = round(sum(values) / len(values), 4)
        return out

    def to_dict(self) -> dict[str, Any]:
        """Render the whole panel for storage."""
        return {
            **{v.judge: v.to_dict() for v in self.verdicts},
            "_combined": self.combined_score,
            "_spread": self.spread,
            "_disagreed": self.disagreed,
        }


def parse_verdict(payload: str) -> dict[str, Any]:
    """Parse a judge's JSON response.

    Returns an empty dict on anything unparseable, which the caller treats as
    "this judge did not answer" rather than as a zero score.

    Example:
        >>> parse_verdict('{"score": 0.9, "reasoning": "ok"}')["score"]
        0.9
        >>> parse_verdict("I think it's pretty good actually")
        {}
    """
    cleaned = _FENCE.sub("", payload.strip())
    try:
        data = json.loads(cleaned)
    except (ValueError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def build_verdict(judge: str, payload: str) -> Verdict | None:
    """Turn a judge's raw response into a verdict, or None if it is unusable.

    Example:
        >>> build_verdict("gpt", '{"score": 0.8, "groundedness": 1.0, "reasoning": "x"}').score
        0.8
        >>> build_verdict("gpt", "not json") is None
        True
    """
    data = parse_verdict(payload)
    score = _clamp(data.get("score"))
    if score is None:
        return None

    dimensions = {}
    for dimension in DIMENSIONS:
        value = _clamp(data.get(dimension))
        if value is not None:
            dimensions[dimension] = value

    return Verdict(
        judge=judge,
        score=score,
        dimensions=dimensions,
        reasoning=str(data.get("reasoning", "")),
    )


def combine(verdicts: Sequence[Verdict], weights: dict[str, float] | None = None) -> Panel:
    """Combine verdicts into a panel score, flagging disagreement.

    Args:
        verdicts: Every judge that answered. Judges that failed are simply absent.
        weights: Calibration weights by judge model. An absent judge weighs 1.0,
            because a judge with no calibration yet should not be silently
            downweighted.

    Example:
        >>> panel = combine([Verdict("a", 0.9, {}, ""), Verdict("b", 0.85, {}, "")])
        >>> panel.disagreed, panel.combined_score
        (False, 0.875)
        >>> combine([Verdict("a", 1.0, {}, ""), Verdict("b", 0.2, {}, "")]).disagreed
        True
        >>> combine([]).combined_score
        0.0
    """
    if not verdicts:
        return Panel((), 0.0, disagreed=False, spread=0.0)

    from src.services.calibration import combine as weighted_combine

    scores = {v.judge: v.score for v in verdicts}
    combined = weighted_combine(scores, weights or {})
    spread = round(max(scores.values()) - min(scores.values()), 4)

    return Panel(
        verdicts=tuple(verdicts),
        combined_score=combined,
        disagreed=spread > DISAGREEMENT_THRESHOLD,
        spread=spread,
    )


def build_prompt(
    *, question: str, answer: str, context: Sequence[str], reference: str | None
) -> str:
    """Build the instruction sent to a judge.

    Two properties matter here and are easy to get wrong.

    The judge is told to score **groundedness in the provided context**, not
    correctness in general. A judge scoring on world knowledge will mark down a
    correct quotation from a document it disagrees with, which measures the
    judge's priors rather than the system.

    The reference answer is offered as *one* acceptable answer rather than as the
    target. Presented as the target, a judge penalises every correct answer that
    is phrased differently, which is most of them.
    """
    numbered = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(context, start=1)) or "(none)"
    reference_block = (
        (
            "\nOne acceptable answer (not the only one; do not require "
            f"matching phrasing):\n{reference}\n"
        )
        if reference
        else ""
    )

    return (
        "You are grading a retrieval-augmented answer. Score only what is asked.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"RETRIEVED CONTEXT:\n{numbered}\n"
        f"{reference_block}\n"
        f"ANSWER:\n{answer}\n\n"
        "Score each dimension from 0.0 to 1.0:\n"
        "- groundedness: is every claim supported by the retrieved context? Judge "
        "support by the context, not by your own knowledge. A correct-sounding "
        "claim the context does not support scores low.\n"
        "- relevance: does it answer the question that was asked?\n"
        "- completeness: does it cover what the context supports, without padding?\n"
        "- citation_quality: do the inline markers point at the passages that "
        "actually support the claims they follow?\n\n"
        "Then give an overall score. If the context does not support an answer, a "
        "refusal is correct and should score high.\n\n"
        'Return JSON only: {"score": float, "groundedness": float, "relevance": '
        'float, "completeness": float, "citation_quality": float, "reasoning": '
        '"one or two sentences"}'
    )


async def judge_answer(
    *,
    router: Any,
    judge_models: Sequence[str],
    question: str,
    answer: str,
    context: Sequence[str],
    reference: str | None = None,
    weights: dict[str, float] | None = None,
) -> Panel:
    """Ask every judge for a verdict and combine them.

    Judges are called concurrently: they are independent, and running them in
    sequence doubles the wall-clock time of every eval run for no benefit.
    """
    import asyncio

    from src.services.llm.types import CompletionRequest, Message

    prompt = build_prompt(question=question, answer=answer, context=context, reference=reference)

    async def ask(model: str) -> Verdict | None:
        """Get one judge's verdict, or None if it could not be obtained."""
        try:
            completion = await router.complete(
                CompletionRequest(
                    messages=(Message(role="user", content=prompt),),
                    model=model,
                    temperature=JUDGE_TEMPERATURE,
                    max_tokens=800,
                    response_schema=JUDGE_SCHEMA,
                    node="judge",
                ),
                allow_fallback=False,
            )
        except Exception:  # noqa: BLE001 - a failed judge abstains, it does not score zero
            return None
        return build_verdict(model, completion.content)

    verdicts = await asyncio.gather(*(ask(model) for model in judge_models))
    return combine([v for v in verdicts if v is not None], weights)


def _clamp(value: Any) -> float | None:
    """Coerce a judge's number into [0, 1], or None if it is not a number.

    Judges regularly return 8 when asked for a score between 0 and 1. Values
    between 1 and 10 are rescaled rather than clamped to 1.0, because clamping
    would silently turn "8 out of 10" into a perfect score.

    Example:
        >>> _clamp(0.8), _clamp(8), _clamp(1), _clamp("x"), _clamp(None)
        (0.8, 0.8, 1.0, None, None)
    """
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None

    if 1.0 < number <= 10.0:
        number /= 10.0
    return round(min(1.0, max(0.0, number)), 4)
