"""Loading and validating the eval sets.

Sets are JSONL — one case per line — so a diff shows which cases changed rather
than reflowing the whole file, and so a set can be appended to without rewriting
it. That matters because the regression set grows from real failures, one line at
a time.

Loading validates. A malformed case raises rather than being skipped: a suite that
quietly shrinks from 340 cases to 338 reports better numbers for the wrong reason,
and nobody notices until the gate lets a regression through.

Example:
    >>> from evals.datasets import SETS
    >>> sorted(SETS)
    ['adversarial', 'golden', 'regression']
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from evals.types import EvalCase

#: Where each set lives, relative to the repository root.
SETS = {
    "golden": "evals/golden_sets/golden.jsonl",
    "regression": "evals/regression_set/regression.jsonl",
    "adversarial": "evals/adversarial_set/adversarial.jsonl",
}

#: Minimum size for each set, asserted on load. The spec commits to these
#: numbers publicly, so a set that has silently shrunk should fail loudly rather
#: than quietly make the README wrong.
MINIMUM_SIZES = {"golden": 150, "regression": 340, "adversarial": 100}


def repository_root() -> Path:
    """The repository root, found by walking up from this file.

    Example:
        >>> (repository_root() / "evals").is_dir()
        True
    """
    return Path(__file__).resolve().parents[1]


def path_for(set_name: str) -> Path:
    """Absolute path to a set's JSONL file.

    Raises:
        KeyError: for an unknown set name.

    Example:
        >>> path_for("golden").name
        'golden.jsonl'
    """
    if set_name not in SETS:
        msg = f"unknown eval set {set_name!r}; known sets are {sorted(SETS)}"
        raise KeyError(msg)
    return repository_root() / SETS[set_name]


def load(set_name: str, *, enforce_size: bool = False) -> list[EvalCase]:
    """Load and validate one set.

    Args:
        set_name: One of :data:`SETS`.
        enforce_size: Whether to require the set's committed minimum size. On in
            CI, off locally, so working on a subset is not blocked.

    Raises:
        FileNotFoundError: when the set has not been generated yet.
        ValueError: when a case is malformed, an id repeats, or the set is
            smaller than its committed minimum.

    Example:
        >>> load("golden") and True  # doctest: +SKIP
        True
    """
    path = path_for(set_name)
    if not path.exists():
        msg = (
            f"{path} does not exist. Generate the sets with "
            f"`python -m evals.scripts.build_sets` before running evals."
        )
        raise FileNotFoundError(msg)

    cases: list[EvalCase] = []
    seen: set[str] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        try:
            raw = json.loads(stripped)
        except ValueError as exc:
            msg = f"{path.name}:{number} is not valid JSON: {exc}"
            raise ValueError(msg) from exc

        case = EvalCase.from_dict(raw)
        if case.id in seen:
            # Duplicate ids silently overwrite each other in every downstream
            # keyed structure — baselines, per-case history, the report.
            msg = f"{path.name}:{number} repeats case id {case.id!r}"
            raise ValueError(msg)
        seen.add(case.id)
        cases.append(case)

    if enforce_size:
        minimum = MINIMUM_SIZES.get(set_name, 0)
        if len(cases) < minimum:
            msg = f"the {set_name} set has {len(cases)} cases; at least {minimum} are required"
            raise ValueError(msg)

    return cases


def append(set_name: str, case: EvalCase) -> None:
    """Append one case to a set.

    Used by the failure explorer to grow the regression set. Appending rather
    than rewriting keeps the diff to a single line, which is what makes a
    reviewer actually read it.

    Raises:
        ValueError: when the id already exists in the set.
    """
    existing = {c.id for c in load(set_name)}
    if case.id in existing:
        msg = f"case id {case.id!r} is already in the {set_name} set"
        raise ValueError(msg)

    path = path_for(set_name)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(case.to_dict(), ensure_ascii=False) + "\n")


def write(set_name: str, cases: list[EvalCase]) -> Path:
    """Write a whole set, sorted by id for a stable diff."""
    path = path_for(set_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(case.to_dict(), ensure_ascii=False) for case in sorted(cases, key=lambda c: c.id)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def describe(cases: list[EvalCase]) -> dict[str, object]:
    """Summarise a set's composition.

    Printed at the top of every run so the stratification is visible rather than
    assumed. A suite that has drifted to 80% easy factual lookups reports a
    number that means something different from what it meant last month.

    Example:
        >>> describe([EvalCase(id="a", query="q")])["count"]
        1
    """
    return {
        "count": len(cases),
        "by_intent": dict(sorted(Counter(str(c.intent) for c in cases).items())),
        "by_difficulty": dict(sorted(Counter(str(c.difficulty) for c in cases).items())),
        "expects_refusal": sum(1 for c in cases if c.expects_refusal),
        "by_attack": dict(sorted(Counter(str(c.attack) for c in cases if c.attack).items())),
    }
