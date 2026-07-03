# ADR 0003 — Two judges, calibrated against humans

**Status:** accepted · **Date:** 2026-08-27

## Context

Groundedness and relevance cannot be measured without a model reading the answer.
But an LLM judge has its own biases — notably, it scores answers from its own
model family generously — and a single judge's score is an opinion presented as a
measurement. A quality gate built on one is built on sand.

## Decision

Three mechanisms, together:

1. **Two judges from different providers.** Cross-family disagreement makes the
   bias visible instead of invisible.
2. **Disagreement goes to a human.** When the judges differ by more than 0.25 the
   case is flagged for review in the admin UI. Those human labels become the
   calibration set.
3. **Weight by measured calibration.** Each judge's expected calibration error is
   computed against the human labels and its vote weighted `1 / (1 + ECE)`.
   Cohen's kappa and a least-squares recalibration are computed alongside.

Recalibration runs weekly. A judge whose ECE has moved materially opens an issue
rather than silently changing every number after it.

## Consequences

**Good.** Judge scores are auditable — the reliability diagram is in the admin
UI, the calibration history is a table. A judge that drifts is detected rather
than trusted. The disagreement queue sends human labelling effort exactly where
the automated score is least trustworthy, instead of spreading it uniformly.

**Bad.** Two judges is twice the cost per case. And the whole apparatus needs a
supply of human labels to mean anything: below 30, the calibration is flagged
unreliable and the weights stay near uniform — honest, but not yet useful.

**Rejected: one strong judge.** Cheaper and simpler, and it produces a number
nobody can defend when asked how they know the judge is right.

**Rejected: human-only evaluation.** The correct answer for a research paper, and
an impossible one for a suite that must run on every pull request.

## Verification

`apps/api/tests/unit/test_calibration.py` — 38 tests including hypothesis
properties, covering the cases where the statistics are undefined rather than
convenient: kappa with no variance, correlation over a constant series, a fitted
line through a vertical point cloud.
