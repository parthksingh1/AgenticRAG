# ADR 0002 — Reciprocal rank fusion, not weighted score fusion

**Status:** accepted · **Date:** 2026-08-19

## Context

Dense retrieval returns cosine similarities in roughly [0, 1]. BM25 returns
unbounded scores whose scale depends on the corpus, the query length and the
index statistics. Combining them means either normalising both onto a common
scale or ignoring the magnitudes entirely.

Min-max normalisation is the usual first attempt, and it is unstable: a
document's normalised score depends on the other documents in the result set, so
the same document scores differently depending on what else came back. A query
returning one strong result and nine weak ones normalises very differently from
one returning ten mediocre ones.

## Decision

Fuse on rank, not score:

```
score(d) = sum over retrievers i of  1 / (k + rank_i(d)),   k = 60
```

`k = 60` is the value from Cormack et al. (2009). It is kept because there is no
evidence for a better one on this corpus, not because it is optimal.

Weighted score fusion is implemented alongside and available per tenant, for
anyone who has actually measured that it helps on their data.

## Consequences

**Good.** No normalisation, no per-corpus tuning, and stability under changes to
either retriever. A third retriever can be added without recalibrating anything.

**Bad.** Rank fusion discards magnitude. A document that is overwhelmingly the
best dense match contributes exactly as much as one that is marginally best. In
practice the cross-encoder reranker recovers this, because it re-scores the fused
top 50 by reading query and document together.

**On changing `k`:** it belongs in the eval suite as an experiment, not in a
config file as an opinion. `evals/` can answer whether a different value helps on
a given corpus.

## Verification

`apps/api/src/retrieval/fusion.py` is held at 100% line and branch coverage, with
property tests asserting that fusion is order-independent, that a document ranked
first by both retrievers always outranks one ranked first by only one, and that
per-document capping cannot starve a source.
