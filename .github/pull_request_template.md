# What and why

<!-- One paragraph. What changes, and what problem it solves. Link the issue. -->

## How it was verified

<!-- Not "tests pass" — what you actually checked, and how someone else could
     check it. If a number appears in this PR, say which command produces it. -->

- [ ] Unit tests cover the new behaviour, including the failure paths
- [ ] Ran `python -m evals.run --set golden` and the gate passed
- [ ] Tenant isolation is unaffected, or a test proves the new path is scoped

## Risk

<!-- What breaks if this is wrong, and how it would be noticed. -->

- [ ] The migration is backward compatible with the running code (a canary runs
      both at once against one database)
- [ ] No new secret, credential or PII reaches a log, an error message or the
      audit trail
- [ ] Any new outbound request goes through `src.core.net.validate_public_url`
- [ ] Any new LLM call has a timeout, retry and fallback

## Checklist

- [ ] Every public function has a docstring with an example
- [ ] Any non-obvious decision is either commented at the point of the decision
      or written up as an ADR under `docs/adr/`
- [ ] No number in the README, the docs or a dashboard is asserted without a
      runnable command under `evals/` or `scripts/` that produces it
