# Deploying the demo

The frontend can be deployed on its own, with no backend, no database and no
provider key. It runs on fixture data and says so on every screen.

This exists because the alternative is worse. The full system is nine services —
Postgres, OpenSearch, Neo4j, Redis, MinIO, Prometheus, Grafana, the API and the
web app — and standing that up publicly costs real money, needs a provider key
that strangers would be spending, and still leaves a URL that is down more often
than a portfolio link should be. A frontend deployed alone without demo mode is
worse still: every page renders an error, which demonstrates nothing.

## What demo mode is

`NEXT_PUBLIC_AGRAG_DEMO=1` at build time swaps the API client for fixtures in
[`apps/web/src/lib/demo.ts`](../apps/web/src/lib/demo.ts). The interface is
entirely real — the same components, the same streaming code, the same Zod
schemas, the same charts. Only the data is canned.

It covers the five conversations from [DEMO.md](DEMO.md), the document list
(including a failed document, because a demo where everything succeeds hides
the error states), the retrieval playground comparison, the graph, and every
admin panel.

Three things it deliberately does not do:

- **It does not hide what it is.** Every page carries a non-dismissible banner
  saying the data is fixture data and the figures are illustrative.
- **It does not fake writes.** Deleting a document returns 403 "This demo is
  read-only" rather than appearing to work and reappearing on reload.
- **It does not present numbers as measurements.** The eval and cost figures are
  shaped like real output so the charts have something to draw. Real numbers come
  from `python -m evals.run` against a running stack, and are never typed in.
  The model label reads `claude-sonnet-5 (fixture)`, not `claude-sonnet-5`.

## Deploying to Vercel

```bash
npm i -g vercel
cd apps/web
vercel            # first run links the project
vercel --prod
```

[`vercel.json`](../apps/web/vercel.json) sets `NEXT_PUBLIC_AGRAG_DEMO=1` for the
build, so there is nothing to configure in the dashboard. `next.config.ts` drops
`output: "standalone"` when Vercel sets `VERCEL=1`, because Vercel builds its own
serverless output and would otherwise warn that the setting is ignored.

Every route is statically prerendered, so this costs nothing to serve and cannot
cold-start.

To deploy from the repository root instead of `apps/web`, set the project's root
directory to `apps/web` in the Vercel dashboard — the monorepo has no workspace
configuration tying the two together.

## Running demo mode locally

```bash
cd apps/web
NEXT_PUBLIC_AGRAG_DEMO=1 npm run build
NEXT_PUBLIC_AGRAG_DEMO=1 npm start
```

Useful on a machine that cannot host the full stack. On 8GB of RAM the compose
stack will not fit — OpenSearch and Neo4j each want a JVM heap — and this is the
way to see the interface anyway.

## Regenerating the screenshots

The images in [screenshots/](screenshots/) are captured from this build, not
mocked up. The capture script is not committed; it is 60 lines of Playwright
that visits each page, asks the five questions, waits for the fixture stream to
render, and screenshots at `deviceScaleFactor: 2`.

## Deploying the real thing

Not done, and [DEMO.md](DEMO.md) says so rather than claiming a URL that does not
exist. [`.github/workflows/deploy.yml`](../.github/workflows/deploy.yml) is
written and tested as a workflow — canary at 10/50/100 with automatic rollback,
Fly.io for the API, Vercel for the frontend, managed stores for the rest — but it
has never been run. Standing it up needs accounts and secrets that only the
repository owner can create.

If you want to: Neon's free tier supports pgvector, Upstash covers Redis, and the
system degrades to dense-only retrieval without OpenSearch. The API is the part
that costs money, and the provider key is the part that costs money unpredictably.
