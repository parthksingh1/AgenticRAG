# Terraform

Provisions the managed stack: Neon (Postgres + pgvector), Upstash (Redis),
Fly.io (API) and Vercel (frontend).

```bash
terraform init -backend-config=backend.hcl
terraform plan  -var-file=production.tfvars
terraform apply -var-file=production.tfvars
```

`backend.hcl` and `*.tfvars` are gitignored — they hold the state bucket and the
provider keys.

## Why managed services

Running Postgres, Redis, OpenSearch and Neo4j yourself is four backup
strategies, four upgrade paths and four things to be paged about, for a system
whose interesting parts are none of those.

OpenSearch and Neo4j are **not** provisioned here. Both are optional: without
them retrieval degrades to dense-only and GraphRAG is unavailable, which
`/readyz` reports. Add Bonsai and Neo4j Aura when the corpus is large enough for
BM25 to earn its cost.

## Two things the module refuses to let you do

**A `:latest` image tag.** A machine rescheduled at 3am must run the same code as
the one it replaced, and `latest` cannot promise that. The variable validation
rejects it.

**One machine.** `min_machines` must be at least 2, because a rolling deploy
needs somewhere to send traffic and a host event should not be an outage.

## After apply

```bash
alembic upgrade head                      # uses the database_url output
python scripts/seed_demo_tenant.py
python scripts/smoke_test.py --url $(terraform output -raw api_url)
```

The smoke test is not optional. Every step above can succeed while the first
real request returns a 500 — that is exactly the failure it exists to catch.
