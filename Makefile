# AgenticRAG developer entrypoints. `make help` lists everything.
SHELL := /bin/bash
COMPOSE := docker compose
API := apps/api

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sort | awk 'BEGIN{FS=":.*?## "}{printf "\033[36m%-24s\033[0m %s\n", $$1, $$2}'

## ── Environment ────────────────────────────────────────────────────────────
.PHONY: env
env: ## Create .env from .env.example if missing
	@test -f .env || (cp .env.example .env && echo "created .env — fill in provider keys")

.PHONY: up
up: env ## Start the core stack (data plane + api + worker + web)
	$(COMPOSE) up -d --build
	@$(MAKE) --no-print-directory wait
	@echo "web http://localhost:3000 | api http://localhost:8000/docs | minio http://localhost:9001"

.PHONY: up-all
up-all: env ## Start everything (core + observability + MCP servers)
	$(COMPOSE) --profile all up -d --build
	@$(MAKE) --no-print-directory wait
	@echo "grafana http://localhost:3001 | jaeger http://localhost:16686 | langfuse http://localhost:3002"

.PHONY: wait
wait: ## Block until the API reports healthy
	@bash scripts/wait_for_stack.sh

.PHONY: down
down: ## Stop the stack (keeps volumes)
	$(COMPOSE) --profile all down

.PHONY: nuke
nuke: ## Stop the stack and delete all volumes (destructive)
	$(COMPOSE) --profile all down -v

.PHONY: logs
logs: ## Tail api + worker logs
	$(COMPOSE) logs -f api worker

## ── Database ───────────────────────────────────────────────────────────────
.PHONY: migrate
migrate: ## Apply Alembic migrations
	$(COMPOSE) exec api alembic upgrade head

.PHONY: migration
migration: ## Autogenerate a migration: make migration m="add foo"
	$(COMPOSE) exec api alembic revision --autogenerate -m "$(m)"

.PHONY: seed
seed: ## Load the demo tenant (papers, handbook, sales CSV, knowledge graph)
	$(COMPOSE) exec api python -m scripts.seed_demo_tenant

## ── Quality ────────────────────────────────────────────────────────────────
.PHONY: fmt
fmt: ## Format Python + TS
	cd $(API) && ruff format . && ruff check --fix .
	cd apps/web && pnpm format

.PHONY: lint
lint: ## Lint everything
	cd $(API) && ruff check . && mypy src
	cd apps/web && pnpm lint && pnpm typecheck

.PHONY: test
test: doctest ## Backend unit tests with coverage gate
	cd $(API) && pytest tests/unit -q --cov=src --cov-report=term-missing

.PHONY: doctest
doctest: ## Run the examples embedded in every public docstring
	cd $(API) && pytest --doctest-modules src/core src/models src/ingestion src/retrieval -q

.PHONY: test-integration
test-integration: ## Integration tests (testcontainers; needs Docker)
	cd $(API) && pytest tests/integration -q

.PHONY: test-e2e
test-e2e: ## Playwright end-to-end tests against the local stack
	cd apps/web && pnpm exec playwright test

.PHONY: security
security: ## bandit + semgrep + pip-audit
	cd $(API) && bandit -q -c pyproject.toml -r src && pip-audit -r requirements.txt || true
	semgrep --config auto --error apps mcp-servers

## ── Evals ──────────────────────────────────────────────────────────────────
.PHONY: eval-golden
eval-golden: ## Run the golden set and write an HTML report
	cd $(API) && python -m evals.run --set golden

.PHONY: eval-all
eval-all: ## Run golden + regression + adversarial sets
	cd $(API) && python -m evals.run --set golden --set regression --set adversarial

.PHONY: calibrate
calibrate: ## Recompute judge calibration coefficients
	cd $(API) && python -m evals.scripts.calibrate_judges

## ── Load & chaos ───────────────────────────────────────────────────────────
.PHONY: load
load: ## k6 load test (50 VUs mixed workload)
	k6 run scripts/load_test.js

.PHONY: chaos
chaos: ## Kill MCP servers / inject latency / corrupt Redis, assert graceful degradation
	python scripts/chaos.py --duration 300
