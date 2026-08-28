# Managed infrastructure for AgenticRAG.
#
# Deliberately managed services rather than self-hosted equivalents. Running
# your own Postgres, Redis, OpenSearch and Neo4j is four backup strategies, four
# upgrade paths and four pager rotations — for a system whose interesting parts
# are none of those.
#
#   terraform init
#   terraform plan  -var-file=prod.tfvars
#   terraform apply -var-file=prod.tfvars
#
# State goes in a remote backend, configured below. Local state means the first
# person to run apply from a different machine destroys everything the state
# file did not know about.

terraform {
  required_version = ">= 1.9"

  required_providers {
    fly    = { source = "fly-apps/fly", version = "~> 0.0.23" }
    neon   = { source = "kislerdm/neon", version = "~> 0.6" }
    upstash = { source = "upstash/upstash", version = "~> 1.5" }
    vercel = { source = "vercel/vercel", version = "~> 2.0" }
    random = { source = "hashicorp/random", version = "~> 3.6" }
  }

  backend "s3" {
    # Populated by `terraform init -backend-config=backend.hcl`, which is not
    # committed. The bucket name is not a secret, but it is environment-specific
    # and hard-coding it makes the module unusable for anyone else.
    key = "agrag/terraform.tfstate"
  }
}

# ── variables ────────────────────────────────────────────────────────────────

variable "environment" {
  type        = string
  description = "Environment name, used in every resource name."
  validation {
    condition     = contains(["staging", "production"], var.environment)
    error_message = "environment must be staging or production."
  }
}

variable "region" {
  type        = string
  default     = "iad"
  description = "Primary region. Put the API next to the database; a cross-region query adds tens of milliseconds to every retrieval, on every request."
}

variable "api_image" {
  type        = string
  description = "Immutable image reference for the API. Never a moving tag."
  validation {
    condition     = !endswith(var.api_image, ":latest")
    error_message = "api_image must be an immutable tag; a rescheduled machine must run the same code as the one it replaced."
  }
}

variable "anthropic_api_key" {
  type      = string
  sensitive = true
}

variable "openai_api_key" {
  type      = string
  sensitive = true
}

variable "min_machines" {
  type        = number
  default     = 2
  description = "Never 1. A single machine means every deploy and every host event is an outage."
  validation {
    condition     = var.min_machines >= 2
    error_message = "min_machines must be at least 2 so a rolling deploy has somewhere to send traffic."
  }
}

locals {
  name = "agrag-${var.environment}"

  tags = {
    project     = "agentic-rag"
    environment = var.environment
    managed_by  = "terraform"
  }
}

# ── secrets ──────────────────────────────────────────────────────────────────

# Generated here rather than supplied, so nobody has to invent one and nobody
# is tempted to reuse one across environments.
resource "random_password" "internal_token" {
  length  = 48
  special = false
}

resource "random_password" "neo4j" {
  length  = 32
  special = true
}

# ── Postgres (Neon) ──────────────────────────────────────────────────────────

resource "neon_project" "main" {
  name      = local.name
  region_id = "aws-us-east-1"

  # pgvector is the one hard dependency; without it nothing works.
  # 15 is what Neon supports with the extension available.
  pg_version = 16

  history_retention_seconds = var.environment == "production" ? 604800 : 86400
}

resource "neon_branch" "main" {
  project_id = neon_project.main.id
  name       = "main"
}

resource "neon_endpoint" "main" {
  project_id = neon_project.main.id
  branch_id  = neon_branch.main.id
  type       = "read_write"

  # Never suspend in production. Neon's scale-to-zero is excellent for staging
  # and adds a multi-second cold start to whichever unlucky user arrives first.
  autoscaling_limit_min_cu = var.environment == "production" ? 0.5 : 0.25
  autoscaling_limit_max_cu = var.environment == "production" ? 4 : 1
  suspend_timeout_seconds  = var.environment == "production" ? 0 : 300
}

# ── Redis (Upstash) ──────────────────────────────────────────────────────────

resource "upstash_redis_database" "cache" {
  database_name = local.name
  region        = "us-east-1"
  tls           = true

  # Eviction is correct here: everything in Redis is a cache, a rate-limit
  # counter or ingestion progress. The durable budget counter lives in Postgres
  # precisely so that an eviction cannot hand a tenant an unlimited budget.
  eviction = true
}

# ── API (Fly.io) ─────────────────────────────────────────────────────────────

resource "fly_app" "api" {
  name = local.name
  org  = "personal"
}

resource "fly_machine" "api" {
  count = var.min_machines

  app    = fly_app.api.name
  region = var.region
  name   = "${local.name}-api-${count.index}"
  image  = var.api_image

  services = [{
    ports = [
      { port = 443, handlers = ["tls", "http"] },
      { port = 80, handlers = ["http"] },
    ]
    protocol      = "tcp"
    internal_port = 8000
  }]

  cpus     = 2
  memorymb = 2048

  env = {
    AGRAG_ENVIRONMENT = var.environment
    AGRAG_LOG_LEVEL   = "INFO"
    OTEL_SERVICE_NAME = local.name
  }

  depends_on = [neon_endpoint.main, upstash_redis_database.cache]
}

# Secrets are set outside the machine definition so that rotating one does not
# show up as a machine replacement in the plan.
resource "fly_secret" "database_url" {
  app   = fly_app.api.name
  name  = "AGRAG_DATABASE_URL"
  value = "postgresql+asyncpg://${neon_project.main.database_user}:${neon_project.main.database_password}@${neon_endpoint.main.host}/${neon_project.main.database_name}?ssl=require"
}

resource "fly_secret" "redis_url" {
  app   = fly_app.api.name
  name  = "AGRAG_REDIS_URL"
  value = "rediss://:${upstash_redis_database.cache.password}@${upstash_redis_database.cache.endpoint}:${upstash_redis_database.cache.port}"
}

resource "fly_secret" "anthropic" {
  app   = fly_app.api.name
  name  = "AGRAG_ANTHROPIC_API_KEY"
  value = var.anthropic_api_key
}

resource "fly_secret" "openai" {
  app   = fly_app.api.name
  name  = "AGRAG_OPENAI_API_KEY"
  value = var.openai_api_key
}

resource "fly_secret" "internal_token" {
  app   = fly_app.api.name
  name  = "AGRAG_INTERNAL_TOKEN"
  value = random_password.internal_token.result
}

# ── Frontend (Vercel) ────────────────────────────────────────────────────────

resource "vercel_project" "web" {
  name           = local.name
  framework      = "nextjs"
  root_directory = "apps/web"

  environment = [{
    key    = "NEXT_PUBLIC_API_URL"
    value  = "https://${fly_app.api.name}.fly.dev"
    target = ["production", "preview"]
  }]
}

# ── outputs ──────────────────────────────────────────────────────────────────

output "api_url" {
  value       = "https://${fly_app.api.name}.fly.dev"
  description = "Point AGRAG_API_URL at this."
}

output "web_url" {
  value = "https://${vercel_project.web.name}.vercel.app"
}

output "database_url" {
  value       = "postgresql+psycopg://${neon_project.main.database_user}:${neon_project.main.database_password}@${neon_endpoint.main.host}/${neon_project.main.database_name}"
  sensitive   = true
  description = "For running `alembic upgrade head` from a shell. Marked sensitive so it is not printed by a plan."
}

output "next_steps" {
  value = <<-EOT
    1. alembic upgrade head          (with the database_url output)
    2. python scripts/seed_demo_tenant.py
    3. python scripts/smoke_test.py --url ${"https://${fly_app.api.name}.fly.dev"}

    The smoke test is not optional. Everything above can succeed while the
    first real request returns a 500.
  EOT
}
