# Helm chart

```bash
helm install agrag infra/helm/agrag \
  --set image.tag=sha-abc123 \
  --set ingress.host=api.example.com \
  --set secrets.existingSecret=agrag-secrets
```

`image.tag` has no default and the chart fails without it. A rescheduled pod
must run the same code as the one it replaced, and `latest` cannot promise that.

## What the defaults assume

Production, not a demo. Three API replicas, pod anti-affinity, a disruption
budget, an HPA, network policies and a read-only root filesystem. A chart whose
defaults are "one replica, no limits, no probes" installs cleanly and falls over
the first time a node is drained.

## Three probes, three jobs

| Probe | Purpose | Checks dependencies? |
|---|---|---|
| `startupProbe` | Survive a slow first boot while models load | no |
| `readinessProbe` | Gate traffic | **yes** — `/readyz` |
| `livenessProbe` | Restart a wedged process | **no** — `/healthz` |

Liveness must not check dependencies. A liveness probe that fails when Postgres
is briefly unreachable restarts every pod at once and turns a blip into an
outage. That distinction is why there are two endpoints.

## Beat runs exactly once

`replicas: 1` with `strategy: Recreate`. Two schedulers means every periodic task
fires twice: two nightly backups racing, two drift snapshots colliding on the
unique constraint, two recalibrations retiring each other's rows. A rolling
update would briefly run two, which is the thing `replicas: 1` exists to prevent.

## Secrets

Referenced from an existing Secret, never inline. A chart that accepts inline
secrets ends up with them in a values file in git. The pod template carries a
checksum of the secret name so a rotation actually rolls the pods — otherwise a
rotated credential sits unused until something unrelated triggers a restart.
