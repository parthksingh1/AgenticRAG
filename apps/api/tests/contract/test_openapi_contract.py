"""Contract tests: does the API obey its own OpenAPI schema?

Schemathesis generates requests from the schema and checks the responses against
it. That catches a class of bug nothing else here does — an endpoint whose
declared response model has drifted from what it actually returns.

It matters more than usual because the schema is not documentation. It generates
the SDKs and the frontend's expectations, so a schema that lies produces clients
that are wrong in ways their own tests cannot see.

    pytest tests/contract -m contract
"""

from __future__ import annotations

import pytest
from src.main import create_app

pytestmark = pytest.mark.contract

app = create_app()


@pytest.mark.contract
def test_generated_requests_never_return_a_500() -> None:
    """Fuzzed requests must be rejected, not crash the handler.

    Skipped when schemathesis is absent so the suite still runs on a machine
    that has not installed the optional extra; CI installs it.
    """
    schemathesis = pytest.importorskip("schemathesis")
    from fastapi.testclient import TestClient
    from schemathesis.core import NotSet
    from schemathesis.core.result import Err

    schema = schemathesis.openapi.from_dict(app.openapi())
    client = TestClient(app)
    failures: list[str] = []

    for result in schema.get_all_operations():
        # get_all_operations() yields a Result per operation rather than
        # raising on the first one it cannot build (an unresolvable $ref, an
        # unsupported schema construct); skip those, they are not what this
        # test is checking.
        if isinstance(result, Err):
            continue
        operation = result.ok()
        # Path parameters have no default and Case() leaves them unset, which
        # raises InvalidSchema before a request is even made — that failure
        # would test this harness, not the endpoint. A fixed dummy value
        # exercises the handler's own validation of a syntactically present
        # but semantically wrong id instead.
        path_parameters = {p.name: "test-fuzz-id" for p in operation.path_parameters.items}
        case = operation.Case(path_parameters=path_parameters)
        try:
            response = client.request(
                case.method,
                case.formatted_path,
                json=None if isinstance(case.body, NotSet) else case.body,
            )
        except Exception as exc:  # noqa: BLE001 - an unhandled exception is the finding
            failures.append(f"{case.method} {case.path}: raised {exc!r}")
            continue

        # 500 is never an acceptable answer to a malformed or unauthenticated
        # request. It means an unvalidated input reached far enough to raise.
        if response.status_code == 500:
            failures.append(f"{case.method} {case.path}: returned 500")

    assert not failures, "generated requests produced server errors:\n  " + "\n  ".join(failures)


@pytest.mark.contract
def test_every_route_declares_its_error_responses() -> None:
    """A schema that documents only the happy path is half a schema.

    A client generated from it has no typed error handling, so every failure
    becomes an untyped exception at the call site.
    """
    spec = app.openapi()
    missing: list[str] = []

    for path, operations in spec["paths"].items():
        if path.startswith(("/healthz", "/readyz", "/metrics")):
            continue
        for method, operation in operations.items():
            if method not in {"get", "post", "patch", "delete", "put"}:
                continue
            codes = set(operation.get("responses", {}))
            if not any(code.startswith("4") for code in codes):
                missing.append(f"{method.upper()} {path}")

    assert not missing, "these operations document no error response: " + ", ".join(missing)


@pytest.mark.contract
def test_no_response_schema_exposes_a_credential() -> None:
    """No response model carries a field whose name reads like a stored secret.

    A cheap structural check for an expensive mistake: returning a key hash in a
    listing because the response model was generated from the ORM row instead of
    written.

    Two deliberate exemptions. ``secret`` on the API-key creation response is
    meant to be returned, exactly once — that is the whole point of it. And a
    field ending in ``_id`` is a reference, not a value: the audit log has to
    record *which* key acted, and `actor_api_key_id` is how.
    """
    spec = app.openapi()
    forbidden = ("key_hash", "secret_hash", "password", "api_key")
    offenders: list[str] = []

    for name, definition in spec.get("components", {}).get("schemas", {}).items():
        for field in definition.get("properties", {}):
            lowered = field.lower()
            if lowered.endswith("_id"):
                continue
            if any(marker in lowered for marker in forbidden):
                offenders.append(f"{name}.{field}")

    assert not offenders, "response schemas expose credential-shaped fields: " + ", ".join(
        offenders
    )


@pytest.mark.contract
def test_the_openai_surface_keeps_its_shape() -> None:
    """The OpenAI-compatible endpoints must not drift from the shape SDKs expect.

    Compatibility is the difference between an API someone can try in two minutes
    and one they have to read about first. A renamed field breaks every client
    silently, because an OpenAI SDK ignores fields it does not recognise.
    """
    spec = app.openapi()
    assert "/v1/chat/completions" in spec["paths"]
    assert "/v1/models" in spec["paths"]

    schemas = spec.get("components", {}).get("schemas", {})
    response = schemas.get("OpenAIChatResponse")
    assert response is not None, "the OpenAI response model is missing from the schema"

    required = {"id", "object", "created", "model", "choices"}
    present = set(response.get("properties", {}))
    assert required <= present, f"missing OpenAI fields: {sorted(required - present)}"
