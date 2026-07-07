# AgenticRAG API

FastAPI service hosting the ingestion pipeline, retrieval strategies, LangGraph
agent, guardrails and the public OpenAI-compatible API.

See [docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md) for the full picture and
[../../README.md](../../README.md) for the quickstart.

```bash
pip install -e ".[dev,evals]"
alembic upgrade head
uvicorn src.main:app --reload
```
