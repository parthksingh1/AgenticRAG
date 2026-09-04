# syntax=docker/dockerfile:1.7
# Multi-stage build: `builder` compiles wheels, `runtime` stays slim and non-root.

FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build

# upgrade before install: the base tag is rebuilt on its own cadence, so
# without this the image ships whatever security patches were outstanding
# when that tag was cut -- which is what the container scan flags.
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
      build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cpu -e ".[dev,evals]"

# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/root/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false

# tesseract → OCR fallback; poppler → PDF rasterisation for vision table extraction;
# libgl/libglib → OpenCV deps pulled in by unstructured/Docling.
# upgrade before install: the base tag is rebuilt on its own cadence, so
# without this the image ships whatever security patches were outstanding
# when that tag was cut -- which is what the container scan flags.
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
      curl tesseract-ocr poppler-utils libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY . .

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=45s --retries=5 \
  CMD curl -sf http://localhost:8000/healthz || exit 1

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
