# syntax=docker/dockerfile:1.7
# --- Base: uv-managed Python, shared by all stages ---------------------------
FROM python:3.11-slim AS base
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /uvx /usr/local/bin/
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

# libgomp1 is required by LightGBM's OpenMP-parallelized boosters at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# --- Dependencies layer (cached independently of source changes) ------------
FROM base AS deps
COPY pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -e .

# --- Runtime image ------------------------------------------------------------
FROM deps AS runtime
COPY src/ ./src/
COPY README.md ./
ENV PYTHONPATH=/app/src

# Non-root user: standard hardening for a production-facing image.
RUN useradd --create-home --uid 1000 finrisk
USER finrisk

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

CMD ["uvicorn", "agent.agent:api", "--host", "0.0.0.0", "--port", "8080"]
