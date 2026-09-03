.PHONY: install dataset train tune eval mlflow-ui mcp mcp-http agent api test lint typecheck fmt docker-build docker-up docker-train docker-eval clean

install:
	uv sync --locked --extra dev

dataset:
	uv run python -m ml_pipeline.make_dataset

# ALIAS picks which registry alias to publish under / read from. It defaults
# to `champion`, the one the MCP server serves. `make train ALIAS=challenger`
# publishes a candidate without touching what is live; `make eval
# ALIAS=challenger` then scores it and records on its own run.
train:
	uv run python -m ml_pipeline.train $(if $(ALIAS),--alias $(ALIAS),)

tune:
	uv run python -m ml_pipeline.train --tune $(if $(ALIAS),--alias $(ALIAS),)

eval:
	uv run python -m ml_pipeline.eval $(if $(ALIAS),--alias $(ALIAS),)

# Browse the tracked runs and the model registry. No --backend-store-uri flag
# needed: the default store is the ./mlruns file store in this directory.
mlflow-ui:
	uv run mlflow ui --port 5000

# stdio: how the agent spawns it locally by default — useful mainly to check
# the server starts. ALIAS=challenger serves a candidate instead of @champion.
mcp:
	uv run python -m mcp_server.server --transport stdio $(if $(ALIAS),--alias $(ALIAS),)

# HTTP: the transport the Docker stack uses. Point the agent at it with
# FINRISK_MCP_URL=http://localhost:8000/mcp.
mcp-http:
	uv run python -m mcp_server.server --transport streamable-http --port 8000 $(if $(ALIAS),--alias $(ALIAS),)

# SME-000182 is in the held-out split — the only clients the server serves —
# for every dataset size the docs use (2k/20k/25k). See EXAMPLE_CLIENTS in
# frontend/src/App.jsx.
agent:
	uv run python -m agent.agent SME-000182 --question "Should we approve this client?"

api:
	uv run uvicorn agent.agent:api --reload --port 8080

test:
	uv run pytest

lint:
	uv run ruff check src tests

typecheck:
	uv run mypy src tests

fmt:
	uv run ruff format src tests

docker-build:
	docker compose -f docker/docker-compose.yml build

# `--build` is not optional here: nothing is bind-mounted at serving time
# (src/ is baked into the image, the dashboard is a prebuilt nginx bundle), so
# a plain `up` silently re-runs the previous images and your code changes never
# reach the containers. Layer caching makes the rebuild cheap when nothing moved.
docker-up:
	docker compose -f docker/docker-compose.yml up --build

# Train/eval inside a container instead of on the host, so the paths MLflow
# bakes into mlruns/ match what `mcp` resolves at /app/mlruns. Required
# before `docker-up` will serve anything — see the `train` service comment
# in docker-compose.yml.
#
# ALIAS is appended as a plain `--alias` flag, which works because both
# services declare an `entrypoint:` rather than a `command:` — see the comment
# above `train` in docker-compose.yml for why that distinction matters.
docker-train:
	docker compose -f docker/docker-compose.yml run --rm train $(if $(ALIAS),--alias $(ALIAS),)

docker-eval:
	docker compose -f docker/docker-compose.yml run --rm eval $(if $(ALIAS),--alias $(ALIAS),)

# Removes the model store too, so `make clean` means re-running `make train`
# before the MCP server or dashboard can start again.
clean:
	rm -rf mlruns .pytest_cache .ruff_cache **/__pycache__
