.PHONY: install dataset train tune eval mcp agent api test lint typecheck fmt docker-build docker-up clean

install:
	uv venv && uv pip install -e ".[dev]"

dataset:
	uv run python -m ml_pipeline.make_dataset

train:
	uv run python -m ml_pipeline.train

tune:
	uv run python -m ml_pipeline.train --tune

eval:
	uv run python -m ml_pipeline.eval

mcp:
	uv run python -m mcp_server.server --transport stdio

mcp-http:
	uv run python -m mcp_server.server --transport streamable-http --port 8000

agent:
	uv run python -m agent.agent SME-000001 --question "Should we approve this client?"

api:
	uv run uvicorn agent.agent:api --reload --port 8080

test:
	uv run pytest

lint:
	uv run ruff check src tests

typecheck:
	uv run mypy src

fmt:
	uv run ruff format src tests

docker-build:
	docker compose -f docker/docker-compose.yml build

docker-up:
	docker compose -f docker/docker-compose.yml up

clean:
	rm -rf models reports .pytest_cache .ruff_cache **/__pycache__
