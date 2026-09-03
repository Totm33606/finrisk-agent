"""Langfuse observability for the FinRisk agent.

Centralizes everything Langfuse-related so `agent.py` stays focused on
orchestration logic. Every agent run is traced end-to-end: the top-level
LLM calls, every MCP tool invocation (name, input, output, latency), and
token cost, all rolled up under one Langfuse trace per analyst request.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from langfuse import Langfuse
from langfuse.callback import CallbackHandler

# `ObservabilityConfig` below reads LANGFUSE_* via `os.getenv(...)` at class
# *definition* time (i.e. at import time), so .env must be loaded before that
# happens. `agent.py` already does this before importing this module, but
# calling it here too makes this module correct on its own — importable
# independently (a notebook, a script, a future test) without silently
# losing tracing because .env was never loaded. Idempotent/safe to repeat.
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ObservabilityConfig:
    """Langfuse connection settings, read from environment by default.

    Required env vars: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY.
    Optional: LANGFUSE_HOST (defaults to Langfuse Cloud EU).
    """

    public_key: str | None = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key: str | None = os.getenv("LANGFUSE_SECRET_KEY")
    host: str = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    enabled: bool = bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))
    release: str = os.getenv("FINRISK_MODEL_VERSION", "dev")


_obs_config = ObservabilityConfig()
_langfuse_client: Langfuse | None = None

if not _obs_config.enabled:
    # Once, at import — not on every call below, which would put one warning
    # line per analyst request into the API log.
    logger.warning("Langfuse credentials not set — running without tracing.")


def get_langfuse_client() -> Langfuse | None:
    """Return a process-wide Langfuse client, or None if credentials are absent.

    Returning None (rather than raising) lets the agent run locally without
    Langfuse configured — observability is layered on, not a hard
    dependency for functional correctness.
    """
    global _langfuse_client
    if not _obs_config.enabled:
        return None
    if _langfuse_client is None:
        _langfuse_client = Langfuse(
            public_key=_obs_config.public_key,
            secret_key=_obs_config.secret_key,
            host=_obs_config.host,
        )
    return _langfuse_client


def build_callback_handler(
    *, session_id: str, user_id: str, client_id: str, question: str
) -> CallbackHandler | None:
    """Build a per-request Langfuse callback handler with rich metadata tags.

    Tags/metadata are chosen to make the Langfuse UI directly useful to a
    financial-analyst audit trail: which analyst (user_id), which
    conversation (session_id), and which client/SME was under review.
    """
    if not _obs_config.enabled:
        return None
    return CallbackHandler(
        public_key=_obs_config.public_key,
        secret_key=_obs_config.secret_key,
        host=_obs_config.host,
        session_id=session_id,
        user_id=user_id,
        release=_obs_config.release,
        tags=["finrisk-agent", "credit-decisioning"],
        metadata={"client_id": client_id, "question": question},
    )


def flush() -> None:
    """Flush pending Langfuse events. Call on FastAPI shutdown / script exit."""
    client = get_langfuse_client()
    if client is not None:
        client.flush()
        logger.info("Langfuse events flushed.")
