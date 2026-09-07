"""Langfuse observability for the FinRisk agent.

Centralizes everything Langfuse-related so `agent.py` stays focused on
orchestration. One trace is opened per analyst request; the LangGraph run
attaches to it, and — the part that makes this an audit trail rather than a
log — the run's *outcome* is written back onto that trace when it finishes.

**The trace is created explicitly here**, rather than letting the callback
handler create one implicitly, and everything below follows from that:

* `Langfuse.trace()` returns a `StatefulTraceClient` whose
  `get_langchain_handler()` binds a handler to *that* trace via the
  `stateful_client` path, which reuses the process-wide client's background
  worker. Constructing `CallbackHandler(public_key=...)` instead builds a
  whole new `Langfuse` client per call — its own worker threads, its own
  HTTP client, its own `atexit` registration, none of them ever released.
* Holding the trace object is what lets `finish()` set the trace's output,
  tag it with the decision, and attach scores once the run is over.
* It is also the supported way to read the trace id back: the handler has
  no `trace_id` attribute, so `getattr(handler, "trace_id", None)` silently
  returned None forever, and the dashboard's cross-link never appeared.

What reaches Langfuse for one run:

* **LLM calls** — typed `GENERATION` observations with prompt, completion,
  token counts, cost and latency. From the LangChain integration.
* **Tool calls** — generic spans, also from the integration. Langfuse v2 has
  no *tool* observation type, so an MCP call is not visually distinct from
  LangGraph's own internal node spans. `ToolTelemetry` below therefore
  records them independently, and `finish()` folds name/latency/status into
  the trace's metadata and scores where they are actually queryable.
* **Outcome** — the decision, the PD and risk band behind it, the served
  model version, and the tool-call tally, as trace output plus metadata.
* **Scores** — `grounded_in_tools`, `decision_matches_model`,
  `tool_success_rate`. Objective and cheap: no LLM judge involved.

Still out of scope: the MCP scoring server is a separate process and emits
application logs only, so tool spans are timed from the agent's side.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import UUID

from dotenv import load_dotenv
from langchain_core.callbacks import AsyncCallbackHandler, BaseCallbackHandler
from langfuse import Langfuse
from langfuse.client import StatefulTraceClient

from common.schemas import AgentAnalysisResult

# `ObservabilityConfig` below reads LANGFUSE_* via `os.getenv(...)` at class
# *definition* time (i.e. at import time), so .env must be loaded before that
# happens. `agent.py` already does this before importing this module, but
# calling it here too makes this module correct on its own — importable
# independently (a notebook, a script, a future test) without silently
# losing tracing because .env was never loaded. Idempotent/safe to repeat.
load_dotenv()

logger = logging.getLogger(__name__)

# The MCP tool the system prompt makes step 1 of every scoring workflow. Named
# here because two of the three scores below are about whether the agent
# actually consulted it before deciding; it is the tool function's own name in
# `mcp_server.server`, which is what MCP exposes over the wire.
SCORE_TOOL_NAME = "get_credit_score"

BASE_TRACE_TAGS = ["finrisk-agent", "credit-decisioning"]


@dataclass(frozen=True)
class ObservabilityConfig:
    """Langfuse connection settings, read from environment by default.

    Required env vars: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY.
    Optional: LANGFUSE_HOST (defaults to Langfuse Cloud EU; point it at
    http://langfuse:3000 to use the self-hosted service in docker-compose).
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
    """Return the process-wide Langfuse client, or None if credentials are absent.

    Process-wide is load-bearing, not just tidy: this one client owns the
    background worker every trace's events are queued on, which is what
    `flush()` can then actually drain. Returning None (rather than raising)
    lets the agent run locally without Langfuse configured — observability is
    layered on, not a hard dependency for functional correctness.
    """
    global _langfuse_client
    if not _obs_config.enabled:
        return None
    if _langfuse_client is None:
        _langfuse_client = Langfuse(
            public_key=_obs_config.public_key,
            secret_key=_obs_config.secret_key,
            host=_obs_config.host,
            # On the client, not on `trace()`, which takes no release.
            release=_obs_config.release,
        )
    return _langfuse_client


# ---------------------------------------------------------------------------
# Tool-call telemetry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolCall:
    """One completed MCP tool invocation, as observed from the agent process."""

    name: str
    status: Literal["success", "error"]
    latency_ms: float
    error: str | None = None


class ToolTelemetry(AsyncCallbackHandler):
    """Records every tool invocation of one agent run, independently of Langfuse.

    Three reasons this exists rather than reading the spans Langfuse
    already emits:

    1. Those spans are untyped and carry no latency we can act on locally —
       `finish()` needs the numbers to put them on the trace and in scores.
    2. Every handler in the Langfuse integration is wrapped in
       `except Exception: log.exception(...)`, so a dropped span is invisible
       to the application. This one runs regardless, including when Langfuse
       is switched off entirely, so the API log always tells you what the
       agent called and whether it worked.
    3. It is ours. Swapping observability vendors — or adding a streaming
       endpoint that emits per-tool events — does not start from scratch.

    One instance per run: `_pending` is keyed by LangChain's `run_id`, and
    the async callbacks are awaited in order on the event loop, so no
    locking is needed. Handler exceptions are swallowed by LangChain, but
    these are kept trivially safe anyway — telemetry must never be able to
    fail a credit analysis.
    """

    def __init__(self) -> None:
        self.calls: list[ToolCall] = []
        self._pending: dict[UUID, tuple[str, float]] = {}

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        name = (serialized or {}).get("name") or kwargs.get("name") or "unknown"
        self._pending[run_id] = (str(name), time.perf_counter())

    async def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self._complete(run_id, status="success")

    async def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self._complete(run_id, status="error", error=str(error))

    def _complete(
        self, run_id: UUID, *, status: Literal["success", "error"], error: str | None = None
    ) -> None:
        started = self._pending.pop(run_id, None)
        if started is None:
            # No matching `on_tool_start`, so there is no latency to report
            # and nothing meaningful to record. Logged rather than guessed at.
            logger.debug("Tool callback for unknown run_id %s ignored.", run_id)
            return
        name, t0 = started
        self.calls.append(
            ToolCall(
                name=name,
                status=status,
                latency_ms=(time.perf_counter() - t0) * 1000,
                error=error,
            )
        )

    @property
    def errors(self) -> list[ToolCall]:
        return [call for call in self.calls if call.status == "error"]

    def summary(self) -> dict[str, Any]:
        """Aggregate view of the run's tool usage, for trace metadata and logs."""
        return {
            "tool_calls": len(self.calls),
            "tool_errors": len(self.errors),
            "tools_used": sorted({call.name for call in self.calls}),
            "tool_latency_ms": {call.name: round(call.latency_ms, 1) for call in self.calls},
            "tool_error_messages": [f"{c.name}: {c.error}" for c in self.errors],
        }


# ---------------------------------------------------------------------------
# Per-run trace
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Score:
    """One quality signal attached to a finished trace."""

    name: str
    value: float
    data_type: Literal["BOOLEAN", "NUMERIC"]
    comment: str


def _served_score_payload(result: AgentAnalysisResult) -> dict[str, Any] | None:
    """The most recent successful `get_credit_score` payload, if the agent called it.

    The single source for every model-side fact worth putting on the trace:
    the PD, the risk band, the served model version and the threshold behind
    the recommendation. Reading the last call rather than the first because a
    what-if question can legitimately score the same client twice.
    """
    for step in reversed(result.steps):
        if step.tool_name == SCORE_TOOL_NAME and step.status == "success" and step.raw_output:
            return dict(step.raw_output)
    return None


def derive_scores(result: AgentAnalysisResult, calls: list[ToolCall]) -> list[_Score]:
    """Quality signals for one run, computed without an LLM judge.

    All three are things a credit reviewer would actually filter on, and all
    three are decidable from the trajectory alone — which is why they are
    worth recording on every run rather than sampling.
    """
    scores: list[_Score] = []

    grounded = any(
        step.tool_name == SCORE_TOOL_NAME and step.status == "success" for step in result.steps
    )
    scores.append(
        _Score(
            name="grounded_in_tools",
            value=float(grounded),
            data_type="BOOLEAN",
            comment=(
                f"{SCORE_TOOL_NAME} was called successfully before deciding."
                if grounded
                else f"Decided without a successful {SCORE_TOOL_NAME} call — the system "
                "prompt makes it step 1, so the PD in the narrative is ungrounded."
            ),
        )
    )

    payload = _served_score_payload(result)
    recommendation = payload.get("recommendation") if payload else None
    if isinstance(recommendation, str):
        # The prompt allows the agent to diverge from the model *with a stated
        # reason*, so this is not a pass/fail — it is the flag that tells a
        # reviewer which decisions need the reasoning read.
        agrees = recommendation == result.decision.value
        scores.append(
            _Score(
                name="decision_matches_model",
                value=float(agrees),
                data_type="BOOLEAN",
                comment=(
                    f"Agent and model both say {recommendation}."
                    if agrees
                    else f"Agent decided {result.decision.value} where the model recommended "
                    f"{recommendation}; the prompt permits this only with a stated reason."
                ),
            )
        )

    if calls:
        succeeded = sum(1 for call in calls if call.status == "success")
        scores.append(
            _Score(
                name="tool_success_rate",
                value=succeeded / len(calls),
                data_type="NUMERIC",
                comment=f"{succeeded}/{len(calls)} MCP tool calls succeeded.",
            )
        )
    return scores


@dataclass
class AgentRunTrace:
    """Observability for one agent run: the callbacks going in, the outcome coming out.

    Always usable, whether or not Langfuse is configured — `_trace` is None
    when it isn't, and the tool telemetry still runs and still reaches the
    application log. That keeps `agent.py` free of `if tracing_enabled`
    branches around its own control flow.
    """

    telemetry: ToolTelemetry = field(default_factory=ToolTelemetry)
    _trace: StatefulTraceClient | None = None
    _handler: BaseCallbackHandler | None = None

    @property
    def callbacks(self) -> list[BaseCallbackHandler]:
        """Handlers to pass to LangGraph. Both see every event; neither depends on the other.

        Order is not significant: LangChain's `ahandle_event` runs non-inline
        handlers through `asyncio.gather`, and wraps each one in its own
        `try/except`, so a handler that throws is logged and skipped without
        touching the others.

        The two are not a division of labour. The Langfuse handler reports
        *everything* it is sent — LLM calls, tool calls, LangGraph's own node
        spans — and is what builds the nested span tree. `ToolTelemetry` only
        implements the three tool events, sends nothing to Langfuse, and
        exists so the agent process holds the tool outcomes itself: `finish()`
        turns them into trace metadata and scores, and the application log
        gets them even when Langfuse is off or silently dropping spans.

        It is also awaited directly on the event loop, being async, whereas
        the Langfuse handler is sync and therefore dispatched to a thread-pool
        executor — so its timings are the tighter of the two.
        """
        handlers: list[BaseCallbackHandler] = [self.telemetry]
        if self._handler is not None:
            handlers.append(self._handler)
        return handlers

    @property
    def trace_id(self) -> str | None:
        """Langfuse trace id, for cross-linking from the dashboard. None when disabled."""
        return str(self._trace.id) if self._trace is not None else None

    def finish(self, result: AgentAnalysisResult) -> None:
        """Write the run's outcome onto the trace and attach its scores.

        Called after `ainvoke` returns, which is the only moment the decision
        exists. Without this the trace records how the agent worked but never
        what it concluded, so "show me every DECLINE last quarter" — the
        first question anyone audits with — has no answer.
        """
        summary = self.telemetry.summary()
        # Logged unconditionally: the tool trail must survive Langfuse being
        # off, unreachable, or silently dropping spans.
        logger.info(
            "Run finished: decision=%s tool_calls=%d tool_errors=%d tools=%s",
            result.decision.value,
            summary["tool_calls"],
            summary["tool_errors"],
            summary["tools_used"],
        )
        for failure in self.telemetry.errors:
            logger.warning("Tool %s failed: %s", failure.name, failure.error)

        if self._trace is None:
            return
        try:
            self._trace.update(
                output={
                    "decision": result.decision.value,
                    "summary": result.summary,
                    "key_drivers": result.key_drivers,
                },
                metadata=self._outcome_metadata(result, summary),
                # A tag, so filtering a quarter's DECLINEs in the Langfuse UI
                # is one click rather than a metadata query.
                tags=[*BASE_TRACE_TAGS, f"decision:{result.decision.value}"],
            )
            for score in derive_scores(result, self.telemetry.calls):
                self._trace.score(
                    name=score.name,
                    value=score.value,
                    data_type=score.data_type,
                    comment=score.comment,
                )
        except Exception:
            # Observability must never fail a credit analysis: the caller
            # already holds a complete result by the time this runs.
            logger.warning("Could not finalize the Langfuse trace.", exc_info=True)

    @staticmethod
    def _outcome_metadata(result: AgentAnalysisResult, summary: dict[str, Any]) -> dict[str, Any]:
        """Tool tally plus the model-side facts behind the decision."""
        metadata: dict[str, Any] = {
            "client_id": result.client_id,
            "total_latency_ms": round(result.total_latency_ms, 1),
            **summary,
        }
        payload = _served_score_payload(result)
        if payload is not None:
            # Recorded on the trace so an audit can ask which model version
            # and which operating point produced a given decision without
            # opening the tool span and parsing its raw output string.
            metadata |= {
                "probability_default": payload.get("probability_default"),
                "risk_band": payload.get("risk_band"),
                "model_version": payload.get("model_version"),
                "decision_threshold": payload.get("decision_threshold"),
                "model_recommendation": payload.get("recommendation"),
            }
        return metadata


def start_run_trace(
    *, session_id: str, user_id: str, client_id: str, question: str
) -> AgentRunTrace:
    """Open the trace for one analyst request and build its callbacks.

    Metadata is chosen to make the Langfuse UI directly useful as an audit
    trail: which analyst (user_id), which conversation (session_id), and
    which client was under review. The run's conclusion is added later by
    `AgentRunTrace.finish`.
    """
    client = get_langfuse_client()
    if client is None:
        return AgentRunTrace()

    trace = client.trace(
        name="finrisk-credit-analysis",
        session_id=session_id,
        user_id=user_id,
        input={"client_id": client_id, "question": question},
        metadata={"client_id": client_id},
        tags=list(BASE_TRACE_TAGS),
    )
    # `update_parent=False`: the LangChain run must not overwrite the trace's
    # output with its raw final message — `finish()` sets the structured
    # decision there instead.
    return AgentRunTrace(_trace=trace, _handler=trace.get_langchain_handler(update_parent=False))


def flush() -> None:
    """Flush pending Langfuse events. Call on FastAPI shutdown / script exit.

    Correct only because every trace is created from the client this returns:
    events are queued on that one client's worker, so draining it drains
    them. Handlers that build their own client — the shape this module used
    to have — queue elsewhere, and this would silently drain an empty queue.
    """
    client = get_langfuse_client()
    if client is not None:
        client.flush()
        logger.info("Langfuse events flushed.")
