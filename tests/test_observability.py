"""Unit tests for `agent.observability`, run without Langfuse credentials.

This module previously had no tests at all, which is exactly why three
defects survived in it: a callback attribute that never existed (so the
trace id was always None), a fresh Langfuse client per request, and a
`flush()` draining a queue nothing wrote to. All three were invisible
because every one of them fails *silently*.

Nothing here reaches the network. The Langfuse client is either absent
(credentials unset, the default in CI) or replaced by a recording stub, so
what is asserted is the shape of what *would* be sent.
"""

from __future__ import annotations

from typing import Any, Literal
from uuid import UUID, uuid4

import pytest

from agent import observability
from agent.observability import (
    SCORE_TOOL_NAME,
    AgentRunTrace,
    ToolTelemetry,
    derive_scores,
    start_run_trace,
)
from common.schemas import AgentAnalysisResult, AgentStep, CreditDecision


def _score_step(
    *, recommendation: str = "APPROVE", status: Literal["success", "error"] = "success"
) -> AgentStep:
    return AgentStep(
        step_index=0,
        tool_name=SCORE_TOOL_NAME,
        tool_input={"client_id": "SME-1"},
        tool_output_summary="PD=12.0%, band=LOW",
        raw_output={
            "probability_default": 0.12,
            "risk_band": "LOW",
            "recommendation": recommendation,
            "model_version": "1 (run abcd1234)",
            "decision_threshold": 0.3,
        },
        status=status,
    )


def _result(
    *, decision: CreditDecision = CreditDecision.APPROVE, steps: list[AgentStep] | None = None
) -> AgentAnalysisResult:
    return AgentAnalysisResult(
        client_id="SME-1",
        question="Should we approve this client?",
        decision=decision,
        summary="PD is low at 12%. APPROVE.",
        key_drivers=["Debt-to-equity within range"],
        steps=[_score_step()] if steps is None else steps,
        total_latency_ms=1234.5,
        langfuse_trace_id=None,
    )


# ---------------------------------------------------------------------------
# Tool telemetry
# ---------------------------------------------------------------------------


async def _run_tool(telemetry: ToolTelemetry, name: str, *, error: str | None = None) -> UUID:
    run_id = uuid4()
    await telemetry.on_tool_start({"name": name}, "{}", run_id=run_id)
    if error is None:
        await telemetry.on_tool_end("{}", run_id=run_id)
    else:
        await telemetry.on_tool_error(RuntimeError(error), run_id=run_id)
    return run_id


@pytest.mark.asyncio
async def test_telemetry_records_successes_and_failures_with_latency() -> None:
    """The success/failure/result trail the Langfuse spans can't be relied on for."""
    telemetry = ToolTelemetry()

    await _run_tool(telemetry, SCORE_TOOL_NAME)
    await _run_tool(telemetry, "get_shap_explanation", error="Unknown client_id: 'SME-9'")

    assert [call.name for call in telemetry.calls] == [SCORE_TOOL_NAME, "get_shap_explanation"]
    assert [call.status for call in telemetry.calls] == ["success", "error"]
    assert all(call.latency_ms >= 0.0 for call in telemetry.calls)
    assert telemetry.errors[0].error is not None
    assert "SME-9" in telemetry.errors[0].error


@pytest.mark.asyncio
async def test_telemetry_summary_counts_calls_and_errors() -> None:
    telemetry = ToolTelemetry()
    await _run_tool(telemetry, SCORE_TOOL_NAME)
    await _run_tool(telemetry, SCORE_TOOL_NAME, error="boom")

    summary = telemetry.summary()

    assert summary["tool_calls"] == 2
    assert summary["tool_errors"] == 1
    assert summary["tools_used"] == [SCORE_TOOL_NAME]


@pytest.mark.asyncio
async def test_telemetry_ignores_an_end_without_a_matching_start() -> None:
    """No start means no latency to report, so there is nothing honest to record."""
    telemetry = ToolTelemetry()

    await telemetry.on_tool_end("{}", run_id=uuid4())

    assert telemetry.calls == []


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------


def _by_name(scores: list[Any]) -> dict[str, Any]:
    return {score.name: score for score in scores}


def test_scores_flag_a_decision_taken_without_calling_the_scoring_tool() -> None:
    """The system prompt makes `get_credit_score` step 1: skipping it means the
    PD in the narrative came from the model's imagination, not from the model."""
    result = _result(steps=[])

    scores = _by_name(derive_scores(result, []))

    assert scores["grounded_in_tools"].value == 0.0
    assert "ungrounded" in scores["grounded_in_tools"].comment
    # Nothing to compare the decision against, so that score is omitted rather
    # than invented.
    assert "decision_matches_model" not in scores


def test_scores_flag_a_decision_diverging_from_the_model_recommendation() -> None:
    """Divergence is permitted by the prompt *with a stated reason* — so this is
    the flag telling a reviewer which narratives actually need reading."""
    result = _result(decision=CreditDecision.DECLINE, steps=[_score_step(recommendation="APPROVE")])

    scores = _by_name(derive_scores(result, []))

    assert scores["grounded_in_tools"].value == 1.0
    assert scores["decision_matches_model"].value == 0.0
    assert "DECLINE" in scores["decision_matches_model"].comment
    assert "APPROVE" in scores["decision_matches_model"].comment


def test_scores_agree_when_the_agent_follows_the_model() -> None:
    result = _result(decision=CreditDecision.APPROVE, steps=[_score_step(recommendation="APPROVE")])

    scores = _by_name(derive_scores(result, []))

    assert scores["decision_matches_model"].value == 1.0


@pytest.mark.asyncio
async def test_tool_success_rate_reflects_failed_calls() -> None:
    telemetry = ToolTelemetry()
    await _run_tool(telemetry, SCORE_TOOL_NAME)
    await _run_tool(telemetry, "get_shap_explanation", error="boom")

    scores = _by_name(derive_scores(_result(), telemetry.calls))

    assert scores["tool_success_rate"].value == 0.5
    assert scores["tool_success_rate"].data_type == "NUMERIC"


# ---------------------------------------------------------------------------
# Trace lifecycle
# ---------------------------------------------------------------------------


class _RecordingTrace:
    """Stand-in for a Langfuse `StatefulTraceClient`, recording what it is sent."""

    id = "trace-abc123"

    def __init__(self) -> None:
        self.updates: list[dict[str, Any]] = []
        self.scores: list[dict[str, Any]] = []
        self.handler_requests: list[bool] = []

    def update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)

    def score(self, **kwargs: Any) -> None:
        self.scores.append(kwargs)

    def get_langchain_handler(self, update_parent: bool = False) -> object:
        self.handler_requests.append(update_parent)
        return object()


def test_run_trace_without_langfuse_still_collects_tool_telemetry() -> None:
    """Tracing is layered on, so the tool trail must survive it being switched off."""
    run_trace = start_run_trace(
        session_id="s1", user_id="analyst@finrisk.local", client_id="SME-1", question="?"
    )

    assert run_trace.trace_id is None
    assert [type(handler) for handler in run_trace.callbacks] == [ToolTelemetry]
    # Must not raise: the caller already holds a complete result by then.
    run_trace.finish(_result())


def test_run_trace_writes_the_decision_and_scores_onto_the_trace() -> None:
    """Without this the trace says how the agent worked but never what it decided,
    so "show me every DECLINE" — the first audit question — has no answer."""
    trace = _RecordingTrace()
    run_trace = AgentRunTrace(_trace=trace)

    run_trace.finish(_result(decision=CreditDecision.DECLINE, steps=[_score_step()]))

    (update,) = trace.updates
    assert update["output"]["decision"] == "DECLINE"
    # Filterable in one click in the Langfuse UI, unlike a metadata field.
    assert "decision:DECLINE" in update["tags"]
    # The model-side facts behind the decision, so an audit needn't parse the
    # tool span's raw output string to learn which version decided what.
    assert update["metadata"]["probability_default"] == 0.12
    assert update["metadata"]["model_version"] == "1 (run abcd1234)"
    assert update["metadata"]["decision_threshold"] == 0.3
    assert {score["name"] for score in trace.scores} == {
        "grounded_in_tools",
        "decision_matches_model",
    }


def test_run_trace_reuses_the_process_client_instead_of_building_one_per_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The thread-leak fix, asserted rather than assumed.

    `CallbackHandler(public_key=...)` builds its own Langfuse client — worker
    threads, HTTP client and an `atexit` registration per request, never
    released. Binding the handler to a trace created from the process-wide
    client is what avoids that, so the handler must come from the trace.
    """
    trace = _RecordingTrace()

    class _Client:
        def trace(self, **kwargs: Any) -> _RecordingTrace:
            trace.creation_kwargs = kwargs  # type: ignore[attr-defined]
            return trace

    monkeypatch.setattr(observability, "get_langfuse_client", lambda: _Client())

    run_trace = start_run_trace(
        session_id="s1", user_id="analyst@finrisk.local", client_id="SME-1", question="Approve?"
    )

    assert trace.handler_requests == [False], "the handler must not overwrite the trace output"
    assert run_trace.trace_id == "trace-abc123"
    assert trace.creation_kwargs["session_id"] == "s1"  # type: ignore[attr-defined]
    assert trace.creation_kwargs["user_id"] == "analyst@finrisk.local"  # type: ignore[attr-defined]


def test_finish_never_propagates_a_tracing_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Observability must not be able to fail a credit analysis."""

    class _BrokenTrace(_RecordingTrace):
        def update(self, **kwargs: Any) -> None:
            raise RuntimeError("langfuse unreachable")

    run_trace = AgentRunTrace(_trace=_BrokenTrace())

    run_trace.finish(_result())
