"""Unit tests for `agent.agent`, run without any LLM API key.

`FinRiskAgentRuntime.analyze` is exercised by stubbing the compiled
LangGraph graph directly (`runtime._graph.ainvoke`) rather than calling
`runtime.start()` — that avoids ever constructing a real `ChatOpenAI` /
`AzureChatOpenAI` client or spawning the MCP subprocess, so no API key or
network access is required. This isolates the logic that *is* worth unit
testing here: turning a LangGraph message trajectory into `AgentStep`s and
parsing the closing message into a `DecisionSynthesis`.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from agent.agent import FinRiskAgentRuntime, _build_llm
from common.schemas import CreditDecision


def _canned_graph_state() -> dict[str, list[AIMessage | ToolMessage]]:
    """A minimal but realistic LangGraph `messages` trajectory: one tool call, then a synthesis."""
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "call_1", "name": "get_credit_score", "args": {"client_id": "SME-1"}}
                ],
            ),
            ToolMessage(
                content='{"probability_default": 0.12, "risk_band": "LOW"}',
                tool_call_id="call_1",
                name="get_credit_score",
            ),
            AIMessage(
                content=(
                    "PD is low at 12%.\n- Debt-to-equity within range\n- No late payments\nAPPROVE."
                )
            ),
        ]
    }


@pytest.fixture
def stubbed_runtime() -> FinRiskAgentRuntime:
    """A `FinRiskAgentRuntime` with its LangGraph graph replaced by a canned async stub.

    No `_build_llm()` call, no MCP subprocess, no API key — `start()` is
    never invoked.
    """
    runtime = FinRiskAgentRuntime()
    runtime._graph = AsyncMock()
    runtime._graph.ainvoke.return_value = _canned_graph_state()
    return runtime


@pytest.mark.asyncio
async def test_analyze_parses_tool_steps_and_decision(
    stubbed_runtime: FinRiskAgentRuntime,
) -> None:
    result = await stubbed_runtime.analyze("SME-1", "Should we approve this client?")

    assert result.decision == CreditDecision.APPROVE
    assert len(result.steps) == 1
    assert result.steps[0].tool_name == "get_credit_score"
    assert result.steps[0].raw_output == {"probability_default": 0.12, "risk_band": "LOW"}
    assert "12" in result.steps[0].tool_output_summary
    stubbed_runtime._graph.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_analyze_without_start_raises() -> None:
    runtime = FinRiskAgentRuntime()

    with pytest.raises(RuntimeError, match="Agent not started"):
        await runtime.analyze("SME-1", "Should we approve this client?")


def test_parse_final_message_defaults_to_review_when_ambiguous() -> None:
    message = AIMessage(content="The client's financials are mixed, hard to call either way.")

    synthesis = FinRiskAgentRuntime._parse_final_message(message, client_id="SME-1")

    assert synthesis.decision == CreditDecision.REVIEW


def test_build_llm_defaults_to_plain_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    """No real network call is made — constructing `ChatOpenAI` only validates config,
    so a fake key is enough to exercise `_build_llm`'s provider-selection logic."""
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")

    llm = _build_llm()

    assert isinstance(llm, ChatOpenAI)
    assert not isinstance(llm, AzureChatOpenAI)


def test_build_llm_prefers_azure_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-not-real")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")

    llm = _build_llm()

    assert isinstance(llm, AzureChatOpenAI)


def test_build_llm_uses_local_server_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """`LOCAL_LLM_BASE_URL` routes `ChatOpenAI` at a local OpenAI-API-compatible
    server (Ollama, LM Studio, ...) instead of api.openai.com — no real key needed."""
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("LOCAL_LLM_MODEL", "qwen2.5:7b-instruct")

    llm = _build_llm()

    assert isinstance(llm, ChatOpenAI)
    assert not isinstance(llm, AzureChatOpenAI)
    assert str(llm.openai_api_base) == "http://localhost:11434/v1"
    assert llm.model_name == "qwen2.5:7b-instruct"


def test_build_llm_local_server_takes_precedence_over_plain_openai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "http://localhost:1234/v1")

    llm = _build_llm()

    assert str(llm.openai_api_base) == "http://localhost:1234/v1"
