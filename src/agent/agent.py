"""FinRisk analyst agent.

A LangGraph ReAct agent that consumes the FinRisk-Agent MCP server's three
tools (`get_credit_score`, `get_shap_explanation`, `simulate_financial_scenario`)
to answer a financial analyst's natural-language question about a client,
producing a structured `AgentAnalysisResult` (decision + narrative + full
tool-call trajectory) with end-to-end Langfuse tracing.

Run standalone:
    python -m agent.agent SME-000123 --question "Should we approve this client?"

Run as an API (consumed by the React dashboard):
    uvicorn agent.agent:api --reload --port 8080
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import typer
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel

# Must run before `agent.observability` is imported: `ObservabilityConfig` reads
# LANGFUSE_* via `os.getenv(...)` at class-definition time (import time), and
# `_build_llm` below does the same for LOCAL_LLM_*/AZURE_OPENAI_*/OPENAI_*.
# Neither `uv run` nor pydantic-settings' MLConfig (which only loads .env for its
# own fields) puts .env values into the real process environment — only this does.
load_dotenv()

from agent.observability import build_callback_handler, flush, get_langfuse_client  # noqa: E402
from common.schemas import AgentAnalysisResult, AgentStep, CreditDecision  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a senior credit risk analyst assistant for FinRisk-Agent.

You have access to tools that score a client's credit risk, explain that \
score with SHAP feature attributions, and simulate what-if financial \
scenarios. Ground every claim in tool output — never invent a probability \
of default, a SHAP driver, or a scenario delta.

Standard workflow for a scoring question:
1. Call get_credit_score to obtain the current PD and risk band.
2. Call get_shap_explanation to identify the top drivers behind that score.
3. If the analyst asks a hypothetical ("what if..."), also call \
simulate_financial_scenario with the relevant parameter deltas.
4. Synthesize a concise, analyst-facing narrative (plain language, no \
jargon dump) and end with a clear APPROVE / REVIEW / DECLINE recommendation \
consistent with the tool's own recommendation field unless you have a \
specific, stated reason to diverge — and if you diverge, say why explicitly.
"""


def _build_llm() -> ChatOpenAI | AzureChatOpenAI:
    """Build the chat model. Priority: Azure OpenAI > a local server > plain OpenAI.

    Mirrors the TOD-agent convention of Azure OpenAI with a reasoning-capable
    model via env-driven configuration, while remaining runnable against
    plain OpenAI for anyone cloning the repo without Azure access.

    Setting `LOCAL_LLM_BASE_URL` switches to a local, OpenAI-API-compatible
    server instead — e.g. Ollama (`http://localhost:11434/v1`), LM Studio
    (`http://localhost:1234/v1`), or llama.cpp's/vLLM's OpenAI-compatible
    server. This reuses `ChatOpenAI` itself (no extra dependency): these
    servers speak the same `/chat/completions` wire format, tool calling
    included, which is what `create_react_agent` needs. Pick a model that
    actually supports tool calling (e.g. `qwen2.5:7b-instruct` or
    `llama3.1:8b-instruct` in Ollama) — not every local model does.
    """
    if os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"):
        return AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            temperature=0,
        )
    local_base_url = os.getenv("LOCAL_LLM_BASE_URL")
    if local_base_url:
        return ChatOpenAI(
            base_url=local_base_url,
            api_key=os.getenv("LOCAL_LLM_API_KEY", "not-needed"),
            model=os.getenv("LOCAL_LLM_MODEL", "qwen2.5:7b-instruct"),
            temperature=0,
        )
    return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4.1"), temperature=0)


class DecisionSynthesis(BaseModel):
    """Structured final synthesis, extracted from the agent's closing message."""

    decision: CreditDecision
    summary: str
    key_drivers: list[str]


class FinRiskAgentRuntime:
    """Owns the MCP client, the compiled LangGraph agent, and the analysis entrypoint.

    One instance is created at process startup (see `lifespan` below) and
    reused across requests — the MCP subprocess and tool bindings are
    expensive to set up per-call.
    """

    def __init__(self, mcp_server_module: str = "mcp_server.server") -> None:
        self._mcp_server_module = mcp_server_module
        self._mcp_client: MultiServerMCPClient | None = None
        self._graph: Any = None

    async def start(self) -> None:
        """Spawn the MCP server as a stdio subprocess and compile the ReAct agent.

        `MultiServerMCPClient` in the pinned `langchain-mcp-adapters` release
        is an async context manager: entering it is what actually spawns the
        stdio subprocess and performs the MCP handshake for every configured
        server. We enter it manually (rather than via `async with`) so the
        subprocess stays alive for the lifetime of the runtime instead of
        being torn down at the end of a `with` block, and close it explicitly
        in `stop()`.
        """
        self._mcp_client = MultiServerMCPClient(
            {
                "finrisk": {
                    "command": "python",
                    "args": ["-m", self._mcp_server_module],
                    "transport": "stdio",
                }
            }
        )
        await self._mcp_client.__aenter__()
        tools = self._mcp_client.get_tools()
        logger.info("Loaded %d MCP tools: %s", len(tools), [t.name for t in tools])

        llm = _build_llm()
        self._graph = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)

    async def analyze(
        self, client_id: str, question: str, *, user_id: str = "analyst@finrisk.local"
    ) -> AgentAnalysisResult:
        """Run the agent for one (client_id, question) pair and return a structured result."""
        if self._graph is None:
            raise RuntimeError("Agent not started — call `await runtime.start()` first.")

        session_id = str(uuid.uuid4())
        handler = build_callback_handler(
            session_id=session_id, user_id=user_id, client_id=client_id, question=question
        )
        config = {"callbacks": [handler]} if handler else {}

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Client: {client_id}\nAnalyst question: {question}"),
        ]

        t0 = time.perf_counter()
        result_state = await self._graph.ainvoke({"messages": messages}, config=config)
        total_latency_ms = (time.perf_counter() - t0) * 1000

        steps = self._extract_steps(result_state["messages"])
        final_message = result_state["messages"][-1]
        synthesis = self._parse_final_message(final_message, client_id=client_id)

        langfuse_client = get_langfuse_client()
        trace_id = getattr(handler, "trace_id", None) if langfuse_client else None

        return AgentAnalysisResult(
            client_id=client_id,
            question=question,
            decision=synthesis.decision,
            summary=synthesis.summary,
            key_drivers=synthesis.key_drivers,
            steps=steps,
            total_latency_ms=total_latency_ms,
            langfuse_trace_id=trace_id,
        )

    @staticmethod
    def _extract_steps(messages: list[Any]) -> list[AgentStep]:
        """Turn the LangGraph message list into the ordered `AgentStep` trace for the UI."""
        steps: list[AgentStep] = []
        pending_calls: dict[str, dict[str, Any]] = {}

        for msg in messages:
            if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                for call in msg.tool_calls:
                    call_id = call.get("id")
                    if call_id is not None:
                        pending_calls[call_id] = {"name": call["name"], "args": call["args"]}
            elif isinstance(msg, ToolMessage):
                call_info = pending_calls.get(msg.tool_call_id, {"name": msg.name, "args": {}})
                steps.append(
                    AgentStep(
                        step_index=len(steps),
                        tool_name=call_info["name"],
                        tool_input=call_info["args"],
                        tool_output_summary=_summarize_tool_output(msg.content),
                        raw_output=_parse_tool_output(msg.content),
                        latency_ms=0.0,  # per-tool latency requires astream_events; see README
                        status="error"
                        if getattr(msg, "status", "success") == "error"
                        else "success",
                    )
                )
        return steps

    @staticmethod
    def _parse_final_message(message: AIMessage, *, client_id: str) -> DecisionSynthesis:
        """Best-effort structured parse of the agent's closing narrative.

        The agent is prompted to ground its recommendation in tool output;
        this parser looks for an explicit decision keyword and falls back to
        REVIEW (the conservative default) if the model's phrasing doesn't
        contain an unambiguous match, rather than guessing.
        """
        content = message.content if isinstance(message.content, str) else str(message.content)
        upper = content.upper()
        if "DECLINE" in upper:
            decision = CreditDecision.DECLINE
        elif "APPROVE" in upper:
            decision = CreditDecision.APPROVE
        else:
            decision = CreditDecision.REVIEW

        bullets = [
            line.strip("-* \t")
            for line in content.splitlines()
            if line.strip().startswith(("-", "*"))
        ]
        return DecisionSynthesis(
            decision=decision,
            summary=content.strip(),
            key_drivers=bullets[:6] or [f"See full narrative for {client_id}'s risk drivers."],
        )

    async def stop(self) -> None:
        """Close the MCP subprocess/session and flush any pending Langfuse events."""
        if self._mcp_client is not None:
            await self._mcp_client.__aexit__(None, None, None)
        flush()


def _parse_tool_output(content: str | list[Any]) -> dict[str, Any] | None:
    """Parse a raw MCP tool result into a plain dict for the frontend, or None if not JSON.

    FastMCP tools return their Pydantic models serialized as JSON text over
    the wire; this recovers the structured payload (e.g. the full
    `CreditScoreResult` fields) so `ScoreCard`/`ShapChart` can render exact
    values rather than the truncated summary string.
    """
    text = content if isinstance(content, str) else json.dumps(content, default=str)
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except (json.JSONDecodeError, TypeError):
        return None


def _summarize_tool_output(content: str | list[Any]) -> str:
    """Truncate/condense a raw tool result for compact display in the trace panel."""
    text = content if isinstance(content, str) else json.dumps(content, default=str)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "probability_default" in parsed:
            return f"PD={parsed['probability_default']:.1%}, band={parsed.get('risk_band')}"
        if isinstance(parsed, dict) and "top_positive_drivers" in parsed:
            return f"Top drivers: {', '.join(parsed['top_positive_drivers'][:3])}"
        if isinstance(parsed, dict) and "pd_delta" in parsed:
            narrative = parsed.get("narrative", text)
            return str(narrative)[:180]
    except (json.JSONDecodeError, TypeError):
        pass
    return text[:180]


# ---------------------------------------------------------------------------
# FastAPI serving layer — consumed by the React dashboard
# ---------------------------------------------------------------------------

runtime = FinRiskAgentRuntime()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    await runtime.start()
    yield
    await runtime.stop()


api = FastAPI(
    title="FinRisk-Agent API",
    description="Agentic layer over the FinRisk credit-scoring MCP server.",
    version="0.1.0",
    lifespan=lifespan,
)
api.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    client_id: str
    question: str = "Should we approve this client's credit request?"


@api.post("/analyze", response_model=AgentAnalysisResult)
async def analyze(payload: AnalyzeRequest) -> AgentAnalysisResult:
    """Run the agent and return the full structured result (used for non-streaming clients)."""
    try:
        return await runtime.analyze(payload.client_id, payload.question)
    except Exception as exc:
        logger.exception("Agent run failed for client_id=%s", payload.client_id)
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@api.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

cli = typer.Typer(add_completion=False)


@cli.command()
def analyze_cli(
    client_id: str, question: str = "Should we approve this client's credit request?"
) -> None:
    """One-shot CLI run: `python -m agent.agent SME-000123 --question "..."`."""

    async def _run() -> None:
        rt = FinRiskAgentRuntime()
        await rt.start()
        try:
            result = await rt.analyze(client_id, question)
            typer.echo(result.model_dump_json(indent=2))
        finally:
            await rt.stop()

    asyncio.run(_run())


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
