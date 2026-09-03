"""FinRisk-Agent MCP server.

Exposes the trained credit-risk model and its SHAP explanations as three
MCP tools, plus a model-card resource, over the Model Context Protocol via
FastMCP. This is the sole boundary between the ML artifacts and everything
downstream (the LangChain agent, and — indirectly — the React dashboard).

Tools:
    get_credit_score(client_id)                         -> CreditScoreResult
    get_shap_explanation(client_id)                      -> ShapExplanation
    simulate_financial_scenario(client_id, params)       -> ScenarioResult

Resources:
    finrisk://model/card                                 -> model metadata (JSON)

Transport:
    Defaults to stdio (the standard choice for a locally-spawned tool
    server consumed by a LangChain/LangGraph agent subprocess). Run with
    `--transport streamable-http` to expose it over HTTP instead, which is
    how the Docker stack reaches it from the agent container.
"""

from __future__ import annotations

import json
import logging
from typing import Annotated, Literal

import typer
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import Field

from common.schemas import CreditScoreResult, ScenarioParams, ScenarioResult, ShapExplanation
from mcp_server.scoring_service import (
    ClientNotFoundError,
    ScoringService,
    build_default_scoring_service,
)
from ml_pipeline.config import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

mcp = FastMCP(
    name="finrisk-agent",
    instructions=(
        "Tools for scoring SME/client credit risk with a trained ML model, "
        "explaining individual predictions via SHAP, and running what-if "
        "financial scenarios. Always call get_credit_score before "
        "get_shap_explanation for the same client in a single turn, and "
        "prefer simulate_financial_scenario over re-deriving numbers "
        "manually when an analyst asks a hypothetical question."
    ),
)

# Loaded lazily on first tool call so `fastmcp dev` / --help don't require
# trained artifacts to exist on disk just to introspect the server.
_service: ScoringService | None = None


def _get_service() -> ScoringService:
    global _service
    if _service is None:
        _service = build_default_scoring_service()
    return _service


@mcp.tool()
def get_credit_score(
    client_id: Annotated[str, Field(description="Client/SME identifier, e.g. 'SME-000123'")],
) -> CreditScoreResult:
    """Return the current credit-risk score (PD, risk band, recommendation) for a client.

    This is the first tool to call for any client-scoring request — it
    returns the probability of default, a discretized risk band, and an
    APPROVE/REVIEW/DECLINE recommendation derived from the model's
    operating threshold.
    """
    try:
        result = _get_service().get_credit_score(client_id)
        logger.info(
            "get_credit_score(%s) -> PD=%.4f band=%s",
            client_id,
            result.probability_default,
            result.risk_band.value,
        )
        return result
    except ClientNotFoundError as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool()
def get_shap_explanation(
    client_id: Annotated[str, Field(description="Client/SME identifier, e.g. 'SME-000123'")],
) -> ShapExplanation:
    """Return the SHAP feature-attribution explanation behind a client's score.

    Use this to answer "why" a client received their score. The response
    includes the ranked top positive (risk-increasing) and negative
    (risk-decreasing) drivers, suitable for direct quoting in an analyst
    summary, plus a base_value (the model's average output with no
    evidence) that the contributions sum away from.
    """
    try:
        explanation = _get_service().get_shap_explanation(client_id)
        logger.info(
            "get_shap_explanation(%s) -> top_positive=%s top_negative=%s",
            client_id,
            explanation.top_positive_drivers,
            explanation.top_negative_drivers,
        )
        return explanation
    except ClientNotFoundError as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool()
def simulate_financial_scenario(
    client_id: Annotated[str, Field(description="Client/SME identifier, e.g. 'SME-000123'")],
    annual_revenue_delta_pct: Annotated[
        float, Field(description="Relative revenue shock, e.g. -0.15 for -15%")
    ] = 0.0,
    total_debt_delta_pct: Annotated[
        float, Field(description="Relative debt change, e.g. 0.20 for +20%")
    ] = 0.0,
    late_payments_12m_override: Annotated[
        int | None, Field(ge=0, description="Override the count of late payments in the last 12m")
    ] = None,
    credit_utilization_override: Annotated[
        float | None, Field(ge=0, le=2.0, description="Override the credit utilization ratio")
    ] = None,
) -> ScenarioResult:
    """Re-score a client under a hypothetical ('what-if') change to their financial profile.

    Use this whenever an analyst asks a hypothetical question ("what if
    revenue drops 15%?", "what if they miss two more payments?") instead of
    reasoning about the direction of the effect yourself — the tool
    re-runs real inference on the perturbed profile and returns the exact
    delta plus a ready-to-quote narrative sentence.
    """
    params = ScenarioParams(
        annual_revenue_delta_pct=annual_revenue_delta_pct,
        total_debt_delta_pct=total_debt_delta_pct,
        late_payments_12m_override=late_payments_12m_override,
        credit_utilization_override=credit_utilization_override,
    )
    try:
        result = _get_service().simulate_financial_scenario(client_id, params)
        logger.info("simulate_financial_scenario(%s) -> pd_delta=%.4f", client_id, result.pd_delta)
        return result
    except ClientNotFoundError as exc:
        raise ToolError(str(exc)) from exc


@mcp.resource("finrisk://model/card")
def model_card() -> str:
    """Model card resource: version, features, and metrics for the currently served model.

    Exposed as a Resource (not a Tool) since it is static, cacheable
    context about the model itself rather than an action performed against
    a specific client — the right MCP primitive for "background reading"
    material the agent can pull into context once per session.

    Everything here comes from the MLflow run behind the resolved registry
    alias, so the card always describes the same model version the three
    tools above are scoring with.
    """
    return json.dumps(_get_service().get_model_card(), indent=2, default=str)


cli = typer.Typer(add_completion=False)


@cli.command()
def main(
    transport: Annotated[
        Literal["stdio", "streamable-http", "sse"],
        typer.Option(help="MCP transport to serve over"),
    ] = "stdio",
    host: Annotated[str, typer.Option(help="Host, for http/sse transports")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port, for http/sse transports")] = 8000,
    alias: Annotated[
        str,
        typer.Option(
            help="Registry alias to serve (default: the configured one, `champion`). "
            "e.g. --alias challenger, to inspect a candidate before promoting it."
        ),
    ] = "",
) -> None:
    """Entry point: `finrisk-mcp --transport streamable-http --port 8000`."""
    if alias:
        # Which version is served is fixed for the server's lifetime, so it
        # has no business in the tool signatures an agent sees — set once
        # here instead; `_get_service()` reads it lazily on first tool call.
        config.mlflow_model_alias = alias
    if transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport=transport, host=host, port=port)


if __name__ == "__main__":
    cli()
