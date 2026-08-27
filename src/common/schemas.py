"""Shared domain schemas for FinRisk-Agent.

These Pydantic models are the single source of truth for the shape of data
that crosses process/tool boundaries: ML pipeline -> MCP server -> Agent ->
Frontend. Keeping them in one module avoids drift between what the model
emits, what the MCP tools return, and what the agent (and UI) expect.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RiskBand(str, Enum):
    """Discretized risk bands used for human-facing communication."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    @classmethod
    def from_probability(cls, probability_default: float) -> RiskBand:
        """Map a raw PD (probability of default) to a discrete risk band.

        Thresholds are illustrative and should be recalibrated against the
        realized bad-rate per decile on a held-out validation set before
        being used for actual credit decisions.
        """
        if probability_default < 0.10:
            return cls.LOW
        if probability_default < 0.30:
            return cls.MEDIUM
        if probability_default < 0.60:
            return cls.HIGH
        return cls.CRITICAL


class ClientFeatures(BaseModel):
    """Raw tabular features for a single client/SME, pre-preprocessing.

    Field names intentionally mirror common credit-bureau / accounting
    statement vocabulary so the dataset reads naturally to a financial
    analyst reviewing the repo.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    client_id: str = Field(..., description="Unique client/SME identifier")
    annual_revenue: float = Field(..., ge=0, description="Annual revenue, EUR")
    total_debt: float = Field(..., ge=0, description="Outstanding total debt, EUR")
    debt_to_equity: float = Field(..., description="Debt-to-equity ratio")
    current_ratio: float = Field(..., ge=0, description="Current assets / current liabilities")
    ebitda_margin: float = Field(..., description="EBITDA / revenue")
    days_payable_outstanding: float = Field(..., ge=0)
    days_sales_outstanding: float = Field(..., ge=0)
    late_payments_12m: int = Field(..., ge=0, description="Count of late payments, last 12 months")
    years_in_business: float = Field(..., ge=0)
    sector: str = Field(..., description="NACE / sector code")
    employees: int = Field(..., ge=0)
    credit_utilization: float = Field(
        ..., ge=0, le=2.0, description="Used credit / granted credit line"
    )


class CreditScoreResult(BaseModel):
    """Output of the scoring model for a single client, as served by MCP."""

    model_config = ConfigDict(frozen=True)

    client_id: str
    probability_default: float = Field(..., ge=0.0, le=1.0, description="Model PD (0-1)")
    risk_band: RiskBand
    model_version: str
    scored_at: datetime
    decision_threshold: float = Field(..., description="Operating threshold used for accept/reject")
    recommendation: Literal["APPROVE", "REVIEW", "DECLINE"]

    @field_validator("recommendation", mode="before")
    @classmethod
    def _derive_recommendation_if_missing(cls, v: str | None) -> str:
        return v or "REVIEW"


class ShapContribution(BaseModel):
    """A single feature's SHAP contribution to one prediction."""

    feature: str
    value: float = Field(..., description="Raw feature value for this client")
    shap_value: float = Field(..., description="Signed contribution to the log-odds output")


class ShapExplanation(BaseModel):
    """Local (per-client) SHAP explanation, ready for waterfall rendering."""

    model_config = ConfigDict(frozen=True)

    client_id: str
    base_value: float = Field(..., description="Expected value (model output with no evidence)")
    contributions: list[ShapContribution]
    top_positive_drivers: list[str] = Field(..., description="Features pushing risk up, sorted")
    top_negative_drivers: list[str] = Field(..., description="Features pushing risk down, sorted")
    plot_path: str | None = Field(
        None, description="Path to a rendered waterfall PNG, if generated"
    )


class ScenarioParams(BaseModel):
    """User-supplied overrides for a what-if financial simulation."""

    model_config = ConfigDict(extra="forbid")

    annual_revenue_delta_pct: float = Field(0.0, description="e.g. -0.15 for -15% revenue shock")
    total_debt_delta_pct: float = Field(0.0)
    late_payments_12m_override: int | None = Field(None, ge=0)
    credit_utilization_override: float | None = Field(None, ge=0, le=2.0)


class ScenarioResult(BaseModel):
    """Delta between baseline and simulated score for a what-if scenario."""

    model_config = ConfigDict(frozen=True)

    client_id: str
    baseline: CreditScoreResult
    simulated: CreditScoreResult
    pd_delta: float = Field(
        ..., description="simulated.probability_default - baseline.probability_default"
    )
    narrative: str = Field(..., description="One-line human-readable delta summary")


class AgentStep(BaseModel):
    """One reasoning/tool-call step in the agent's trajectory.

    This is the payload rendered live by the frontend's Chain-of-Thought
    panel (`AgentTrace.jsx`) — one entry per MCP tool invocation, in order.
    """

    step_index: int
    tool_name: str
    tool_input: dict[str, object]
    tool_output_summary: str = Field(
        ..., description="Short, display-ready summary of the tool result"
    )
    raw_output: dict[str, object] | None = Field(
        None,
        description="Full parsed tool payload (e.g. CreditScoreResult as dict), for UI rendering",
    )
    latency_ms: float
    status: Literal["success", "error"] = "success"


class CreditDecision(str, Enum):
    """The agent's final recommended credit decision."""

    APPROVE = "APPROVE"
    REVIEW = "REVIEW"
    DECLINE = "DECLINE"


class AgentAnalysisResult(BaseModel):
    """Final structured output of one agent run over a client."""

    model_config = ConfigDict(frozen=True)

    client_id: str
    question: str
    decision: CreditDecision
    summary: str = Field(..., description="Analyst-facing narrative synthesis, 3-6 sentences")
    key_drivers: list[str] = Field(..., description="Bullet-point risk drivers, plain language")
    steps: list[AgentStep]
    total_latency_ms: float
    langfuse_trace_id: str | None = Field(
        None, description="Trace id for cross-linking to Langfuse UI"
    )
