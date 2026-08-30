"""Scoring service: the single place that turns raw client rows into
`CreditScoreResult` / `ShapExplanation` / `ScenarioResult` objects.

Kept separate from `server.py` so the MCP tool functions stay thin
(argument validation + delegation), and so this logic is independently
unit-testable without spinning up an MCP transport.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from common.schemas import (
    CreditScoreResult,
    RiskBand,
    ScenarioParams,
    ScenarioResult,
    ShapExplanation,
)
from ml_pipeline.config import MLConfig, config
from ml_pipeline.models import Classifier
from ml_pipeline.preprocessing import get_feature_names, to_feature_frame
from ml_pipeline.shap_explainer import CreditRiskExplainer

logger = logging.getLogger(__name__)


class ClientNotFoundError(KeyError):
    """Raised when a `client_id` does not exist in the client store."""


def _is_numeric(value: object) -> bool:
    """True for values safe to coerce to float for SHAP display (excludes e.g. the sector string)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


@dataclass(frozen=True)
class ModelBundle:
    """All artifacts required to score a client, loaded once at startup."""

    model: Classifier
    preprocessor: ColumnTransformer
    explainer: CreditRiskExplainer
    feature_names: list[str]
    model_version: str


def load_model_bundle(cfg: MLConfig = config) -> ModelBundle:
    """Load the trained model + preprocessor + SHAP explainer from disk.

    Raises FileNotFoundError with an actionable message if artifacts are
    missing (a common first-run pitfall this makes explicit rather than
    surfacing an opaque joblib traceback).
    """
    if not cfg.model_path.exists() or not cfg.preprocessor_path.exists():
        raise FileNotFoundError(
            f"Model artifacts not found under {cfg.model_dir}. "
            "Run `python -m ml_pipeline.make_dataset` then "
            "`python -m ml_pipeline.train` before starting the MCP server."
        )
    model = joblib.load(cfg.model_path)
    preprocessor = joblib.load(cfg.preprocessor_path)
    feature_names = get_feature_names(preprocessor)
    background_path = cfg.model_dir / "shap_background.joblib"
    background = joblib.load(background_path) if background_path.exists() else None
    explainer = CreditRiskExplainer(
        model=model,
        feature_names=feature_names,
        model_type=cfg.model_type,
        background=background,
    )
    logger.info(
        "Loaded model bundle: %d features, version=%s", len(feature_names), cfg.model_version
    )
    return ModelBundle(
        model=model,
        preprocessor=preprocessor,
        explainer=explainer,
        feature_names=feature_names,
        model_version=cfg.model_version,
    )


class ClientStore:
    """Read-only lookup of client feature rows, keyed by `client_id`.

    Backed by the held-out test split (`holdout_test_path`), not the full
    training corpus — every client scoreable at runtime is therefore one the
    currently-deployed model never trained on. Loading the full
    `raw_data_path` instead would mean ~80% of demo-able client_ids were
    memorized during training, quietly making the live demo look more
    accurate than genuine out-of-sample performance. In production this
    would be a call to a data warehouse / feature store (e.g. a Feast online
    store) instead of a flat file.
    """

    def __init__(
        self, data_path: Path = config.holdout_test_path, id_column: str = config.id_column
    ) -> None:
        if not data_path.exists():
            raise FileNotFoundError(
                f"No held-out client dataset at {data_path}. Run `python -m ml_pipeline.train` "
                "first — it writes this file alongside the model artifacts."
            )
        self._df = pd.read_parquet(data_path).set_index(id_column, drop=False)
        self._id_column = id_column

    def get_row(self, client_id: str) -> pd.Series:
        if client_id not in self._df.index:
            raise ClientNotFoundError(f"Unknown client_id: {client_id!r}")
        row = self._df.loc[client_id]
        assert isinstance(row, pd.Series), (
            f"Expected a single row for client_id={client_id!r}, found duplicates in the index."
        )
        return row


class ScoringService:
    """High-level API used directly by MCP tool implementations."""

    def __init__(self, bundle: ModelBundle, store: ClientStore, cfg: MLConfig = config) -> None:
        self._bundle = bundle
        self._store = store
        self._cfg = cfg

    # -- internal helpers ------------------------------------------------

    def _predict_proba(self, features_df: pd.DataFrame) -> float:
        transformed = self._bundle.preprocessor.transform(features_df)
        named = to_feature_frame(transformed, self._bundle.feature_names)
        proba = np.asarray(self._bundle.model.predict_proba(named))[:, 1]
        return float(proba[0])

    def _to_score_result(self, client_id: str, probability_default: float) -> CreditScoreResult:
        threshold = self._cfg.decision_threshold
        if probability_default < threshold * 0.5:
            recommendation = "APPROVE"
        elif probability_default < threshold:
            recommendation = "REVIEW"
        else:
            recommendation = "DECLINE"
        return CreditScoreResult(
            client_id=client_id,
            probability_default=probability_default,
            risk_band=RiskBand.from_probability(probability_default),
            model_version=self._bundle.model_version,
            scored_at=datetime.now(UTC),
            decision_threshold=threshold,
            recommendation=recommendation,
        )

    # -- public API, mirrors the MCP tool surface -------------------------

    def get_credit_score(self, client_id: str) -> CreditScoreResult:
        """Score a single client by id. Raises `ClientNotFoundError` if unknown."""
        row = self._store.get_row(client_id)
        features_df = row.to_frame().T
        proba = self._predict_proba(features_df)
        return self._to_score_result(client_id, proba)

    def get_shap_explanation(self, client_id: str, render_plot: bool = True) -> ShapExplanation:
        """Compute the local SHAP explanation for a client, optionally rendering a waterfall PNG."""
        row = self._store.get_row(client_id)
        features_df = row.to_frame().T
        transformed = self._bundle.preprocessor.transform(features_df)
        raw = row.drop(labels=["defaulted_12m"], errors="ignore").to_dict()
        raw_values = {str(k): float(v) for k, v in raw.items() if _is_numeric(v)}

        explanation = self._bundle.explainer.explain_row(
            transformed[0], client_id=client_id, raw_values=raw_values
        )
        if render_plot:
            path = self._bundle.explainer.render_waterfall(
                transformed[0], client_id=client_id, out_dir=self._cfg.shap_plots_dir
            )
            explanation = explanation.model_copy(update={"plot_path": str(path)})
        return explanation

    def simulate_financial_scenario(self, client_id: str, params: ScenarioParams) -> ScenarioResult:
        """Apply a what-if perturbation to a client's features and re-score.

        Used by the agent to answer analyst questions like "what if this
        client's revenue drops 15% and they have one more late payment?"
        without needing a second model — it simply re-runs inference on a
        perturbed copy of the client's row.
        """
        baseline = self.get_credit_score(client_id)
        row = self._store.get_row(client_id).copy()

        if params.annual_revenue_delta_pct:
            row["annual_revenue"] *= 1 + params.annual_revenue_delta_pct
        if params.total_debt_delta_pct:
            row["total_debt"] *= 1 + params.total_debt_delta_pct
        if params.late_payments_12m_override is not None:
            row["late_payments_12m"] = params.late_payments_12m_override
        if params.credit_utilization_override is not None:
            row["credit_utilization"] = params.credit_utilization_override

        features_df = row.to_frame().T
        simulated_proba = self._predict_proba(features_df)
        simulated = self._to_score_result(client_id, simulated_proba)

        pd_delta = simulated.probability_default - baseline.probability_default
        direction = "increases" if pd_delta > 0 else "decreases"
        narrative = (
            f"Under the simulated scenario, {client_id}'s probability of default "
            f"{direction} by {abs(pd_delta):.1%} "
            f"({baseline.probability_default:.1%} -> {simulated.probability_default:.1%}), "
            f"moving the risk band from {baseline.risk_band.value} to {simulated.risk_band.value}."
        )
        return ScenarioResult(
            client_id=client_id,
            baseline=baseline,
            simulated=simulated,
            pd_delta=pd_delta,
            narrative=narrative,
        )


def build_default_scoring_service() -> ScoringService:
    """Convenience factory used by both the MCP server and tests."""
    bundle = load_model_bundle()
    store = ClientStore()
    return ScoringService(bundle=bundle, store=store)
