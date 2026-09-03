"""Scoring service: the single place that turns raw client rows into
`CreditScoreResult` / `ShapExplanation` / `ScenarioResult` objects.

Kept separate from `server.py` so the MCP tool functions stay thin
(argument validation + delegation), and so this logic is independently
unit-testable without spinning up an MCP transport.
"""

from __future__ import annotations

import json
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
from ml_pipeline.tracking import (
    HOLDOUT_ARTIFACT,
    METADATA_ARTIFACT,
    METRICS_ARTIFACT,
    PREPROCESSOR_ARTIFACT,
    SHAP_BACKGROUND_ARTIFACT,
    RegisteredModel,
    resolve_model,
)

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
    run_id: str
    artifacts_dir: Path


def build_model_bundle(cfg: MLConfig, target: RegisteredModel) -> ModelBundle:
    """Assemble the scoring bundle from one registry-resolved model version.

    The preprocessor and SHAP background come from the artifacts of the run
    that produced this exact version, so the transform applied at serving
    time is provably the one the model was fitted with.

    Trust boundary: these artifacts are unpickled, so the tracking store must
    be trusted like code — pointing `FINRISK_MLFLOW_TRACKING_URI` at a store
    you don't control is equivalent to running whatever it contains.
    """
    preprocessor: ColumnTransformer = joblib.load(target.artifact(PREPROCESSOR_ARTIFACT))
    background_path = target.artifacts_dir / SHAP_BACKGROUND_ARTIFACT
    background: np.ndarray | None = (
        joblib.load(background_path) if background_path.exists() else None
    )

    feature_names = get_feature_names(preprocessor)
    explainer = CreditRiskExplainer(
        model=target.model,
        feature_names=feature_names,
        model_type=cfg.model_type,
        background=background,
    )
    logger.info(
        "Loaded model bundle: %d features, %s v%s (@%s, run %s)",
        len(feature_names),
        cfg.mlflow_registered_model,
        target.version,
        target.alias,
        target.run_id,
    )
    return ModelBundle(
        model=target.model,
        preprocessor=preprocessor,
        explainer=explainer,
        feature_names=feature_names,
        model_version=target.display_version,
        run_id=target.run_id,
        artifacts_dir=target.artifacts_dir,
    )


def load_model_bundle(cfg: MLConfig = config, *, alias: str | None = None) -> ModelBundle:
    """Resolve a registry alias (default `champion`) and build its scoring bundle.

    Raises `MlflowException` if the alias doesn't resolve and `FileNotFoundError`
    if the run behind it is missing an artifact — both loudly, at startup: a
    scoring server that can't load its declared model must not start rather
    than answer from something else.
    """
    return build_model_bundle(cfg, resolve_model(cfg, alias=alias))


class ClientStore:
    """Read-only lookup of client feature rows, keyed by `client_id`.

    Backed by the held-out split logged inside the served model's own MLflow
    run, not the full training corpus — every client scoreable at runtime is
    one that specific model never trained on. Serving the full dataset
    instead would mean ~80% of demo-able client_ids were memorized during
    training, quietly making the live demo look more accurate than genuine
    out-of-sample performance. In production this would be a call to a data
    warehouse / feature store instead of a flat file.
    """

    def __init__(self, data_path: Path, id_column: str = config.id_column) -> None:
        if not data_path.exists():
            raise FileNotFoundError(
                f"No held-out client dataset at {data_path}. Run "
                "`python -m ml_pipeline.train` first — it logs this file into the run."
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

    def get_shap_explanation(self, client_id: str) -> ShapExplanation:
        """Compute the local SHAP explanation for a client.

        Returns data, not files — an earlier version also rendered a
        waterfall PNG per call that nothing read and that grew unboundedly.
        The per-model equivalent (the global SHAP summary) is logged into
        the MLflow run by `ml_pipeline.eval` instead.
        """
        row = self._store.get_row(client_id)
        features_df = row.to_frame().T
        transformed = self._bundle.preprocessor.transform(features_df)
        raw = row.drop(labels=[self._cfg.target_column], errors="ignore").to_dict()
        raw_values = {str(k): float(v) for k, v in raw.items() if _is_numeric(v)}

        return self._bundle.explainer.explain_row(
            transformed[0], client_id=client_id, raw_values=raw_values
        )

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

    def get_model_card(self) -> dict[str, object]:
        """Metadata + held-out metrics of the served model, read from its own run.

        Backs the `finrisk://model/card` MCP resource. Both files are
        artifacts of the run the alias resolves to, so the card can never
        describe a different model than the one answering the tools —
        `metrics.json` is absent until `ml_pipeline.eval` has been run
        against this version, which the empty dict makes visible rather than
        papering over with someone else's numbers.
        """
        artifacts = self._bundle.artifacts_dir

        def _read(name: str) -> dict[str, object]:
            path = artifacts / name
            return dict(json.loads(path.read_text())) if path.exists() else {}

        return {
            "registered_model": self._cfg.mlflow_registered_model,
            "alias": self._cfg.mlflow_model_alias,
            "model_version": self._bundle.model_version,
            "run_id": self._bundle.run_id,
            "metadata": _read(METADATA_ARTIFACT),
            "metrics": _read(METRICS_ARTIFACT),
        }


def build_default_scoring_service(cfg: MLConfig = config) -> ScoringService:
    """Build the service from a single registry resolution.

    Resolving once — rather than once for the model and again for the client
    store — is what guarantees the served model and the client rows it scores
    come from the same run.
    """
    target = resolve_model(cfg)
    return ScoringService(
        bundle=build_model_bundle(cfg, target),
        store=ClientStore(target.artifact(HOLDOUT_ARTIFACT), id_column=cfg.id_column),
        cfg=cfg,
    )
