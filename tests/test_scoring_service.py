"""Unit tests for `mcp_server.scoring_service`.

Trains a tiny real LightGBM model on synthetic data within the test fixture
(no reliance on pre-existing artifacts on disk) so these tests are
hermetic and fast — a deliberate contrast with `eval.py`, which evaluates
the *actual* persisted production artifact.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import joblib
import lightgbm as lgb
import pytest
from sklearn.linear_model import LogisticRegression

from common.schemas import ScenarioParams
from mcp_server.scoring_service import (
    ClientNotFoundError,
    ClientStore,
    ModelBundle,
    ScoringService,
    load_model_bundle,
)
from ml_pipeline.config import MLConfig, config
from ml_pipeline.make_dataset import _simulate
from ml_pipeline.preprocessing import build_preprocessor, get_feature_names
from ml_pipeline.shap_explainer import CreditRiskExplainer
from ml_pipeline.train import _split


@pytest.fixture
def tmp_cfg(tmp_path: Path) -> MLConfig:
    cfg = MLConfig(
        data_dir=tmp_path,
        raw_data_path=tmp_path / "clients.parquet",
        model_dir=tmp_path / "models",
        shap_plots_dir=tmp_path / "reports" / "shap",
    )
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    df = _simulate(n_clients=300, seed=cfg.random_state)
    df.to_parquet(cfg.raw_data_path, index=False)
    return cfg


@pytest.fixture
def scoring_service(tmp_cfg: MLConfig) -> ScoringService:
    from ml_pipeline.preprocessing import load_raw_dataset

    df = load_raw_dataset(tmp_cfg)
    preprocessor = build_preprocessor(tmp_cfg)
    X = preprocessor.fit_transform(df)
    y = df[tmp_cfg.target_column].to_numpy()

    model = lgb.LGBMClassifier(n_estimators=50, num_leaves=7, min_child_samples=5, verbosity=-1)
    model.fit(X, y, feature_name=get_feature_names(preprocessor))

    joblib.dump(model, tmp_cfg.model_path)
    joblib.dump(preprocessor, tmp_cfg.preprocessor_path)

    explainer = CreditRiskExplainer(model=model, feature_names=get_feature_names(preprocessor))
    bundle = ModelBundle(
        model=model,
        preprocessor=preprocessor,
        explainer=explainer,
        feature_names=get_feature_names(preprocessor),
        model_version="test",
    )
    store = ClientStore(data_path=tmp_cfg.raw_data_path, id_column=tmp_cfg.id_column)
    return ScoringService(bundle=bundle, store=store, cfg=tmp_cfg)


def test_get_credit_score_returns_valid_probability(scoring_service: ScoringService) -> None:
    result = scoring_service.get_credit_score("SME-000001")

    assert 0.0 <= result.probability_default <= 1.0
    assert result.recommendation in {"APPROVE", "REVIEW", "DECLINE"}


def test_get_credit_score_unknown_client_raises(scoring_service: ScoringService) -> None:
    with pytest.raises(ClientNotFoundError):
        scoring_service.get_credit_score("SME-999999")


def test_shap_explanation_contributions_sum_close_to_margin(
    scoring_service: ScoringService,
) -> None:
    explanation = scoring_service.get_shap_explanation("SME-000002", render_plot=False)

    assert len(explanation.contributions) > 0
    assert explanation.plot_path is None  # render_plot=False was respected


def test_simulate_scenario_revenue_shock_moves_pd(scoring_service: ScoringService) -> None:
    params = ScenarioParams(annual_revenue_delta_pct=-0.5, late_payments_12m_override=5)

    scenario = scoring_service.simulate_financial_scenario("SME-000003", params)

    assert scenario.simulated.probability_default != scenario.baseline.probability_default
    assert scenario.client_id == "SME-000003"
    assert "SME-000003" in scenario.narrative


def test_load_model_bundle_logistic_regression_end_to_end(tmp_path: Path) -> None:
    """`load_model_bundle` must work identically for `model_type="logistic_regression"`,
    including loading the SHAP background sample `train.py` persists for non-tree models."""
    cfg = MLConfig(
        model_type="logistic_regression",
        data_dir=tmp_path,
        raw_data_path=tmp_path / "clients.parquet",
        model_dir=tmp_path / "models",
        shap_plots_dir=tmp_path / "reports" / "shap",
    )
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    df = _simulate(n_clients=300, seed=cfg.random_state)
    df.to_parquet(cfg.raw_data_path, index=False)

    preprocessor = build_preprocessor(cfg)
    X = preprocessor.fit_transform(df)
    y = df[cfg.target_column].to_numpy()
    model = LogisticRegression(**cfg.logreg_params).fit(X, y)

    joblib.dump(model, cfg.model_path)
    joblib.dump(preprocessor, cfg.preprocessor_path)
    joblib.dump(X[:50], cfg.model_dir / "shap_background.joblib")

    bundle = load_model_bundle(cfg)
    service = ScoringService(
        bundle=bundle,
        store=ClientStore(data_path=cfg.raw_data_path, id_column=cfg.id_column),
        cfg=cfg,
    )

    assert isinstance(bundle.model, LogisticRegression)
    score = service.get_credit_score("SME-000001")
    assert 0.0 <= score.probability_default <= 1.0

    explanation = service.get_shap_explanation("SME-000002", render_plot=False)
    assert len(explanation.contributions) > 0


def test_load_model_bundle_raises_when_artifacts_missing(tmp_path: Path) -> None:
    cfg = MLConfig(model_dir=tmp_path / "models")

    with pytest.raises(FileNotFoundError):
        load_model_bundle(cfg)


def test_client_store_raises_when_data_file_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ClientStore(data_path=tmp_path / "does_not_exist.parquet", id_column="client_id")


def test_to_score_result_recommendation_bands(scoring_service: ScoringService) -> None:
    threshold = scoring_service._cfg.decision_threshold

    approve = scoring_service._to_score_result("SME-A", threshold * 0.1)
    review = scoring_service._to_score_result("SME-B", threshold * 0.75)
    decline = scoring_service._to_score_result("SME-C", threshold * 1.5)

    assert approve.recommendation == "APPROVE"
    assert review.recommendation == "REVIEW"
    assert decline.recommendation == "DECLINE"


def test_get_shap_explanation_render_plot_true_sets_plot_path(
    scoring_service: ScoringService,
) -> None:
    explanation = scoring_service.get_shap_explanation("SME-000004", render_plot=True)

    assert explanation.plot_path is not None


def test_simulate_scenario_debt_and_utilization_overrides_move_pd(
    scoring_service: ScoringService,
) -> None:
    params = ScenarioParams(total_debt_delta_pct=0.5, credit_utilization_override=1.3)

    scenario = scoring_service.simulate_financial_scenario("SME-000005", params)

    assert scenario.simulated.probability_default != scenario.baseline.probability_default


def test_client_store_default_data_path_is_holdout_test_not_raw_data() -> None:
    """ClientStore's default must point at the held-out split, not the full training
    corpus, so a live server never serves clients the deployed model was trained on."""
    default = inspect.signature(ClientStore.__init__).parameters["data_path"].default

    assert default == config.holdout_test_path
    assert default != config.raw_data_path


def test_client_store_only_serves_holdout_clients_not_training_clients(
    tmp_path: Path,
) -> None:
    """Reproduces the actual fix end-to-end: a client_id that was part of the training
    fold must be unreachable through the store that backs the live MCP tools, while a
    held-out client_id works normally."""
    cfg = MLConfig(
        data_dir=tmp_path,
        raw_data_path=tmp_path / "clients.parquet",
        model_dir=tmp_path / "models",
        shap_plots_dir=tmp_path / "reports" / "shap",
    )
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    df = _simulate(n_clients=300, seed=cfg.random_state)
    df.to_parquet(cfg.raw_data_path, index=False)

    train_df, test_df = _split(cfg, df)
    test_df.to_parquet(cfg.holdout_test_path, index=False)

    store = ClientStore(data_path=cfg.holdout_test_path, id_column=cfg.id_column)

    train_only_id = next(iter(set(train_df["client_id"]) - set(test_df["client_id"])))
    with pytest.raises(ClientNotFoundError):
        store.get_row(train_only_id)

    holdout_id = str(test_df["client_id"].iloc[0])
    assert store.get_row(holdout_id)["client_id"] == holdout_id
