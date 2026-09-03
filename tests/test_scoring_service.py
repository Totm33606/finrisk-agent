"""Unit tests for `mcp_server.scoring_service`.

Trains tiny real models on synthetic data inside the fixtures, and — for
anything touching artifact resolution — publishes them to a real MLflow
file store rooted in `tmp_path`. Mocking the registry here would test the
mock; the file store is fast enough that there's no reason to.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd
import pytest
from mlflow.exceptions import MlflowException
from sklearn.linear_model import LogisticRegression

from common.schemas import ScenarioParams
from mcp_server.scoring_service import (
    ClientNotFoundError,
    ClientStore,
    ModelBundle,
    ScoringService,
    build_default_scoring_service,
    load_model_bundle,
)
from ml_pipeline import tracking
from ml_pipeline.config import MLConfig
from ml_pipeline.make_dataset import _simulate
from ml_pipeline.preprocessing import build_preprocessor, get_feature_names, to_feature_frame
from ml_pipeline.shap_explainer import CreditRiskExplainer
from ml_pipeline.train import _split


@pytest.fixture
def tmp_cfg(tmp_path: Path) -> MLConfig:
    cfg = MLConfig(
        data_dir=tmp_path,
        raw_data_path=tmp_path / "clients.parquet",
        mlflow_dir=tmp_path / "mlruns",
    )
    df = _simulate(n_clients=300, seed=cfg.random_state)
    df.to_parquet(cfg.raw_data_path, index=False)
    return cfg


@pytest.fixture
def scoring_service(tmp_cfg: MLConfig, tmp_path: Path) -> ScoringService:
    """A service assembled in memory, bypassing the registry.

    These tests exercise scoring/SHAP/scenario logic, not artifact resolution
    — that's covered separately by the registry tests below.
    """
    from ml_pipeline.preprocessing import load_raw_dataset

    df = load_raw_dataset(tmp_cfg)
    preprocessor = build_preprocessor(tmp_cfg)
    X = preprocessor.fit_transform(df)
    y = df[tmp_cfg.target_column].to_numpy()

    model = lgb.LGBMClassifier(n_estimators=50, num_leaves=7, min_child_samples=5, verbosity=-1)
    model.fit(X, y, feature_name=get_feature_names(preprocessor))

    explainer = CreditRiskExplainer(model=model, feature_names=get_feature_names(preprocessor))
    bundle = ModelBundle(
        model=model,
        preprocessor=preprocessor,
        explainer=explainer,
        feature_names=get_feature_names(preprocessor),
        model_version="test",
        run_id="0" * 32,
        artifacts_dir=tmp_path / "artifacts",
    )
    store = ClientStore(tmp_cfg.raw_data_path, id_column=tmp_cfg.id_column)
    return ScoringService(bundle=bundle, store=store, cfg=tmp_cfg)


def test_get_credit_score_returns_valid_probability(scoring_service: ScoringService) -> None:
    result = scoring_service.get_credit_score("SME-000001")

    assert 0.0 <= result.probability_default <= 1.0
    assert result.recommendation in {"APPROVE", "REVIEW", "DECLINE"}


def test_get_credit_score_unknown_client_raises(scoring_service: ScoringService) -> None:
    with pytest.raises(ClientNotFoundError):
        scoring_service.get_credit_score("SME-999999")


def test_shap_explanation_covers_every_model_feature(scoring_service: ScoringService) -> None:
    explanation = scoring_service.get_shap_explanation("SME-000002")

    assert [c.feature for c in explanation.contributions] == scoring_service._bundle.feature_names


def test_simulate_scenario_revenue_shock_moves_pd(scoring_service: ScoringService) -> None:
    params = ScenarioParams(annual_revenue_delta_pct=-0.5, late_payments_12m_override=5)

    scenario = scoring_service.simulate_financial_scenario("SME-000003", params)

    assert scenario.simulated.probability_default != scenario.baseline.probability_default
    assert scenario.client_id == "SME-000003"
    assert "SME-000003" in scenario.narrative


def _publish_champion(cfg: MLConfig, tmp_path: Path, *, with_holdout: bool = True) -> str:
    """Fit a tiny logistic regression and publish it the way `train.py` does.

    Mirrors the real publishing contract — model, preprocessor, SHAP
    background and held-out split all logged into one run, then registered
    and aliased — so the serving tests below exercise a realistic run rather
    than a hand-built one.
    """
    df = _simulate(n_clients=300, seed=cfg.random_state)
    df.to_parquet(cfg.raw_data_path, index=False)
    train_df, test_df = _split(cfg, df)

    preprocessor = build_preprocessor(cfg)
    X = preprocessor.fit_transform(train_df)
    y = train_df[cfg.target_column].to_numpy()
    model = LogisticRegression(**cfg.logreg_params).fit(X, y)

    staging = tmp_path / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, staging / tracking.PREPROCESSOR_ARTIFACT)
    joblib.dump(X[:50], staging / tracking.SHAP_BACKGROUND_ARTIFACT)
    if with_holdout:
        test_df.to_parquet(staging / tracking.HOLDOUT_ARTIFACT, index=False)

    with tracking.start_run(cfg) as run_id:
        tracking.log_model(cfg, model, to_feature_frame(X, get_feature_names(preprocessor)))
        for artifact in staging.iterdir():
            tracking.log_artifact(artifact)
    tracking.register_version(cfg, run_id)
    return run_id


def test_build_default_scoring_service_serves_the_registry_champion(tmp_path: Path) -> None:
    """The whole serving path comes from the registry: model, transform and clients.

    Nothing is written to a local artifact directory, so there is no
    possibility of silently falling back to a stale file — if resolution
    broke, this test would fail rather than quietly pass on the wrong model.
    """
    cfg = MLConfig(
        model_type="logistic_regression",
        data_dir=tmp_path,
        raw_data_path=tmp_path / "clients.parquet",
        mlflow_dir=tmp_path / "mlruns",
        mlflow_experiment="test-serving",
    )
    run_id = _publish_champion(cfg, tmp_path)

    service = build_default_scoring_service(cfg)

    # The registry identity replaces the static config version, so every
    # CreditScoreResult states which run produced it.
    assert service._bundle.model_version == f"1 (run {run_id[:8]})"
    assert service._bundle.run_id == run_id
    assert isinstance(service._bundle.model, LogisticRegression)

    holdout_id = str(
        pd.read_parquet(tmp_path / "staging" / tracking.HOLDOUT_ARTIFACT)["client_id"].iloc[0]
    )
    score = service.get_credit_score(holdout_id)
    assert 0.0 <= score.probability_default <= 1.0
    assert score.model_version == service._bundle.model_version
    assert len(service.get_shap_explanation(holdout_id).contributions) > 0


def test_serving_never_writes_to_the_mlflow_store(tmp_path: Path) -> None:
    """docker-compose mounts the store read-only, so serving must not write to it.

    Two things made it write before: selecting the experiment during
    resolution (which creates it), and loading via a `models:/name@alias` URI
    (which drops a `registered_model_meta` provenance file next to the model).
    Both would crash the container at startup, so this asserts on the whole
    file listing rather than on either symptom.
    """
    cfg = MLConfig(
        model_type="logistic_regression",
        data_dir=tmp_path,
        raw_data_path=tmp_path / "clients.parquet",
        mlflow_dir=tmp_path / "mlruns",
        mlflow_experiment="test-readonly",
    )
    _publish_champion(cfg, tmp_path)

    def snapshot() -> dict[str, float]:
        return {
            str(p.relative_to(cfg.mlflow_dir)): p.stat().st_mtime
            for p in cfg.mlflow_dir.rglob("*")
            if p.is_file()
        }

    before = snapshot()
    assert before, "the store should have been populated by the publish above"

    service = build_default_scoring_service(cfg)
    client_id = str(service._store._df["client_id"].iloc[0])
    service.get_credit_score(client_id)
    service.get_shap_explanation(client_id)
    service.get_model_card()

    assert snapshot() == before


def test_scoring_service_client_store_holds_only_the_champion_run_holdout(
    tmp_path: Path,
) -> None:
    """The served client set is the holdout of *the served model's own run*.

    This is what makes the demo honest: a client the deployed model was
    trained on must be unreachable through the MCP tools, and it must be
    unreachable because the run says so — not because a local parquet
    happened to hold the right rows.
    """
    cfg = MLConfig(
        model_type="logistic_regression",
        data_dir=tmp_path,
        raw_data_path=tmp_path / "clients.parquet",
        mlflow_dir=tmp_path / "mlruns",
        mlflow_experiment="test-holdout-only",
    )
    _publish_champion(cfg, tmp_path)
    df = pd.read_parquet(cfg.raw_data_path)
    train_df, test_df = _split(cfg, df)

    service = build_default_scoring_service(cfg)

    train_only_id = next(iter(set(train_df["client_id"]) - set(test_df["client_id"])))
    with pytest.raises(ClientNotFoundError):
        service.get_credit_score(train_only_id)

    holdout_id = str(test_df["client_id"].iloc[0])
    assert service.get_credit_score(holdout_id).client_id == holdout_id


def test_get_model_card_reports_the_served_run(tmp_path: Path) -> None:
    """The card must describe the same version the tools score with."""
    cfg = MLConfig(
        model_type="logistic_regression",
        data_dir=tmp_path,
        raw_data_path=tmp_path / "clients.parquet",
        mlflow_dir=tmp_path / "mlruns",
        mlflow_experiment="test-model-card",
    )
    run_id = _publish_champion(cfg, tmp_path)

    card = build_default_scoring_service(cfg).get_model_card()

    assert card["run_id"] == run_id
    assert card["alias"] == cfg.mlflow_model_alias
    assert card["registered_model"] == cfg.mlflow_registered_model
    # `metrics.json` only exists once `ml_pipeline.eval` has run against this
    # version — an empty dict is the honest answer, not someone else's numbers.
    assert card["metrics"] == {}


def test_load_model_bundle_rejects_a_run_without_its_preprocessor(tmp_path: Path) -> None:
    """A model whose run has no preprocessor can't be served without risking skew."""
    cfg = MLConfig(
        model_type="logistic_regression",
        mlflow_dir=tmp_path / "mlruns",
        mlflow_experiment="test-serving-incomplete",
    )
    X = pd.DataFrame({"a": [0.0, 1.0, 0.3, 0.8], "b": [1.0, 0.0, 0.7, 0.2]})
    model = LogisticRegression().fit(X, [0, 1, 0, 1])

    with tracking.start_run(cfg) as run_id:
        tracking.log_model(cfg, model, X)
    tracking.register_version(cfg, run_id)

    with pytest.raises(FileNotFoundError, match=r"no 'preprocessor\.joblib' artifact"):
        load_model_bundle(cfg)


def test_build_default_scoring_service_rejects_a_run_without_its_holdout(
    tmp_path: Path,
) -> None:
    """No holdout in the run means no honest client set — refuse rather than
    fall back to the full training corpus."""
    cfg = MLConfig(
        model_type="logistic_regression",
        data_dir=tmp_path,
        raw_data_path=tmp_path / "clients.parquet",
        mlflow_dir=tmp_path / "mlruns",
        mlflow_experiment="test-serving-no-holdout",
    )
    _publish_champion(cfg, tmp_path, with_holdout=False)

    with pytest.raises(FileNotFoundError, match=r"no 'holdout_test\.parquet' artifact"):
        build_default_scoring_service(cfg)


def test_load_model_bundle_raises_when_no_champion_is_registered(tmp_path: Path) -> None:
    """With no local fallback left, an unresolvable alias must fail loudly at startup."""
    cfg = MLConfig(mlflow_dir=tmp_path / "mlruns", mlflow_experiment="test-empty-registry")

    with pytest.raises(MlflowException):
        load_model_bundle(cfg)


def test_client_store_raises_when_data_file_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ClientStore(tmp_path / "does_not_exist.parquet", id_column="client_id")


def test_to_score_result_recommendation_bands(scoring_service: ScoringService) -> None:
    threshold = scoring_service._cfg.decision_threshold

    approve = scoring_service._to_score_result("SME-A", threshold * 0.1)
    review = scoring_service._to_score_result("SME-B", threshold * 0.75)
    decline = scoring_service._to_score_result("SME-C", threshold * 1.5)

    assert approve.recommendation == "APPROVE"
    assert review.recommendation == "REVIEW"
    assert decline.recommendation == "DECLINE"


def test_get_shap_explanation_writes_no_files(
    scoring_service: ScoringService, tmp_path: Path
) -> None:
    """Explaining a client returns data, never files.

    An earlier version rendered a waterfall PNG per call that nothing read —
    unbounded disk growth, and the only reason the scoring container needed a
    writable mount. Asserting on the whole tree, rather than on the absence of
    one filename, is what would catch any new stray output.
    """
    before = {p for p in tmp_path.rglob("*") if p.is_file()}

    explanation = scoring_service.get_shap_explanation("SME-000004")

    assert len(explanation.contributions) > 0
    assert {p for p in tmp_path.rglob("*") if p.is_file()} == before


def test_simulate_scenario_debt_and_utilization_overrides_move_pd(
    scoring_service: ScoringService,
) -> None:
    params = ScenarioParams(total_debt_delta_pct=0.5, credit_utilization_override=1.3)

    scenario = scoring_service.simulate_financial_scenario("SME-000005", params)

    assert scenario.simulated.probability_default != scenario.baseline.probability_default


def test_client_store_only_serves_the_rows_it_was_given(tmp_path: Path) -> None:
    """A client_id outside the store's parquet must raise, not fall through to
    some other source — the store has exactly one backing file and no default."""
    cfg = MLConfig(data_dir=tmp_path, raw_data_path=tmp_path / "clients.parquet")
    df = _simulate(n_clients=300, seed=cfg.random_state)
    train_df, test_df = _split(cfg, df)
    holdout_path = tmp_path / "holdout.parquet"
    test_df.to_parquet(holdout_path, index=False)

    store = ClientStore(holdout_path, id_column=cfg.id_column)

    train_only_id = next(iter(set(train_df["client_id"]) - set(test_df["client_id"])))
    with pytest.raises(ClientNotFoundError):
        store.get_row(train_only_id)

    holdout_id = str(test_df["client_id"].iloc[0])
    assert store.get_row(holdout_id)["client_id"] == holdout_id
