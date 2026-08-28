"""Unit tests for `ml_pipeline.shap_explainer`."""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from ml_pipeline.config import MLConfig
from ml_pipeline.make_dataset import _simulate
from ml_pipeline.preprocessing import build_preprocessor, get_feature_names
from ml_pipeline.shap_explainer import CreditRiskExplainer


@pytest.fixture
def explainer_and_row() -> tuple[CreditRiskExplainer, np.ndarray]:
    cfg = MLConfig()
    df = _simulate(n_clients=300, seed=cfg.random_state)
    preprocessor = build_preprocessor(cfg)
    X = preprocessor.fit_transform(df)
    y = df[cfg.target_column].to_numpy()
    feature_names = get_feature_names(preprocessor)

    model = lgb.LGBMClassifier(n_estimators=50, num_leaves=7, min_child_samples=5, verbosity=-1)
    model.fit(X, y, feature_name=feature_names)

    explainer = CreditRiskExplainer(model=model, feature_names=feature_names)
    return explainer, X[0]


def test_explain_row_contributions_match_feature_count(
    explainer_and_row: tuple[CreditRiskExplainer, np.ndarray],
) -> None:
    explainer, row = explainer_and_row

    explanation = explainer.explain_row(row, client_id="SME-000001")

    assert len(explanation.contributions) == len(explainer._feature_names)
    assert explanation.client_id == "SME-000001"


def test_explain_row_uses_raw_values_when_provided(
    explainer_and_row: tuple[CreditRiskExplainer, np.ndarray],
) -> None:
    explainer, row = explainer_and_row
    feature = explainer._feature_names[0]

    explanation = explainer.explain_row(row, client_id="SME-000002", raw_values={feature: 999.0})

    matching = next(c for c in explanation.contributions if c.feature == feature)
    assert matching.value == 999.0


def test_explain_row_drivers_are_disjoint_and_correctly_signed(
    explainer_and_row: tuple[CreditRiskExplainer, np.ndarray],
) -> None:
    explainer, row = explainer_and_row

    explanation = explainer.explain_row(row, client_id="SME-000003")

    assert set(explanation.top_positive_drivers).isdisjoint(explanation.top_negative_drivers)
    by_feature = {c.feature: c.shap_value for c in explanation.contributions}
    assert all(by_feature[f] > 0 for f in explanation.top_positive_drivers)
    assert all(by_feature[f] < 0 for f in explanation.top_negative_drivers)


def test_render_waterfall_writes_png(
    tmp_path: Path, explainer_and_row: tuple[CreditRiskExplainer, np.ndarray]
) -> None:
    explainer, row = explainer_and_row

    out_path = explainer.render_waterfall(row, client_id="SME-000004", out_dir=tmp_path)

    assert out_path.exists()
    assert out_path.name == "waterfall_SME-000004.png"


def test_render_summary_writes_png(tmp_path: Path) -> None:
    cfg = MLConfig()
    df = _simulate(n_clients=150, seed=cfg.random_state)
    preprocessor = build_preprocessor(cfg)
    X = preprocessor.fit_transform(df)
    y = df[cfg.target_column].to_numpy()
    feature_names = get_feature_names(preprocessor)

    model = lgb.LGBMClassifier(n_estimators=30, num_leaves=7, verbosity=-1)
    model.fit(X, y, feature_name=feature_names)
    explainer = CreditRiskExplainer(model=model, feature_names=feature_names)

    # sample_size < len(X) exercises the subsampling branch too
    out_path = explainer.render_summary(X, out_dir=tmp_path, sample_size=50)

    assert out_path.exists()
    assert out_path.name == "shap_summary.png"


def test_logistic_regression_model_requires_background_data() -> None:
    cfg = MLConfig()
    df = _simulate(n_clients=200, seed=cfg.random_state)
    preprocessor = build_preprocessor(cfg)
    X = preprocessor.fit_transform(df)
    y = df[cfg.target_column].to_numpy()
    feature_names = get_feature_names(preprocessor)

    model = LogisticRegression(max_iter=1000).fit(X, y)

    with pytest.raises(ValueError, match="background"):
        CreditRiskExplainer(
            model=model, feature_names=feature_names, model_type="logistic_regression"
        )


def test_logistic_regression_explain_row_uses_linear_explainer() -> None:
    cfg = MLConfig()
    df = _simulate(n_clients=200, seed=cfg.random_state)
    preprocessor = build_preprocessor(cfg)
    X = preprocessor.fit_transform(df)
    y = df[cfg.target_column].to_numpy()
    feature_names = get_feature_names(preprocessor)

    model = LogisticRegression(max_iter=1000).fit(X, y)
    background = X[:50]
    explainer = CreditRiskExplainer(
        model=model,
        feature_names=feature_names,
        model_type="logistic_regression",
        background=background,
    )

    explanation = explainer.explain_row(X[0], client_id="SME-000001")

    assert len(explanation.contributions) == len(feature_names)
    assert explanation.client_id == "SME-000001"
