"""Unit tests for `ml_pipeline.models` — the model_type-driven classifier factory."""

from __future__ import annotations

import lightgbm as lgb
from sklearn.linear_model import LogisticRegression

from ml_pipeline.config import MLConfig
from ml_pipeline.models import build_model


def test_build_model_lightgbm_returns_lgbm_classifier() -> None:
    cfg = MLConfig(model_type="lightgbm")

    model = build_model(cfg, cfg.lgbm_params)

    assert isinstance(model, lgb.LGBMClassifier)


def test_build_model_logistic_regression_returns_logreg() -> None:
    cfg = MLConfig(model_type="logistic_regression")

    model = build_model(cfg, cfg.logreg_params)

    assert isinstance(model, LogisticRegression)


def test_build_model_passes_through_hyperparameters() -> None:
    cfg = MLConfig(model_type="logistic_regression")

    model = build_model(cfg, {**cfg.logreg_params, "C": 0.5})

    assert isinstance(model, LogisticRegression)
    assert model.C == 0.5
