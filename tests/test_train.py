"""Unit tests for `ml_pipeline.train`."""

from __future__ import annotations

from typing import Any

import optuna
import pandas as pd
import pytest

from ml_pipeline.config import MLConfig
from ml_pipeline.make_dataset import _simulate
from ml_pipeline.preprocessing import build_preprocessor
from ml_pipeline.train import _cv_average_precision, _optuna_search, _split


@pytest.fixture
def small_cfg() -> MLConfig:
    # Explicit model_type: these tests assert LightGBM-shaped behavior
    # (n_estimators, num_leaves, ...) and must not depend on whatever
    # MLConfig.model_type's global default currently is.
    return MLConfig(model_type="lightgbm", n_cv_folds=2, optuna_n_trials=2)


@pytest.fixture
def small_logreg_cfg() -> MLConfig:
    return MLConfig(model_type="logistic_regression", n_cv_folds=2, optuna_n_trials=2)


@pytest.fixture
def raw_df(small_cfg: MLConfig) -> pd.DataFrame:
    return _simulate(n_clients=400, seed=small_cfg.random_state)


def test_split_is_stratified_on_target(small_cfg: MLConfig, raw_df: pd.DataFrame) -> None:
    train_df, test_df = _split(small_cfg, raw_df)

    assert len(train_df) + len(test_df) == len(raw_df)
    train_rate = train_df[small_cfg.target_column].mean()
    test_rate = test_df[small_cfg.target_column].mean()
    assert abs(train_rate - test_rate) < 0.05


def test_cv_average_precision_returns_valid_score(
    small_cfg: MLConfig, raw_df: pd.DataFrame
) -> None:
    preprocessor = build_preprocessor(small_cfg)
    X = preprocessor.fit_transform(raw_df)
    y = raw_df[small_cfg.target_column].to_numpy()
    params: dict[str, Any] = {**small_cfg.lgbm_params, "n_estimators": 50, "num_leaves": 7}

    score = _cv_average_precision(small_cfg, X, y, params)

    assert 0.0 <= score <= 1.0


def test_optuna_search_uses_an_explicit_seeded_tpe_sampler(
    monkeypatch: pytest.MonkeyPatch, small_cfg: MLConfig, raw_df: pd.DataFrame
) -> None:
    """`_optuna_search` must pass an explicit sampler to `create_study` rather than
    relying on Optuna's implicit default — this is what makes `--tune` runs reproducible."""
    captured: dict[str, Any] = {}
    real_create_study = optuna.create_study

    def spy_create_study(*args: Any, **kwargs: Any) -> optuna.Study:
        captured.update(kwargs)
        return real_create_study(*args, **kwargs)

    monkeypatch.setattr(optuna, "create_study", spy_create_study)

    preprocessor = build_preprocessor(small_cfg)
    X = preprocessor.fit_transform(raw_df)
    y = raw_df[small_cfg.target_column].to_numpy()
    best_params = _optuna_search(small_cfg, X, y)

    sampler = captured.get("sampler")
    assert isinstance(sampler, optuna.samplers.TPESampler)
    assert isinstance(best_params, dict)
    assert "n_estimators" in best_params


def test_cv_average_precision_returns_valid_score_for_logistic_regression(
    small_logreg_cfg: MLConfig, raw_df: pd.DataFrame
) -> None:
    preprocessor = build_preprocessor(small_logreg_cfg)
    X = preprocessor.fit_transform(raw_df)
    y = raw_df[small_logreg_cfg.target_column].to_numpy()

    score = _cv_average_precision(small_logreg_cfg, X, y, small_logreg_cfg.logreg_params)

    assert 0.0 <= score <= 1.0


def test_optuna_search_logistic_regression_returns_valid_hyperparameters(
    small_logreg_cfg: MLConfig, raw_df: pd.DataFrame
) -> None:
    preprocessor = build_preprocessor(small_logreg_cfg)
    X = preprocessor.fit_transform(raw_df)
    y = raw_df[small_logreg_cfg.target_column].to_numpy()

    best_params = _optuna_search(small_logreg_cfg, X, y)

    assert best_params["penalty"] in {"l1", "l2"}
    assert best_params["solver"] == "liblinear"
    assert best_params["C"] > 0
