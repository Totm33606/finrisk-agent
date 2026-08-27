"""Train the credit-risk model (LightGBM by default, or logistic regression).

Pipeline:
    1. Load raw data, split off a held-out test set (stratified on target).
    2. Fit the sklearn preprocessor on the training fold only (no leakage).
    3. (Optional) Optuna search over the selected model's hyperparameters,
       optimizing mean out-of-fold PR-AUC under `StratifiedKFold` — PR-AUC is
       preferred over ROC-AUC as the *tuning* objective because credit
       default is a rare-event problem where the positive class is what
       matters.
    4. Refit on the full training set with the best params.
    5. Persist model + preprocessor + metadata via joblib.

Which model gets trained is controlled by `cfg.model_type`
(`FINRISK_MODEL_TYPE=lightgbm|logistic_regression`) — see `ml_pipeline.models`.

Run:
    python -m ml_pipeline.train --tune
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import typer
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from ml_pipeline.config import MLConfig, config
from ml_pipeline.models import build_model
from ml_pipeline.preprocessing import (
    build_preprocessor,
    get_feature_names,
    load_raw_dataset,
    to_feature_frame,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

app = typer.Typer(add_completion=False)


def _split(cfg: MLConfig, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split on the target to preserve the (low) default rate.

    Note: if a `snapshot_date` column is available in your production data,
    prefer a *temporal* split (train on past, test on future) over this
    stratified split — it more honestly simulates deployment, where the
    model always predicts forward in time. The synthetic dataset here has
    no time dimension, so stratified sampling is the correct fallback.
    """
    train_df, test_df = train_test_split(
        df,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=df[cfg.target_column],
    )
    return train_df, test_df


def _cv_average_precision(
    cfg: MLConfig, X: np.ndarray, y: np.ndarray, params: dict[str, Any]
) -> float:
    """Mean out-of-fold Average Precision (AP) under stratified K-fold CV, for a given param set.

    Uses `average_precision_score` rather than `auc(recall, precision)` from
    `precision_recall_curve` — the latter linearly interpolates between PR
    points, which is not a valid operation in PR space (unlike ROC space) and
    is known to be an overly optimistic estimator (Davis & Goadrich, 2006).
    AP instead integrates via a non-interpolated step function, which is why
    scikit-learn recommends it as *the* scalar summary of a PR curve. It's
    colloquially called "PR-AUC" throughout this codebase (a near-universal
    shorthand — it's literally the string `scoring="average_precision"` in
    scikit-learn's own CV utilities) but is not numerically identical to a
    trapezoidal-rule PR AUC; expect the two to differ, with AP being the
    lower, more honest number.
    """
    skf = StratifiedKFold(n_splits=cfg.n_cv_folds, shuffle=True, random_state=cfg.random_state)
    scores = []
    for train_idx, valid_idx in skf.split(X, y):
        model = build_model(cfg, params)
        if cfg.model_type == "lightgbm":
            model.fit(
                X[train_idx],
                y[train_idx],
                eval_set=[(X[valid_idx], y[valid_idx])],
                eval_metric="average_precision",
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
            )
        else:
            model.fit(X[train_idx], y[train_idx])
        proba = np.asarray(model.predict_proba(X[valid_idx]))[:, 1]
        scores.append(average_precision_score(y[valid_idx], proba))
    return float(np.mean(scores))


def _optuna_search(cfg: MLConfig, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Bayesian hyperparameter search (Optuna TPE) maximizing mean CV PR-AUC.

    The sampler is explicit (`TPESampler`, rather than relying on Optuna's
    implicit default) and seeded with `cfg.random_state` so repeated runs
    with the same config produce the same trial sequence.
    """

    def objective(trial: optuna.Trial) -> float:
        params: dict[str, Any]
        if cfg.model_type == "lightgbm":
            params = {
                **cfg.lgbm_params,
                "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=100),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
            }
        else:
            params = {
                **cfg.logreg_params,
                "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
                "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
                "solver": "liblinear",  # the one solver supporting both l1 and l2 here
            }
        return _cv_average_precision(cfg, X, y, params)

    sampler = optuna.samplers.TPESampler(seed=cfg.random_state)
    study = optuna.create_study(
        direction="maximize", study_name=f"finrisk_{cfg.model_type}", sampler=sampler
    )
    study.optimize(objective, n_trials=cfg.optuna_n_trials, timeout=cfg.optuna_timeout_s)
    logger.info("Optuna best PR-AUC=%.4f params=%s", study.best_value, study.best_params)
    base_params = cfg.lgbm_params if cfg.model_type == "lightgbm" else cfg.logreg_params
    return {**base_params, **study.best_params}


@app.command()
def run(
    tune: bool = typer.Option(False, help="Run Optuna hyperparameter search before final fit"),
) -> None:
    """Train the model end-to-end and persist artifacts to `cfg.model_dir`."""
    cfg = config
    cfg.model_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw_dataset(cfg)
    train_df, test_df = _split(cfg, df)
    logger.info(
        "Train rows=%d Test rows=%d Default rate(train)=%.2f%%",
        len(train_df),
        len(test_df),
        100 * train_df[cfg.target_column].mean(),
    )

    preprocessor = build_preprocessor(cfg)
    X_train_raw = preprocessor.fit_transform(train_df)
    y_train = train_df[cfg.target_column].to_numpy()
    feature_names = get_feature_names(preprocessor)

    default_params = cfg.lgbm_params if cfg.model_type == "lightgbm" else cfg.logreg_params
    params = _optuna_search(cfg, X_train_raw, y_train) if tune else default_params

    X_train = to_feature_frame(X_train_raw, feature_names)
    model = build_model(cfg, params)
    model.fit(X_train, y_train)

    joblib.dump(model, cfg.model_path)
    joblib.dump(preprocessor, cfg.preprocessor_path)
    test_df.to_parquet(cfg.model_dir / "holdout_test.parquet", index=False)

    if cfg.model_type != "lightgbm":
        # shap.LinearExplainer needs a background sample to estimate the
        # expected-value baseline (TreeExplainer doesn't need this, so it's
        # only persisted for non-tree models — see shap_explainer.py).
        rng = np.random.default_rng(cfg.random_state)
        sample_size = min(200, len(X_train_raw))
        background_idx = rng.choice(len(X_train_raw), size=sample_size, replace=False)
        joblib.dump(X_train_raw[background_idx], cfg.model_dir / "shap_background.joblib")

    metadata = {
        "model_version": cfg.model_version,
        "model_type": cfg.model_type,
        "trained_at": datetime.now(UTC).isoformat(),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "params": params,
        "feature_names": feature_names,
        "tuned_with_optuna": tune,
    }
    (cfg.model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    logger.info("Artifacts written to %s", cfg.model_dir)


if __name__ == "__main__":
    app()
