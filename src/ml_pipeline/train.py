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
    5. Publish model + preprocessor + held-out split + metadata as artifacts
       of one MLflow run, and move the `champion` registry alias onto it.
       Nothing is written outside the MLflow store — see `ml_pipeline.tracking`.

Which model gets trained is controlled by `cfg.model_type`
(`FINRISK_MODEL_TYPE=lightgbm|logistic_regression`) — see `ml_pipeline.models`.

Run:
    python -m ml_pipeline.train --tune
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
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
from ml_pipeline.tracking import (
    HOLDOUT_ARTIFACT,
    METADATA_ARTIFACT,
    PREPROCESSOR_ARTIFACT,
    SHAP_BACKGROUND_ARTIFACT,
    log_artifact,
    log_metrics,
    log_model,
    log_params,
    nested_run,
    register_version,
    start_run,
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

    Uses `average_precision_score` rather than `auc(recall, precision)`: the
    latter linearly interpolates between PR points, which overstates the
    score in PR space (Davis & Goadrich, 2006), while AP integrates via a
    non-interpolated step function — scikit-learn's own recommended scalar
    summary of a PR curve. Called "PR-AUC" throughout this codebase as a
    near-universal shorthand, but expect it to read slightly lower than a
    trapezoidal PR AUC.
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
        with nested_run(run_name=f"trial-{trial.number:03d}"):
            return _run_trial(trial)

    def _run_trial(trial: optuna.Trial) -> float:
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
        score = _cv_average_precision(cfg, X, y, params)
        log_params(params)
        log_metrics({"cv_pr_auc": score})
        return score

    sampler = optuna.samplers.TPESampler(seed=cfg.random_state)
    study = optuna.create_study(
        direction="maximize", study_name=f"finrisk_{cfg.model_type}", sampler=sampler
    )
    study.optimize(objective, n_trials=cfg.optuna_n_trials, timeout=cfg.optuna_timeout_s)
    logger.info("Optuna best PR-AUC=%.4f params=%s", study.best_value, study.best_params)
    # Logged after `optimize` returns, so the active run is the parent again:
    # the best score belongs on the training run, the per-trial scores on the
    # nested children.
    log_metrics({"cv_pr_auc": float(study.best_value), "n_trials": len(study.trials)})
    base_params = cfg.lgbm_params if cfg.model_type == "lightgbm" else cfg.logreg_params
    return {**base_params, **study.best_params}


@app.command()
def run(
    tune: bool = typer.Option(False, help="Run Optuna hyperparameter search before final fit"),
    alias: str = typer.Option(
        "",
        help="Registry alias to publish under (default: the configured one, `champion`). "
        "e.g. --alias challenger, to publish a candidate without touching what is served.",
    ),
) -> None:
    """Train the model end-to-end and publish it as a single MLflow run.

    Model, preprocessor, held-out split, SHAP background and metadata are all
    logged as artifacts of one run, and the requested alias is moved onto the
    resulting registry version — nothing is left on the local filesystem.
    `staging` is a scratch directory that exists only because
    `mlflow.log_artifact` takes a file path; it's deleted on the way out.
    """
    cfg = config

    with start_run(cfg, run_name="train") as run_id, TemporaryDirectory() as tmp:
        staging = Path(tmp)
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

        log_params({**params, "tuned_with_optuna": tune, **_split_params(cfg)})
        log_metrics(
            {
                "n_train": len(train_df),
                "n_test": len(test_df),
                "train_default_rate": float(train_df[cfg.target_column].mean()),
            }
        )

        X_train = to_feature_frame(X_train_raw, feature_names)
        model = build_model(cfg, params)
        model.fit(X_train, y_train)

        # Everything below travels into the run *beside* the model, so anything
        # resolving this version later gets the exact transform it was fitted
        # with and the exact rows it was never trained on — the train/serve
        # skew guarantee, now enforced by there being no second copy anywhere.
        joblib.dump(preprocessor, staging / PREPROCESSOR_ARTIFACT)
        test_df.to_parquet(staging / HOLDOUT_ARTIFACT, index=False)
        if cfg.model_type != "lightgbm":
            # shap.LinearExplainer needs a background sample to estimate the
            # expected-value baseline (TreeExplainer doesn't, so this is only
            # persisted for non-tree models — see shap_explainer.py).
            rng = np.random.default_rng(cfg.random_state)
            sample_size = min(200, len(X_train_raw))
            background_idx = rng.choice(len(X_train_raw), size=sample_size, replace=False)
            joblib.dump(X_train_raw[background_idx], staging / SHAP_BACKGROUND_ARTIFACT)
            log_artifact(staging / SHAP_BACKGROUND_ARTIFACT)

        log_model(cfg, model, X_train)
        log_artifact(staging / PREPROCESSOR_ARTIFACT)
        log_artifact(staging / HOLDOUT_ARTIFACT)

        # Registration comes before the metadata artifact so the version number
        # it assigns can be recorded inside the run it describes.
        registry_version = register_version(cfg, run_id, alias=alias or None)

        metadata = {
            "model_version": cfg.model_version,
            "model_type": cfg.model_type,
            "trained_at": datetime.now(UTC).isoformat(),
            "n_train": len(train_df),
            "n_test": len(test_df),
            "params": params,
            "feature_names": feature_names,
            "tuned_with_optuna": tune,
            "mlflow_run_id": run_id,
            "mlflow_model_version": registry_version,
        }
        (staging / METADATA_ARTIFACT).write_text(json.dumps(metadata, indent=2, default=str))
        log_artifact(staging / METADATA_ARTIFACT)
        logger.info(
            "Published %s v%s @%s (run %s) — nothing written outside the MLflow store.",
            cfg.mlflow_registered_model,
            registry_version,
            alias or cfg.mlflow_model_alias,
            run_id,
        )


def _split_params(cfg: MLConfig) -> dict[str, Any]:
    """The split/threshold settings worth recording next to the hyperparameters.

    Recorded because they change what a metric *means*: the same `pr_auc` is
    not comparable across two runs that used different test sizes, seeds or
    decision thresholds.
    """
    return {
        "model_type": cfg.model_type,
        "test_size": cfg.test_size,
        "n_cv_folds": cfg.n_cv_folds,
        "random_state": cfg.random_state,
        "decision_threshold": cfg.decision_threshold,
        "target_column": cfg.target_column,
    }


if __name__ == "__main__":
    app()
