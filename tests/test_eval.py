"""Unit tests for `ml_pipeline.eval`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from mlflow.tracking import MlflowClient

from ml_pipeline import eval as eval_module
from ml_pipeline import train as train_module
from ml_pipeline.config import MLConfig
from ml_pipeline.eval import compute_metrics
from ml_pipeline.make_dataset import _simulate
from ml_pipeline.tracking import (
    DIAGNOSTICS_DIR,
    HOLDOUT_ARTIFACT,
    METADATA_ARTIFACT,
    METRICS_ARTIFACT,
    PREPROCESSOR_ARTIFACT,
)


def test_compute_metrics_perfect_separation_scores_maximal() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_proba = np.array([0.05, 0.1, 0.15, 0.85, 0.9, 0.95])

    metrics = compute_metrics(y_true, y_proba, threshold=0.5)

    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["precision_at_threshold"] == 1.0
    assert metrics["recall_at_threshold"] == 1.0
    assert metrics["f1_at_threshold"] == 1.0
    assert metrics["confusion_matrix"] == [[3, 0], [0, 3]]
    assert metrics["n_samples"] == 6
    assert metrics["base_rate"] == 0.5


def test_compute_metrics_threshold_changes_precision_recall_tradeoff() -> None:
    y_true = np.array([0, 0, 1, 1, 1])
    y_proba = np.array([0.2, 0.6, 0.3, 0.7, 0.9])

    loose = compute_metrics(y_true, y_proba, threshold=0.1)  # flags everyone positive
    strict = compute_metrics(y_true, y_proba, threshold=0.99)  # flags no one positive

    assert loose["recall_at_threshold"] == 1.0
    assert strict["recall_at_threshold"] == 0.0
    assert strict["precision_at_threshold"] == 0.0  # zero_division=0, not a NaN/exception


def _tmp_cfg(tmp_path: Path, experiment: str) -> MLConfig:
    return MLConfig(
        data_dir=tmp_path,
        raw_data_path=tmp_path / "clients.parquet",
        mlflow_dir=tmp_path / "mlruns",
        n_cv_folds=2,
        mlflow_experiment=experiment,
    )


def test_run_end_to_end_records_one_shared_mlflow_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`train.run()` then `eval.run()` must land on a *single* MLflow run.

    This is the property the whole design hangs on: a `pr_auc` that isn't
    attached to the hyperparameters that produced it can't be compared across
    runs. Asserting on run *count* is what would catch a regression to two
    detached runs — metrics alone wouldn't.

    Both modules read the module-level `config` singleton, so it is
    monkeypatched to a tmp-path config; the tracking URI is derived under
    `mlflow_dir`, so this never writes into the developer's ./mlruns.
    """
    cfg = _tmp_cfg(tmp_path, "test-end-to-end")
    _simulate(n_clients=300, seed=cfg.random_state).to_parquet(cfg.raw_data_path, index=False)

    monkeypatch.setattr(train_module, "config", cfg)
    train_module.run(tune=False, alias="")
    monkeypatch.setattr(eval_module, "config", cfg)
    # `alias=""` is passed explicitly because typer's default is an OptionInfo
    # object when the command function is called directly rather than via the
    # CLI — same reason `train.run(tune=False)` spells its argument out.
    eval_module.run(alias="")

    client = MlflowClient(tracking_uri=cfg.mlflow_tracking_uri)
    experiment = client.get_experiment_by_name(cfg.mlflow_experiment)
    assert experiment is not None, "train.run() should have created the experiment"
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1, "eval.py opened a second run instead of resuming the training one"
    run_id = runs[0].info.run_id

    run = client.get_run(run_id)
    # Training-side and evaluation-side facts on the same run:
    assert run.data.params["model_type"] == cfg.model_type
    assert run.data.metrics["n_train"] > 0
    assert 0.0 <= run.data.metrics["roc_auc"] <= 1.0
    assert run.data.metrics["cm_true_positive"] >= 0

    logged = {artifact.path for artifact in client.list_artifacts(run_id)}
    assert {
        "model",
        PREPROCESSOR_ARTIFACT,
        HOLDOUT_ARTIFACT,
        METADATA_ARTIFACT,
        METRICS_ARTIFACT,
    } <= logged
    # `shap_summary.png` is the per-*model* explanation ("what matters overall"),
    # which belongs in the run — unlike a per-client waterfall, which describes
    # one request and is no longer rendered anywhere.
    diagnostics = {Path(a.path).name for a in client.list_artifacts(run_id, DIAGNOSTICS_DIR)}
    assert diagnostics == {
        "roc_curve.png",
        "pr_curve.png",
        "confusion_matrix.png",
        "shap_summary.png",
    }

    # And the registry alias points at that same run.
    alias = client.get_model_version_by_alias(cfg.mlflow_registered_model, cfg.mlflow_model_alias)
    assert alias.run_id == run_id


def test_run_leaves_no_artifacts_outside_the_mlflow_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The single-source-of-truth guarantee, asserted rather than assumed.

    A stray `models/` or `reports/*.png` written alongside the run is exactly
    the stale second copy this design exists to eliminate, so the test fails
    on any file appearing outside `mlruns/` and the input dataset.
    """
    cfg = _tmp_cfg(tmp_path, "test-no-local-artifacts")
    _simulate(n_clients=300, seed=cfg.random_state).to_parquet(cfg.raw_data_path, index=False)

    monkeypatch.setattr(train_module, "config", cfg)
    train_module.run(tune=False, alias="")
    monkeypatch.setattr(eval_module, "config", cfg)
    eval_module.run(alias="")

    stray = {
        path.relative_to(tmp_path).as_posix()
        for path in tmp_path.rglob("*")
        if path.is_file() and not path.is_relative_to(cfg.mlflow_dir)
    }
    assert stray == {"clients.parquet"}


def test_run_evaluates_the_alias_it_is_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--alias` must send the results to that version's run, not the champion's.

    Walks the real champion/challenger workflow: publish a champion, publish a
    candidate under `challenger`, then evaluate the candidate. The champion's
    run must come out without held-out metrics — proving the results followed
    the alias rather than the most recent training.
    """
    cfg = _tmp_cfg(tmp_path, "test-alias-routing")
    _simulate(n_clients=300, seed=cfg.random_state).to_parquet(cfg.raw_data_path, index=False)
    monkeypatch.setattr(train_module, "config", cfg)
    train_module.run(tune=False, alias="")
    train_module.run(tune=False, alias="challenger")

    client = MlflowClient(tracking_uri=cfg.mlflow_tracking_uri)
    champion = client.get_model_version_by_alias(cfg.mlflow_registered_model, "champion")
    challenger = client.get_model_version_by_alias(cfg.mlflow_registered_model, "challenger")
    # The second training published `challenger` without moving `champion`.
    # `str()` because the file store hands versions back as ints.
    assert str(champion.version) == "1"
    assert str(challenger.version) == "2"

    monkeypatch.setattr(eval_module, "config", cfg)
    eval_module.run(alias="challenger")

    assert "roc_auc" in client.get_run(challenger.run_id).data.metrics
    assert "roc_auc" not in client.get_run(champion.run_id).data.metrics
