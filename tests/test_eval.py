"""Unit tests for `ml_pipeline.eval`."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ml_pipeline import eval as eval_module
from ml_pipeline import train as train_module
from ml_pipeline.config import MLConfig
from ml_pipeline.eval import _load_artifacts, compute_metrics
from ml_pipeline.make_dataset import _simulate


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


def test_load_artifacts_raises_when_model_missing(tmp_path: Path) -> None:
    cfg = MLConfig(model_dir=tmp_path / "models")

    with pytest.raises(FileNotFoundError):
        _load_artifacts(cfg)


def test_run_end_to_end_writes_metrics_and_diagnostic_plots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercises `train.run()` + `eval.run()` back to back, the way `make train eval` does —
    both read the module-level `config` singleton, so it's monkeypatched to a tmp-path config."""
    cfg = MLConfig(
        data_dir=tmp_path,
        raw_data_path=tmp_path / "clients.parquet",
        model_dir=tmp_path / "models",
        shap_plots_dir=tmp_path / "reports" / "shap",
        n_cv_folds=2,
    )
    df = _simulate(n_clients=300, seed=cfg.random_state)
    df.to_parquet(cfg.raw_data_path, index=False)

    monkeypatch.setattr(train_module, "config", cfg)
    train_module.run(tune=False)

    monkeypatch.setattr(eval_module, "config", cfg)
    eval_module.run()

    assert cfg.metrics_path.exists()
    metrics = json.loads(cfg.metrics_path.read_text())
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0

    reports_dir = cfg.shap_plots_dir.parent
    assert (reports_dir / "roc_curve.png").exists()
    assert (reports_dir / "pr_curve.png").exists()
    assert (reports_dir / "confusion_matrix.png").exists()
