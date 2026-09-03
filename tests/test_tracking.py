"""Unit tests for `ml_pipeline.tracking`.

These run against a real MLflow file store rooted in `tmp_path` (via
`MLConfig(mlflow_dir=...)`, which derives `mlflow_tracking_uri` under it) —
mocking MLflow here would test the mock, not the integration, and the file
store is fast enough that there's no reason to.

MLflow is the storage layer, so the behaviour that matters most is failing
loudly: an unregistered alias or a missing artifact must raise rather than
be papered over, since there is no local copy to fall back to.
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression

from ml_pipeline import tracking
from ml_pipeline.config import MLConfig


@pytest.fixture
def cfg(tmp_path: Path) -> MLConfig:
    """A config whose MLflow store lives under tmp_path, not the developer's ./mlruns.

    `model_type` is explicit because `fitted_model` below is a
    `LogisticRegression`: `log_model` dispatches the MLflow flavor on this
    field, so inheriting whatever the global default happens to be would
    hand a sklearn estimator to the lightgbm flavor.
    """
    return MLConfig(
        model_type="logistic_regression",
        mlflow_dir=tmp_path / "mlruns",
        mlflow_experiment="test-experiment",
    )


@pytest.fixture
def fitted_model() -> tuple[LogisticRegression, pd.DataFrame]:
    X = pd.DataFrame({"a": [0.0, 1.0, 0.2, 0.9, 0.4], "b": [1.0, 0.0, 0.8, 0.1, 0.6]})
    y = [0, 1, 0, 1, 0]
    return LogisticRegression().fit(X, y), X


def test_tracking_uri_is_derived_under_an_overridden_mlflow_dir(tmp_path: Path) -> None:
    """The guard that stops a test run from writing into the real project store."""
    cfg = MLConfig(mlflow_dir=tmp_path / "mlruns")

    assert cfg.mlflow_tracking_uri.startswith("file:")
    assert "mlruns" in cfg.mlflow_tracking_uri
    assert str(tmp_path.resolve()).replace("\\", "/") in cfg.mlflow_tracking_uri.replace("%20", " ")


def test_an_explicit_tracking_uri_wins_over_mlflow_dir(tmp_path: Path) -> None:
    """A remote server must not be silently rewritten into a local file store."""
    cfg = MLConfig(mlflow_dir=tmp_path / "mlruns", mlflow_tracking_uri="http://mlflow.test:5000")

    assert cfg.mlflow_tracking_uri == "http://mlflow.test:5000"


def test_start_run_records_tags_and_yields_a_usable_run_id(cfg: MLConfig) -> None:
    with tracking.start_run(cfg, run_name="unit-test") as run_id:
        assert run_id
        tracking.log_params({"C": 1.0, "penalty": "l2"})
        tracking.log_metrics({"cv_pr_auc": 0.42})

    run = MlflowClient(tracking_uri=cfg.mlflow_tracking_uri).get_run(run_id)
    assert run.data.params["C"] == "1.0"
    assert run.data.metrics["cv_pr_auc"] == pytest.approx(0.42)
    assert run.data.tags["model_type"] == cfg.model_type
    assert run.data.tags["model_version"] == cfg.model_version


def test_start_run_rejects_an_experiment_with_a_foreign_artifact_location(
    cfg: MLConfig,
) -> None:
    """Reproduces training from two different absolute-path contexts (e.g. host
    then Docker) against the same mlruns/: the reused experiment's stale
    artifact_location must fail loudly here, not several mlflow frames down
    inside a bare PermissionError/FileNotFoundError from the artifact writer."""
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.create_experiment(cfg.mlflow_experiment, artifact_location="file:///elsewhere")

    with pytest.raises(RuntimeError, match="already has artifacts under"), tracking.start_run(cfg):
        pass


def test_log_helpers_outside_a_run_do_not_create_orphan_runs(cfg: MLConfig) -> None:
    """MLflow's fluent API auto-creates a run when none is active — `_active` blocks that."""
    tracking.log_params({"stray": 1})
    tracking.log_metrics({"stray": 1.0})

    assert mlflow.active_run() is None
    client = MlflowClient(tracking_uri=cfg.mlflow_tracking_uri)
    experiment = client.get_experiment_by_name(cfg.mlflow_experiment)
    assert experiment is None or not client.search_runs([experiment.experiment_id])


def test_resume_run_appends_to_the_same_run(cfg: MLConfig) -> None:
    """`eval.py`'s core requirement: held-out metrics land on the training run."""
    with tracking.start_run(cfg) as run_id:
        tracking.log_metrics({"cv_pr_auc": 0.42})

    with tracking.resume_run(cfg, run_id):
        tracking.log_metrics({"roc_auc": 0.91})

    run = MlflowClient(tracking_uri=cfg.mlflow_tracking_uri).get_run(run_id)
    assert set(run.data.metrics) == {"cv_pr_auc", "roc_auc"}


def test_resume_run_raises_on_an_unknown_run_id(cfg: MLConfig) -> None:
    """No silent no-op: results must never be quietly dropped on the floor."""
    with pytest.raises(MlflowException), tracking.resume_run(cfg, "0" * 32):
        pass  # pragma: no cover — the context manager raises on entry


def test_nested_run_is_a_noop_without_an_active_parent(cfg: MLConfig) -> None:
    with tracking.nested_run(run_name="trial-000"):
        assert mlflow.active_run() is None


def test_nested_run_attaches_trials_to_the_parent(cfg: MLConfig) -> None:
    with tracking.start_run(cfg) as parent_id:
        for trial in range(2):
            with tracking.nested_run(run_name=f"trial-{trial:03d}"):
                tracking.log_metrics({"cv_pr_auc": 0.1 * trial})

    client = MlflowClient(tracking_uri=cfg.mlflow_tracking_uri)
    experiment = client.get_experiment_by_name(cfg.mlflow_experiment)
    assert experiment is not None, "start_run should have created the experiment"
    children = client.search_runs(
        [experiment.experiment_id], filter_string=f"tags.mlflow.parentRunId = '{parent_id}'"
    )
    assert len(children) == 2


def test_register_version_moves_the_alias_onto_the_new_run(
    cfg: MLConfig, fitted_model: tuple[LogisticRegression, pd.DataFrame]
) -> None:
    model, X = fitted_model
    versions = []
    for _ in range(2):
        with tracking.start_run(cfg) as run_id:
            tracking.log_model(cfg, model, X)
        versions.append(tracking.register_version(cfg, run_id))

    # `register_version` normalizes to str: the file store hands back an int
    # here while the SQL store hands back a str, and `ModelBundle.model_version`
    # downstream is typed `str`.
    assert versions == ["1", "2"]
    client = MlflowClient(tracking_uri=cfg.mlflow_tracking_uri)
    alias = client.get_model_version_by_alias(cfg.mlflow_registered_model, cfg.mlflow_model_alias)
    assert str(alias.version) == "2"


def test_register_version_propagates_registry_failures(
    cfg: MLConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The alias is the only statement of what gets served, so a failed write
    must not let `train.py` report success while serving keeps the old model."""

    def boom(*_args: object, **_kwargs: object) -> None:
        # MlflowException's __init__ is untyped upstream.
        raise MlflowException("registry unavailable")  # type: ignore[no-untyped-call]

    # Patched on the mlflow module itself, which is the same object
    # `tracking` calls through — `tracking.mlflow` is an implicit re-export.
    monkeypatch.setattr(mlflow, "register_model", boom)

    with pytest.raises(MlflowException):
        tracking.register_version(cfg, "0" * 32)


def test_resolve_model_round_trips_model_and_run_artifacts(
    cfg: MLConfig, fitted_model: tuple[LogisticRegression, pd.DataFrame], tmp_path: Path
) -> None:
    """The serving contract: the resolved version also yields its own preprocessor."""
    model, X = fitted_model
    side_artifact = tmp_path / tracking.PREPROCESSOR_ARTIFACT
    side_artifact.write_bytes(b"stand-in for the fitted ColumnTransformer")

    with tracking.start_run(cfg) as run_id:
        tracking.log_model(cfg, model, X)
        tracking.log_artifact(side_artifact)
    tracking.register_version(cfg, run_id)

    resolved = tracking.resolve_model(cfg)

    assert resolved.run_id == run_id
    assert resolved.version == "1"
    assert resolved.alias == cfg.mlflow_model_alias
    assert resolved.display_version == f"1 (run {run_id[:8]})"
    assert resolved.artifact(tracking.PREPROCESSOR_ARTIFACT).read_bytes() == (
        side_artifact.read_bytes()
    )
    # `np.asarray` for the same reason as the pipeline code: `predict_proba`
    # doesn't narrow to an ndarray across the `Classifier` union.
    assert np.asarray(resolved.model.predict_proba(X)).shape == (len(X), 2)


def test_resolve_model_follows_an_explicit_alias(
    cfg: MLConfig, fitted_model: tuple[LogisticRegression, pd.DataFrame]
) -> None:
    """`--alias challenger` must reach a different version than `champion`.

    This is what lets `eval.py` score a candidate without disturbing whatever
    is currently being served.
    """
    model, X = fitted_model
    run_ids = []
    for publish_alias in (None, "challenger"):
        with tracking.start_run(cfg) as run_id:
            tracking.log_model(cfg, model, X)
        tracking.register_version(cfg, run_id, alias=publish_alias)
        run_ids.append(run_id)

    # The second publish created `challenger` from nothing and left `champion`
    # where it was — no manual alias setup anywhere in this test.
    assert tracking.resolve_model(cfg).run_id == run_ids[0]
    challenger = tracking.resolve_model(cfg, alias="challenger")
    assert challenger.run_id == run_ids[1]
    assert challenger.alias == "challenger"
    assert challenger.version == "2"


def test_resolve_model_raises_when_the_alias_does_not_resolve(cfg: MLConfig) -> None:
    """Unlike the fire-and-forget log helpers, this is on the critical serving path:
    failing to load a model must be loud, not silently fall back to something else."""
    with pytest.raises(MlflowException):
        tracking.resolve_model(cfg)


def test_resolve_model_error_tells_you_how_to_publish_a_model(
    cfg: MLConfig, fitted_model: tuple[LogisticRegression, pd.DataFrame]
) -> None:
    """An unpublished model is the most common first-run stumble; MLflow's own
    'Registered Model not found' says nothing about how to fix it."""
    with pytest.raises(MlflowException, match=r"ml_pipeline\.train`") as champion_error:
        tracking.resolve_model(cfg)
    assert "make_dataset" in str(champion_error.value)

    # For a non-default alias the hint must name it, since plain `train`
    # would publish under `champion` and not fix the problem.
    model, X = fitted_model
    with tracking.start_run(cfg) as run_id:
        tracking.log_model(cfg, model, X)
    tracking.register_version(cfg, run_id)

    with pytest.raises(MlflowException, match=r"--alias challenger") as challenger_error:
        tracking.resolve_model(cfg, alias="challenger")
    assert "challenger" in str(challenger_error.value)


def test_registered_model_artifact_reports_a_missing_file_actionably(
    cfg: MLConfig, fitted_model: tuple[LogisticRegression, pd.DataFrame]
) -> None:
    """The diagnosis must point at an incomplete run, not a corrupt file."""
    model, X = fitted_model
    with tracking.start_run(cfg) as run_id:
        tracking.log_model(cfg, model, X)
    tracking.register_version(cfg, run_id)

    resolved = tracking.resolve_model(cfg)

    with pytest.raises(FileNotFoundError, match=r"re-run"):
        resolved.artifact(tracking.HOLDOUT_ARTIFACT)
