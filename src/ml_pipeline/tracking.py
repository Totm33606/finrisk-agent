"""MLflow tracking & model registry for the FinRisk ML pipeline.

Centralizes everything MLflow-related so `train.py` / `eval.py` /
`scoring_service.py` stay focused on their own job.

MLflow is the storage layer, not optional instrumentation: there is no
local-artifact fallback. A run's model, preprocessor, held-out split,
metrics and plots exist in exactly one place, and the registry alias is the
only statement of which run is served — so a stale local file can never
disagree with the tracked metrics that describe it. `train.py` opens the run
and `eval.py` resumes the same one (rather than a detached second run), so
hyperparameters and held-out metrics always share one id.
"""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from ml_pipeline.config import MLConfig, config
from ml_pipeline.models import Classifier

logger = logging.getLogger(__name__)

# Artifact names inside a run. Single source of truth: `train.py` writes them,
# `eval.py` and `scoring_service.py` read them back, and a typo here fails
# loudly in one place instead of silently in three.
MODEL_ARTIFACT_PATH = "model"
PREPROCESSOR_ARTIFACT = "preprocessor.joblib"
HOLDOUT_ARTIFACT = "holdout_test.parquet"
SHAP_BACKGROUND_ARTIFACT = "shap_background.joblib"
METADATA_ARTIFACT = "metadata.json"
METRICS_ARTIFACT = "metrics.json"
DIAGNOSTICS_DIR = "diagnostics"


def _git_sha() -> str | None:
    """Short git SHA of the working tree, or None outside a repo / without git."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    return completed.stdout.strip() or None


def _point_at_store(cfg: MLConfig) -> None:
    """Point the MLflow client at the configured store. Creates nothing.

    Separate from `_configure`: the scoring server mounts the store
    read-only, and selecting an experiment (which `_configure` also does)
    creates it on first use — that would crash a read-only mount.
    """
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)


def _configure(cfg: MLConfig) -> None:
    """Point at the store *and* select the experiment, creating it if absent."""
    _point_at_store(cfg)
    mlflow.set_experiment(cfg.mlflow_experiment)
    _check_experiment_matches_store(cfg)


def _check_experiment_matches_store(cfg: MLConfig) -> None:
    """Catch a stale experiment before a write fails deep inside mlflow's artifact writer.

    An experiment's artifact_location is fixed at creation time. Train once
    on the host, then again in a container against the same mlruns/ (or the
    reverse), and `set_experiment` above silently reuses the old experiment
    with its now-foreign artifact_location — the next write then fails with
    a bare PermissionError/FileNotFoundError several mlflow frames down, with
    nothing pointing at the actual cause (reproduced: training in Docker
    after a host run left a Windows path baked into the experiment tried to
    `mkdir` a literal `/C:` inside the container).

    Scoped to the local file store: a remote tracking server legitimately
    uses a different artifact_location scheme (e.g. mlflow-artifacts:/), so
    this would false-positive there.
    """
    if not cfg.mlflow_tracking_uri.startswith("file://"):
        return
    experiment = mlflow.get_experiment_by_name(cfg.mlflow_experiment)
    if experiment is not None and not experiment.artifact_location.startswith(
        cfg.mlflow_tracking_uri
    ):
        raise RuntimeError(
            f"Experiment {cfg.mlflow_experiment!r} already has artifacts under "
            f"{experiment.artifact_location!r}, which isn't under this process's store "
            f"({cfg.mlflow_tracking_uri!r}). This happens when the same mlruns/ directory "
            "is trained into from two different absolute-path contexts (host, then "
            "Docker, or the reverse) — fix by deleting mlruns/ and retraining from one "
            "context only."
        )


@contextmanager
def start_run(cfg: MLConfig = config, *, run_name: str = "train") -> Iterator[str]:
    """Open a top-level MLflow run and yield its run id."""
    _configure(cfg)
    with mlflow.start_run(run_name=run_name) as run:
        tags = {
            "model_type": cfg.model_type,
            "model_version": cfg.model_version,
            "decision_threshold": str(cfg.decision_threshold),
        }
        sha = _git_sha()
        if sha is not None:
            tags["git_sha"] = sha
        mlflow.set_tags(tags)
        run_id = str(run.info.run_id)
        logger.info("MLflow run %s started (%s)", run_id, cfg.mlflow_tracking_uri)
        yield run_id


@contextmanager
def resume_run(cfg: MLConfig, run_id: str) -> Iterator[None]:
    """Re-open an existing run so `eval.py` (a separate process from
    `train.py`) can log held-out metrics onto the same run as the model."""
    # The run id already names its experiment, so there is nothing to select.
    _point_at_store(cfg)
    with mlflow.start_run(run_id=run_id):
        yield None


@contextmanager
def nested_run(*, run_name: str) -> Iterator[None]:
    """Child run under the currently-active run — one per Optuna trial.

    A no-op when nothing is active, so `_optuna_search` stays callable on its
    own (as the unit tests call it) without scattering orphan runs.
    """
    if mlflow.active_run() is None:
        yield None
        return
    with mlflow.start_run(run_name=run_name, nested=True):
        yield None


def _active() -> bool:
    """True when there is a live run to log into.

    MLflow's fluent `log_*` calls silently create a run when none is active,
    so the `log_*` wrappers below all guard on this rather than let logging
    outside a `start_run` block scatter orphan runs across the store.
    """
    return mlflow.active_run() is not None


def log_params(params: Mapping[str, Any]) -> None:
    """Log hyperparameters/settings, coercing values to MLflow-safe strings."""
    if not _active():
        return
    mlflow.log_params({key: str(value) for key, value in params.items()})


def log_metrics(metrics: Mapping[str, float]) -> None:
    """Log scalar metrics; non-numeric values are skipped, not fatal."""
    if not _active():
        return
    clean = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
    if clean:
        mlflow.log_metrics(clean)


def log_artifact(path: Path, *, artifact_path: str | None = None) -> None:
    """Attach a file already written to disk (plots, joblib bundles, JSON reports)."""
    if not _active() or not path.exists():
        return
    mlflow.log_artifact(str(path), artifact_path=artifact_path)


def log_model(cfg: MLConfig, model: Classifier, X_sample: pd.DataFrame) -> None:
    """Log the fitted estimator with an inferred signature and an input example.

    Dispatches on `cfg.model_type` so a LightGBM booster is stored under the
    native `lightgbm` flavor (inspectable) rather than opaquely pickled by
    the sklearn one — both remain loadable via `mlflow.pyfunc`.
    """
    if not _active():
        return
    example = X_sample.head(5)
    # `np.asarray` for the same reason as train.py/eval.py: LightGBM's and
    # sklearn's `predict_proba` return types don't narrow to an ndarray for
    # the type checker across the `Classifier` union.
    signature = infer_signature(example, np.asarray(model.predict_proba(example))[:, 1])
    flavor = mlflow.lightgbm if cfg.model_type == "lightgbm" else mlflow.sklearn
    flavor.log_model(
        model,
        artifact_path=MODEL_ARTIFACT_PATH,
        signature=signature,
        input_example=example,
    )


def register_version(cfg: MLConfig, run_id: str, *, alias: str | None = None) -> str:
    """Register this run's model, move `alias` onto it, and return the new version.

    `alias` defaults to `cfg.mlflow_model_alias` (`champion`, the one the
    scoring server resolves) — pass e.g. `challenger` to publish without
    touching what is currently served.

    Unlike the fire-and-forget `log_*` helpers above, this raises
    `MlflowException` on failure rather than swallowing it: the alias is the
    only statement of which model is served, so a silent failure here would
    let `train.py` report success while serving kept answering with the
    previous model.
    """
    resolved_alias = alias or cfg.mlflow_model_alias
    version = mlflow.register_model(
        f"runs:/{run_id}/{MODEL_ARTIFACT_PATH}", cfg.mlflow_registered_model
    )
    client = MlflowClient(tracking_uri=cfg.mlflow_tracking_uri)
    client.set_registered_model_alias(cfg.mlflow_registered_model, resolved_alias, version.version)
    resolved = str(version.version)
    logger.info(
        "Registered %s v%s and moved alias @%s onto it.",
        cfg.mlflow_registered_model,
        resolved,
        resolved_alias,
    )
    return resolved


@dataclass(frozen=True)
class RegisteredModel:
    """A registry-resolved model version, plus the artifacts of the run behind it."""

    model: Classifier
    run_id: str
    version: str
    alias: str
    artifacts_dir: Path

    @property
    def display_version(self) -> str:
        """Human-readable identity carried through to `CreditScoreResult.model_version`."""
        return f"{self.version} (run {self.run_id[:8]})"

    def artifact(self, name: str) -> Path:
        """Path to one artifact of this model's run, checked to exist.

        Raises here rather than letting `joblib.load`/`read_parquet` fail on
        a missing file: the actual problem is an incomplete run, not a
        corrupt file, and the message says so.
        """
        path = self.artifacts_dir / name
        if not path.exists():
            raise FileNotFoundError(
                f"Run {self.run_id} (version {self.version}, @{self.alias}) has no {name!r} "
                "artifact. It was produced by an older `train.py`; re-run "
                "`python -m ml_pipeline.train` to publish a self-contained run."
            )
        return path


def resolve_model(cfg: MLConfig = config, *, alias: str | None = None) -> RegisteredModel:
    """Load `models:/{registered_model}@{alias}` and locate that run's artifacts.

    `alias` defaults to `cfg.mlflow_model_alias` (`champion`); passing e.g.
    `challenger` lets `eval.py` score a candidate without touching what's
    served. Resolves the whole run artifact directory, not just the model,
    so callers can also pick up the preprocessor, SHAP background and
    held-out split that belong to this exact version.

    Raises `MlflowException` when the alias doesn't resolve, deliberately:
    this is the critical serving path, so it must fail loudly at startup
    rather than fall back to something else.
    """
    # Read-only on purpose: the lookup is by registered-model name and alias,
    # so there is no experiment to select — and the serving store is mounted
    # read-only in docker-compose.
    _point_at_store(cfg)
    resolved_alias = alias or cfg.mlflow_model_alias
    client = MlflowClient(tracking_uri=cfg.mlflow_tracking_uri)
    try:
        version = client.get_model_version_by_alias(cfg.mlflow_registered_model, resolved_alias)
    except MlflowException as exc:
        # Re-raised with an actionable message: an unpublished model is the
        # most common first-run stumble, and MLflow's own "Registered Model
        # not found" traceback says nothing about how to fix it.
        publish = "" if resolved_alias == cfg.mlflow_model_alias else f" --alias {resolved_alias}"
        # MlflowException's own __init__ is untyped upstream; the ignore is
        # scoped to this call rather than relaxing --strict for the module.
        raise MlflowException(  # type: ignore[no-untyped-call]
            f"No model registered as {cfg.mlflow_registered_model}@{resolved_alias} in "
            f"{cfg.mlflow_tracking_uri}. Run `python -m ml_pipeline.make_dataset` then "
            f"`python -m ml_pipeline.train{publish}` first — MLflow is the only artifact "
            "store, so there is no local model to fall back on."
        ) from exc
    flavor = mlflow.lightgbm if cfg.model_type == "lightgbm" else mlflow.sklearn
    # Loaded from the version's own `source` rather than a `models:/name@alias`
    # URI. Both reach the same artifact, but the latter makes MLflow write a
    # `registered_model_meta` provenance file into the run's artifacts — which
    # fails against the read-only store the scoring server mounts. The alias
    # was already resolved above, so nothing is lost.
    model: Classifier = flavor.load_model(version.source)
    artifacts_dir = Path(mlflow.artifacts.download_artifacts(run_id=version.run_id))
    logger.info(
        "Resolved %s@%s -> v%s (run %s)",
        cfg.mlflow_registered_model,
        resolved_alias,
        version.version,
        version.run_id,
    )
    return RegisteredModel(
        model=model,
        run_id=str(version.run_id),
        version=str(version.version),
        alias=resolved_alias,
        artifacts_dir=artifacts_dir,
    )


__all__ = [
    "DIAGNOSTICS_DIR",
    "HOLDOUT_ARTIFACT",
    "METADATA_ARTIFACT",
    "METRICS_ARTIFACT",
    "MODEL_ARTIFACT_PATH",
    "PREPROCESSOR_ARTIFACT",
    "SHAP_BACKGROUND_ARTIFACT",
    "RegisteredModel",
    "log_artifact",
    "log_metrics",
    "log_model",
    "log_params",
    "nested_run",
    "register_version",
    "resolve_model",
    "resume_run",
    "start_run",
]
