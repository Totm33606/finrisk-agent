"""Evaluate a registered model on its own held-out test set.

Reports both threshold-free ranking metrics (ROC-AUC, PR-AUC) and
threshold-dependent decision metrics (confusion matrix, precision/recall/F1
at `cfg.decision_threshold`) — the latter is what actually matters for the
APPROVE/REVIEW/DECLINE policy the agent will apply downstream.

The model under evaluation is selected by registry alias (`--alias`,
defaulting to `champion`), and every output is written back into that
model's MLflow run. Nothing is read from or written to a local artifact
directory — see `ml_pipeline.tracking` for why.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import matplotlib
from sklearn.compose import ColumnTransformer

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ml_pipeline.config import MLConfig, config
from ml_pipeline.models import Classifier
from ml_pipeline.preprocessing import get_feature_names, to_feature_frame
from ml_pipeline.shap_explainer import CreditRiskExplainer
from ml_pipeline.tracking import (
    DIAGNOSTICS_DIR,
    HOLDOUT_ARTIFACT,
    METRICS_ARTIFACT,
    PREPROCESSOR_ARTIFACT,
    SHAP_BACKGROUND_ARTIFACT,
    RegisteredModel,
    log_artifact,
    log_metrics,
    resolve_model,
    resume_run,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False)


def _load_artifacts(
    target: RegisteredModel,
) -> tuple[Classifier, ColumnTransformer, pd.DataFrame]:
    """Take the model, its fitted preprocessor and its own held-out split from one run.

    The held-out split comes from the run rather than from a shared file on
    disk, which is what makes the evaluation honest for *any* alias: each
    registered version is scored on the rows that version never saw, not on
    whichever split the most recent training happened to leave behind.
    """
    return (
        target.model,
        joblib.load(target.artifact(PREPROCESSOR_ARTIFACT)),
        pd.read_parquet(target.artifact(HOLDOUT_ARTIFACT)),
    )


def compute_metrics(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float
) -> dict[str, float | list[list[int]]]:
    """Compute the full metric bundle for a set of predictions at a fixed threshold.

    `pr_auc` is `average_precision_score` (Average Precision), not a
    trapezoidal-rule `auc(recall, precision)` — see `_cv_average_precision`
    in `train.py` for why AP is the correct scalar summary of a PR curve.
    The key is kept as `pr_auc` since that's the near-universal name for
    this exact metric in ML tooling and literature.
    """
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "precision_at_threshold": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_at_threshold": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_at_threshold": float(f1_score(y_true, y_pred, zero_division=0)),
        "decision_threshold": threshold,
        "confusion_matrix": cm,  # [[TN, FP], [FN, TP]]
        "base_rate": float(y_true.mean()),
        "n_samples": len(y_true),
    }


def _plot_diagnostics(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float, out_dir: Path
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax)
    ax.set_title("ROC Curve — FinRisk-Agent")
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    PrecisionRecallDisplay.from_predictions(y_true, y_proba, ax=ax)
    ax.set_title("Precision-Recall Curve — FinRisk-Agent")
    fig.tight_layout()
    fig.savefig(out_dir / "pr_curve.png", dpi=150)
    plt.close(fig)

    y_pred = (y_proba >= threshold).astype(int)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, ax=ax, display_labels=["No Default", "Default"], cmap="Blues"
    )
    ax.set_title(f"Confusion Matrix @ threshold={threshold}")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def _plot_shap_summary(
    cfg: MLConfig,
    target: RegisteredModel,
    model: Classifier,
    feature_names: list[str],
    X: np.ndarray,
    out_dir: Path,
) -> Path | None:
    """Render the "what matters overall" SHAP beeswarm for this model version.

    A per-*model* artifact, unlike a per-client waterfall — so it belongs
    beside the metrics it explains. Returns None rather than raising if SHAP
    can't run: losing the plot must not cost the run its metrics.
    """
    background_path = target.artifacts_dir / SHAP_BACKGROUND_ARTIFACT
    try:
        explainer = CreditRiskExplainer(
            model=model,
            feature_names=feature_names,
            model_type=cfg.model_type,
            background=joblib.load(background_path) if background_path.exists() else None,
        )
        return explainer.render_summary(X, out_dir=out_dir)
    except Exception:
        logger.warning("Could not render the SHAP summary plot; metrics unaffected.", exc_info=True)
        return None


def _mlflow_metrics(metrics: dict[str, float | list[list[int]]]) -> dict[str, float]:
    """Flatten `compute_metrics` output into the scalars MLflow accepts.

    The confusion matrix is the only non-scalar; it is split into its four
    cells rather than dropped, since TN/FP/FN/TP are what a credit reviewer
    actually argues about when a threshold is being challenged.
    """
    flat = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
    cm = metrics.get("confusion_matrix")
    if isinstance(cm, list) and len(cm) == 2 and all(len(row) == 2 for row in cm):
        flat |= {
            "cm_true_negative": float(cm[0][0]),
            "cm_false_positive": float(cm[0][1]),
            "cm_false_negative": float(cm[1][0]),
            "cm_true_positive": float(cm[1][1]),
        }
    return flat


@app.command()
def run(
    alias: str = typer.Option(
        "",
        help="Registry alias to evaluate (default: the configured one, `champion`). "
        "e.g. --alias challenger",
    ),
) -> None:
    """Evaluate a registered model on its own held-out split and record the results.

    Which model is evaluated is a registry alias, not whatever files happen
    to sit on disk — `--alias challenger` scores a candidate without
    touching what's currently served. Results are written back into that
    model's own run (resumed, not a second one), so `pr_auc` always stays
    attached to the hyperparameters that produced it.
    """
    cfg = config
    target = resolve_model(cfg, alias=alias or None)
    model, preprocessor, test_df = _load_artifacts(target)

    transformed = preprocessor.transform(test_df)
    feature_names = get_feature_names(preprocessor)
    X_test = to_feature_frame(transformed, feature_names)
    y_test = test_df[cfg.target_column].to_numpy()
    y_proba = np.asarray(model.predict_proba(X_test))[:, 1]

    metrics = compute_metrics(y_test, y_proba, cfg.decision_threshold)

    # Scratch space: these files exist only to be handed to `log_artifact`,
    # which takes a path. The run is the only place they are kept.
    with TemporaryDirectory() as tmp:
        staging = Path(tmp)
        (staging / METRICS_ARTIFACT).write_text(json.dumps(metrics, indent=2))
        _plot_diagnostics(y_test, y_proba, cfg.decision_threshold, staging)
        summary_plot = _plot_shap_summary(cfg, target, model, feature_names, transformed, staging)

        with resume_run(cfg, target.run_id):
            log_metrics(_mlflow_metrics(metrics))
            log_artifact(staging / METRICS_ARTIFACT)
            for plot in ("roc_curve.png", "pr_curve.png", "confusion_matrix.png"):
                log_artifact(staging / plot, artifact_path=DIAGNOSTICS_DIR)
            if summary_plot is not None:
                log_artifact(summary_plot, artifact_path=DIAGNOSTICS_DIR)

    logger.info(
        "ROC-AUC=%.4f  PR-AUC=%.4f  Precision=%.3f  Recall=%.3f  F1=%.3f",
        metrics["roc_auc"],
        metrics["pr_auc"],
        metrics["precision_at_threshold"],
        metrics["recall_at_threshold"],
        metrics["f1_at_threshold"],
    )
    logger.info("Confusion matrix [[TN, FP], [FN, TP]] = %s", metrics["confusion_matrix"])
    logger.info(
        "Results recorded on run %s (%s v%s, @%s)",
        target.run_id,
        cfg.mlflow_registered_model,
        target.version,
        target.alias,
    )


if __name__ == "__main__":
    app()
