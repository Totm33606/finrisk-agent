"""Evaluate the trained model on the held-out test set.

Reports both threshold-free ranking metrics (ROC-AUC, PR-AUC) and
threshold-dependent decision metrics (confusion matrix, precision/recall/F1
at `cfg.decision_threshold`) — the latter is what actually matters for the
APPROVE/REVIEW/DECLINE policy the agent will apply downstream.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False)


def _load_artifacts(
    cfg: MLConfig,
) -> tuple[Classifier, ColumnTransformer, pd.DataFrame]:
    if not cfg.model_path.exists():
        raise FileNotFoundError(
            f"No trained model at {cfg.model_path}. Run `python -m ml_pipeline.train` first."
        )
    model = joblib.load(cfg.model_path)
    preprocessor = joblib.load(cfg.preprocessor_path)
    test_df = pd.read_parquet(cfg.model_dir / "holdout_test.parquet")
    return model, preprocessor, test_df


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


def _plot_diagnostics(y_true: np.ndarray, y_proba: np.ndarray, out_dir: Path) -> None:
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

    y_pred = (y_proba >= config.decision_threshold).astype(int)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, ax=ax, display_labels=["No Default", "Default"], cmap="Blues"
    )
    ax.set_title(f"Confusion Matrix @ threshold={config.decision_threshold}")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)


@app.command()
def run() -> None:
    """Evaluate the persisted model on the held-out test set and write a metrics report."""
    cfg = config
    model, preprocessor, test_df = _load_artifacts(cfg)

    X_test = preprocessor.transform(test_df)
    X_test = to_feature_frame(X_test, get_feature_names(preprocessor))
    y_test = test_df[cfg.target_column].to_numpy()
    y_proba = np.asarray(model.predict_proba(X_test))[:, 1]

    metrics = compute_metrics(y_test, y_proba, cfg.decision_threshold)
    cfg.model_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.metrics_path.write_text(json.dumps(metrics, indent=2))

    _plot_diagnostics(y_test, y_proba, cfg.shap_plots_dir.parent)

    logger.info(
        "ROC-AUC=%.4f  PR-AUC=%.4f  Precision=%.3f  Recall=%.3f  F1=%.3f",
        metrics["roc_auc"],
        metrics["pr_auc"],
        metrics["precision_at_threshold"],
        metrics["recall_at_threshold"],
        metrics["f1_at_threshold"],
    )
    logger.info("Confusion matrix [[TN, FP], [FN, TP]] = %s", metrics["confusion_matrix"])
    logger.info("Full report written to %s", cfg.metrics_path)


if __name__ == "__main__":
    app()
