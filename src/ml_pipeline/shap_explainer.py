"""SHAP-based local & global explainability for the credit-risk model.

Two consumers:
    1. `eval.py` / notebooks — global summary plots for model documentation.
    2. `mcp_server.server` — per-client structured contributions, consumed
       by the agent's `get_shap_explanation` tool and rendered by the React
       waterfall component.

`shap.TreeExplainer` is used because it is exact (not sampling-based) and
fast for tree ensembles like LightGBM, computing SHAP values via the
recursive tree-traversal algorithm rather than Kernel SHAP's model-agnostic
but much slower perturbation approach.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap

from common.schemas import ShapContribution, ShapExplanation

logger = logging.getLogger(__name__)


class CreditRiskExplainer:
    """Thin, cached wrapper around a SHAP explainer for the trained model.

    Instantiated once (e.g. at MCP server startup) and reused across
    requests. Picks the SHAP algorithm appropriate for `model_type`:
    `TreeExplainer` (exact, no background data needed) for LightGBM, or
    `LinearExplainer` (needs a background sample to estimate the baseline
    expected value) for logistic regression — both expose the same
    `explainer(X)` call interface, so the rest of this class stays
    model-agnostic.
    """

    def __init__(
        self,
        model: object,
        feature_names: list[str],
        model_type: str = "lightgbm",
        background: np.ndarray | None = None,
    ) -> None:
        self._model = model
        self._feature_names = feature_names
        if model_type == "lightgbm":
            self._explainer = shap.TreeExplainer(model)
        else:
            if background is None:
                raise ValueError(
                    f"model_type={model_type!r} requires background data for shap.LinearExplainer"
                )
            self._explainer = shap.LinearExplainer(model, background)

    def explain_row(
        self, x_row: np.ndarray, client_id: str, raw_values: dict[str, float] | None = None
    ) -> ShapExplanation:
        """Compute a local SHAP explanation for a single (already-preprocessed) row.

        Args:
            x_row: 1-D transformed feature vector (output of the fitted preprocessor).
            client_id: identifier propagated into the returned explanation.
            raw_values: optional pre-transform feature values, used only for
                display in `ShapContribution.value` when provided (falls back
                to the transformed value otherwise, e.g. for one-hot columns).
        """
        explanation = self._explainer(x_row.reshape(1, -1))
        shap_values = explanation.values[0]
        base_value = float(np.asarray(explanation.base_values).flatten()[0])

        contributions = [
            ShapContribution(
                feature=name,
                value=float((raw_values or {}).get(name, x_row[i])),
                shap_value=float(shap_values[i]),
            )
            for i, name in enumerate(self._feature_names)
        ]
        ranked = sorted(contributions, key=lambda c: c.shap_value, reverse=True)
        top_positive = [c.feature for c in ranked if c.shap_value > 0][:5]
        top_negative = [c.feature for c in ranked[::-1] if c.shap_value < 0][:5]

        return ShapExplanation(
            client_id=client_id,
            base_value=base_value,
            contributions=contributions,
            top_positive_drivers=top_positive,
            top_negative_drivers=top_negative,
            plot_path=None,
        )

    def render_waterfall(self, x_row: np.ndarray, client_id: str, out_dir: Path) -> Path:
        """Render and save a SHAP waterfall plot PNG for one client. Returns the file path."""
        out_dir.mkdir(parents=True, exist_ok=True)
        explanation = self._explainer(x_row.reshape(1, -1))
        explanation.feature_names = self._feature_names

        fig = plt.figure(figsize=(9, 6))
        shap.plots.waterfall(explanation[0], show=False, max_display=12)
        out_path = out_dir / f"waterfall_{client_id}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Waterfall plot for %s written to %s", client_id, out_path)
        return out_path

    def render_summary(self, X: np.ndarray, out_dir: Path, sample_size: int = 2000) -> Path:
        """Render a global SHAP summary (beeswarm) plot over a sample of the dataset."""
        out_dir.mkdir(parents=True, exist_ok=True)
        if len(X) > sample_size:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X), size=sample_size, replace=False)
            X = X[idx]

        shap_values = self._explainer(X)
        shap_values.feature_names = self._feature_names

        fig = plt.figure(figsize=(9, 7))
        shap.plots.beeswarm(shap_values, show=False, max_display=15)
        out_path = out_dir / "shap_summary.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Global SHAP summary plot written to %s", out_path)
        return out_path
