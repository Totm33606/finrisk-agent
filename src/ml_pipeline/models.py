"""Model factory: builds the classifier selected by `cfg.model_type`.

`LightGBM` and `LogisticRegression` both implement the same scikit-learn
classifier surface (`fit` / `predict_proba`), so this factory is the single
place that decides which one to instantiate — everything downstream
(cross-validation scoring, the persisted model artifact, `ScoringService`,
SHAP explanation) works against either one identically, which is exactly
what the model↔agent MCP boundary was designed to make swappable.
"""

from __future__ import annotations

from typing import Any, TypeAlias

import lightgbm as lgb
from sklearn.linear_model import LogisticRegression

from ml_pipeline.config import MLConfig

Classifier: TypeAlias = lgb.LGBMClassifier | LogisticRegression


def build_model(cfg: MLConfig, params: dict[str, Any]) -> Classifier:
    """Instantiate the classifier selected by `cfg.model_type` with the given hyperparameters."""
    model: Classifier
    if cfg.model_type == "lightgbm":
        model = lgb.LGBMClassifier(**params)
    else:
        model = LogisticRegression(**params)
    return model
