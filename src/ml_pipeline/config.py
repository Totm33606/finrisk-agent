"""Centralized, typed configuration for the credit-risk ML pipeline.

Kept as a single Pydantic settings object so `train.py`, `eval.py` and the
MCP server all read the exact same paths/thresholds — a common source of
train/serve skew is config drift between scripts, and this file exists to
remove that failure mode.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class MLConfig(BaseSettings):
    """Environment-overridable settings. Prefix: FINRISK_."""

    model_config = SettingsConfigDict(env_prefix="FINRISK_", env_file=".env", extra="ignore")

    # --- Paths -------------------------------------------------------
    data_dir: Path = PROJECT_ROOT / "data"
    raw_data_path: Path = PROJECT_ROOT / "data" / "clients.parquet"
    model_dir: Path = PROJECT_ROOT / "models"
    model_path: Path = PROJECT_ROOT / "models" / "lgbm_credit_risk.joblib"
    preprocessor_path: Path = PROJECT_ROOT / "models" / "preprocessor.joblib"
    metrics_path: Path = PROJECT_ROOT / "models" / "metrics.json"
    holdout_test_path: Path = PROJECT_ROOT / "models" / "holdout_test.parquet"
    shap_plots_dir: Path = PROJECT_ROOT / "reports" / "shap"

    @model_validator(mode="before")
    @classmethod
    def _derive_dependent_paths(cls, data: Any) -> Any:
        """Make `raw_data_path`/`model_path`/`preprocessor_path`/`metrics_path`/
        `holdout_test_path` follow an overridden `data_dir`/`model_dir` unless
        explicitly overridden themselves.

        Without this, e.g. `MLConfig(model_dir=tmp_path)` silently leaves `model_path`
        pointing at the real `PROJECT_ROOT/models/...` — every caller that overrides only
        the *_dir field (tests included) would otherwise read/write real project artifacts.
        """
        if not isinstance(data, dict):
            return data
        data_dir = data.get("data_dir")
        if data_dir is not None and "raw_data_path" not in data:
            data["raw_data_path"] = Path(data_dir) / "clients.parquet"
        model_dir = data.get("model_dir")
        if model_dir is not None:
            if "model_path" not in data:
                data["model_path"] = Path(model_dir) / "lgbm_credit_risk.joblib"
            if "preprocessor_path" not in data:
                data["preprocessor_path"] = Path(model_dir) / "preprocessor.joblib"
            if "metrics_path" not in data:
                data["metrics_path"] = Path(model_dir) / "metrics.json"
            if "holdout_test_path" not in data:
                data["holdout_test_path"] = Path(model_dir) / "holdout_test.parquet"
        return data

    # --- Target / split ------------------------------------------------
    target_column: str = "defaulted_12m"
    id_column: str = "client_id"
    test_size: float = 0.2
    n_cv_folds: int = 5
    random_state: int = 42

    # --- Feature groups --------------------------------------------------
    numeric_features: list[str] = Field(
        default_factory=lambda: [
            "annual_revenue",
            "total_debt",
            "debt_to_equity",
            "current_ratio",
            "ebitda_margin",
            "days_payable_outstanding",
            "days_sales_outstanding",
            "late_payments_12m",
            "years_in_business",
            "employees",
            "credit_utilization",
        ]
    )
    categorical_features: list[str] = Field(default_factory=lambda: ["sector"])

    # --- Model selection -------------------------------------------------
    # LightGBM is the default (best accuracy on this kind of tabular data);
    # logistic_regression is available as a simpler, natively-interpretable
    # alternative — both are built by `ml_pipeline.models.build_model` and
    # served identically downstream (ScoringService, SHAP explanation).
    model_type: Literal["lightgbm", "logistic_regression"] = "logistic_regression"

    # --- Model hyperparameters (LightGBM) -------------------------------
    lgbm_params: dict[str, object] = Field(
        default_factory=lambda: {
            "objective": "binary",
            "metric": "average_precision",
            "boosting_type": "gbdt",
            "n_estimators": 600,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 30,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "is_unbalance": True,
            "random_state": 42,
            "n_jobs": -1,
        }
    )

    # --- Model hyperparameters (logistic regression) ----------------------
    logreg_params: dict[str, object] = Field(
        default_factory=lambda: {
            "penalty": "l2",
            "C": 1.0,
            "solver": "liblinear",
            "max_iter": 1000,
            "class_weight": "balanced",
            "random_state": 42,
        }
    )

    # --- Decisioning ------------------------------------------------------
    decision_threshold: float = Field(
        0.30, description="PD above which a client moves out of auto-approve"
    )

    # --- Optuna --------------------------------------------------------
    optuna_n_trials: int = 40
    optuna_timeout_s: int = 900

    model_version: str = "1.0.0"


config = MLConfig()
