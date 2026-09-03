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
    # The raw dataset is the only path left outside the MLflow store: it's
    # the pipeline's entry point, so it has to come from somewhere. Every
    # produced artifact — model, preprocessor, held-out split, metrics,
    # plots — lives in MLflow and nowhere else.
    data_dir: Path = PROJECT_ROOT / "data"
    raw_data_path: Path = PROJECT_ROOT / "data" / "clients.parquet"

    # --- MLflow tracking / registry ---------------------------------------
    # `mlflow_dir` is a plain file store rather than `sqlite:///`: MLflow
    # 2.x's file store already implements the registry surface used here
    # (register, aliases, `models:/name@alias`), so a database would add a
    # migration for nothing. Point `mlflow_tracking_uri` at an `http://`
    # server to centralize runs for a team; leave it empty to use the local
    # `mlflow_dir` file store.
    mlflow_dir: Path = PROJECT_ROOT / "mlruns"
    mlflow_tracking_uri: str = ""
    mlflow_experiment: str = "finrisk-credit-risk"
    mlflow_registered_model: str = "finrisk-credit-risk"
    mlflow_model_alias: str = "champion"

    @model_validator(mode="before")
    @classmethod
    def _derive_dependent_paths(cls, data: Any) -> Any:
        """Derive `raw_data_path` from `data_dir`, and the tracking URI from `mlflow_dir`.

        Without this, a test overriding only `mlflow_dir` would still write
        into the developer's real `./mlruns` store.
        """
        if not isinstance(data, dict):
            return data
        data_dir = data.get("data_dir")
        if data_dir is not None and "raw_data_path" not in data:
            data["raw_data_path"] = Path(data_dir) / "clients.parquet"
        if not data.get("mlflow_tracking_uri"):
            # `.resolve()` before `.as_uri()`: the latter rejects relative paths.
            mlflow_dir = Path(data.get("mlflow_dir") or PROJECT_ROOT / "mlruns")
            data["mlflow_tracking_uri"] = mlflow_dir.resolve().as_uri()
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
    # LightGBM is the default: better precision/F1 at the operating threshold,
    # which is what the decision policy acts on (logistic regression edges it
    # on the threshold-free ranking metrics instead, so this isn't a clean
    # win — see README). logistic_regression remains available as a simpler,
    # natively-interpretable alternative; both are built by
    # `ml_pipeline.models.build_model` and served identically downstream.
    # Also selects the MLflow flavor `tracking.log_model` uses, so tests that
    # build a model by hand should set this explicitly.
    model_type: Literal["lightgbm", "logistic_regression"] = "lightgbm"

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
