"""Unit tests for `ml_pipeline.preprocessing`."""

from __future__ import annotations

import pandas as pd
import pytest

from ml_pipeline.config import MLConfig
from ml_pipeline.preprocessing import build_preprocessor, get_feature_names


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "annual_revenue": [100_000.0, 250_000.0, None],
            "total_debt": [20_000.0, 80_000.0, 15_000.0],
            "debt_to_equity": [0.5, 1.2, 0.3],
            "current_ratio": [1.1, 0.8, 2.0],
            "ebitda_margin": [0.05, -0.02, 0.12],
            "days_payable_outstanding": [30.0, 45.0, 20.0],
            "days_sales_outstanding": [40.0, 60.0, 25.0],
            "late_payments_12m": [0, 3, 1],
            "years_in_business": [5.0, 1.0, 12.0],
            "employees": [10, 3, 50],
            "credit_utilization": [0.4, 0.9, 0.1],
            "sector": ["retail", "tech_services", None],
        }
    )


def test_preprocessor_handles_missing_values(sample_df: pd.DataFrame) -> None:
    cfg = MLConfig()
    preprocessor = build_preprocessor(cfg)

    transformed = preprocessor.fit_transform(sample_df)

    assert transformed.shape[0] == len(sample_df)
    assert not pd.isna(transformed).any(), "Imputation should remove all NaNs"


def test_preprocessor_handles_unseen_category_at_transform_time(sample_df: pd.DataFrame) -> None:
    cfg = MLConfig()
    preprocessor = build_preprocessor(cfg)
    preprocessor.fit(sample_df)

    unseen = sample_df.copy()
    unseen.loc[0, "sector"] = "space_mining"  # not seen during fit

    transformed = preprocessor.transform(unseen)  # must not raise
    assert transformed.shape[0] == len(unseen)


def test_feature_names_are_stable_and_non_empty(sample_df: pd.DataFrame) -> None:
    cfg = MLConfig()
    preprocessor = build_preprocessor(cfg)
    preprocessor.fit(sample_df)

    names = get_feature_names(preprocessor)

    assert len(names) > 0
    assert len(names) == len(set(names)), "Feature names must be unique"
    assert all(isinstance(n, str) for n in names)
