"""Unit tests for `ml_pipeline.make_dataset`."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ml_pipeline import make_dataset as make_dataset_module
from ml_pipeline.config import MLConfig
from ml_pipeline.make_dataset import SECTORS, _simulate


def test_simulate_produces_expected_schema_and_row_count() -> None:
    df = _simulate(n_clients=500, seed=42)

    assert len(df) == 500
    assert df["client_id"].is_unique
    assert set(df["sector"]).issubset(set(SECTORS))
    assert df["defaulted_12m"].isin([0, 1]).all()


def test_simulate_is_deterministic_given_same_seed() -> None:
    first = _simulate(n_clients=200, seed=7)
    second = _simulate(n_clients=200, seed=7)

    pd.testing.assert_frame_equal(first, second)


def test_simulate_different_seeds_produce_different_data() -> None:
    first = _simulate(n_clients=200, seed=1)
    second = _simulate(n_clients=200, seed=2)

    assert not first["annual_revenue"].equals(second["annual_revenue"])


def test_simulate_default_rate_is_plausible_and_risk_ordered() -> None:
    """Higher-leverage clients should default more often than lower-leverage ones, on average —
    the causal structure the module's docstring claims for the latent risk score."""
    df = _simulate(n_clients=20_000, seed=42)

    default_rate = df["defaulted_12m"].mean()
    assert 0.02 < default_rate < 0.5

    median_dte = df["debt_to_equity"].median()
    high_leverage_rate = df.loc[df["debt_to_equity"] > median_dte, "defaulted_12m"].mean()
    low_leverage_rate = df.loc[df["debt_to_equity"] <= median_dte, "defaulted_12m"].mean()
    assert high_leverage_rate > low_leverage_rate


def test_build_command_writes_parquet_to_configured_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = MLConfig(data_dir=tmp_path, raw_data_path=tmp_path / "clients.parquet")
    monkeypatch.setattr(make_dataset_module, "config", cfg)

    make_dataset_module.build(n_clients=100, seed=7)

    assert cfg.raw_data_path.exists()
    df = pd.read_parquet(cfg.raw_data_path)
    assert len(df) == 100
