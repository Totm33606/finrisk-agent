"""Generate a synthetic but structurally realistic SME credit dataset.

Real credit-bureau data can't be published in a public showcase repo, so
this module produces a synthetic dataset with the same feature schema and
plausible causal structure (financially weaker profiles are more likely to
default), which lets `train.py` / `eval.py` / the MCP server / the agent all
run out of the box via `make dataset`.

Not intended as a substitute for a real underwriting dataset — see the
README's "Data" section for how to swap in your own source.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import typer

from ml_pipeline.config import config

app = typer.Typer(add_completion=False)

SECTORS = ["retail", "manufacturing", "construction", "hospitality", "tech_services", "logistics"]


def _zscore(x: np.ndarray) -> np.ndarray:
    return np.asarray((x - x.mean()) / (x.std() + 1e-9))


def _simulate(n_clients: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    years_in_business = rng.gamma(shape=2.2, scale=4.0, size=n_clients).clip(0.1, 40)
    annual_revenue = rng.lognormal(mean=13.5, sigma=1.1, size=n_clients).clip(20_000, 50_000_000)
    ebitda_margin = rng.normal(loc=0.08, scale=0.09, size=n_clients).clip(-0.4, 0.45)
    debt_to_equity = rng.gamma(shape=1.6, scale=1.1, size=n_clients).clip(0, 12)
    current_ratio = rng.gamma(shape=3.0, scale=0.5, size=n_clients).clip(0.1, 6)
    total_debt = (annual_revenue * rng.uniform(0.05, 0.9, size=n_clients)).round(2)
    dpo = rng.gamma(shape=3.0, scale=15.0, size=n_clients).clip(0, 180)
    dso = rng.gamma(shape=3.0, scale=14.0, size=n_clients).clip(0, 180)
    late_payments_12m = rng.poisson(lam=0.6, size=n_clients)
    employees = rng.poisson(lam=annual_revenue / 120_000 + 1)
    credit_utilization = rng.beta(a=2.0, b=2.5, size=n_clients) * 1.4
    sector = rng.choice(SECTORS, size=n_clients)

    # Latent risk score: a weighted combination of z-scored financial stress
    # signals, transformed through a logistic link, drives realized default.
    # Z-scoring (rather than dividing by the sample max) keeps each
    # coefficient's effective influence stable regardless of outliers, and
    # a moderate noise term keeps the problem learnable-but-imperfect, the
    # way real credit risk is.
    z = (
        -4.4
        + 1.7 * _zscore(debt_to_equity)
        - 1.6 * _zscore(ebitda_margin)
        - 1.1 * _zscore(current_ratio)
        + 0.75 * late_payments_12m
        + 1.5 * _zscore(credit_utilization)
        - 0.7 * _zscore(np.log1p(years_in_business))
        - 0.5 * _zscore(np.log1p(annual_revenue))
        + rng.normal(0, 0.35, size=n_clients)
    )
    probability_default = 1 / (1 + np.exp(-z))
    defaulted_12m = rng.binomial(1, probability_default.clip(0.01, 0.95))

    df = pd.DataFrame(
        {
            "client_id": [f"SME-{i:06d}" for i in range(n_clients)],
            "annual_revenue": annual_revenue.round(2),
            "total_debt": total_debt,
            "debt_to_equity": debt_to_equity.round(3),
            "current_ratio": current_ratio.round(3),
            "ebitda_margin": ebitda_margin.round(3),
            "days_payable_outstanding": dpo.round(1),
            "days_sales_outstanding": dso.round(1),
            "late_payments_12m": late_payments_12m,
            "years_in_business": years_in_business.round(1),
            "sector": sector,
            "employees": employees.clip(1, None),
            "credit_utilization": credit_utilization.round(3),
            "defaulted_12m": defaulted_12m,
        }
    )
    return df


@app.command()
def build(
    n_clients: int = typer.Option(20_000, help="Number of synthetic SME records to generate"),
    seed: int = typer.Option(config.random_state, help="RNG seed for reproducibility"),
) -> None:
    """Build the synthetic dataset and write it to `config.raw_data_path`."""
    df = _simulate(n_clients=n_clients, seed=seed)
    config.data_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.raw_data_path, index=False)
    typer.echo(
        f"Wrote {len(df):,} rows ({df['defaulted_12m'].mean():.1%} default rate) "
        f"to {config.raw_data_path}"
    )


if __name__ == "__main__":
    app()
