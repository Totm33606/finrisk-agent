"""Unit tests for `common.schemas`."""

from __future__ import annotations

from common.schemas import RiskBand


def test_risk_band_from_probability_covers_all_bands() -> None:
    assert RiskBand.from_probability(0.05) == RiskBand.LOW
    assert RiskBand.from_probability(0.15) == RiskBand.MEDIUM
    assert RiskBand.from_probability(0.45) == RiskBand.HIGH
    assert RiskBand.from_probability(0.75) == RiskBand.CRITICAL


def test_risk_band_from_probability_boundaries_are_exclusive_upper() -> None:
    assert RiskBand.from_probability(0.10) == RiskBand.MEDIUM
    assert RiskBand.from_probability(0.30) == RiskBand.HIGH
    assert RiskBand.from_probability(0.60) == RiskBand.CRITICAL
