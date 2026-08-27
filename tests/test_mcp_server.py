"""Tests for `mcp_server.server` — the FastMCP tool/resource layer.

FastMCP's `@mcp.tool()` / `@mcp.resource()` decorators return the original
function unchanged (specifically so decorated tools stay directly callable
in tests, without spinning up an MCP transport), so these tests call
`get_credit_score` etc. exactly like `test_scoring_service.py` calls
`ScoringService` directly. The module-level service singleton is swapped
via `monkeypatch` so no trained model artifacts need to exist on disk.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import joblib
import lightgbm as lgb
import pytest
from fastmcp.exceptions import ToolError

import mcp_server.server as server_module
from common.schemas import CreditScoreResult, ScenarioResult, ShapExplanation
from mcp_server.scoring_service import ClientStore, ModelBundle, ScoringService
from ml_pipeline.config import MLConfig
from ml_pipeline.make_dataset import _simulate
from ml_pipeline.preprocessing import build_preprocessor, get_feature_names, load_raw_dataset
from ml_pipeline.shap_explainer import CreditRiskExplainer


@pytest.fixture
def tmp_cfg(tmp_path: Path) -> MLConfig:
    cfg = MLConfig(
        data_dir=tmp_path,
        raw_data_path=tmp_path / "clients.parquet",
        model_dir=tmp_path / "models",
        shap_plots_dir=tmp_path / "reports" / "shap",
    )
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    df = _simulate(n_clients=300, seed=cfg.random_state)
    df.to_parquet(cfg.raw_data_path, index=False)
    return cfg


@pytest.fixture
def wired_service(tmp_cfg: MLConfig, monkeypatch: pytest.MonkeyPatch) -> Iterator[ScoringService]:
    """Build a real (tiny, in-memory) `ScoringService` and wire it into `server._service`."""
    df = load_raw_dataset(tmp_cfg)
    preprocessor = build_preprocessor(tmp_cfg)
    X = preprocessor.fit_transform(df)
    y = df[tmp_cfg.target_column].to_numpy()

    model = lgb.LGBMClassifier(n_estimators=50, num_leaves=7, min_child_samples=5, verbosity=-1)
    model.fit(X, y, feature_name=get_feature_names(preprocessor))

    joblib.dump(model, tmp_cfg.model_path)
    joblib.dump(preprocessor, tmp_cfg.preprocessor_path)

    explainer = CreditRiskExplainer(model=model, feature_names=get_feature_names(preprocessor))
    bundle = ModelBundle(
        model=model,
        preprocessor=preprocessor,
        explainer=explainer,
        feature_names=get_feature_names(preprocessor),
        model_version="test",
    )
    store = ClientStore(data_path=tmp_cfg.raw_data_path, id_column=tmp_cfg.id_column)
    service = ScoringService(bundle=bundle, store=store, cfg=tmp_cfg)

    monkeypatch.setattr(server_module, "_service", service)
    yield service
    monkeypatch.setattr(server_module, "_service", None)


def test_get_credit_score_returns_result(wired_service: ScoringService) -> None:
    result = server_module.get_credit_score("SME-000001")

    assert isinstance(result, CreditScoreResult)
    assert 0.0 <= result.probability_default <= 1.0
    assert result.recommendation in {"APPROVE", "REVIEW", "DECLINE"}


def test_get_credit_score_unknown_client_raises_tool_error(
    wired_service: ScoringService,
) -> None:
    with pytest.raises(ToolError):
        server_module.get_credit_score("SME-999999")


def test_get_shap_explanation_returns_drivers(wired_service: ScoringService) -> None:
    explanation = server_module.get_shap_explanation("SME-000002", render_plot=False)

    assert isinstance(explanation, ShapExplanation)
    assert len(explanation.contributions) > 0
    assert explanation.plot_path is None  # render_plot=False was respected


def test_simulate_financial_scenario_moves_pd(wired_service: ScoringService) -> None:
    result = server_module.simulate_financial_scenario(
        "SME-000003", annual_revenue_delta_pct=-0.5, late_payments_12m_override=5
    )

    assert isinstance(result, ScenarioResult)
    assert result.simulated.probability_default != result.baseline.probability_default


def test_get_service_is_memoized(wired_service: ScoringService) -> None:
    assert server_module._get_service() is wired_service


def test_model_card_returns_valid_json_with_expected_keys() -> None:
    payload = json.loads(server_module.model_card())

    assert set(payload) == {"metadata", "metrics"}
    assert isinstance(payload["metadata"], dict)
    assert isinstance(payload["metrics"], dict)


def test_get_shap_explanation_unknown_client_raises_tool_error(
    wired_service: ScoringService,
) -> None:
    with pytest.raises(ToolError):
        server_module.get_shap_explanation("SME-999999")


def test_simulate_financial_scenario_unknown_client_raises_tool_error(
    wired_service: ScoringService,
) -> None:
    with pytest.raises(ToolError):
        server_module.simulate_financial_scenario("SME-999999")


def test_main_stdio_runs_mcp_with_stdio_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}
    monkeypatch.setattr(server_module.mcp, "run", lambda **kwargs: calls.update(kwargs))

    server_module.main(transport="stdio")

    assert calls == {"transport": "stdio"}


def test_main_http_runs_mcp_with_host_and_port(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}
    monkeypatch.setattr(server_module.mcp, "run", lambda **kwargs: calls.update(kwargs))

    server_module.main(transport="streamable-http", host="0.0.0.0", port=9000)

    assert calls == {"transport": "streamable-http", "host": "0.0.0.0", "port": 9000}
