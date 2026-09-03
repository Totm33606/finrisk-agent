"""Tests for `mcp_server.server` — the FastMCP tool/resource layer.

FastMCP's `@mcp.tool()` / `@mcp.resource()` decorators return the original
function unchanged (specifically so decorated tools stay directly callable
in tests, without spinning up an MCP transport), so these tests call
`get_credit_score` etc. exactly like `test_scoring_service.py` calls
`ScoringService` directly. The module-level service singleton is swapped
via `monkeypatch` so no trained model artifacts need to exist on disk.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Iterator
from pathlib import Path

import lightgbm as lgb
import pytest
from fastmcp.exceptions import ToolError

import mcp_server.server as server_module
from common.schemas import CreditScoreResult, ScenarioResult, ShapExplanation
from mcp_server.scoring_service import ClientStore, ModelBundle, ScoringService
from ml_pipeline.config import MLConfig, config
from ml_pipeline.make_dataset import _simulate
from ml_pipeline.preprocessing import build_preprocessor, get_feature_names, load_raw_dataset
from ml_pipeline.shap_explainer import CreditRiskExplainer
from ml_pipeline.tracking import METADATA_ARTIFACT, METRICS_ARTIFACT


@pytest.fixture
def tmp_cfg(tmp_path: Path) -> MLConfig:
    cfg = MLConfig(
        data_dir=tmp_path,
        raw_data_path=tmp_path / "clients.parquet",
        mlflow_dir=tmp_path / "mlruns",
    )
    df = _simulate(n_clients=300, seed=cfg.random_state)
    df.to_parquet(cfg.raw_data_path, index=False)
    return cfg


@pytest.fixture
def wired_service(
    tmp_cfg: MLConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[ScoringService]:
    """Build a real (tiny, in-memory) `ScoringService` and wire it into `server._service`."""
    df = load_raw_dataset(tmp_cfg)
    preprocessor = build_preprocessor(tmp_cfg)
    X = preprocessor.fit_transform(df)
    y = df[tmp_cfg.target_column].to_numpy()

    model = lgb.LGBMClassifier(n_estimators=50, num_leaves=7, min_child_samples=5, verbosity=-1)
    model.fit(X, y, feature_name=get_feature_names(preprocessor))

    # The model card reads its two JSON files out of the bundle's artifacts
    # dir, so point it at a stand-in directory rather than a real run — these
    # tests are about the tool layer, not artifact resolution.
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / METADATA_ARTIFACT).write_text(json.dumps({"model_type": tmp_cfg.model_type}))
    (artifacts / METRICS_ARTIFACT).write_text(json.dumps({"roc_auc": 0.9}))

    explainer = CreditRiskExplainer(model=model, feature_names=get_feature_names(preprocessor))
    bundle = ModelBundle(
        model=model,
        preprocessor=preprocessor,
        explainer=explainer,
        feature_names=get_feature_names(preprocessor),
        model_version="test",
        run_id="0" * 32,
        artifacts_dir=artifacts,
    )
    store = ClientStore(tmp_cfg.raw_data_path, id_column=tmp_cfg.id_column)
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
    explanation = server_module.get_shap_explanation("SME-000002")

    assert isinstance(explanation, ShapExplanation)
    assert len(explanation.contributions) > 0


def test_simulate_financial_scenario_moves_pd(wired_service: ScoringService) -> None:
    result = server_module.simulate_financial_scenario(
        "SME-000003", annual_revenue_delta_pct=-0.5, late_payments_12m_override=5
    )

    assert isinstance(result, ScenarioResult)
    assert result.simulated.probability_default != result.baseline.probability_default


def test_get_service_is_memoized(wired_service: ScoringService) -> None:
    assert server_module._get_service() is wired_service


def test_model_card_describes_the_served_model(wired_service: ScoringService) -> None:
    """The card must name the version the tools are actually scoring with —
    it now reads the served run's own artifacts, not whatever is on disk."""
    payload = json.loads(server_module.model_card())

    assert set(payload) == {
        "registered_model",
        "alias",
        "model_version",
        "run_id",
        "metadata",
        "metrics",
    }
    assert payload["model_version"] == wired_service._bundle.model_version
    assert payload["run_id"] == wired_service._bundle.run_id
    assert payload["metadata"]["model_type"] == wired_service._cfg.model_type
    assert payload["metrics"]["roc_auc"] == 0.9


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


def test_main_alias_option_selects_the_served_version(monkeypatch: pytest.MonkeyPatch) -> None:
    """`--alias` picks which registry version the server resolves on first use.

    Asserted on the shared `config` object rather than `server_module.config`,
    which is the same object reached through an implicit re-export.
    """
    monkeypatch.setattr(server_module.mcp, "run", lambda **kwargs: None)
    monkeypatch.setattr(config, "mlflow_model_alias", "champion")

    server_module.main(transport="stdio", alias="challenger")

    assert config.mlflow_model_alias == "challenger"


def test_main_without_alias_leaves_the_configured_one_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Omitting the flag must not clobber FINRISK_MLFLOW_MODEL_ALIAS."""
    monkeypatch.setattr(server_module.mcp, "run", lambda **kwargs: None)
    monkeypatch.setattr(config, "mlflow_model_alias", "staging")

    server_module.main(transport="stdio")

    assert config.mlflow_model_alias == "staging"


def test_mcp_tools_expose_no_alias_parameter() -> None:
    """The served version is startup state, not a per-call argument.

    Threading the alias through `_get_service()` would have put it in all
    three tool signatures — extra surface for the agent to get wrong, for
    something that cannot change while the server is running.
    """
    tools = (
        server_module.get_credit_score,
        server_module.get_shap_explanation,
        server_module.simulate_financial_scenario,
    )

    for tool in tools:
        assert "alias" not in inspect.signature(tool).parameters
