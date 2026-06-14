"""Fast e2e smoke contracts for the public demo flows."""

# Tests import application modules inside test bodies so monkeypatching and
# safe environment setup happen before module initialization.

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VALID_PREDICTION_PAYLOAD = {
    "Age": 30,
    "Gender": 1,
    "AnnualIncome": 50000,
    "NumberOfPurchases": 10,
    "TimeSpentOnWebsite": 120.5,
    "LoyaltyProgram": 1,
    "DiscountsAvailed": 2,
    "PurchaseStatus": 1,
}


def test_make_targets_cover_supported_fast_demo_configs() -> None:
    """Make targets should wire the supported smoke paths without executing them.

    This intentionally reads the Makefile instead of `make --dry-run`. GNU Make
    can execute recursive recipes under dry-run when a logical recipe line
    contains `$(MAKE)`, which is unsafe for `up-dev-run`.
    """
    makefile = _makefile_text()

    assert "$(MAKE) train MODEL=randomforest MAX_EVALS=0 CLEARML=false" in makefile
    assert "poetry run select-model" in makefile
    assert "--models $(MODELS)" in makefile
    assert "--max-evals $(MAX_EVALS)" in makefile
    assert "$(if $(filter true,$(CLEARML)),--clearml,)" in makefile
    assert "$(if $(filter true,$(SMOTE)),--smote,--no-smote)" in makefile
    assert "$(MAKE) up-dev-run LLM_PROVIDER=ollama" in makefile
    assert 'OLLAMA_MODEL="$(FINETUNED_OLLAMA_MODEL)"' in makefile
    assert "OLLAMA_SKIP_PULL=true" in makefile
    assert "OLLAMA_DOCKER=false" in makefile


def test_up_dev_run_starts_services_apis_and_has_cleanup_contract() -> None:
    """`up-dev-run` should start dependencies, APIs, chat, and cleanup on exit."""
    output = _makefile_text()

    assert "OPENAI_API_KEY is required for LLM_PROVIDER=openai" in output
    assert "docker compose up postgres" in output
    assert "clearml-apiserver" in output
    assert "clearml-webserver" in output
    assert "clearml-serving-up" in output
    assert "uvicorn interface.api.app:app" in output
    assert "adk web" in output
    assert "docker compose stop" in output
    assert "Stopping dev stack" in output


def test_api_routes_available_with_cached_predictor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """API routes should expose health/model/predict and reuse the predictor."""
    from interface.api import app as api_app

    _reset_api_cache(api_app)
    artifact_dir = tmp_path / "classical_ml" / "smoke_run"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "model.joblib").write_text("mock-model")

    fake_predictor = SimpleNamespace(
        predict_one=lambda _features: {
            "class_id": 1,
            "label": "Clothing",
            "confidence": 0.95,
            "probabilities": [],
        }
    )
    predictor_calls: list[Path | None] = []

    def fake_predictor_factory(artifact_dir: Path | None = None) -> Any:
        predictor_calls.append(artifact_dir)
        return fake_predictor

    monkeypatch.setattr(api_app, "latest_model_dir", lambda _root: artifact_dir)
    monkeypatch.setattr(api_app, "ProductCategoryPredictor", fake_predictor_factory)

    client = TestClient(api_app.app)
    try:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/models/latest").json() == {
            "available": True,
            "artifact_dir": str(artifact_dir),
        }

        first_response = client.post("/predict", json=VALID_PREDICTION_PAYLOAD)
        second_response = client.post("/predict", json=VALID_PREDICTION_PAYLOAD)

        assert first_response.status_code == 200
        assert first_response.json()["label"] == "Clothing"
        assert second_response.status_code == 200
        assert predictor_calls == [artifact_dir]
    finally:
        _reset_api_cache(api_app)


def test_adk_chat_model_contracts_for_openai_ollama_and_finetuned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ADK chat should select the right model wrapper for supported providers."""
    from interface.chat.ecommerce_chat import agent

    monkeypatch.setattr(agent, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(agent, "OPENAI_MODEL", "gpt-4o-mini")
    assert "openai/gpt-4o-mini" in _model_name(agent.build_model())

    monkeypatch.setattr(agent, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(agent, "OLLAMA_MODEL", "llama3.1:8b")
    assert "ollama_chat/llama3.1:8b" in _model_name(agent.build_model())

    monkeypatch.setattr(agent, "OLLAMA_MODEL", "finetuned-tinyllama_1.1b")
    assert "ollama_chat/finetuned-tinyllama_1.1b" in _model_name(agent.build_model())


@pytest.mark.parametrize(
    ("provider", "model_name", "expected_model"),
    [
        ("openai", "gpt-4o-mini", "openai/gpt-4o-mini"),
        ("ollama", "llama3.1:8b", "ollama_chat/llama3.1:8b"),
        ("ollama", "finetuned-tinyllama_1.1b", "ollama_chat/finetuned-tinyllama_1.1b"),
    ],
    ids=["openai", "ollama", "ollama-finetuned"],
)
def test_agent_completed_conversation_invokes_tool_and_returns_user_answer(
    provider: str,
    model_name: str,
    expected_model: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Completed chat flow should invoke the tool and produce a user answer.

    The LLM call itself is mocked by using a deterministic completed transcript.
    This keeps the test fast while validating the contract used by all
    `make up-dev*` chat providers.
    """
    from interface.chat.ecommerce_chat import agent, tools

    if provider == "openai":
        monkeypatch.setattr(agent, "LLM_PROVIDER", "openai")
        monkeypatch.setattr(agent, "OPENAI_MODEL", model_name)
    else:
        monkeypatch.setattr(agent, "LLM_PROVIDER", "ollama")
        monkeypatch.setattr(agent, "OLLAMA_MODEL", model_name)

    assert expected_model in _model_name(agent.build_model())
    assert _agent_has_prediction_tool(agent.root_agent)
    _assert_instruction_matches_feature_encoding(agent.load_instruction())

    captured_request: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: int) -> _FakeHttpResponse:
        captured_request["timeout"] = timeout
        captured_request["payload"] = json.loads(request.data.decode("utf-8"))
        return _FakeHttpResponse(
            {
                "class_id": 1,
                "label": "Clothing",
                "confidence": 0.93,
                "probabilities": [],
            }
        )

    monkeypatch.setattr(tools.urllib.request, "urlopen", fake_urlopen)

    tool_payload = _features_from_completed_transcript()
    tool_result = tools.predict_product_category(**tool_payload)
    final_answer = _render_agent_prediction_answer(tool_result)

    assert captured_request["timeout"] == 30
    assert captured_request["payload"] == tool_payload
    assert captured_request["payload"]["Gender"] == 1
    assert captured_request["payload"]["LoyaltyProgram"] == 1
    assert captured_request["payload"]["PurchaseStatus"] == 1
    assert tool_result["label"] == "Clothing"
    assert "Clothing" in final_answer
    assert "recommend" in final_answer.lower()
    assert "api" not in final_answer.lower()
    assert "model" not in final_answer.lower()
    assert "0.93" not in final_answer
    assert "{" not in final_answer


@pytest.mark.parametrize("model", ["llama3.1:8b", "finetuned-tinyllama_1.1b"])
def test_ollama_chat_provider_contract(
    model: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ollama provider should call `/api/chat` with the selected runtime model."""
    from src.providers.base import LLMMessage
    from src.providers.ollama import OllamaProvider
    import src.providers.ollama as ollama_module

    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: int) -> _FakeHttpResponse:
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _FakeHttpResponse(
            {"message": {"content": f"mock response from {model}"}}
        )

    monkeypatch.setattr(ollama_module.urllib.request, "urlopen", fake_urlopen)

    provider = OllamaProvider(
        base_url="http://mock-ollama:11434",
        model=model,
        timeout_seconds=7,
    )
    response = provider.generate(
        [LLMMessage(role="user", content="collect ecommerce features")],
        system_prompt="You are a test agent.",
    )

    assert response == f"mock response from {model}"
    assert captured["url"] == "http://mock-ollama:11434/api/chat"
    assert captured["timeout"] == 7
    assert captured["payload"]["model"] == model
    assert captured["payload"]["stream"] is False
    assert captured["payload"]["messages"][0]["role"] == "system"
    assert captured["payload"]["messages"][1]["role"] == "user"


def test_training_and_model_selection_clearml_contracts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Training/model-selection should pass ClearML flags without running ML."""
    from interface.cli import training_commands

    training_calls: list[dict[str, Any]] = []

    class FakeTrainingWorkflow:
        """Fake model workflow for CLI contract assertions."""

        model_name = "randomforest"

        def run(self, **kwargs: Any) -> SimpleNamespace:
            """Collect training kwargs and return a fake result."""
            training_calls.append(kwargs)
            artifact_dir = tmp_path / "artifact"
            artifact_dir.mkdir()
            return SimpleNamespace(
                model_name="randomforest",
                run_id="smoke_run",
                artifact_dir=artifact_dir,
                metrics={"f1_macro": 1.0},
                report_path=artifact_dir / "report.md",
            )

    monkeypatch.setattr(
        training_commands, "get_workflow", lambda _model: FakeTrainingWorkflow()
    )

    training_result = training_commands.run_training(
        model="randomforest",
        data_path=tmp_path / "dataset.csv",
        max_evals=0,
        clearml_enabled=True,
    )

    assert training_result["model_name"] == "randomforest"
    assert training_calls[0]["max_evals"] == 0
    assert training_calls[0]["clearml_enabled"] is True

    selection_calls: list[dict[str, Any]] = []

    class FakeSelectionWorkflow:
        """Fake selection workflow for CLI contract assertions."""

        def run(self, **kwargs: Any) -> dict[str, Any]:
            """Collect selection kwargs and return a fake result."""
            selection_calls.append(kwargs)
            return {"best_model": "randomforest", "runs": []}

    monkeypatch.setattr(
        training_commands, "ModelSelectionWorkflow", FakeSelectionWorkflow
    )

    selection_result = training_commands.run_model_selection(
        models="randomforest",
        data_path=tmp_path / "dataset.csv",
        max_evals=0,
        clearml_enabled=True,
    )

    assert selection_result["best_model"] == "randomforest"
    assert selection_calls[0]["models"] == "randomforest"
    assert selection_calls[0]["max_evals"] == 0
    assert selection_calls[0]["clearml_enabled"] is True


def test_clearml_tracker_reports_metrics_without_external_clearml() -> None:
    """Tracker should send metrics/tables/text to the active task logger."""
    from evaluation.tracking.clearml_tracker import ClearMLTracker

    fake_logger = _FakeClearMLLogger()
    tracker = ClearMLTracker(enabled=True)
    tracker._logger = fake_logger
    tracker.task = object()

    tracker.report_metrics("training", "randomforest", {"f1_macro": 0.91})
    tracker.report_table(
        "training/metrics",
        "randomforest",
        pd.DataFrame([{"metric": "f1_macro", "value": 0.91}]),
    )
    tracker.report_text("training completed")

    assert fake_logger.scalars == [
        {
            "title": "training/f1_macro",
            "series": "randomforest",
            "value": 0.91,
            "iteration": 0,
        }
    ]
    assert fake_logger.tables[0]["title"] == "training/metrics"
    assert fake_logger.texts == ["training completed"]


def test_finetuning_clearml_and_provider_paths_do_not_train(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fine-tuning commands should route to mocked ClearML/provider paths."""
    from interface.cli import llm_commands
    import src.finetuning.clearml_pipeline as clearml_pipeline

    clearml_calls: list[dict[str, Any]] = []

    def fake_clearml_pipeline(**kwargs: Any) -> dict[str, Any]:
        clearml_calls.append(kwargs)
        return {
            "provider": "ollama",
            "adapter_or_job": "data/runs/llm_finetuning/smoke/adapter",
            "clearml_task_id": "mock-task",
        }

    monkeypatch.setattr(
        clearml_pipeline,
        "run_clearml_ollama_finetuning_pipeline",
        fake_clearml_pipeline,
    )

    clearml_result = llm_commands.run_finetuning(
        provider="ollama",
        model="tinyllama:1.1b",
        base_model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        data_path="mock-dataset",
        epochs=1,
        batch_size=1,
        max_seq_length=128,
        max_train_samples=1,
        clearml_enabled=True,
    )

    assert clearml_result["clearml_task_id"] == "mock-task"
    assert clearml_calls[0]["ollama_model"] == "tinyllama:1.1b"
    assert clearml_calls[0]["max_train_samples"] == 1

    provider_calls: list[dict[str, Any]] = []

    class FakeProvider:
        """Fake LLM provider for fine-tuning routing assertions."""

        model = "tinyllama:1.1b"

        def finetune(self, **kwargs: Any) -> str:
            """Collect fine-tuning kwargs and return a fake adapter path."""
            provider_calls.append(kwargs)
            return "data/runs/llm_finetuning/smoke/adapter"

    monkeypatch.setattr(
        llm_commands,
        "build_llm_provider",
        lambda _provider, allow_fallback: FakeProvider(),
    )

    provider_result = llm_commands.run_finetuning(
        provider="ollama",
        model="tinyllama:1.1b",
        data_path="mock-dataset",
        epochs=1,
        max_train_samples=1,
        clearml_enabled=False,
    )

    assert provider_result["adapter_or_job"].endswith("/adapter")
    assert provider_calls[0]["epochs"] == 1
    assert provider_calls[0]["max_train_samples"] == 1
    assert provider_calls[0]["clearml_enabled"] is False


def test_adapter_registration_clearml_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """Adapter registration should request ClearML logging without retraining."""
    from interface.cli import llm_commands

    calls: list[dict[str, Any]] = []

    def fake_register_existing_lora_adapter(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {
            "adapter_dir": kwargs["adapter_path"],
            "finetuned_ollama_model": "finetuned-tinyllama_1.1b",
        }

    fake_local_qlora = ModuleType("src.finetuning.local_qlora")
    fake_local_qlora.register_existing_lora_adapter = (
        fake_register_existing_lora_adapter
    )
    monkeypatch.setitem(sys.modules, "src.finetuning.local_qlora", fake_local_qlora)

    result = llm_commands.run_adapter_registration(
        adapter_path="data/runs/llm_finetuning/smoke/adapter",
        model="tinyllama:1.1b",
        base_model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        clearml_enabled=True,
    )

    assert result["finetuned_ollama_model"] == "finetuned-tinyllama_1.1b"
    assert calls[0]["clearml_enabled"] is True
    assert calls[0]["ollama_model"] == "tinyllama:1.1b"


@pytest.mark.docker
def test_docker_compose_config_is_valid_without_using_real_env(tmp_path: Path) -> None:
    """Docker Compose config should render using only mock env values."""
    if shutil.which("docker") is None:
        pytest.skip("Docker CLI is not installed.")
    version_result = subprocess.run(
        ["docker", "compose", "version"],
        cwd=PROJECT_ROOT,
        env=_safe_env(),
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if version_result.returncode != 0:
        pytest.skip(f"Docker Compose is not available: {version_result.stderr}")

    compose_file = tmp_path / "docker-compose.yml"
    shutil.copy(PROJECT_ROOT / "docker-compose.yml", compose_file)
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "POSTGRES_PASSWORD=mock-postgres-password",
                "CLEARML_API_ACCESS_KEY=mock-clearml-key",
                "CLEARML_API_SECRET_KEY=mock-clearml-secret",
                "OLLAMA_HOST_PORT=11434",
            ]
        )
        + "\n"
    )

    result = subprocess.run(
        [
            "docker",
            "compose",
            "--project-directory",
            str(tmp_path),
            "-f",
            str(compose_file),
            "config",
            "--quiet",
        ],
        cwd=tmp_path,
        env=_safe_env(),
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def _makefile_text() -> str:
    """Read the Makefile for static smoke-contract assertions.

    Returns:
        Makefile contents.
    """
    return (PROJECT_ROOT / "Makefile").read_text()


def _safe_env() -> dict[str, str]:
    """Build a minimal environment that avoids reading local secrets.

    Returns:
        Environment mapping for subprocess tests.
    """
    keys = ["PATH", "HOME", "LANG", "LC_ALL", "PYTHONPATH"]
    return {key: value for key in keys if (value := os.environ.get(key))}


def _reset_api_cache(api_app: Any) -> None:
    """Reset FastAPI predictor cache between tests.

    Args:
        api_app: Imported API app module.
    """
    api_app.reset_predictor_cache()


def _model_name(model_config: object) -> str:
    """Return the model name from an ADK/LiteLLM model config.

    Args:
        model_config: ADK model object or model string.

    Returns:
        Model name string.
    """
    return str(getattr(model_config, "model", model_config))


def _agent_has_prediction_tool(root_agent: Any) -> bool:
    """Check that the ADK agent exposes the product-category tool.

    Args:
        root_agent: Imported ADK root agent.

    Returns:
        Whether the tool is attached to the agent.
    """
    tools = getattr(root_agent, "tools", []) or []
    return any(
        getattr(tool, "__name__", getattr(getattr(tool, "func", None), "__name__", ""))
        == "predict_product_category"
        for tool in tools
    )


def _assert_instruction_matches_feature_encoding(instruction: str) -> None:
    """Assert the prompt matches API feature encodings.

    Args:
        instruction: Agent instruction text.
    """
    assert "encode female as 0 and male as 1" in instruction
    assert "encode yes as 1 and no as 0" in instruction
    assert "Once and only once all 8 fields are collected" in instruction
    assert "provide a friendly, natural response" in instruction
    assert "without mentioning any percentages" in instruction


def _features_from_completed_transcript() -> dict[str, int | float]:
    """Return features collected from a completed smoke conversation.

    Returns:
        Tool payload using the API/model feature encoding.
    """
    return {
        "Age": 34,
        "Gender": 1,
        "AnnualIncome": 82000.0,
        "NumberOfPurchases": 12,
        "TimeSpentOnWebsite": 47.5,
        "LoyaltyProgram": 1,
        "DiscountsAvailed": 2,
        "PurchaseStatus": 1,
    }


def _render_agent_prediction_answer(tool_result: dict[str, Any]) -> str:
    """Render the user-facing response expected from the agent contract.

    Args:
        tool_result: Prediction tool result.

    Returns:
        Natural language response.
    """
    label = tool_result["label"]
    return (
        f"Based on what you shared, I recommend exploring {label}. "
        "That category fits your shopping profile well."
    )


class _FakeHttpResponse:
    """Context-manager HTTP response for mocked urllib calls."""

    def __init__(self, payload: dict[str, Any]) -> None:
        """Initialize fake response.

        Args:
            payload: JSON response payload.
        """
        self._payload = payload

    def __enter__(self) -> "_FakeHttpResponse":
        """Enter context manager.

        Returns:
            This response object.
        """
        return self

    def __exit__(self, *_args: object) -> None:
        """Exit context manager."""

    def read(self) -> bytes:
        """Return encoded JSON payload.

        Returns:
            Response bytes.
        """
        return json.dumps(self._payload).encode("utf-8")


class _FakeClearMLLogger:
    """Collect ClearML logger calls for assertions."""

    def __init__(self) -> None:
        """Initialize call storage."""
        self.scalars: list[dict[str, Any]] = []
        self.tables: list[dict[str, Any]] = []
        self.texts: list[str] = []

    def report_scalar(
        self, title: str, series: str, value: float, iteration: int
    ) -> None:
        """Collect a scalar report call.

        Args:
            title: ClearML metric title.
            series: ClearML metric series.
            value: Metric value.
            iteration: Metric iteration.
        """
        self.scalars.append(
            {
                "title": title,
                "series": series,
                "value": value,
                "iteration": iteration,
            }
        )

    def report_table(
        self, title: str, series: str, iteration: int, table_plot: pd.DataFrame
    ) -> None:
        """Collect a table report call.

        Args:
            title: ClearML table title.
            series: ClearML table series.
            iteration: Table iteration.
            table_plot: Reported table.
        """
        self.tables.append(
            {
                "title": title,
                "series": series,
                "iteration": iteration,
                "table": table_plot,
            }
        )

    def report_text(self, message: str) -> None:
        """Collect a text report call.

        Args:
            message: Text message.
        """
        self.texts.append(message)
