import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import publish_dataset as publish_module
import run_eval as run_eval_module
from model_config import ModelConfig


def evaluation_config(provider):
    return ModelConfig(
        provider=provider,
        agent_model="agent-deployment" if provider == "foundry" else "vendor/agent",
        judge_model="judge-deployment" if provider == "foundry" else "vendor/judge",
        endpoint=(
            "https://offline-resource.openai.azure.com/openai/v1/"
            if provider == "foundry"
            else "https://openrouter.ai/api/v1"
        ),
        api_key="offline-provider-secret",
    )


class FakeEvaluationLogger:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.predictions = []
        self.summary_logged = False
        self.failure = None
        self.ui_url = "https://example.invalid/offline-evaluation"

    @contextmanager
    def log_prediction(self, **kwargs):
        prediction = SimpleNamespace(
            evaluate_call=SimpleNamespace(id="offline-eval-id"),
            predict_and_score_call=SimpleNamespace(id="offline-prediction-id"),
        )
        self.predictions.append((kwargs, prediction))
        yield prediction

    def log_summary(self):
        self.summary_logged = True

    def fail(self, error):
        self.failure = error


def install_evaluation_fakes(monkeypatch, *, predict_error=None, scorer_error=None):
    observed = {"scorers": [], "predictions": []}

    def create_logger(**kwargs):
        observed["logger"] = FakeEvaluationLogger(**kwargs)
        return observed["logger"]

    async def predict(self, **kwargs):
        observed["predictions"].append(kwargs)
        if predict_error is not None:
            raise predict_error
        return {"slide_text": "Offline slides.", "tool_calls": []}

    async def apply_scorer(**kwargs):
        observed["scorers"].append(kwargs["scorer"])
        if scorer_error is not None:
            raise scorer_error

    monkeypatch.setattr(run_eval_module.weave, "EvaluationLogger", create_logger)
    monkeypatch.setattr(run_eval_module.SlideAgentModel, "predict", predict)
    monkeypatch.setattr(run_eval_module, "apply_and_log_scorer", apply_scorer)
    return observed


def offline_dataset():
    return SimpleNamespace(
        rows=[
            {
                "arxiv_id": "offline-paper",
                "paper_url": "https://example.invalid/offline-paper",
                "source_text": "Offline source.",
                "expected_tools": ["execute", "generate_pptx"],
            }
        ]
    )


@pytest.mark.parametrize("provider", ["openrouter", "foundry"])
async def test_evaluation_uses_one_config_for_model_metadata_and_all_judges(
    monkeypatch, provider
):
    observed = install_evaluation_fakes(monkeypatch)
    config = evaluation_config(provider)
    dataset = offline_dataset()
    url, errors = await run_eval_module.run_evaluation(
        variant="baseline", dataset=dataset, config=config
    )
    logger = observed["logger"]
    model = logger.kwargs["model"]
    for name, value in config.metadata().items():
        assert getattr(model, name) == value
    assert config.api_key not in model.model_dump_json()
    assert logger.kwargs["dataset"] is dataset
    assert logger.kwargs["scorers"] == [
        "tool_correctness", "summarization", "hallucination_free", "SlideQualityScorer"
    ]
    assert logger.summary_logged is True
    assert logger.failure is None
    assert url == logger.ui_url
    assert errors == 0
    assert len(observed["predictions"]) == 1
    context = observed["predictions"][0]["eval_context"]
    assert context["weave.eval.run_id"] == "offline-eval-id"
    assert context["weave.eval.evaluation_name"] == "baseline"
    assert len(observed["scorers"]) == 4
    for scorer in observed["scorers"][1:]:
        assert scorer.provider == config.provider
        assert scorer.endpoint == config.endpoint
        assert scorer.judge_model == config.judge_model_id


@pytest.mark.parametrize("error_source", ["prediction", "scorer"])
async def test_evaluation_counts_errors_without_printing_provider_credentials(
    monkeypatch, capsys, error_source
):
    config = evaluation_config("foundry")
    error = RuntimeError(f"Provider error including {config.api_key}")
    observed = install_evaluation_fakes(
        monkeypatch,
        predict_error=error if error_source == "prediction" else None,
        scorer_error=error if error_source == "scorer" else None,
    )
    _, errors = await run_eval_module.run_evaluation(
        variant="baseline", dataset=offline_dataset(), config=config
    )
    assert errors == (1 if error_source == "prediction" else 4)
    assert observed["logger"].summary_logged is True
    output = capsys.readouterr().out
    assert "RuntimeError" in output
    assert "Provider error including [redacted]" in output
    assert config.api_key not in output


def test_publish_dataset_only_requires_wandb_settings(monkeypatch, capsys):
    monkeypatch.setenv("LLM_PROVIDER", "unsupported-provider")
    monkeypatch.delenv("OPENROUTER_API_KEY")
    observed = {}

    def init(project):
        observed["project"] = project

    def publish(dataset):
        observed["dataset"] = dataset
        return SimpleNamespace(uri=lambda: "weave:///offline/dataset", digest="offline")

    rows = [{"arxiv_id": "offline-paper", "source_text": "Offline source."}]
    monkeypatch.setattr(publish_module.weave, "init", init)
    monkeypatch.setattr(publish_module.weave, "publish", publish)
    monkeypatch.setattr(publish_module, "build_dataset_rows", lambda: rows)
    publish_module.main()
    assert observed["project"] == f"offline-entity/{publish_module.DATASET_NAME}"
    assert observed["dataset"].rows[0]["arxiv_id"] == "offline-paper"
    assert "offline-wandb-key" not in capsys.readouterr().out


def test_eval_cli_validates_provider_before_any_weave_connection(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "foundry")
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "baseline"])

    def reject_connection(*args, **kwargs):
        raise AssertionError("Invalid model configuration must fail before Weave init.")

    monkeypatch.setattr(run_eval_module.weave, "init", reject_connection)
    with pytest.raises(SystemExit, match="FOUNDRY_ENDPOINT"):
        run_eval_module.main()


def test_eval_cli_passes_dotenv_resolved_config_and_reports_model_identity(
    monkeypatch, capsys
):
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "baseline"])
    monkeypatch.setenv("OPENROUTER_AGENT_MODEL", "vendor/agent-override")
    monkeypatch.setenv("OPENROUTER_JUDGE_MODEL", "vendor/judge-override")
    observed = {}
    dataset = SimpleNamespace(ref=None)

    async def run_evaluation(**kwargs):
        observed.update(kwargs)
        return "https://example.invalid/offline-evaluation", 0

    monkeypatch.setattr(
        run_eval_module.weave, "init",
        lambda project: SimpleNamespace(entity="offline-entity", project="offline-project"),
    )
    monkeypatch.setattr(
        run_eval_module.weave, "ref",
        lambda uri: SimpleNamespace(get=lambda: dataset),
    )
    monkeypatch.setattr(run_eval_module, "run_evaluation", run_evaluation)
    run_eval_module.main()
    assert observed["dataset"] is dataset
    assert observed["config"].agent_model == "vendor/agent-override"
    assert observed["config"].judge_model == "vendor/judge-override"
    output = capsys.readouterr().out
    assert "provider=openrouter" in output
    assert "agent=vendor/agent-override" in output
    assert "judge=vendor/judge-override" in output
    assert observed["config"].api_key not in output
