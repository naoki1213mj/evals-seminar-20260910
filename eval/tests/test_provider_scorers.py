import asyncio
import json
import os
from types import SimpleNamespace

import httpx
import httpx2
import pytest
from weave.flow.scorer import get_scorer_attributes
from weave.trace.object_record import pydantic_object_record

import scorers as scorers_module
from agent_model import SlideAgentModel
from model_config import model_config_from_env
from scorers import build_scorers

OUTPUT = {
    "slide_text": "Attention is the main contribution.",
    "tool_calls": [{"name": "execute"}, {"name": "generate_pptx"}],
}
SOURCE_TEXT = "The paper proposes attention."


def provider_config(provider, *, key=None, judge=None):
    if provider == "foundry":
        return model_config_from_env(
            {
                "LLM_PROVIDER": provider,
                "FOUNDRY_ENDPOINT": "https://offline-resource.services.ai.azure.com",
                "FOUNDRY_API_KEY": key or "offline-foundry-key",
                "FOUNDRY_AGENT_DEPLOYMENT": "agent-deployment",
                "FOUNDRY_JUDGE_DEPLOYMENT": judge or "judge-deployment",
            }
        )
    return model_config_from_env(
        {
            "LLM_PROVIDER": provider,
            "OPENROUTER_API_KEY": key or "offline-openrouter-key",
            "OPENROUTER_AGENT_MODEL": "vendor/agent",
            "OPENROUTER_JUDGE_MODEL": judge or "vendor/judge",
        }
    )


def verdict_for_format(response_format):
    name = (
        response_format.__name__
        if isinstance(response_format, type)
        else response_format.get("json_schema", {}).get("name")
    )
    if name == "EntityExtractionResponse":
        return {"entities": ["attention"]}
    if name == "SummarizationEvaluationResponse":
        return {
            "think_step_by_step": "The main contribution is preserved.",
            "summarization_evaluation": "excellent",
        }
    if name == "HallucinationResponse":
        return {
            "chain_of_thought": "The claim is supported.",
            "reasonings": [],
            "conclusion": "No hallucinations.",
            "has_hallucination": False,
        }
    assert response_format == {"type": "json_object"}
    return {"score": 8, "reason": "The main message is clear."}


def completion_response(verdict):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=json.dumps(verdict))
            )
        ]
    )


@pytest.mark.parametrize("provider", ["openrouter", "foundry"])
async def test_all_judges_use_selected_request_config_and_real_preset_semantics(
    monkeypatch, provider
):
    config = provider_config(provider)
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return completion_response(verdict_for_format(kwargs["response_format"]))

    monkeypatch.setattr(scorers_module.litellm, "acompletion", fake_acompletion)
    scorers = build_scorers(config)
    assert [get_scorer_attributes(s).scorer_name for s in scorers] == [
        "tool_correctness",
        "summarization",
        "hallucination_free",
        "SlideQualityScorer",
    ]
    summary = await scorers[1].score(output=OUTPUT, source_text=SOURCE_TEXT)
    hallucination = await scorers[2].score(output=OUTPUT, source_text=SOURCE_TEXT)
    quality = await scorers[3].score(output=OUTPUT, source_text=SOURCE_TEXT)

    assert summary == {
        "summarization_eval_score": 1.0,
        "llm_eval_reasoning": "The main contribution is preserved.",
        "is_entity_dense": True,
        "entity_density": 0.2,
    }
    assert hallucination["has_hallucination"] is False
    assert hallucination["conclusion"] == "No hallucinations."
    assert quality["score"] == 0.8
    assert quality["passed"] is True
    assert len(calls) == 4
    for call in calls:
        assert call["model"] == config.judge_model_id
        assert call["api_base"] == config.endpoint
        assert call["api_key"] == config.api_key
        assert call["num_retries"] == 3
    if provider == "foundry":
        assert all("temperature" not in call for call in calls)
        assert all("max_tokens" not in call for call in calls)
        assert sorted(call["max_completion_tokens"] for call in calls[:3]) == [
            1024, 1024, 4096
        ]
    else:
        assert [call["temperature"] for call in calls] == [0.7, 0.7, 0.7, 0]
        assert sorted(call["max_tokens"] for call in calls[:3]) == [1024, 1024, 4096]
    assert calls[-1]["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize(
    ("provider", "judge"),
    [
        ("openrouter", "openai/gpt-5.4"),
        ("foundry", "judge-deployment"),
        ("foundry", "gpt-5.4"),
        ("foundry", "o3"),
    ],
)
async def test_real_litellm_and_sdk_send_v1_requests_to_the_selected_provider(
    monkeypatch, provider, judge
):
    config = provider_config(provider, judge=judge)
    requests = []

    def respond(request, response_type):
        assert str(request.url) == f"{config.endpoint.rstrip('/')}/chat/completions"
        assert request.headers["authorization"] == f"Bearer {config.api_key}"
        body = json.loads(request.content)
        assert body["model"] == config.judge_model
        requests.append(body)
        verdict = verdict_for_format(body["response_format"])
        return response_type(
            200,
            request=request,
            json={
                "id": "offline-chat-completion",
                "object": "chat.completion",
                "created": 1,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(verdict),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 10,
                    "total_tokens": 30,
                },
            },
        )

    async def fake_send(self, request, **kwargs):
        return respond(request, httpx.Response)

    async def fake_send_v2(self, request, **kwargs):
        return respond(request, httpx2.Response)

    monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)
    monkeypatch.setattr(httpx2.AsyncClient, "send", fake_send_v2)
    monkeypatch.setattr(scorers_module.litellm, "telemetry", False)
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-openai-key")
    monkeypatch.setenv("OPENAI_API_BASE", "https://unrelated-openai.example/v1")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://unrelated-openai.example/v1")
    monkeypatch.setenv("AZURE_API_KEY", "unrelated-azure-key")
    monkeypatch.setenv("AZURE_API_BASE", "https://unrelated-azure.example")

    for scorer in build_scorers(config)[1:]:
        await scorer.score(output=OUTPUT, source_text=SOURCE_TEXT)
    assert len(requests) == 4
    assert [request["response_format"]["type"] for request in requests] == [
        "json_schema", "json_schema", "json_schema", "json_object"
    ]
    if provider == "foundry":
        assert all("temperature" not in request for request in requests)
        assert all("max_tokens" not in request for request in requests)
        assert sorted(request["max_completion_tokens"] for request in requests[:3]) == [
            1024, 1024, 4096
        ]


@pytest.mark.parametrize("provider", ["openrouter", "foundry"])
@pytest.mark.parametrize("scorer_index", [1, 2, 3])
async def test_every_judge_propagates_api_errors(monkeypatch, provider, scorer_index):
    failure = RuntimeError("Offline judge API failure.")

    async def reject_completion(**kwargs):
        raise failure

    monkeypatch.setattr(scorers_module.litellm, "acompletion", reject_completion)
    scorer = build_scorers(provider_config(provider))[scorer_index]
    with pytest.raises(RuntimeError) as error:
        await scorer.score(output=OUTPUT, source_text=SOURCE_TEXT)
    assert error.value is failure


@pytest.mark.parametrize("provider", ["openrouter", "foundry"])
async def test_every_judge_skips_missing_slide_text(monkeypatch, provider):
    async def reject_completion(**kwargs):
        raise AssertionError("Missing slides must not call a judge.")

    monkeypatch.setattr(scorers_module.litellm, "acompletion", reject_completion)
    for scorer in build_scorers(provider_config(provider))[1:]:
        result = await scorer.score(
            output={"slide_text": "", "failure_reason": "No slides."},
            source_text=SOURCE_TEXT,
        )
        assert result["passed"] is False
        assert result["failure_reason"] == "No slides."


async def test_provider_switching_keeps_clients_independent_without_global_changes(
    monkeypatch,
):
    calls = []
    unrelated = {
        "OPENAI_API_KEY": "unrelated-openai-key",
        "OPENAI_BASE_URL": "https://unrelated.example",
        "AZURE_API_KEY": "unrelated-azure-key",
        "AZURE_API_BASE": "https://unrelated-azure.example",
    }
    for name, value in unrelated.items():
        monkeypatch.setenv(name, value)
    original_retries = scorers_module.litellm.num_retries

    async def fake_acompletion(**kwargs):
        calls.append((kwargs["api_base"], kwargs["api_key"], kwargs["model"]))
        return completion_response({"score": 7, "reason": "Clear."})

    monkeypatch.setattr(scorers_module.litellm, "acompletion", fake_acompletion)
    configs = [
        provider_config("openrouter"),
        provider_config("foundry"),
        provider_config("openrouter", key="new-openrouter-key", judge="vendor/new-judge"),
    ]
    scorers = [build_scorers(config)[3] for config in configs]
    await asyncio.gather(
        *(scorer.score(output=OUTPUT, source_text=SOURCE_TEXT) for scorer in scorers)
    )
    assert set(calls) == {
        (config.endpoint, config.api_key, config.judge_model_id) for config in configs
    }
    assert {name: os.environ[name] for name in unrelated} == unrelated
    assert scorers_module.litellm.num_retries == original_retries


@pytest.mark.parametrize("provider", ["openrouter", "foundry"])
def test_weave_model_and_scorer_serialization_excludes_credentials(provider):
    config = provider_config(provider)
    model = SlideAgentModel.configured(variant="baseline", config=config)
    objects = [model]
    for scorer in build_scorers(config)[1:]:
        objects.append(scorer)
        assert scorer.provider == config.provider
        assert scorer.judge_model == config.judge_model_id
        assert scorer.endpoint == config.endpoint
        if hasattr(scorer, "_preset"):
            objects.append(scorer._preset)
    for obj in objects:
        assert config.api_key not in obj.model_dump_json()
        assert config.api_key not in repr(obj)
        record = pydantic_object_record(obj).__dict__
        assert config.api_key not in json.dumps(record, default=str)
        assert "api_key" not in record
        assert "_completion" not in record
        assert "_preset" not in record
        assert "_runtime_config" not in record
    assert model.agent_model == config.agent_model
    assert model.judge_model == config.judge_model


def test_agent_and_judge_deployments_distinguish_weave_models():
    foundry = provider_config("foundry")
    other_judge = provider_config("foundry", judge="another-judge-deployment")
    openrouter = provider_config("openrouter")
    serialized = {
        SlideAgentModel(variant="baseline", **config.metadata()).model_dump_json()
        for config in (foundry, other_judge, openrouter)
    }
    assert len(serialized) == 3
