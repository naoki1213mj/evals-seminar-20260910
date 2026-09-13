import json

import pytest

import dataset as dataset_module
import model_config as model_config_module
from model_config import (
    DEFAULT_AGENT_MODEL,
    DEFAULT_JUDGE_MODEL,
    OPENROUTER_ENDPOINT,
    model_config_from_env,
)

FOUNDRY_ENV = {
    "LLM_PROVIDER": "foundry",
    "FOUNDRY_ENDPOINT": "https://test-resource.openai.azure.com",
    "FOUNDRY_API_KEY": "offline-foundry-key",
    "FOUNDRY_AGENT_DEPLOYMENT": "agent-deployment",
    "FOUNDRY_JUDGE_DEPLOYMENT": "judge-deployment",
}


@pytest.mark.parametrize("provider", [None, "", " \t ", "openrouter", " OpenRouter "])
def test_openrouter_defaults(provider):
    environ = {"OPENROUTER_API_KEY": " offline-openrouter-key "}
    if provider is not None:
        environ["LLM_PROVIDER"] = provider
    config = model_config_from_env(environ)
    assert config.provider == "openrouter"
    assert config.agent_model == DEFAULT_AGENT_MODEL
    assert config.judge_model == DEFAULT_JUDGE_MODEL
    assert config.judge_model_id == f"openrouter/{DEFAULT_JUDGE_MODEL}"
    assert config.endpoint == OPENROUTER_ENDPOINT
    assert config.api_key == "offline-openrouter-key"


def test_openrouter_model_overrides_are_raw_model_ids():
    config = model_config_from_env(
        {
            "OPENROUTER_API_KEY": "offline-key",
            "OPENROUTER_AGENT_MODEL": " vendor/agent-model ",
            "OPENROUTER_JUDGE_MODEL": " vendor/judge-model ",
        }
    )
    assert config.agent_model == "vendor/agent-model"
    assert config.judge_model == "vendor/judge-model"
    assert config.judge_model_id == "openrouter/vendor/judge-model"


def test_blank_openrouter_model_overrides_use_defaults():
    config = model_config_from_env(
        {
            "OPENROUTER_API_KEY": "offline-key",
            "OPENROUTER_AGENT_MODEL": " ",
            "OPENROUTER_JUDGE_MODEL": "",
        }
    )
    assert config.agent_model == DEFAULT_AGENT_MODEL
    assert config.judge_model == DEFAULT_JUDGE_MODEL


@pytest.mark.parametrize("provider", ["azure", "openai", "bad-provider-value"])
def test_unknown_provider_fails_without_echoing_value(provider):
    with pytest.raises(ValueError, match="LLM_PROVIDER") as error:
        model_config_from_env({"LLM_PROVIDER": provider})
    assert provider not in str(error.value)


@pytest.mark.parametrize("provider", ["foundry", " Foundry ", "FOUNDRY"])
@pytest.mark.parametrize("suffix", ["", "/", "/openai/v1", "/openai/v1/"])
@pytest.mark.parametrize("domain", ["openai.azure.com", "services.ai.azure.com"])
def test_foundry_resource_and_v1_urls_are_normalized(provider, suffix, domain):
    config = model_config_from_env(
        {
            **FOUNDRY_ENV,
            "LLM_PROVIDER": provider,
            "FOUNDRY_ENDPOINT": f" https://test-resource.{domain}{suffix} ",
        }
    )
    assert config.provider == "foundry"
    assert config.endpoint == f"https://test-resource.{domain}/openai/v1/"
    assert config.agent_model == "agent-deployment"
    assert config.judge_model == "judge-deployment"
    assert config.judge_model_id == "openai/judge-deployment"
    assert config.api_key == "offline-foundry-key"


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://invalid-resource.example",
        "ftp://invalid-resource.example",
        "invalid-resource.example",
        "https://",
        "https://invalid-resource.example/api/projects/example-project",
        "https://invalid-resource.example/models",
        "https://invalid-resource.example/openai/deployments/judge",
        "https://invalid-resource.example/openai/v1/chat/completions",
        "https://invalid-resource.example/openai/v1//",
        "https://invalid-resource.example/?api-key=offline-secret",
        "https://invalid-resource.example?",
        "https://invalid-resource.example/#offline-secret",
        "https://invalid-resource.example#",
        "https://offline-user:offline-password@invalid-resource.example",
        "https://@invalid-resource.example",
        "https://invalid-resource.example\\openai\\v1",
        "https://invalid-\nresource.example",
        "https://invalid-\x00resource.example",
        "https://invalid-\x7fresource.example",
        "https://invalid-resource.example/other/../",
        "https://invalid-resource.example:bad-port",
        "https://invalid-resource.example:99999",
        "https://[invalid-ipv6",
    ],
)
def test_foundry_rejects_invalid_endpoints_without_echoing_them(endpoint):
    with pytest.raises(ValueError, match="FOUNDRY_ENDPOINT") as error:
        model_config_from_env({**FOUNDRY_ENV, "FOUNDRY_ENDPOINT": endpoint})
    assert endpoint not in str(error.value)
    assert "offline-secret" not in str(error.value)
    assert "offline-password" not in str(error.value)


@pytest.mark.parametrize(
    "name",
    [
        "FOUNDRY_ENDPOINT",
        "FOUNDRY_API_KEY",
        "FOUNDRY_AGENT_DEPLOYMENT",
        "FOUNDRY_JUDGE_DEPLOYMENT",
        "OPENROUTER_API_KEY",
    ],
)
@pytest.mark.parametrize("value", [None, "", " \t ", "your_value", "YOUR_VALUE", "<key>", "a>b"])
def test_required_provider_settings_reject_missing_and_placeholder_values(name, value):
    environ = (
        {"LLM_PROVIDER": "openrouter"} if name == "OPENROUTER_API_KEY" else dict(FOUNDRY_ENV)
    )
    if value is None:
        environ.pop(name, None)
    else:
        environ[name] = value
    with pytest.raises(ValueError, match=name):
        model_config_from_env(environ)


@pytest.mark.parametrize("active", ["openrouter", "foundry"])
def test_inactive_provider_settings_are_not_read(active):
    inactive_prefix = "FOUNDRY_" if active == "openrouter" else "OPENROUTER_"

    class SelectedProviderOnly(dict):
        def get(self, key, default=None):
            assert not key.startswith(inactive_prefix)
            return super().get(key, default)

    environ = SelectedProviderOnly(
        FOUNDRY_ENV
        if active == "foundry"
        else {"OPENROUTER_API_KEY": "offline-openrouter-key"}
    )
    assert model_config_from_env(environ).provider == active


def test_model_config_repr_and_metadata_do_not_include_credentials():
    config = model_config_from_env(FOUNDRY_ENV)
    assert config.api_key not in repr(config)
    assert config.api_key not in json.dumps(config.metadata())
    assert config.metadata() == {
        "provider": "foundry",
        "agent_model": "agent-deployment",
        "judge_model": "judge-deployment",
        "endpoint": "https://test-resource.openai.azure.com/openai/v1/",
    }


def test_model_config_loads_dotenv_before_reading_any_settings(monkeypatch):
    calls = []

    def fake_load_dotenv(path, *, override):
        calls.append((path, override))
        for name, value in FOUNDRY_ENV.items():
            monkeypatch.setenv(name, value)

    monkeypatch.setattr(model_config_module, "load_dotenv", fake_load_dotenv)
    config = model_config_module.load_model_config()
    assert calls == [(model_config_module.ROOT / ".env", True)]
    assert config.provider == "foundry"
    assert config.judge_model == "judge-deployment"


@pytest.mark.parametrize("provider", ["foundry", "openrouter", "unsupported-provider"])
def test_dataset_only_settings_do_not_validate_model_provider(monkeypatch, provider):
    monkeypatch.setenv("LLM_PROVIDER", provider)
    monkeypatch.delenv("OPENROUTER_API_KEY")

    def reject_provider_validation(*args, **kwargs):
        raise AssertionError("Dataset publishing must not resolve model settings.")

    monkeypatch.setattr(
        dataset_module, "model_config_from_env", reject_provider_validation
    )
    settings = dataset_module.load_settings(require_model=False)
    assert settings.weave_project == f"offline-entity/{dataset_module.DATASET_NAME}"
    assert settings.model_config is None
    assert "offline-wandb-key" not in repr(settings)


def test_eval_settings_load_dotenv_before_provider_settings(monkeypatch):
    def fake_load_dotenv(*args, **kwargs):
        for name, value in FOUNDRY_ENV.items():
            monkeypatch.setenv(name, value)

    monkeypatch.setattr(dataset_module, "load_dotenv", fake_load_dotenv)
    settings = dataset_module.load_settings(require_model=True)
    assert settings.model_config.provider == "foundry"
    assert settings.model_config.agent_model == "agent-deployment"


def test_eval_settings_report_missing_provider_configuration(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "foundry")
    with pytest.raises(SystemExit, match="FOUNDRY_ENDPOINT"):
        dataset_module.load_settings(require_model=True)
