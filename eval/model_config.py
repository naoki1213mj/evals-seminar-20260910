"""OpenRouter / Foundryの設定を、利用するproviderについてだけ解決する。"""

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AGENT_MODEL = "deepseek/deepseek-v4-flash"
DEFAULT_JUDGE_MODEL = "openai/gpt-5.4"
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1"


def is_placeholder(value: str) -> bool:
    return value.lower().startswith("your_") or "<" in value or ">" in value


def required_setting(environ: Mapping[str, str], name: str) -> str:
    value = environ.get(name, "").strip()
    if not value or is_placeholder(value):
        raise ValueError(f"{name} must be set to a non-placeholder value.")
    return value


def normalize_foundry_endpoint(value: str) -> str:
    message = (
        "FOUNDRY_ENDPOINT must be an HTTPS resource root or /openai/v1/ URL "
        "without credentials, query, or fragment; project endpoints are not supported."
    )
    try:
        if (
            not value
            or is_placeholder(value)
            or "\\" in value
            or "?" in value
            or "#" in value
            or any(
                character.isspace() or ord(character) < 32 or ord(character) == 127
                for character in value
            )
        ):
            raise ValueError
        parsed = urlsplit(value)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in ("", "/", "/openai/v1", "/openai/v1/")
        ):
            raise ValueError
        host = parsed.hostname
        if ":" in host:
            host = f"[{host}]"
        else:
            host = host.encode("idna").decode("ascii")
        port = parsed.port
        authority = host if port in (None, 443) else f"{host}:{port}"
        return urlunsplit(("https", authority, "/openai/v1/", "", ""))
    except (ValueError, UnicodeError):
        raise ValueError(message) from None


@dataclass(frozen=True)
class ModelConfig:
    provider: Literal["openrouter", "foundry"]
    agent_model: str
    judge_model: str
    endpoint: str
    api_key: str = field(repr=False)

    @property
    def judge_model_id(self) -> str:
        prefix = "openrouter" if self.provider == "openrouter" else "openai"
        return f"{prefix}/{self.judge_model}"

    def metadata(self) -> dict[str, str]:
        return {
            "provider": self.provider,
            "agent_model": self.agent_model,
            "judge_model": self.judge_model,
            "endpoint": self.endpoint,
        }

    def agent_environment(self) -> dict[str, str]:
        settings = {"LLM_PROVIDER": self.provider}
        if self.provider == "foundry":
            settings.update(
                FOUNDRY_ENDPOINT=self.endpoint,
                FOUNDRY_API_KEY=self.api_key,
                FOUNDRY_AGENT_DEPLOYMENT=self.agent_model,
                FOUNDRY_JUDGE_DEPLOYMENT=self.judge_model,
            )
        else:
            settings.update(
                OPENROUTER_API_KEY=self.api_key,
                OPENROUTER_AGENT_MODEL=self.agent_model,
                OPENROUTER_JUDGE_MODEL=self.judge_model,
            )
        return {**os.environ, **settings}


def model_config_from_env(environ: Mapping[str, str]) -> ModelConfig:
    provider = environ.get("LLM_PROVIDER", "").strip().lower() or "openrouter"
    if provider == "openrouter":
        api_key = required_setting(environ, "OPENROUTER_API_KEY")
        models = {
            "OPENROUTER_AGENT_MODEL": DEFAULT_AGENT_MODEL,
            "OPENROUTER_JUDGE_MODEL": DEFAULT_JUDGE_MODEL,
        }
        resolved = {}
        for name, default in models.items():
            value = environ.get(name, "").strip() or default
            resolved[name] = value
        return ModelConfig(
            provider="openrouter",
            agent_model=resolved["OPENROUTER_AGENT_MODEL"],
            judge_model=resolved["OPENROUTER_JUDGE_MODEL"],
            endpoint=OPENROUTER_ENDPOINT,
            api_key=api_key,
        )
    if provider == "foundry":
        return ModelConfig(
            provider="foundry",
            endpoint=normalize_foundry_endpoint(
                required_setting(environ, "FOUNDRY_ENDPOINT")
            ),
            api_key=required_setting(environ, "FOUNDRY_API_KEY"),
            agent_model=required_setting(environ, "FOUNDRY_AGENT_DEPLOYMENT"),
            judge_model=required_setting(environ, "FOUNDRY_JUDGE_DEPLOYMENT"),
        )
    raise ValueError("LLM_PROVIDER must be openrouter or foundry.")


def load_model_config() -> ModelConfig:
    load_dotenv(ROOT / ".env", override=True)
    return model_config_from_env(os.environ)
