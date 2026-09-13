"""Keep every evaluation test independent of local credentials and the network."""

import os
import socket
import sys
from pathlib import Path

import pytest

os.environ["WEAVE_DISABLED"] = "true"
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dataset as dataset_module
import model_config as model_config_module


@pytest.fixture(autouse=True)
def offline_environment(monkeypatch):
    monkeypatch.setattr(dataset_module, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        model_config_module, "load_dotenv", lambda *args, **kwargs: None
    )
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-openrouter-key")
    monkeypatch.setenv("WANDB_API_KEY", "offline-wandb-key")
    monkeypatch.setenv("WANDB_ENTITY", "offline-entity")
    for name in (
        "OPENROUTER_AGENT_MODEL",
        "OPENROUTER_JUDGE_MODEL",
        "FOUNDRY_ENDPOINT",
        "FOUNDRY_API_KEY",
        "FOUNDRY_AGENT_DEPLOYMENT",
        "FOUNDRY_JUDGE_DEPLOYMENT",
        "WANDB_PROJECT",
    ):
        monkeypatch.delenv(name, raising=False)

    connect = socket.socket.connect
    connect_ex = socket.socket.connect_ex

    def guard_network(method):
        def guarded(sock, address):
            # Windows asyncio uses a loopback socket pair for its event loop.
            if address[0] not in ("127.0.0.1", "::1"):
                raise AssertionError("Evaluation tests must not access external networks.")
            return method(sock, address)

        return guarded

    monkeypatch.setattr(socket.socket, "connect", guard_network(connect))
    monkeypatch.setattr(socket.socket, "connect_ex", guard_network(connect_ex))
