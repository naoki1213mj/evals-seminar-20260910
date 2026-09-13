"""SlideAgentModelとTypeScriptエージェントとのsubprocess境界。

Weave EvaluationLoggerのDataset行ごとに`agent-run/eval.ts`を1回起動し、
run workspaceへ書かれたevaluation-result.jsonを読み取ってModel出力にする。
プロセスの起動失敗・タイムアウト・JSONプロトコル違反はインフラエラーとして
例外にし、Weave上でもその行の実行をerrorにする。
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import weave
from model_config import ModelConfig, load_model_config
from pydantic import PrivateAttr

ROOT = Path(__file__).resolve().parents[1]
TSX_BIN = ROOT / "node_modules" / ".bin" / "tsx"
EVAL_ENTRYPOINT = ROOT / "agent-run" / "eval.ts"
WORKSPACES_DIR = ROOT / "tmp" / "workspaces"
AGENT_TIMEOUT_SECONDS = 900
REQUIRED_RESULT_KEYS = [
    "generation_success",
    "slides",
    "slide_text",
    "tool_calls",
    "final_text",
    "duration_ms",
    "conversation_id",
]


async def run_agent_process(
    *,
    variant: str,
    arxiv_id: str,
    paper_url: str,
    eval_context: dict[str, Any],
    model_config: ModelConfig | None = None,
) -> dict:
    """TypeScriptエージェントを評価試行内で1回実行し、結果ファイルを読む。"""
    run_id = uuid4().hex
    context = f"variant={variant}, arxiv_id={arxiv_id}, run_id={run_id}"

    process = await asyncio.create_subprocess_exec(
        str(TSX_BIN),
        str(EVAL_ENTRYPOINT),
        "--arxiv-id",
        arxiv_id,
        "--paper-url",
        paper_url,
        "--variant",
        variant,
        "--run-id",
        run_id,
        "--eval-context",
        json.dumps(eval_context, ensure_ascii=False),
        cwd=ROOT,
        env=model_config.agent_environment() if model_config is not None else None,
    )
    try:
        returncode = await asyncio.wait_for(
            process.wait(), timeout=AGENT_TIMEOUT_SECONDS
        )
    except TimeoutError:
        process.kill()
        await process.wait()
        raise RuntimeError(
            f"エージェント実行が{AGENT_TIMEOUT_SECONDS}秒でタイムアウトしました ({context})"
        ) from None

    if returncode != 0:
        raise RuntimeError(
            f"エージェントプロセスがexit code {returncode}で終了しました ({context})"
        )

    matches = sorted(WORKSPACES_DIR.glob(f"*-{variant}-{run_id}"))
    if len(matches) != 1:
        raise RuntimeError(
            f"run workspaceを一意に特定できません: {len(matches)}件 ({context})"
        )

    result_file = matches[0] / "evaluation-result.json"
    if not result_file.exists():
        raise RuntimeError(f"evaluation-result.jsonがありません: {result_file}")

    result = json.loads(result_file.read_text())
    missing = [key for key in REQUIRED_RESULT_KEYS if key not in result]
    if missing:
        raise RuntimeError(
            f"evaluation-result.jsonに必須キーがありません: {missing} ({context})"
        )
    return result


class SlideAgentModel(weave.Model):
    variant: str
    provider: Literal["openrouter", "foundry"]
    agent_model: str
    judge_model: str
    endpoint: str
    _runtime_config: ModelConfig | None = PrivateAttr(default=None)

    @classmethod
    def configured(cls, *, variant: str, config: ModelConfig) -> "SlideAgentModel":
        model = cls(variant=variant, **config.metadata())
        model._runtime_config = config
        return model

    @weave.op()
    async def predict(
        self,
        arxiv_id: str,
        paper_url: str,
        eval_context: dict[str, Any],
    ) -> dict:
        config = self._runtime_config or load_model_config()
        if any(
            getattr(self, name) != value for name, value in config.metadata().items()
        ):
            raise ValueError(
                "Runtime provider/model settings do not match this Weave Model."
            )
        return await run_agent_process(
            variant=self.variant,
            arxiv_id=arxiv_id,
            paper_url=paper_url,
            eval_context=eval_context,
            model_config=config,
        )
