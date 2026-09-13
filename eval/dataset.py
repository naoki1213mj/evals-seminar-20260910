"""Weave Dataset行の組み立てと評価共通の設定。

行データ(source_textを含む)はリポジトリで固定した論文ID・取得ロジックから
組み立てる。source_text自体はコミットせず、publish時にar5ivから取得して
Dataset versionへ保存することで過去評価の再現性を担保する。
"""

import html as html_lib
import os
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from model_config import ModelConfig, is_placeholder, model_config_from_env

ROOT = Path(__file__).resolve().parents[1]

DATASET_NAME = "evals-seminar-20260910"
PAPER_IDS = ["1706.03762", "2512.07828", "2603.03303"]
VARIANTS = ["baseline", "improvement-1", "improvement-2", "improvement-3"]
EXPECTED_TOOLS = ["execute", "generate_pptx"]
MAX_SOURCE_CHARS = 120_000


@dataclass(frozen=True)
class Settings:
    wandb_api_key: str = field(repr=False)
    weave_project: str
    model_config: ModelConfig | None = field(default=None, repr=False)


def load_settings(*, require_model: bool) -> Settings:
    """`.env`を読み込み、評価に必要な環境変数を検証する。"""
    load_dotenv(ROOT / ".env", override=True)
    wandb_api_key = os.environ.get("WANDB_API_KEY", "").strip()
    wandb_entity = os.environ.get("WANDB_ENTITY", "").strip()
    wandb_project = os.environ.get("WANDB_PROJECT", "").strip()
    if wandb_entity and not is_placeholder(wandb_entity):
        weave_project = f"{wandb_entity}/{wandb_project or DATASET_NAME}"
    else:
        weave_project = ""
    missing = []
    if not wandb_api_key or is_placeholder(wandb_api_key):
        missing.append("WANDB_API_KEY")
    if not weave_project:
        missing.append("WANDB_ENTITY（必要ならWANDB_PROJECTも設定）")
    if missing:
        raise SystemExit(
            "次の環境変数を.envへ設定してください: " + ", ".join(missing)
        )

    try:
        model_config = model_config_from_env(os.environ) if require_model else None
    except ValueError as error:
        raise SystemExit(str(error)) from None

    return Settings(
        wandb_api_key=wandb_api_key,
        weave_project=weave_project,
        model_config=model_config,
    )


def fetch_paper_text(arxiv_id: str) -> str:
    """ar5ivから論文本文をテキスト化して取得する"""
    url = f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (sd36-eval)"})
    html = urllib.request.urlopen(req, timeout=90).read().decode("utf-8", "ignore")
    # 数式はalttext(LaTeX)に置き換えて残す
    html = re.sub(
        r'(?is)<math[^>]*?alttext="([^"]*)"[^>]*>.*?</math>',
        lambda m: f" {html_lib.unescape(m.group(1))} ",
        html,
    )
    html = re.sub(r"(?is)<(script|style|math|nav|header|footer).*?</\1>", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    text = re.sub(r"&[a-zA-Z#0-9]+;", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # 参考文献以降は本文後半に出現した場合のみ打ち切る
    refs = list(re.finditer(r"\bReferences\b", text))
    if refs and refs[-1].start() > len(text) * 0.5:
        text = text[: refs[-1].start()]
    return text[:MAX_SOURCE_CHARS]


def build_dataset_rows() -> list[dict]:
    """3論文の評価行を組み立てる。source_textはar5ivからライブ取得する。"""
    return [
        {
            "arxiv_id": paper_id,
            "paper_url": f"https://arxiv.org/abs/{paper_id}",
            "source_text": fetch_paper_text(paper_id),
            "expected_tools": list(EXPECTED_TOOLS),
        }
        for paper_id in PAPER_IDS
    ]
