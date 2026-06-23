"""Shared LLM client helpers for langgraph-study.

This follows the same OpenAI-compatible calling style used in:
D:\dev\Python\AI\langchain-study
"""

from __future__ import annotations

import os

from langchain_openai import ChatOpenAI

BASE_URL = "https://codex.ysaikeji.cn/v1"
API_KEY_ENV = "K_CODEX"
DEFAULT_MODEL = os.environ.get("K_CODEX_MODEL", "gpt-5.4")


def get_chat_model(*, temperature: float = 0) -> ChatOpenAI:
    """Return a LangChain chat model using the shared OpenAI-compatible endpoint."""
    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        raise RuntimeError(
            f"请先设置环境变量 {API_KEY_ENV}，例如："
            f"$env:{API_KEY_ENV} = \"你的 API Key\""
        )

    return ChatOpenAI(
        model=DEFAULT_MODEL,
        temperature=temperature,
        api_key=api_key,
        base_url=BASE_URL,
    )
