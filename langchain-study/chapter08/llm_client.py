import os
from collections.abc import Iterable
from typing import Any

from openai import OpenAI


BASE_URL = "https://codex.ysaikeji.cn/v1"
WIRE_API = "responses"
API_KEY_ENV = "K_CODEX"
DEFAULT_MODEL = os.environ.get("K_CODEX_MODEL", "gpt-5.4")


def get_client() -> OpenAI:
    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        raise RuntimeError(
            f"请先设置环境变量 {API_KEY_ENV}，例如："
            f"$env:{API_KEY_ENV} = \"你的 API Key\""
        )
    return OpenAI(api_key=api_key, base_url=BASE_URL)


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "role": str(message.get("role", "user")),
            "content": str(message.get("content", "")),
        }
        for message in messages
    ]


def extract_response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text

    chunks = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                chunks.append(text)
    return "".join(chunks)


def call_responses(
    input_data: str | list[dict[str, Any]],
    *,
    model: str = DEFAULT_MODEL,
    temperature: float | None = None,
) -> str:
    kwargs: dict[str, Any] = {
        "model": model,
        "input": normalize_messages(input_data) if isinstance(input_data, list) else input_data,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature

    response = get_client().responses.create(**kwargs)
    return extract_response_text(response)


def stream_responses(
    input_data: str | list[dict[str, Any]],
    *,
    model: str = DEFAULT_MODEL,
    temperature: float | None = None,
) -> Iterable[str]:
    kwargs: dict[str, Any] = {
        "model": model,
        "input": normalize_messages(input_data) if isinstance(input_data, list) else input_data,
        "stream": True,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature

    with get_client().responses.stream(**kwargs) as stream:
        for event in stream:
            if getattr(event, "type", "") == "response.output_text.delta":
                delta = getattr(event, "delta", "")
                if delta:
                    yield delta

