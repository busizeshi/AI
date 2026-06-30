"""GetStart 示例共用的千问大模型客户端。"""

from __future__ import annotations

import os
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

DEFAULT_BASE_URL = os.environ.get(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)

DEFAULT_MODEL = "qwen-turbo"


def get_chat_model(*, temperature: float = 0) -> ChatOpenAI:
    """返回一个可复用的千问聊天模型实例。"""
    return ChatOpenAI(
        model=DEFAULT_MODEL,
        temperature=temperature,
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=DEFAULT_BASE_URL,
    )


def invoke_llm(
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0,
) -> str:
    """执行一次简单文本调用，并返回模型输出内容。"""
    model = get_chat_model(temperature=temperature)
    messages = []

    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=prompt))

    response = model.invoke(messages)
    return response.text.strip()
