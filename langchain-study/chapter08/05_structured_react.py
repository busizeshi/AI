import json
import sys

from llm_client import call_responses


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


PAGES = {
    "https://example.com/langchain-agent": (
        "LangChain Agent 可以根据用户目标选择工具、检索资料、执行计算，并在多步骤任务中持续调整策略。"
    ),
    "https://example.com/react": (
        "ReAct 通过 Reasoning 与 Action 结合，让模型先思考，再调用工具，并根据观察结果继续推理。"
    ),
}


def structured_react(payload: dict) -> str:
    task = payload.get("task")
    if task != "summarize_page":
        return "当前示例只支持 summarize_page 任务。"

    url = payload["url"]
    language = payload.get("language", "zh")
    max_words = int(payload.get("max_words", 80))
    page_text = PAGES.get(url, "没有找到该 URL 的本地页面内容。")

    return call_responses(
        f"请用 {language} 在 {max_words} 字以内总结网页内容。\nURL：{url}\n内容：{page_text}",
        temperature=0.3,
    )


def main():
    payload = {
        "task": "summarize_page",
        "url": "https://example.com/react",
        "language": "中文",
        "max_words": 60,
    }
    print("== 结构化输入 ==")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print("\n== Agent 输出 ==")
    print(structured_react(payload))


if __name__ == "__main__":
    main()

