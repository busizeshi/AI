import sys

from langchain_core.runnables import RunnableLambda, RunnableParallel

from llm_client import call_responses


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def summarize(data: dict) -> str:
    return call_responses(f"请用 60 字以内总结这个主题：{data['topic']}", temperature=0.3)


def make_title(data: dict) -> str:
    return call_responses(f"请为「{data['topic']}」生成一个 12 字以内中文标题。", temperature=0.3)


def extract_keywords(data: dict) -> list[str]:
    text = call_responses(
        f"请提取「{data['topic']}」的 4 个中文关键词，用逗号分隔。",
        temperature=0.2,
    )
    return [item.strip() for item in text.replace("，", ",").split(",") if item.strip()]


def maybe_fail(data: dict) -> str:
    if data.get("fail"):
        raise RuntimeError("模拟某个并行分支失败")
    return "风险检查通过"


def safe_task(fn):
    def wrapper(data: dict) -> dict:
        try:
            return {"ok": True, "value": fn(data)}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    return RunnableLambda(wrapper)


def build_parallel_chain():
    return RunnableParallel(
        summary=RunnableLambda(summarize),
        title=RunnableLambda(make_title),
        keywords=RunnableLambda(extract_keywords),
    )


def build_safe_parallel_chain():
    return RunnableParallel(
        summary=safe_task(summarize),
        risk_check=safe_task(maybe_fail),
    )


def build_dynamic_parallel(topics: list[str]):
    tasks = {
        topic: RunnableLambda(
            lambda _data, topic=topic: call_responses(
                f"请为「{topic}」生成一个一句话学习任务。",
                temperature=0.3,
            )
        )
        for topic in topics
    }
    return RunnableParallel(tasks)


def main():
    print("== 基础并行 ==")
    parallel = build_parallel_chain()
    print(parallel.invoke({"topic": "LangChain 表达式语言"}))

    print("\n== batch：多个输入批量跑同一组并行分支 ==")
    inputs = [{"topic": "LCEL"}, {"topic": "LangSmith"}]
    print(parallel.batch(inputs))

    print("\n== 安全并行：单个分支失败也返回结构化结果 ==")
    safe_parallel = build_safe_parallel_chain()
    print(safe_parallel.invoke({"topic": "回退机制", "fail": True}))

    print("\n== 动态创建并行任务 ==")
    dynamic = build_dynamic_parallel(["流式处理", "并行执行", "回退机制"])
    print(dynamic.invoke({}))


if __name__ == "__main__":
    main()
