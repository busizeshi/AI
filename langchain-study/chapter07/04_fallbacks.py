import sys

from langchain_core.runnables import RunnableLambda

from llm_client import call_responses


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


class RateLimitError(Exception):
    pass


def primary_model(data: dict) -> str:
    if data.get("simulate_rate_limit"):
        raise RateLimitError("主模型触发速率限制")
    if data.get("simulate_timeout"):
        raise TimeoutError("主模型超时")
    return call_responses(
        f"请回答这个问题：{data['question']}",
        temperature=0.3,
    )


def fallback_model(data: dict) -> str:
    return call_responses(
        f"主链路失败，请用稳妥、简短的方式回答：{data['question']}",
        temperature=0.1,
    )


def main():
    primary_chain = RunnableLambda(primary_model)
    fallback_chain = RunnableLambda(fallback_model)

    print("== 默认成功：不会触发回退 ==")
    robust_chain = primary_chain.with_fallbacks([fallback_chain])
    print(robust_chain.invoke({"question": "LCEL 是什么？"}))

    print("\n== 主链失败：自动触发回退 ==")
    print(
        robust_chain.invoke(
            {"question": "主模型限流时怎么办？", "simulate_rate_limit": True}
        )
    )

    print("\n== 只处理指定异常：RateLimitError 会回退 ==")
    rate_limit_only = primary_chain.with_fallbacks(
        [fallback_chain],
        exceptions_to_handle=(RateLimitError,),
    )
    print(
        rate_limit_only.invoke(
            {"question": "如何指定回退错误类型？", "simulate_rate_limit": True}
        )
    )

    print("\n== 只处理指定异常：TimeoutError 不会回退 ==")
    try:
        rate_limit_only.invoke(
            {"question": "超时是否回退？", "simulate_timeout": True}
        )
    except TimeoutError as exc:
        print(f"捕获到未回退异常：{exc}")


if __name__ == "__main__":
    main()
