import sys

from llm_client import call_responses
from tools import search, search_product_info


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def summarize_space_exploration() -> str:
    context = search("太空探索")
    return call_responses(
        f"请基于资料，用 80 字以内总结太空探索。\n资料：{context}",
        temperature=0.3,
    )


def recommend_product(product_name: str) -> str:
    product_info = search_product_info(product_name)
    return call_responses(
        f"请基于产品信息生成推荐总结。\n产品：{product_name}\n信息：{product_info}",
        temperature=0.3,
    )


def main():
    print("== Agent 多步骤任务：搜索后总结 ==")
    print(summarize_space_exploration())

    print("\n== Agent 应用场景：产品推荐 ==")
    print(recommend_product("智能手机"))


if __name__ == "__main__":
    main()

