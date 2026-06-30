from __future__ import annotations

import sys
from pathlib import Path
from typing import TypedDict, Any

from langgraph.graph import END, START, StateGraph
from typing_extensions import Literal
# 确保你的环境里能正常导入此模块
from llm_client import invoke_llm


class ChainingState(TypedDict, total=False):
    """在整个链式工作流中流转的状态。"""
    topic: str
    joke: str
    improved_joke: str
    final_joke: str


def call_qwen(prompt: str) -> str:
    """统一封装一次千问调用，方便多个节点重复使用。"""
    return invoke_llm(
        prompt,
        system_prompt=(
            "你是一名擅长写英文冷笑话的助手。"
            "请直接返回笑话内容，不要补充解释。"
        ),
        temperature=0.7,
    )


def generate_joke(state: ChainingState) -> ChainingState:
    """节点 1：根据主题生成一个初稿笑话。"""
    print("--- [Node 1: 生成初稿笑话] ---")
    joke = call_qwen(f"Write a short joke about {state['topic']}.")
    return {"joke": joke}


def check_punchline(state: ChainingState) -> Literal["Pass", "Fail"]:
    """条件边：检查初稿是否足够完整。"""
    print("--- [Edge: 检查笑话质量] ---")
    if len(state.get("joke", "")) >= 60:
        return "Pass"
    return "Fail"


def improve_joke(state: ChainingState) -> ChainingState:
    """节点 2：把初稿改得更有趣一些。"""
    print("--- [Node 2: 改进笑话] ---")
    improved_joke = call_qwen(
        "Make this joke funnier and slightly smarter, while keeping it concise:\n"
        f"{state['joke']}"
    )
    return {"improved_joke": improved_joke}


def polish_joke(state: ChainingState) -> ChainingState:
    """节点 3：继续润色，并加入一点反转。"""
    print("--- [Node 3: 润色笑话] ---")
    final_joke = call_qwen(
        "Add a surprising twist to the following joke and keep it natural:\n"
        f"{state['improved_joke']}"
    )
    return {"final_joke": final_joke}


def build_chain():
    """构建并编译 Prompt Chaining 示例图。"""
    builder = StateGraph(ChainingState)  # type: ignore
    builder.add_node("generate_joke", generate_joke)
    builder.add_node("improve_joke", improve_joke)
    builder.add_node("polish_joke", polish_joke)

    builder.add_edge(START, "generate_joke")
    builder.add_conditional_edges(
        "generate_joke",
        check_punchline,
        {"Fail": "improve_joke", "Pass": END},
    )
    builder.add_edge("improve_joke", "polish_joke")
    builder.add_edge("polish_joke", END)
    return builder.compile()


def main() -> None:
    """运行一个完整的 Prompt Chaining 示例并绘制流程图。"""
    chain = build_chain()

    try:
        # 获取图片的二进制数据
        png_data = chain.get_graph().draw_mermaid_png()
        output_path = Path(__file__).parent / "graph.png"

        # 将二进制数据写入本地文件
        output_path.write_bytes(png_data)
        print(f"【系统提示】工作流可视化图已成功保存至: {output_path.resolve()}")
    except Exception as e:
        print(f"【系统提示】生成图片失败（可能是因为网络问题或缺少依赖包）: {e}")

    # 运行原本的 LLM 工作流
    result = chain.invoke({"topic": "science"})

    # 如果初稿直接通过质量检查，就没有 final_joke，此时回退到 joke。
    final_output = result.get("final_joke") or result.get("joke", "")
    print(f"\n最终输出的笑话:\n{final_output}")


if __name__ == "__main__":
    main()
