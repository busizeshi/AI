from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from pathlib import Path

# 确保你的环境里能正常导入此模块
from prompt_chaining import call_qwen


class ParallelState(TypedDict):
    topic: str
    joke: str
    story: str
    poem: str
    combined_output: str


def generate_parallel_joke(state: ParallelState) -> ParallelState:
    print("---生成笑话---")
    # 修复 1：修正了 f-string 的笔误
    resp = call_qwen(f"给我写一个笑话，关于{state['topic']}")
    # 修复 2：resp 本身就是字符串，直接返回即可
    return {"joke": resp}  # type: ignore


def generate_parallel_story(state: ParallelState) -> ParallelState:
    print("---生成故事---")
    resp = call_qwen(f"给我生成一个故事，关于{state['topic']}")
    return {"story": resp}  # type: ignore


def generate_parallel_poem(state: ParallelState) -> ParallelState:
    print("---生成诗歌---")
    resp = call_qwen(f"给我生成一首诗歌，关于{state['topic']}")
    return {"poem": resp}  # type: ignore


def aggregator(state: ParallelState) -> ParallelState:
    print("---合并所有并发分支的结果---")
    combined = (
        f"--- 合并报告 - 主题：{state['topic']} ---\n"
        f"笑话：{state.get('joke', '')}\n"
        f"故事：{state.get('story', '')}\n"
        f"诗歌：{state.get('poem', '')}\n"
    )
    return {"combined_output": combined}  # type: ignore


# 构建图
parallel_builder = StateGraph(ParallelState)  # type: ignore
parallel_builder.add_node("call_joke", generate_parallel_joke)
parallel_builder.add_node("call_story", generate_parallel_story)
parallel_builder.add_node("call_poem", generate_parallel_poem)
parallel_builder.add_node("aggregator", aggregator)

# 开启并行分支
parallel_builder.add_edge(START, "call_joke")
parallel_builder.add_edge(START, "call_story")
parallel_builder.add_edge(START, "call_poem")

# 汇聚到聚合节点
parallel_builder.add_edge("call_joke", "aggregator")
parallel_builder.add_edge("call_story", "aggregator")
parallel_builder.add_edge("call_poem", "aggregator")
parallel_builder.add_edge("aggregator", END)

# 编译工作流
parallel_workflow = parallel_builder.compile()

output_path = Path(__file__).parent / "output/parallel.png"
output_path.parent.mkdir(parents=True, exist_ok=True)

try:
    png_data = parallel_workflow.get_graph().draw_mermaid_png()
    output_path.write_bytes(png_data)
    print(f"工作流图已成功保存至: {output_path.resolve()}")
except Exception as e:
    print(f"生成图片失败: {e}")

# 启动工作流
output = parallel_workflow.invoke({"topic": "AI Code Agent"})
print("\n" + output["combined_output"])