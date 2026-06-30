import json
from pathlib import Path
from typing import TypedDict, List
from typing_extensions import Annotated
from llm_client import invoke_llm
from langgraph.graph import StateGraph, START, END


# 单个任务定义
class SubTask(TypedDict):
    worker_id: str
    instruction: str
    result: str


def merge_plan(left: list, right: list) -> list:
    """自定义 Reducer：确保并行分支同时更新 plan 列表时按 worker_id 完美合并，不发生覆盖"""
    task_map = {task["worker_id"]: task.copy() for task in left}
    for task in right:
        w_id = task["worker_id"]
        if w_id in task_map:
            if task.get("result"):
                task_map[w_id]["result"] = task["result"]
            if task.get("instruction"):
                task_map[w_id]["instruction"] = task["instruction"]
        else:
            task_map[w_id] = task
    return list(task_map.values())


class OrchestratorState(TypedDict):
    user_query: str
    plan: Annotated[List[SubTask], merge_plan]  # 动态任务规划表，挂载合并器
    final_report: str


# =================【节点 1：规划者 (Orchestrator)】=================
def orchestrator_node(state: OrchestratorState):
    print("--- [Orchestrator] 正在利用大模型分解总任务并分配人员... ---")

    system_prompt = (
        "你是一个极其严谨的任务编排专家。你需要将用户的总调研任务拆解为两个专职子任务：\n"
        "1. 分配给 'Tech_Worker'：负责研究该技术的底层架构、并发吞吐或性能瓶颈。\n"
        "2. 分配给 'Market_Worker'：负责研究该技术的商业价值、应用前景或行业痛点。\n\n"
        "【极其重要】你必须严格按照以下 JSON 数组格式回复，绝对不要包含任何其他说明文字或 Markdown 代码块标记（如 ```json ）：\n"
        "[\n"
        '  {"worker_id": "Tech_Worker", "instruction": "具体的技术分析指导要求", "result": ""},\n'
        '  {"worker_id": "Market_Worker", "instruction": "具体的市场调研指导要求", "result": ""}\n'
        "]"
    )

    user_prompt = f"用户希望调研的主题是：{state['user_query']}"

    response = invoke_llm(user_prompt, system_prompt=system_prompt, temperature=0.2)

    # 解析动态规划 JSON
    try:
        clean_res = response.replace("```json", "").replace("```", "").strip()
        dynamic_plan = json.loads(clean_res)
    except Exception as e:
        print(f"--- [Orchestrator] JSON 解析失败，启用静态方案兜底。错误: {e} ---")
        dynamic_plan = [
            {"worker_id": "Tech_Worker", "instruction": "分析该技术的底层并发性能", "result": ""},
            {"worker_id": "Market_Worker", "instruction": "评估该技术的商业市场应用前景", "result": ""}
        ]

    return {"plan": dynamic_plan}


# =================【节点 2：并行执行的专业 Workers】=================
def tech_worker_node(state: OrchestratorState):
    print("--- [Tech Worker] 正在利用大模型执行并发与吞吐分析... ---")

    # 从规划表中提取属于自己的具体指令
    instruction = "分析该技术的底层并发性能"
    for task in state["plan"]:
        if task["worker_id"] == "Tech_Worker":
            instruction = task["instruction"]
            break

    system_prompt = "你是一个技术死磕派的资深系统架构师。请针对给出的主题和细分任务，写一份技术过硬、包含具体考量层面的深度技术报告。"
    user_prompt = f"核心主题: {state['user_query']}\n你的具体分析任务: {instruction}"

    result = invoke_llm(user_prompt, system_prompt=system_prompt, temperature=0.5)

    # 仅返回自己这一部分更新，通过 Reducer 机制合并入主状态
    return {"plan": [{"worker_id": "Tech_Worker", "instruction": instruction, "result": result}]}


def market_worker_node(state: OrchestratorState):
    print("--- [Market Worker] 正在利用大模型执行市场调研... ---")

    instruction = "评估该技术的商业市场应用前景"
    for task in state["plan"]:
        if task["worker_id"] == "Market_Worker":
            instruction = task["instruction"]
            break

    system_prompt = "你是一个拥有敏锐商业嗅觉的行业分析总监。请针对给出的主题和细分任务，写一份包含市场痛点、落地前景及趋势的商业分析报告。"
    user_prompt = f"核心主题: {state['user_query']}\n你的具体调研任务: {instruction}"

    result = invoke_llm(user_prompt, system_prompt=system_prompt, temperature=0.5)

    return {"plan": [{"worker_id": "Market_Worker", "instruction": instruction, "result": result}]}


# =================【节点 3：结果合成器】=================
def synthesizer_node(state: OrchestratorState):
    print("--- [Synthesizer] 正在汇总各路大模型情报并提炼最终长文... ---")

    # 抽取并行节点的产出拼接成素材上下文
    expert_inputs = ""
    for task in state["plan"]:
        expert_inputs += f"=== 专家 [{task['worker_id']}] 的深度调研成果 ===\n{task['result']}\n\n"

    system_prompt = (
        "你是一个顶级战略咨询顾问。你需要将多个领域的专家报告融合成一份天衣无缝、结构严密的综合报告。\n"
        "请消除各小节拼接的生硬感，将其润色为一份具备核心摘要、技术深度洞察、市场价值论证和落地建议的专业级综合长文。"
    )
    user_prompt = f"研究总主题：{state['user_query']}\n\n以下是收集到的各路专家素材：\n{expert_inputs}"

    final_report = invoke_llm(user_prompt, system_prompt=system_prompt, temperature=0.3)
    return {"final_report": final_report}


# =================【构建编排图结构】=================
orch_builder = StateGraph(OrchestratorState)
orch_builder.add_node("orchestrator", orchestrator_node)
orch_builder.add_node("tech_worker", tech_worker_node)
orch_builder.add_node("market_worker", market_worker_node)
orch_builder.add_node("synthesizer", synthesizer_node)

orch_builder.add_edge(START, "orchestrator")
# 派发至并行的 Workers
orch_builder.add_edge("orchestrator", "tech_worker")
orch_builder.add_edge("orchestrator", "market_worker")
# 并行结束，收束
orch_builder.add_edge("tech_worker", "synthesizer")
orch_builder.add_edge("market_worker", "synthesizer")
orch_builder.add_edge("synthesizer", END)

orch_graph = orch_builder.compile()

graph_png_path = Path(__file__).parent / "output/orch_graph.png"
png_data = orch_graph.get_graph().draw_mermaid_png()
print(f"【系统提示】编排图已成功保存至: {graph_png_path.resolve()}\n")
graph_png_path.write_bytes(png_data)

# =================【运行测试】=================
if __name__ == "__main__":
    report_out = orch_graph.invoke({"user_query": "LangGraph 企业级落地可行性"})
    print("\n====================================")
    print("      ✨ 生成的综合调研报告 ✨     ")
    print("====================================")
    print(report_out["final_report"])
