import json
from typing import TypedDict

from llm_client import invoke_llm
from typing_extensions import Literal
from pathlib import Path
from langgraph.graph import StateGraph, START, END


class OptimizationState(TypedDict):
    task: str
    draft: str
    feedback: str
    score: int
    iterations: int


# 节点一：生成器
def generator_node(state: OptimizationState) -> OptimizationState:
    iters = state.get("iterations", 0) + 1
    print(f"--- [Generator] 开始进行第{iters}次代码生成/优化... ---")

    system_prompt = (
        "你是一个精通 Python 的高级开发工程师。请根据要求编写高质量、符合 PEP8 规范的代码。\n"
        "【极其重要】请直接输出代码内容，务必包含类型提示(Type Hints)和完整的 Docstring 注释。"
    )

    # 区分：是初次生成，还是根据反馈进行迭代修正
    if iters == 1:
        user_prompt = f"请编写满足以下要求的 Python 代码：\n{state['task']}"
    else:
        user_prompt = (
            f"目标任务: {state['task']}\n\n"
            f"你上一版编写的代码:\n{state['draft']}\n\n"
            f"评审专家的意见反馈:\n{state['feedback']}\n\n"
            f"请根据反馈意见彻底修改并优化你的代码，修复其中的不足，使其达到完美状态。"
        )

    draft = invoke_llm(user_prompt, system_prompt=system_prompt, temperature=0.5)
    return {"draft": draft, "iterations": iters}  # type: ignore


# =================【节点 2：评审器 (Evaluator)】=================
def evaluator_node(state: OptimizationState) -> OptimizationState:
    print("--- [Evaluator] 正在利用 LLM 对当前草稿进行深度质量评审... ---")

    system_prompt = (
        "你是一个极其严格的 Python 代码评审专家。你需要对用户提交的代码进行盲审打分（1-10分）并给出具体的改进反馈。\n"
        "评审标准：\n"
        "1. 代码功能是否完全满足目标任务？\n"
        "2. 是否包含完善的类型提示(Type Hints)？\n"
        "3. 是否有清晰的 Docstring 注释说明？\n\n"
        "【极其重要】你必须严格按照以下 JSON 格式回复，绝对不要包含任何其他说明文字或 Markdown 代码块标记（如 ```json ）：\n"
        "{\n"
        '  "score": 整数(1-10),\n'
        '  "feedback": "具体、严厉的改进意见或指出遗漏点"\n'
        "}"
    )

    user_prompt = f"要求实现的任务: {state['task']}\n\n当前提交的代码:\n{state['draft']}"

    response = invoke_llm(user_prompt, system_prompt=system_prompt, temperature=0.2)

    # 解析大模型返回的结构化 JSON 数据
    try:
        # 兼容处理大模型可能附带的 Markdown 代码块标签
        clean_res = response.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_res)
        score = int(data.get("score", 1))
        feedback = data.get("feedback", "未提供具体反馈")
    except Exception as e:
        # 兜底解析失败的逻辑
        score = 1
        feedback = f"解析评审 JSON 结果失败，大模型原始输出为: {response}。错误: {str(e)}"

    print(f"--- [Evaluator] 评审完毕。给分: {score}，反馈: {feedback} ---")
    return {"score": score, "feedback": feedback}  # type: ignore


# =================【闭环控制逻辑：检查质量分与最大限制】=================
def check_satisfaction(state: OptimizationState) -> Literal["optimize", "terminate"]:
    score = state.get("score", 0)
    iters = state.get("iterations", 0)

    print(f"--- [控制台] 决策阶段 -> 当前得分: {score}/10，已迭代轮次: {iters}/3 ---")

    # 退出条件：分数达到 8 分以上，或者达到最大安全迭代轮次（3次）
    if score >= 8 or iters >= 3:
        if score >= 8:
            print("--- [控制台] 🎉 代码质量达标，批准通过！ ---")
        else:
            print("--- [控制台] ⚠️ 已达到最大迭代次数，强制触发安全终止。 ---")
        return "terminate"

    print("--- [控制台] ❌ 质量未达标，发回生成器继续优化... ---\n")
    return "optimize"


# =================【构建闭环图】=================
loop_builder = StateGraph(OptimizationState)  # type: ignore
loop_builder.add_node("generator", generator_node)
loop_builder.add_node("evaluator", evaluator_node)

loop_builder.add_edge(START, "generator")
loop_builder.add_edge("generator", "evaluator")

# 根据分数和轮次动态回弹或终止
loop_builder.add_conditional_edges(
    "evaluator",
    check_satisfaction,
    {
        "optimize": "generator",
        "terminate": END
    }
)

loop_graph = loop_builder.compile()

# =================【自动保存工作流图】=================
output_path = Path(__file__).parent / "output/optimization_loop.png"
output_path.parent.mkdir(parents=True, exist_ok=True)
try:
    png_data = loop_graph.get_graph().draw_mermaid_png()
    output_path.write_bytes(png_data)
    print(f"【系统提示】闭环优化工作流图已成功保存至: {output_path.resolve()}\n")
except Exception as e:
    print(f"【系统提示】生成流程图图片失败: {e}\n")

# =================【运行测试】=================
if __name__ == "__main__":
    # 我们故意给一个复杂的任务，并要求“严苛”，看它如何自我纠错
    test_task = "编写一个计算斐波那契数列第 n 项的函数，要求必须处理输入为负数的异常情况，包含类型提示和规范的 docstring。"

    print("====== 启动大模型自我演进闭环 ======")
    final_state = loop_graph.invoke({
        "task": test_task,
        "draft": "",
        "iterations": 0,
        "score": 0,
        "feedback": ""
    })

    print("\n====================================")
    print(f"✨ 最终演进出的代码 (得分: {final_state['score']}):")
    print("====================================")
    print(final_state['draft'])
