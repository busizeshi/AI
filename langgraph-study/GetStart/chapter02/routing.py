"""
@Description: 条件路由
"""
from typing import TypedDict, Literal

from langgraph.graph import StateGraph, START, END
from llm_client import invoke_llm
from pathlib import Path


class RoutingState(TypedDict):
    query: str
    category: str
    response: str


def classify_intent(state: RoutingState) -> RoutingState:
    """
    使用llm分析用户请求意图
    """
    # 构建精准分类的 Prompt
    system_prompt = (
        "你是一个精确定位用户意图的路由器。请分析用户输入的文本，将其分类到以下三类之一：\n"
        "- 'technical': 技术故障、报错、Bug、代码异常、环境配置等问题。\n"
        "- 'billing': 价格咨询、发票、退款、订阅方案、购买等财务问题。\n"
        "- 'general': 通用日常问候、聊天、不属于上述两类的常规咨询。\n"
        "【极其重要】你只能返回 'technical'、'billing' 或 'general' 这三个英文单词中的一个，绝对不能包含任何其他多余文本、标点、空格或解释。"
    )

    category = invoke_llm(state["query"], system_prompt=system_prompt, temperature=0.2)

    # 清洗返回文本
    cleaned_category = category.strip().lower()
    print(f"---- LLM 判定意图分类为: {cleaned_category} ---")

    return {"category": cleaned_category}


def router_decision(state: RoutingState) -> Literal["handle_tech", "handle_billing", "handle_general"]:
    """
    解析LLM路由判断
    """
    category = state.get("category", "general")
    if "technical" in category:
        return "handle_tech"
    elif "billing" in category:
        return "handle_billing"
    return "handle_general"


# =================【节点 2-A：技术问题分支】=================
def handle_tech(state: RoutingState) -> RoutingState:
    print("--- [技术专区] LLM 正在调取系统日志并组织技术解答... ---")
    system_prompt = "你是一名资深的技术支持工程师。请专业、清晰地解答用户的技术问题或报错信息，并给出具体的排查方向。"
    response = invoke_llm(state["query"], system_prompt=system_prompt, temperature=0.5)
    return {"response": response}  # type: ignore


# =================【节点 2-B：账单财务分支】=================
def handle_billing(state: RoutingState) -> RoutingState:
    print("--- [财务专区] LLM 正在查询产品定价和发票系统... ---")
    system_prompt = "你是一名亲切的财务账单专员。请解答用户关于价格、订阅、发票或退款的疑问，语气要礼貌、专业。"
    response = invoke_llm(state["query"], system_prompt=system_prompt, temperature=0.5)
    return {"response": response}  # type: ignore


# =================【节点 2-C：通用问题分支】=================
def handle_general(state: RoutingState) -> RoutingState:
    print("--- [客服大厅] LLM 正在调用通用知识库解答... ---")
    system_prompt = "你是一名优秀的在线客服。请热情、友好地回应用户的日常问候或常规咨询。"
    response = invoke_llm(state["query"], system_prompt=system_prompt, temperature=0.7)
    return {"response": response}  # type: ignore


# 编排Graph
router_builder = StateGraph(RoutingState)
router_builder.add_node("classify_intent", classify_intent)
router_builder.add_node("handle_tech", handle_tech)
router_builder.add_node("handle_billing", handle_billing)
router_builder.add_node("handle_general", handle_general)

router_builder.add_edge(START, "classify_intent")
router_builder.add_conditional_edges(
    "classify_intent",
    router_decision,
    {
        "handle_tech": "handle_tech",
        "handle_billing": "handle_billing",
        "handle_general": "handle_general"
    }
)
router_builder.add_edge("handle_tech", END)
router_builder.add_edge("handle_billing", END)
router_builder.add_edge("handle_general", END)

# 编译工作流
router_graph = router_builder.compile()

output_path = Path(__file__).parent / "output/router.png"

try:
    png_data = router_graph.get_graph().draw_mermaid_png()
    output_path.write_bytes(png_data)
except Exception as e:
    print(e)

# =================【运行测试】=================
if __name__ == "__main__":
    # 测试 1：技术报错
    print("====== 测试 1：技术问题 ======")
    res_tech = router_graph.invoke({"query": "I got an ImportError in line 10 when running LangGraph!"})
    print(f"最终回复:\n{res_tech['response']}\n")

    # 测试 2：账单问题
    print("====== 测试 2：账单问题 ======")
    res_billing = router_graph.invoke({"query": "How much does the annual subscription cost? Can I get a discount?"})
    print(f"最终回复:\n{res_billing['response']}\n")
