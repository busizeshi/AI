from dataclasses import dataclass
from typing import List
from typing_extensions import Literal
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import MessagesState, START, END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import ToolRuntime
from pathlib import Path

# 从同级目录的 llm_client 导入获取大模型实例的函数
from llm_client import get_chat_model


# 1. 声明系统的运行时全局 Context 结构
@dataclass
class SecurityContext:
    organization_id: str
    user_role: str


# 2. 声明 Graph State（继承自自带的消息状态）
class CustomAgentState(MessagesState):
    user_id: str  # 拓展自定义字段


# 3. 编写一个不仅能读入参，还能深度读取 Graph State 和 Run Context 的高级工具
@tool
def query_secure_database(query_str: str, runtime: ToolRuntime[SecurityContext, CustomAgentState]) -> str:
    """查询企业内部安全机密数据库。"""
    # 动态抓取图状态中的用户标识
    user_id = runtime.state["user_id"]
    # 动态抓取与图实例绑定的上下文租户标识
    org_id = runtime.context.organization_id
    role = runtime.context.user_role

    print(f"\n[Tool Execution] 收到查询请求: '{query_str}'")
    print(f"  └─ 鉴权成功: 用户 {user_id} | 组织 {org_id} | 角色 {role}")

    if role != "admin":
        return f"警告：用户 {user_id} 无权限执行机密数据查询。"
    return f"【机密结果】针对 {query_str} 的查询结果已加载。组织 {org_id} 安全通道已放行。"


# =================【新规：Agent 智能决策节点】=================
def agent_node(state: CustomAgentState):
    print("--- [Agent] 正在分析上下文并决定下一步行动... ---")

    # 1. 初始化千问大模型
    model = get_chat_model(temperature=0)

    # 2. 将安全数据库查询工具“武器化”绑定给大模型，让模型感知到此工具的存在
    model_with_tools = model.bind_tools([query_secure_database])

    # 3. 传入所有历史消息，让模型产出决策（普通文本响应或携带 tool_calls 的响应）
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# =================【条件路由：根据大模型输出流向不同节点】=================
def should_continue(state: CustomAgentState) -> Literal["tools", "__end__"]:
    last_message = state["messages"][-1]

    # 如果大模型生成的最新消息中包含 tool_calls，说明大模型认为需要调用工具
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # 否则，说明大模型已经回答完毕或无需工具，直接结束流程
    return "__end__"


# 4. 构建图网络
builder = StateGraph(CustomAgentState, context_schema=SecurityContext)

# 注册节点：加入大模型决策节点与原有的工具节点
builder.add_node("agent", agent_node)
secure_tools_node = ToolNode([query_secure_database])
builder.add_node("tools_node", secure_tools_node)

# =================【重构图网络拓扑拓扑关系】=================
# 从起点出发，首先交由大模型思考
builder.add_edge(START, "agent")

# 根据大模型的思考结果，动态选择是去执行工具，还是直接结束
builder.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools_node",
        "__end__": END
    }
)

# 【关键闭环】：工具执行完毕后，必须将结果重新投递给大模型，让大模型基于工具返回的结果进行“润色和最终总结”
builder.add_edge("tools_node", "agent")

agent_graph = builder.compile()

graph_png_path = Path(__file__).parent / "output/agent_graph.png"
png_data = agent_graph.get_graph().draw_mermaid_png()
graph_png_path.write_bytes(png_data)

# 5. 运行测试：采用真实的自然语言输入测试
if __name__ == "__main__":
    print("=== 开始运行具备 Context 状态注入的高级 Agent ===")

    # 抛弃之前硬编码的 mock_ai_message，换成用户自然的业务提问
    user_query = "你好，请帮我调取 2026年Q2核心财报指标 并在数据库中检索它。"

    result = agent_graph.invoke(
        {
            "messages": [HumanMessage(content=user_query)],
            "user_id": "staff_member_alex",
        },
        context=SecurityContext(organization_id="tencent_corp", user_role="admin")
    )

    print("\n--- 图执行结束完整的消息流链条 ---")
    for msg in result["messages"]:
        role = msg.__class__.__name__
        # 优化打印可读性
        if isinstance(msg, HumanMessage):
            role = "User"
        elif isinstance(msg, AIMessage):
            role = "Agent (Tool Call)" if msg.tool_calls else "Agent (Final Reply)"

        content = msg.content if msg.content else f"发起工具请求 -> {msg.tool_calls}"
        print(f"[{role}]: {content}")
