# src/agent.py
import os
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_openai import ChatOpenAI


def assistant_node(state: MessagesState):
    """一个最基础的聊天节点"""
    # 动态适配千问大模型或标准 OpenAI
    model = ChatOpenAI(
        model=os.environ.get("DEFAULT_MODEL", "qwen-turbo"),
        api_key=os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    )

    response = model.invoke(state["messages"])
    # 必须返回要更新的 State 字典
    return {"messages": [response]}


# 创建图
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant_node)
builder.add_edge(START, "assistant")
builder.add_edge("assistant", END)

# 【极其重要】这里的变量名 graph 必须与 langgraph.json 里的声明完全一致
graph = builder.compile()