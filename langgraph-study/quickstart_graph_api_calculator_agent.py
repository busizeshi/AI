"""LangGraph quickstart demo using the Graph API.

Based on the official calculator-agent example in:
docs/LangGraph-学习笔记-01-入门与核心思维.md

Required environment:
    K_CODEX=...

Optional environment:
    K_CODEX_MODEL=gpt-5.4
"""

from __future__ import annotations

import operator
from typing import Literal

from langchain.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain.tools import tool
from langgraph.graph import END, START, StateGraph
from typing_extensions import Annotated, TypedDict

from llm_client import get_chat_model


@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`."""
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Add `a` and `b`."""
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide `a` and `b`."""
    return a / b


class MessagesState(TypedDict):
    """Shared state for the calculator agent."""

    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


def build_graph():
    """Compile and return the calculator graph."""
    model = get_chat_model(temperature=0.7)
    tools = [add, multiply, divide]
    tools_by_name = {registered_tool.name: registered_tool for registered_tool in tools}
    model_with_tools = model.bind_tools(tools)

    def llm_call(state: MessagesState) -> dict:
        """Ask the model whether it should answer directly or call a tool."""
        return {
            "messages": [
                model_with_tools.invoke(
                    [
                        SystemMessage(
                            content=(
                                "You are a helpful assistant tasked with performing "
                                "arithmetic on a set of inputs."
                            )
                        )
                    ]
                    + state["messages"]
                )
            ],
            "llm_calls": state.get("llm_calls", 0) + 1,
        }

    def tool_node(state: MessagesState) -> dict:
        """Execute the tool calls requested by the model."""
        result = []
        for tool_call in state["messages"][-1].tool_calls:
            selected_tool = tools_by_name[tool_call["name"]]
            observation = selected_tool.invoke(tool_call["args"])
            result.append(
                ToolMessage(content=str(observation), tool_call_id=tool_call["id"])
            )
        return {"messages": result}

    def should_continue(state: MessagesState) -> Literal["tool_node", END]:
        """Route to tool execution when the model emits tool calls."""
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "tool_node"
        return END

    agent_builder = StateGraph(MessagesState)
    agent_builder.add_node("llm_call", llm_call)
    agent_builder.add_node("tool_node", tool_node)
    agent_builder.add_edge(START, "llm_call")
    agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
    agent_builder.add_edge("tool_node", "llm_call")
    return agent_builder.compile()


def main() -> None:
    """Run a sample calculator interaction."""
    agent = build_graph()
    messages = [HumanMessage(content="Add 3 and 4. Then multiply the result by 5.")]
    result = agent.invoke({"messages": messages, "llm_calls": 0})

    print("=== Calculator Agent Output ===")
    for index, message in enumerate(result["messages"], start=1):
        print(f"\n[{index}] {message.__class__.__name__}")
        if hasattr(message, "pretty_print"):
            message.pretty_print()
        else:
            print(message)
    print(f"\nTotal llm_calls: {result['llm_calls']}")


if __name__ == "__main__":
    main()
