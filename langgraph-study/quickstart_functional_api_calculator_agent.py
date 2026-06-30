"""LangGraph quickstart demo using the Functional API.

Based on the official calculator-agent example in:
docs/LangGraph-学习笔记-01-入门与核心思维.md

Required environment:
    K_CODEX=...
"""

from __future__ import annotations

from langchain.messages import HumanMessage, SystemMessage, ToolCall
from langchain.tools import tool
from langchain_core.messages import BaseMessage
from langgraph.func import entrypoint, task
from langgraph.graph import add_messages

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


model = get_chat_model(temperature=0)
tools = [add, multiply, divide]
tools_by_name = {registered_tool.name: registered_tool for registered_tool in tools}
llm_with_tools = model.bind_tools(tools)


@task
def call_llm(messages: list[BaseMessage]):
    """Ask the model for the next step in the tool loop."""
    return llm_with_tools.invoke(
        [
            SystemMessage(
                content=(
                    "You are a helpful assistant tasked with performing arithmetic "
                    "on a set of inputs."
                )
            )
        ]
        + messages
    )


@task
def call_tool(tool_call: ToolCall):
    """Execute a single tool call requested by the model."""
    selected_tool = tools_by_name[tool_call["name"]]
    return selected_tool.invoke(tool_call)


@entrypoint()
def calculator_agent(messages: list[BaseMessage]):
    """Run the functional calculator agent until no more tools are needed."""
    llm_response = call_llm(messages).result()

    while True:
        if not llm_response.tool_calls:
            break

        tool_result_futures = [
            call_tool(tool_call) for tool_call in llm_response.tool_calls
        ]
        tool_results = [future.result() for future in tool_result_futures]
        messages = add_messages(messages, [llm_response, *tool_results])
        llm_response = call_llm(messages).result()

    messages = add_messages(messages, llm_response)
    return messages


def main() -> None:
    """Run a sample functional-agent interaction."""
    input_messages = [
        HumanMessage(content="Add 10 and 15, then divide the result by 5.")
    ]
    result_messages = calculator_agent.invoke(input_messages)

    print("=== Functional Calculator Agent Output ===")
    for index, message in enumerate(result_messages, start=1):
        print(f"\n[{index}] {message.__class__.__name__}")
        if hasattr(message, "pretty_print"):
            message.pretty_print()
        else:
            print(message)


if __name__ == "__main__":
    main()
