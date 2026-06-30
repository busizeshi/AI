"""Minimal LangGraph interrupt/checkpointer demo.

This is a compact runnable version of the human-review example discussed in:
docs/LangGraph-学习笔记-01-入门与核心思维.md

It does not call an external LLM. The goal is to demonstrate:
1. checkpoint-backed pause with interrupt()
2. resume with Command(resume=...)
3. stable thread_id based recovery
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class EmailState(TypedDict):
    """State for the human-review demo."""

    email_content: str
    response_text: str | None


def human_review_node(state: EmailState):
    """Pause execution and wait for a human approval payload."""
    interrupt(
        {
            "approved": False,
            "edited_response": state.get("response_text") or "",
            "original_email": state["email_content"],
        }
    )
    return {"response_text": "placeholder"}


def build_app():
    """Create and compile the minimal interrupt demo graph."""
    return (
        StateGraph(EmailState)
        .add_node("human_review", human_review_node)
        .add_edge(START, "human_review")
        .add_edge("human_review", END)
        .compile(checkpointer=InMemorySaver())
    )


def main() -> None:
    """Run the interrupt demo from pause to resume."""
    app = build_app()
    initial_state = {
        "email_content": "I was charged twice for my subscription! This is urgent!",
        "response_text": "Draft response: We are checking your billing issue.",
    }

    config = {"configurable": {"thread_id": "customer_123"}}
    first_run = app.stream_events(initial_state, config, version="v3")
    _ = first_run.output

    print("=== Interrupt Demo: Paused State ===")
    print(first_run.interrupts)

    human_response = Command(
        resume={
            "approved": True,
            "edited_response": (
                "We sincerely apologize for the double charge. "
                "I've initiated an immediate refund."
            ),
        }
    )

    resumed_run = app.stream_events(human_response, config, version="v3")
    final_state = resumed_run.output

    print("\n=== Interrupt Demo: Resumed State ===")
    print(final_state)


if __name__ == "__main__":
    main()
