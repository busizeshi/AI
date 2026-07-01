# LangGraph 学习笔记 08：Streaming 流式输出

> 来源：<https://docs.langchain.com/oss/python/langgraph/streaming>
>
> 这一章讲的是“怎么把图的输出边跑边吐出来”，重点是 `stream()` 和不同 `stream_mode` 的组合。

## 一句话理解

- `stream()` 是面向结果消费的主入口。
- `stream_mode` 决定你看到的是状态、增量、消息、还是自定义事件。
- `version="v2"` 是官方当前强调的输出结构格式。

## 常见 stream mode

- `values`：每一步的完整值。
- `updates`：每个节点更新了什么。
- `messages`：LLM token 或 message chunk。
- `custom`：你自己往流里塞的事件。
- `checkpoints`：检查点事件。
- `tasks`：任务调度信息。
- `debug`：调试信息。

## 完整 Demo

```python
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


class State(TypedDict):
    topic: str
    joke: str


def generate_joke(state: State):
    writer = get_stream_writer()
    writer({"status": "thinking of a joke..."})
    return {"joke": f"Why did the {state['topic']} go to school? To get a sundae education!"}


graph = (
    StateGraph(State)
    .add_node(generate_joke)
    .add_edge(START, "generate_joke")
    .add_edge("generate_joke", END)
    .compile()
)

for chunk in graph.stream(
    {"topic": "ice cream"},
    stream_mode=["updates", "custom"],
    version="v2",
):
    if chunk["type"] == "updates":
        for node_name, state in chunk["data"].items():
            print(f"Node {node_name} updated: {state}")
    elif chunk["type"] == "custom":
        print(f"Status: {chunk['data']['status']}")
```

## 这章最值得记住的点

- `custom` 流适合你自己定义过程消息。
- `updates` 流适合 UI 增量渲染。
- `messages` 流适合聊天类应用的 token 级输出。
- `values` 流适合看最终状态收敛过程。

## 和 event streaming 的区别

- `streaming` 更偏“怎么喂给前端”。
- `event-streaming` 更偏“协议层发生了什么”。
- 一个偏消费，一个偏观测。

