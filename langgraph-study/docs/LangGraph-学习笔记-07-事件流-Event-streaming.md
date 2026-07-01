# LangGraph 学习笔记 07：Event streaming 事件流

> 来源：<https://docs.langchain.com/oss/python/langgraph/event-streaming>
>
> 这一章讲的是“运行中的图到底发生了什么”。它比普通 `stream()` 更接近底层协议。

## 一句话理解

- `stream_events()` 会把运行过程拆成可消费的事件。
- 你可以按 `values`、`messages`、`subgraphs` 等视角去读同一条运行线。
- 它很适合 UI、调试器、观察面板、审计系统。

## 你会在 `stream_events` 里看到什么

- `stream.values`：状态快照。
- `stream.messages`：LLM 或节点消息块。
- `stream.subgraphs`：子图事件。
- `stream.output`：最终输出。

## 完整 Demo

下面这个例子不依赖外部模型，主要演示状态级事件流：

```python
from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    topic: str
    joke: str


def generate_joke(state: State):
    return {"joke": f"Why did {state['topic']} cross the road? To get to the other side."}


builder = StateGraph(State)
builder.add_node("generate_joke", generate_joke)
builder.add_edge(START, "generate_joke")
builder.add_edge("generate_joke", END)
graph = builder.compile()

stream = graph.stream_events({"topic": "ice cream"}, version="v3")

for snapshot in stream.values:
    print(snapshot)

print(stream.output)
```

## 如果你接的是 LLM

- 你可以把 `stream.messages` 当成 token / message chunk 流。
- 文档里的 `messages` 适合边生成边显示到 UI。
- 如果 graph 里还有 subgraph，`stream.subgraphs` 可以把层级也一起暴露出来。

## 适合什么场景

- 聊天界面需要实时显示进度。
- 复杂工作流需要可观测性。
- 你想区分“节点更新”与“模型 token”。

## 本章结论

- `stream_events` 更像事件总线。
- `stream()` 更像面向输出的消费接口。
- 两者不是重复，而是不同观察层级。

