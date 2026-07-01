# LangGraph 学习笔记 04：Checkpointers 检查点

> 来源：<https://docs.langchain.com/oss/python/langgraph/checkpointers>
>
> 这一章把“图为什么能续跑、回放、回滚”讲透了。你可以把 checkpointer 理解成 LangGraph 的时间轴。

## 一句话理解

- `thread_id` 代表一次独立的运行线。
- 每次 super-step 都会形成一个 checkpoint。
- checkpoint 保存了 state、写入记录、下一步要执行的节点、父 checkpoint 信息。
- `checkpoint_ns` 是子图和父图之间的命名空间隔离层。

## 你一定要记住的四个概念

- `Thread`：一次任务/对话的逻辑会话。
- `Checkpoint`：某一时刻的状态快照。
- `Super-step`：图按批次推进的一小步。
- `Namespace`：子图或嵌套运行时的状态隔离空间。

## 官方页面最重要的几个 API

- `graph.get_state(config)`：拿最新快照。
- `graph.get_state_history(config)`：拿历史快照列表。
- `graph.update_state(config, values=...)`：在某个快照上 fork。
- `graph.invoke(None, fork_config)`：从 fork 出来的状态继续跑。

## 完整 Demo

```python
from operator import add
from typing_extensions import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    foo: str
    bar: Annotated[list[str], add]


def node_a(state: State):
    return {"foo": "a", "bar": ["a"]}


def node_b(state: State):
    return {"foo": "b", "bar": ["b"]}


builder = StateGraph(State)
builder.add_node("node_a", node_a)
builder.add_node("node_b", node_b)
builder.add_edge(START, "node_a")
builder.add_edge("node_a", "node_b")
builder.add_edge("node_b", END)

graph = builder.compile(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "thread-1"}}

print(graph.invoke({"foo": "", "bar": []}, config))

latest = graph.get_state(config)
print(latest.values)

history = list(graph.get_state_history(config))
before_node_b = next(snapshot for snapshot in history if snapshot.next == ("node_b",))

fork_config = graph.update_state(before_node_b.config, {"foo": "forked"})
fork_result = graph.invoke(None, fork_config)
print(fork_result)
```

## 读历史时要看什么

- `values`：当前 state 的最终值。
- `next`：下一步要执行的节点。
- `metadata["source"]`：这个快照是输入、循环、还是 update 产生的。
- `parent_config`：这个 checkpoint 从哪里分叉出来。

## 官方提到的 durability 模式

- `sync`：更稳，写入确认后再往下走。
- `async`：更快，适合更高吞吐但要接受更弱的即时持久化保障。
- `exit`：偏向在运行结束时统一落盘。

## 这章的隐藏重点

- 你不仅能“看状态”，还能“改状态以后继续跑”。
- 这就是 LangGraph 比普通函数调用更像工作流引擎的地方。
- 一旦理解 checkpoint，你就理解了后面的 interrupt、time travel、subgraph persistence。

## 配套图

![checkpoints overview](D:/jwd-dev/study/AI/langgraph-study/docs/assets/langgraph/checkpoints.jpg)

![checkpoints full story](D:/jwd-dev/study/AI/langgraph-study/docs/assets/langgraph/checkpoints_full_story.jpg)

![get state](D:/jwd-dev/study/AI/langgraph-study/docs/assets/langgraph/get_state.jpg)

![replay](D:/jwd-dev/study/AI/langgraph-study/docs/assets/langgraph/re_play.png)

