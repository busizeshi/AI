# LangGraph 学习笔记 03：Persistence 持久化

> 来源：<https://docs.langchain.com/oss/python/langgraph/persistence>
>
> 这一章讲的是：让 graph 的运行状态能“记住”，并且把短期状态和长期共享数据分开管理。

## 一眼看懂

- `checkpointer` 负责保存图的状态快照，核心是“按 thread 记住这次对话/任务的进度”。
- `store` 负责保存跨线程、跨会话的共享数据，核心是“把长期记忆放到可复用的 namespace 里”。
- `thread_id` 是短期状态的主键。
- `namespace` 是长期存储的主键空间，通常按 `user_id`、`agent_id`、`project_id` 分层。

## 核心区别

- `checkpointer` 更像“运行日志 + 可回放快照”，适合短期记忆、断点恢复、time-travel、interrupt。
- `store` 更像“应用数据库里的共享记忆层”，适合用户画像、偏好、项目知识、检索型记忆。
- `checkpointer` 解决“这次跑到哪了”。
- `store` 解决“以后别的线程也能查到这条信息”。

## 官方快速起步的意思

官方快速起步的关键写法就是把两者一起传给 `compile()`：

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

checkpointer = InMemorySaver()
store = InMemoryStore()
graph = builder.compile(checkpointer=checkpointer, store=store)
```

之后运行时再带上：

```python
config = {"configurable": {"thread_id": "thread-1"}}
result = graph.invoke({"messages": [{"role": "user", "content": "Hi"}]}, config)
```

## 完整 Demo

下面这个 demo 同时演示短期持久化和长期存储：

```python
from operator import add
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.memory import InMemoryStore
from typing_extensions import Annotated, TypedDict


class State(TypedDict):
    messages: Annotated[list[str], add]


checkpointer = InMemorySaver()
store = InMemoryStore()


def greet(state: State):
    return {"messages": ["hello from LangGraph"]}


def save_profile(state: State):
    # 把长期记忆写到共享 store，namespace 由 user_id + 类别组成
    user_ns = ("alice", "profile")
    store.put(user_ns, str(uuid4()), {"favorite_food": "pizza", "city": "Shanghai"})
    return {"messages": ["profile saved"]}


builder = StateGraph(State)
builder.add_node("greet", greet)
builder.add_node("save_profile", save_profile)
builder.add_edge(START, "greet")
builder.add_edge("greet", "save_profile")
builder.add_edge("save_profile", END)

graph = builder.compile(checkpointer=checkpointer, store=store)
config = {"configurable": {"thread_id": "thread-1"}}

print(graph.invoke({"messages": []}, config))
print(graph.invoke({"messages": ["second turn"]}, config))

# 长期记忆可以被另一个线程或另一个 graph 查询
memories = store.search(("alice", "profile"), query="What does Alice like to eat?", limit=3)
print(memories[0].value)
```

## 关键实践

- 生产环境别用 `InMemorySaver` / `InMemoryStore`，进程重启后数据会丢。
- PostgresSaver 的 `thread_id` 不要太长，官方特别提醒过长度限制。
- 检查点会持续增长，记得做清理策略。
- 如果你在子图里读取父图状态，要注意 `checkpoint_ns`，那是子图隔离的关键。

## 本章结论

- 短期状态靠 `checkpointer`。
- 长期共享记忆靠 `store`。
- 真实项目里，两者通常是一起用的，而不是二选一。

