# LangGraph 学习笔记 11：Memory 记忆

> 来源：<https://docs.langchain.com/oss/python/langgraph/add-memory>
>
> 这一章把记忆拆成两层：短期记忆和长期记忆。

## 一句话理解

- 短期记忆 = checkpoint 里的线程状态。
- 长期记忆 = store 里的跨线程信息。
- `MessagesState` 很适合短期聊天记忆。
- `store` 很适合用户画像、偏好、摘要和检索型记忆。

## 你会在官方页里看到的几个动作

- Add short-term memory。
- Add long-term memory。
- Trim messages。
- Delete messages。
- Summarize messages。
- Manage checkpoints。

## 完整 Demo

```python
from operator import add
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.memory import InMemoryStore
from typing_extensions import Annotated, TypedDict


class State(TypedDict):
    user_id: str
    messages: Annotated[list[str], add]


store = InMemoryStore()
checkpointer = InMemorySaver()


def chat_node(state: State):
    last_message = state["messages"][-1]
    if "love" in last_message.lower():
        store.put(
            (state["user_id"], "memories"),
            str(uuid4()),
            {"memory": last_message},
            index=["memory"],
        )
    return {"messages": [f"I noted: {last_message}"]}


builder = StateGraph(State)
builder.add_node("chat_node", chat_node)
builder.add_edge(START, "chat_node")
builder.add_edge("chat_node", END)
graph = builder.compile(checkpointer=checkpointer, store=store)

config = {"configurable": {"thread_id": "thread-1"}}
print(graph.invoke({"user_id": "alice", "messages": ["I love ramen"]}, config))
print(graph.invoke({"user_id": "alice", "messages": ["what do you remember about me?"]}, config))

memories = store.search(("alice", "memories"), query="What does Alice like to eat?", limit=3)
print([item.value for item in memories])
```

## 短期记忆怎么管

- 太长就 trim。
- 关键信息可以 summarize。
- 不需要了就 delete。
- 线程级别的历史可以直接从 checkpoint 里读。

## 长期记忆怎么管

- 用户偏好、项目事实、个人设定放到 store。
- 存储前先设计 namespace。
- 需要语义召回的字段再做 embedding。
- 不要把所有原始聊天记录都塞进长期存储。

## 配套图

![memory summary](D:/jwd-dev/study/AI/langgraph-study/docs/assets/langgraph/summary.png)

