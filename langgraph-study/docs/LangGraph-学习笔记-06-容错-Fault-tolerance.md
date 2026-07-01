# LangGraph 学习笔记 06：Fault tolerance 容错

> 来源：<https://docs.langchain.com/oss/python/langgraph/fault-tolerance>
>
> 这一章的重点是：节点失败时不要把整张图打崩，LangGraph 给你三层工具来兜底。

## 三层保护

- `RetryPolicy`：先重试。
- `TimeoutPolicy`：单次执行别跑太久。
- Error handler / `NodeError`：重试都没救时再进入兜底逻辑。

## 你应该怎么理解执行顺序

- 先发生异常。
- 再看这个异常值不值得重试。
- 重试耗尽后，才会走错误处理。
- 如果是 `interrupt()` 或 graceful shutdown，则属于另一类“可恢复暂停”，不是单纯失败。

## 完整 Demo

```python
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import RetryPolicy, TimeoutPolicy
from typing_extensions import TypedDict


class State(TypedDict):
    result: str


def flaky_node(state: State, runtime: Runtime):
    # 第一次尝试故意失败，模拟短暂网络问题
    if runtime.execution_info.node_attempt == 1:
        raise RuntimeError("transient network error")
    return {"result": f"ok on attempt {runtime.execution_info.node_attempt}"}


def slow_node(state: State):
    return {"result": f"finished with {state['result']}"}


builder = StateGraph(State)
builder.add_node("flaky_node", flaky_node, retry_policy=RetryPolicy(max_attempts=3))
builder.add_node("slow_node", slow_node, timeout=TimeoutPolicy(run_timeout=5, idle_timeout=2))
builder.add_edge(START, "flaky_node")
builder.add_edge("flaky_node", "slow_node")
builder.add_edge("slow_node", END)

graph = builder.compile()
print(graph.invoke({"result": ""}))
```

## 官方页里值得关注的细节

- `custom retry logic`：可以用 `retry_on` 改写哪些异常要重试。
- `inspect retry state`：你能在 runtime 里看到当前是第几次尝试。
- `Heartbeat mode`：长任务可以主动发心跳，避免 idle timeout。
- `NodeTimeoutError`：超时也会进入容错链路。
- `set_node_defaults()`：适合给所有节点统一配置 retry / timeout / error handler。

## 生产视角

- 外部 API 调用几乎都应该配 retry。
- 长任务要么分拆，要么发 heartbeat。
- 如果你要优雅停机，别靠“直接杀进程”，用 graceful shutdown 的思路。
