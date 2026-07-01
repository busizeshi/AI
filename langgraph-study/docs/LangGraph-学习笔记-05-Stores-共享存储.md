# LangGraph 学习笔记 05：Stores 共享存储

> 来源：<https://docs.langchain.com/oss/python/langgraph/stores>
>
> 这一章讲的是“跨线程的长期记忆”，也就是多个 graph run 之间共享的数据层。

## 一句话理解

- `store` 是长期记忆仓库。
- `namespace` 是分区，通常按用户、项目、Agent、业务域来设计。
- `key` 是 namespace 里的具体条目 id。
- `search()` 既能做普通检索，也能做语义检索。

## 先记住这几个操作

- `put(namespace, key, value)`：写入。
- `get(namespace, key)`：读取单条。
- `search(namespace, query=..., limit=...)`：搜索。
- `list_namespaces(prefix=..., max_depth=...)`：列出命名空间。

## 完整 Demo

```python
from uuid import uuid4

from langchain.embeddings import init_embeddings
from langgraph.store.memory import InMemoryStore


namespace_for_memory = ("alice", "memories")

store = InMemoryStore(
    index={
        "embed": init_embeddings("openai:text-embedding-3-small"),
        "dims": 1536,
        "fields": ["food_preference", "$"],
    }
)

# 写入普通记忆
store.put(namespace_for_memory, str(uuid4()), {"food_preference": "I like pizza"})
store.put(namespace_for_memory, str(uuid4()), {"food_preference": "I also enjoy sushi"})
store.put(namespace_for_memory, str(uuid4()), {"system_info": "Last updated: 2024-01-01"}, index=False)

# 列出同一前缀下的命名空间
print(store.list_namespaces(prefix=("alice",), max_depth=2))

# 普通搜索
all_items = store.search(namespace_for_memory, limit=10)
print(all_items[-1].dict())

# 语义搜索
semantic_items = store.search(
    namespace_for_memory,
    query="What does Alice like to eat?",
    limit=3,
)
for item in semantic_items:
    print(item.key, item.value)
```

## 在 LangGraph 里怎么用

- 在 node 里通过 `runtime.store` 读写共享记忆。
- 如果是用户画像类数据，namespace 通常长这样：`(user_id, "profile")` 或 `(user_id, "memories")`。
- 如果是团队知识库，可以用 `(org_id, "knowledge")`。

## 实战建议

- 不要把所有东西都塞进同一个 namespace。
- 需要语义召回的字段才做 embedding，没必要的字段用 `index=False`。
- 只读长文本没问题，但高频写入的结构化小数据更适合直接 key-value。
- 如果你要做生产级 store，要关注分页、清理和索引成本。

## 配套图

![shared state](assets/langgraph/shared_state.png)
