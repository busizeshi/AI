# LangGraph 学习笔记 01：入门、核心概念与建图思维

> 说明：
> 本笔记主要基于 `docs/Get-started` 下的官方文档整理，尽量保持官方原始结构与术语。
> 内容大约 90% 参考官方文档，10% 加入了面向学习者的中文解释、串联和经验提示。
> 建议学习顺序：`Install` -> `Quickstart` -> `Thinking in LangGraph`。

---

## 1. 这批官方文档在讲什么

当前 `Get-started` 目录中的重点页面有：

1. `Install.html`
2. `Quickstart.html`
3. `Thinking-in-LangGraph.html`
4. `Workflows-and-agents.html`
5. `Run-a-local-server.html`
6. `Changelog.html`

如果目标是“系统学习 LangGraph”，最重要的主线不是按文件名机械阅读，而是按下面的认知顺序学习：

1. 先知道 LangGraph 怎么安装、依赖什么。
2. 再通过一个最小可运行示例理解“图是怎么跑起来的”。
3. 然后建立 LangGraph 的核心思维方式：节点、状态、路由、错误处理、人类介入。
4. 最后再看各种 workflow/agent 模式，理解它为什么适合复杂 AI 应用。

这也是本套笔记的组织方式。

---

## 2. LangGraph 是什么

根据官方文档的整体表述，LangGraph 的核心定位可以概括为：

- 它是一个用“图”来组织 LLM 应用、Agent 和工作流的框架。
- 你把一个复杂任务拆成多个节点 `node`。
- 节点之间通过边 `edge` 连接。
- 所有节点共享一个状态 `state`，每个节点既能读状态，也能向状态写回更新。
- 图执行时可以根据节点输出动态决定下一步去哪，这使它既能表示固定工作流，也能表示更灵活的 agent。

如果用一句更适合学习者的话来解释：

> LangGraph 的本质，不是“再造一个聊天机器人框架”，而是给 Agent 系统提供一个可控制、可观察、可恢复、可暂停、可继续执行的运行时结构。

这也是它和“单次 prompt 调用”最大的不同。

---

## 3. 安装：官方建议与隐含前提

### 3.1 基础安装

官方 `Install` 页面给出的最基础安装方式是：

```bash
pip install -U langgraph
```

或：

```bash
uv add langgraph
```

如果你准备按官方示例继续学习，通常还会安装 `langchain`：

```bash
pip install -U langchain
# Requires Python 3.10+
```

或：

```bash
uv add langchain
# Requires Python 3.10+
```

这说明：

- `langgraph` 本身是核心框架包。
- 只装它可以开始使用图能力。
- 但要真正做 agent，通常还需要接入模型与工具。

### 3.2 为什么官方文档经常同时安装 LangChain

官方文档明确提到，使用 LangGraph 时，通常还需要：

- 访问 LLM
- 定义工具

文档里推荐的一种常见方式是配合 LangChain 使用，因此还会安装：

```bash
pip install -U langchain
```

这背后的含义是：

- LangGraph 负责“图式编排”和运行机制。
- LangChain 负责“模型抽象、消息对象、工具机制、结构化输出”等常见组件。

所以在学习上，最好把两者关系理解成：

- LangChain：更像模型和工具层
- LangGraph：更像执行编排层

### 3.3 模型供应商包要单独安装

官方文档特别提醒：

- 如果你要接具体模型供应商，还得安装各自的 provider 包。
- 比如 Anthropic、OpenAI 等都需要单独装。

这对初学者很重要，因为很多“跑不起来”的问题都不是 LangGraph 本身的问题，而是：

1. 没装 provider 包
2. 没配 API Key
3. Python 版本不对

### 3.4 学习阶段的建议环境

结合官方文档内容，建议你准备：

- Python 3.10+ 用于常规 LangGraph / LangChain 示例
- Python 3.11+ 用于本地 LangGraph CLI / 本地 server 部分
- 独立虚拟环境
- 至少一个可调用的模型 API Key

---

## 4. Quickstart：第一个 LangGraph Agent 到底在做什么

官方 `Quickstart` 用“计算器 Agent”演示了两种写法：

1. Graph API
2. Functional API

这篇文档非常关键，因为它不是只教你“怎么写代码”，而是在隐含地告诉你：一个 LangGraph Agent 的最小闭环是什么。

### 4.1 Quickstart 的任务结构

示例中的 agent 要做的事非常简单：

- 接收用户问题
- 让模型决定是否需要调用工具
- 如果需要，就调用数学工具
- 再把结果交回模型
- 重复，直到模型不再需要工具
- 输出最终答案

你会发现，这其实就是很多工具型 agent 的基本循环：

1. LLM 思考
2. 是否用工具
3. 调用工具
4. 将工具结果反馈给 LLM
5. 结束或继续

### 4.2 第一步：定义模型和工具

官方示例先定义：

- `add`
- `multiply`
- `divide`

再把这些工具绑定给模型：

```python
model_with_tools = model.bind_tools(tools)
```

这里的学习重点不是数学本身，而是理解：

- 工具不是图的一部分之前，LLM 只是普通模型。
- 把工具绑到模型上后，模型开始具备“生成工具调用意图”的能力。
- 但“真正执行工具”这件事，不是模型自己完成的，而是图中的某个节点负责执行。

这是 Agent 设计中一个非常重要的职责分离：

- 模型负责“决定做什么”
- 节点负责“真的去做”

下面是官方 quickstart 中最核心的模型与工具定义代码：

```python
from langchain.tools import tool
from langchain.chat_models import init_chat_model


model = init_chat_model(
    "claude-sonnet-4-6",
    temperature=0
)


@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`."""
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Adds `a` and `b`."""
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide `a` and `b`."""
    return a / b


tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)
```

这段代码在图中的角色是：

- `model` 是推理核心
- `tools` 是可执行动作
- `model_with_tools` 是“知道可以调用哪些工具”的模型

### 4.3 第二步：定义状态

官方示例定义了一个 `MessagesState`，至少包含：

- `messages`
- `llm_calls`

这里最核心的点在于 `messages` 的设计：

```python
messages: Annotated[list[AnyMessage], operator.add]
```

官方解释的重点是：

- 状态会在整个图执行期间持续存在。
- `Annotated + operator.add` 让新消息不是覆盖旧消息，而是追加到原消息列表。

这就是 LangGraph 中“状态累积”的典型写法。

从学习角度看，你需要真正明白两件事：

1. `state` 不是临时变量，它是整个 graph 的共享记忆。
2. 消息历史之所以能连续存在，是因为状态更新被设计成“追加式”而不是“替换式”。

官方 quickstart 的状态定义代码如下：

```python
from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
```

### 4.4 第三步：模型节点

模型节点 `llm_call` 的作用是：

- 读取当前 `state["messages"]`
- 在前面加一条系统提示
- 调用绑定了工具的模型
- 把模型新返回的一条消息写回状态

这个节点本质上代表：

> “让 LLM 基于当前上下文做下一次决策”

换句话说，图中的 LLM 节点通常不是“最终回答节点”，而是“决策节点”。

官方 quickstart 中的模型节点代码如下：

```python
from langchain.messages import SystemMessage


def llm_call(state: dict):
    """LLM decides whether to call a tool or not"""

    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }
```

### 4.5 第四步：工具节点

工具节点 `tool_node` 的作用是：

- 读取最后一条 AI 消息里的 `tool_calls`
- 找到对应工具
- 执行工具
- 把结果包装成 `ToolMessage`
- 追加回消息状态

这个节点代表的是：

> “把模型刚才提出的动作意图，真正转成可执行行为，并把结果反馈给系统”

这一步是 Agent 从“会说”到“会做”的关键。

官方 quickstart 的工具节点代码如下：

```python
from langchain.messages import ToolMessage


def tool_node(state: dict):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}
```

### 4.6 第五步：条件路由

官方的 `should_continue` 判断逻辑很简单：

- 如果最后一条消息包含工具调用，则跳去 `tool_node`
- 否则结束

这是 LangGraph 最典型的一个思想：

- 图的节点负责产出信息
- 图的边负责连接流程
- 条件边负责决定下一步路径

所以 LangGraph 并不是“写一个大 while 循环”，而是把循环结构显式建模成图。

官方 quickstart 中的条件路由函数如下：

```python
from typing import Literal
from langgraph.graph import StateGraph, START, END


def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop"""

    messages = state["messages"]
    last_message = messages[-1]

    if last_message.tool_calls:
        return "tool_node"

    return END
```

### 4.7 第六步：编译图

官方通过 `StateGraph`：

1. 添加节点
2. 添加开始边
3. 添加条件边
4. 添加工具节点返回模型节点的边
5. `compile()`

这说明 LangGraph 的标准构建流程大致是：

1. 定义状态 schema
2. 定义节点函数
3. 定义边和分支
4. 编译成可运行图

这个流程建议你牢牢记住，因为之后几乎所有高级模式都只是这四步的扩展。

官方 quickstart 中 Graph API 的组图、编译和调用代码如下：

```python
agent_builder = StateGraph(MessagesState)

agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)
agent_builder.add_edge("tool_node", "llm_call")

agent = agent_builder.compile()

from IPython.display import Image, display
display(Image(agent.get_graph(xray=True).draw_mermaid_png()))

from langchain.messages import HumanMessage
messages = [HumanMessage(content="Add 3 and 4.")]
messages = agent.invoke({"messages": messages})
for m in messages["messages"]:
    m.pretty_print()
```

如果你要把 quickstart 真正吃透，建议你先亲手把这段 Graph API 版本敲一遍。

---

## 5. Graph API 和 Functional API 的区别

官方 quickstart 特别强调了两种 API 风格。

### 5.1 Graph API

Graph API 更适合：

- 你明确想把系统拆成图
- 你希望直观看到节点和边
- 你更重视复杂控制流、显式路由和可视化结构

优点：

- 结构清晰
- 对复杂 agent 更自然
- 更容易映射到“工作流图”的思维模式

### 5.2 Functional API

Functional API 更适合：

- 你更喜欢函数式写法
- 希望保留部分顺序编程体验
- 任务逻辑还没有复杂到需要把所有结构显式展开成图

优点：

- 写法更紧凑
- 对简单任务上手更快

### 5.3 学习建议

如果你的目标是“真正学会 LangGraph”，建议优先掌握 Graph API。原因是：

- LangGraph 的核心心智模型本来就是 graph。
- Functional API 更像 Graph API 的另一种表达形式。
- 当进入条件路由、子图、持久化、中断恢复、复杂 agent 模式时，Graph API 的理解会更扎实。

官方 quickstart 也给出了 Functional API 版本。这里保留最关键的骨架代码，方便你对照两种写法：

```python
from langgraph.func import entrypoint, task
from langgraph.graph import add_messages
from langchain.messages import SystemMessage, HumanMessage, ToolCall
from langchain_core.messages import BaseMessage


@task
def call_llm(messages: list[BaseMessage]):
    return llm_with_tools.invoke(
        [
            SystemMessage(
                content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
            )
        ]
        + messages
    )


@task
def call_tool(tool_call: ToolCall):
    tool = tools_by_name[tool_call["name"]]
    return tool.invoke(tool_call)


@entrypoint()
def agent(messages: list[BaseMessage]):
    llm_response = call_llm(messages).result()

    while True:
        if not llm_response.tool_calls:
            break

        tool_result_futures = [
            call_tool(tool_call) for tool_call in llm_response.tool_calls
        ]
        tool_results = [fut.result() for fut in tool_result_futures]
        messages = add_messages(messages, [llm_response, *tool_results])
        llm_response = call_llm(messages).result()

    messages = add_messages(messages, llm_response)
    return messages
```

---

## 6. Thinking in LangGraph：真正重要的是“建图思维”

`Thinking in LangGraph` 是这批资料里最值得反复读的一篇，因为它讲的是方法论。

官方通过“客户支持邮件 Agent”带出 LangGraph 的五步思考法。

### 6.1 第一步：先从你要自动化的流程开始

官方先给出业务需求，比如：

- 读取客户邮件
- 判断紧急度和主题
- 搜索相关文档
- 生成回复
- 必要时升级给人工
- 需要时安排后续跟进

这一步其实是在强调：

> 不要从“我该调用哪个框架 API”开始，而要先从“业务流程有哪些离散步骤”开始。

LangGraph 的第一原则是：先拆业务流程，再映射成节点。

### 6.2 第二步：把流程拆成离散步骤

官方把示例拆成以下节点：

- `Read Email`
- `Classify Intent`
- `Doc Search`
- `Bug Track`
- `Draft Reply`
- `Human Review`
- `Send Reply`

这里最重要的学习点是：

- 一个节点应当尽量只做一件事。
- 有些节点只负责处理。
- 有些节点还负责决定下一步去哪。

这会带来三个好处：

1. 更容易调试
2. 更容易重试
3. 更容易持久化和恢复

### 6.3 第三步：先判断每个节点属于什么类型

官方把节点大致分成四类：

1. LLM steps
2. Data steps
3. Action steps
4. User input steps

这个分类特别适合拿来做建模检查。

#### LLM steps

例如：

- 分类意图
- 草拟回复

这类节点通常要考虑：

- 静态 prompt
- 动态上下文
- 期望输出格式

#### Data steps

例如：

- 搜索文档
- 查询客户历史

这类节点通常要考虑：

- 查询参数
- 重试策略
- 缓存策略

#### Action steps

例如：

- 发邮件
- 创建 bug 工单

这类节点通常要考虑：

- 是否幂等
- 失败后是否可重试
- 是否需要补偿机制

#### User input steps

例如：

- 人工审核

这类节点通常要考虑：

- 暂停点要暴露哪些上下文
- 人类返回数据格式是什么
- 审核之后路由去哪里

这个分类法是官方文档里非常实用的一部分，我建议你以后设计任何 agent 都先做一次这种分类。

---

## 7. State 设计：这是 LangGraph 最关键的能力之一

官方把 state 称为所有节点都能访问的共享记忆。

### 7.1 什么应该放进 state

官方给出的判断原则非常重要：

- 如果数据需要跨步骤持久存在，就放进 state。
- 如果数据能从别的信息推导出来，就尽量不要存，按需计算。

例如在邮件 agent 里，适合放进 state 的有：

- 原始邮件内容
- 发件人信息
- 分类结果
- 检索结果
- 客户历史
- 草稿回复
- 执行元数据

### 7.2 官方特别强调：state 存原始数据，不存格式化 prompt

这是整篇文档最值得记的一条原则之一：

> 在 state 中存 raw data，不存为了某个 prompt 临时拼好的文本。

这样做的好处有：

1. 不同节点可以按自己的方式格式化同一份数据。
2. 改 prompt 模板时，不需要改 state 结构。
3. 调试时更清楚地看到真实输入数据。
4. 系统演进时更稳。

### 7.3 为什么这条原则非常重要

从工程角度补充一点经验：

- 把格式化文本长期放进 state，会让状态变脆。
- prompt 一改，旧状态可能就不兼容。
- 还会造成状态冗余膨胀，特别是在长线程或多轮执行里。

所以官方这条建议不仅是“代码风格”，也是长期维护能力的一部分。

官方在 `Thinking in LangGraph` 中给出的邮件 Agent 状态定义很值得参考：

```python
from typing import TypedDict, Literal


class EmailClassification(TypedDict):
    intent: Literal["question", "bug", "billing", "feature", "complex"]
    urgency: Literal["low", "medium", "high", "critical"]
    topic: str
    summary: str


class EmailAgentState(TypedDict):
    email_content: str
    sender_email: str
    email_id: str

    classification: EmailClassification | None

    search_results: list[str] | None
    customer_history: dict | None

    draft_response: str | None
    messages: list[str] | None
```

---

## 8. 节点不是抽象对象，本质上就是函数

官方明确说：

> A node in LangGraph is just a Python function that takes the current state and returns updates to it.

这句话可以直接翻译为：

- 节点本质上就是函数。
- 输入是当前状态。
- 输出是对状态的更新，或者是“更新 + 路由命令”。

这意味着你在写 LangGraph 时，不需要把节点想得过于神秘。
真正需要你花心思的，是：

1. 这个函数的职责边界是否清楚
2. 它依赖 state 的哪些字段
3. 它可能失败在哪
4. 它完成后应该把控制权交给谁

---

## 9. 错误处理：官方给出的四类错误思维非常实用

`Thinking in LangGraph` 对错误处理讲得很系统，这一点非常值得学。

官方把错误大致分成几类：

1. 瞬时错误
2. LLM 可恢复错误
3. 用户可修复错误
4. 重试后仍失败但可恢复的错误
5. 意外错误

### 9.1 瞬时错误

例如：

- 网络波动
- 限流

策略：

- 自动重试

### 9.2 LLM 可恢复错误

例如：

- 工具失败
- 解析失败

策略：

- 把错误写入 state
- 再回到模型，让模型根据错误信息调整策略

### 9.3 用户可修复错误

例如：

- 缺信息
- 指令模糊

策略：

- 用 `interrupt()` 暂停
- 等待用户/人工补全

### 9.4 重试耗尽后的可恢复错误

策略：

- 使用 `error_handler`
- 走补偿或恢复分支

### 9.5 意外错误

策略：

- 直接抛出
- 让开发者调试，而不是强行吞掉

这个分类特别像“生产级 Agent 的故障地图”。很多初学者写 agent 时会把所有异常都 `try/except` 掉，结果系统表面稳定，实际上丢信息、难调试。官方文档的态度更成熟：

- 能自动恢复的，就自动恢复
- 需要人处理的，就明确停下
- 真异常就暴露出来

官方在组图时也给了一个很直接的 `RetryPolicy` 示例：

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import RetryPolicy

workflow = StateGraph(EmailAgentState)

workflow.add_node("read_email", read_email)
workflow.add_node("classify_intent", classify_intent)
workflow.add_node(
    "search_documentation",
    search_documentation,
    retry_policy=RetryPolicy(max_attempts=3)
)
workflow.add_node("bug_tracking", bug_tracking)
workflow.add_node("draft_response", draft_response)
workflow.add_node("human_review", human_review)
workflow.add_node("send_reply", send_reply)

workflow.add_edge(START, "read_email")
workflow.add_edge("read_email", "classify_intent")
workflow.add_edge("send_reply", END)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
```

---

## 10. Human-in-the-loop：人工介入不是补丁，而是一等能力

官方在 `Thinking in LangGraph` 中用 `interrupt()` 演示人工审核。

这里的核心思想是：

- 图执行到某一节点时可以暂停
- 状态会被保存
- 以后可以基于同一线程恢复

这和很多“把人类审核写成额外接口逻辑”的做法不一样。LangGraph 把人类介入作为图执行流程本身的一部分。

### 10.1 `interrupt()` 的意义

它不是简单的 `input()`，而是：

- 可持久化的暂停点
- 可恢复的执行断点
- 有明确上下文的数据交换点

### 10.2 官方特别提醒的一点

如果一个节点中同时有：

- `interrupt()`
- 其他逻辑

那 `interrupt()` 应尽量放在最前面，因为恢复时该节点前面的代码可能会重新执行。

这条看起来像小细节，其实很关键。尤其是节点里如果有副作用操作，放错顺序会造成重复执行。

官方邮件 Agent 中 `human_review` 节点的关键写法如下：

```python
from langgraph.types import Command, interrupt
from typing import Literal


def human_review(state: EmailAgentState) -> Command[Literal["send_reply", END]]:
    """Pause for human review using interrupt and route based on decision"""

    classification = state.get("classification", {})

    human_decision = interrupt({
        "email_id": state.get("email_id", ""),
        "original_email": state.get("email_content", ""),
        "draft_response": state.get("draft_response", ""),
        "urgency": classification.get("urgency"),
        "intent": classification.get("intent"),
        "action": "Please review and approve/edit this response"
    })

    if human_decision.get("approved"):
        return Command(
            update={
                "draft_response": human_decision.get(
                    "edited_response",
                    state.get("draft_response", "")
                )
            },
            goto="send_reply"
        )
    else:
        return Command(update={}, goto=END)
```

这段代码非常值得细看，因为它展示了 `interrupt()`、`Command`、人工决策、恢复后路由这几个 LangGraph 的核心点是如何放在一起工作的。

---

## 11. Step 5：为什么图结构往往可以很“少”

官方在邮件 agent 示例里有个很重要的观点：

- 图中只保留必要的边
- 更复杂的路由可以由节点通过 `Command` 返回来控制

也就是说，LangGraph 不要求你把所有路由逻辑都塞在边上。

一种常见模式是：

- 基础边只负责主流程连通
- 节点在运行时通过 `Command(update=..., goto=...)` 同时完成：
  - 更新状态
  - 指定下一跳

这会带来一个好处：

> 图结构保持简洁，而动态路由逻辑仍然是显式且可追踪的。

---

## 12. Checkpointer 与持久化：为什么 LangGraph 适合长流程 Agent

官方在 `Thinking in LangGraph` 中提到，想使用 human-in-the-loop，需要编译时配 `checkpointer`。

例如：

```python
app = workflow.compile(checkpointer=memory)
```

这背后的意义是：

- LangGraph 可以在节点边界保存执行状态
- 中断后可以恢复
- 失败后可以从某个检查点继续

这和很多“单次请求式”的 LLM 程序差异非常大。

从学习角度，你可以把它理解为：

- 普通脚本：更像一次性运行
- LangGraph：更像“可暂停、可恢复的流程机”

官方在 `Thinking in LangGraph` 里还给了一个非常适合学习的最小 `interrupt()` 测试示例：

```python
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class EmailState(TypedDict):
    email_content: str
    response_text: str | None


def human_review_node(state: EmailState):
    interrupt(
        {
            "approved": False,
            "edited_response": state.get("response_text") or "",
        }
    )
    return {"response_text": "placeholder"}


app = (
    StateGraph(EmailState)
    .add_node("human_review", human_review_node)
    .add_edge(START, "human_review")
    .add_edge("human_review", END)
    .compile(checkpointer=InMemorySaver())
)

initial_state = {
    "email_content": "I was charged twice for my subscription! This is urgent!",
    "response_text": "Draft response",
}

config = {"configurable": {"thread_id": "customer_123"}}
stream = app.stream_events(initial_state, config, version="v3")
_ = stream.output
print(f"human review interrupt:{stream.interrupts}")

human_response = Command(
    resume={
        "approved": True,
        "edited_response": "We sincerely apologize for the double charge. I've initiated an immediate refund...",
    }
)

resumed = app.stream_events(human_response, config, version="v3")
final_state = resumed.output
print("Email sent successfully!")
```

---

## 13. 这一篇文档最值得记住的五条原则

综合 `Thinking in LangGraph`，最值得反复记忆的是：

1. 先拆业务流程，再写框架代码。
2. 节点尽量单一职责。
3. state 存原始数据，不存 prompt 成品。
4. 错误是流程的一部分，不是异常角落。
5. 人类输入应被纳入图的显式控制流。

如果你真正掌握了这五条，后面看复杂模式时会轻松很多。

---

## 14. 针对这批文档的学习建议

### 14.1 第一轮学习目标

第一轮不要追求“会用所有 API”，只追求三件事：

1. 明白什么是 node / edge / state
2. 能看懂 quickstart 的 tool loop
3. 能自己画出一个小业务流程图

### 14.2 第二轮学习目标

第二轮重点是：

1. 学会判断 state 里该放什么
2. 学会把流程拆成更合理的节点
3. 学会区分固定 workflow 和动态 agent

### 14.3 第三轮学习目标

第三轮再进入：

1. 持久化
2. 中断恢复
3. 本地 server
4. 更复杂的 workflow patterns

---

## 15. 这一篇笔记对应的官方文档来源

本篇主要参考以下官方页面：

1. `docs/Get-started/Install.html`
2. `docs/Get-started/Quickstart.html`
3. `docs/Get-started/Thinking-in-LangGraph.html`

其中：

- 安装部分主要来自 `Install`
- 第一个 agent 的结构主要来自 `Quickstart`
- 方法论、状态设计、错误处理、人类介入主要来自 `Thinking in LangGraph`

---

## 16. 下一步应该读什么

建议紧接着阅读第二篇笔记，重点进入：

- workflow 与 agent 的区别
- Prompt chaining / Parallelization / Routing 等模式
- Orchestrator-worker
- Evaluator-optimizer
- ToolNode
- 本地 server 与调试入口

这部分会更偏“模式库”和“工程实践”。
