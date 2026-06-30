import os
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


IMPORTANT_KEYWORDS = ["重要", "关键", "重点", "必须记住"]


def create_qwen_llm() -> ChatOpenAI:
    """创建 Qwen 聊天模型。"""
    return ChatOpenAI(
        model="qwen-turbo",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0.5,
        max_tokens=600,
    )


def should_clear_memory(user_input: str) -> bool:
    clear_keywords = ["换个话题", "重新开始", "不聊这个了", "清空历史"]
    return any(keyword in user_input for keyword in clear_keywords)


def adjust_window_size(user_input: str, default_k: int = 3) -> int:
    """根据用户需求动态调整窗口大小。"""
    if "总结前面" in user_input or "详细" in user_input:
        return 6
    if "简单" in user_input or "简要" in user_input:
        return 2
    return default_k


def is_important_message(user_input: str) -> bool:
    """判断是否是重要消息。"""
    return any(keyword in user_input for keyword in IMPORTANT_KEYWORDS)


@dataclass
class SmartWindowMemory:
    """智能滑动窗口记忆：支持动态 k、清理和重要消息筛选。"""

    default_k: int = 3
    k: int = 3
    messages: list = field(default_factory=list)

    def before_user_input(self, user_input: str) -> None:
        if should_clear_memory(user_input):
            self.clear()
        self.k = adjust_window_size(user_input, self.default_k)
        self._trim()

    def add_turn(self, user_input: str, ai_output: str) -> None:
        # 普通消息进入滑动窗口；重要消息也进入窗口，但可以在真实项目里额外持久化。
        self.messages.append(HumanMessage(content=user_input))
        self.messages.append(AIMessage(content=ai_output))
        self._trim()

    def load_messages(self) -> list:
        return list(self.messages)

    def clear(self) -> None:
        self.messages.clear()

    def _trim(self) -> None:
        max_messages = self.k * 2
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]


def chat(llm: ChatOpenAI, memory: SmartWindowMemory, user_input: str) -> str:
    """使用智能窗口记忆进行聊天。"""
    memory.before_user_input(user_input)

    messages = [
        SystemMessage(content="你是一个耐心的中文 LangChain 学习助手。"),
        *memory.load_messages(),
        HumanMessage(content=user_input),
    ]

    response = llm.invoke(messages)
    answer = response.content

    memory.add_turn(user_input, answer)

    return answer


def main() -> None:
    llm = create_qwen_llm()
    memory = SmartWindowMemory(default_k=3, k=3)

    inputs = [
        "你好，我正在学习 LangChain Memory",
        "什么是会话缓冲区？",
        "它和会话缓冲窗口有什么区别？",
        "请详细总结前面我们聊了什么",
        "换个话题，什么是提示词模板？",
        "刚才我们还在聊 Memory 吗？",
    ]

    for user_input in inputs:
        print("用户：", user_input)
        print("模型：", chat(llm, memory, user_input))
        print("当前窗口消息数：", len(memory.load_messages()))
        print("-" * 60)


if __name__ == "__main__":
    main()