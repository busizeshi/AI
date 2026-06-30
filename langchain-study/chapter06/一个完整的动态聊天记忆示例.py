import os
from dataclasses import dataclass, field
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


def create_qwen_llm() -> ChatOpenAI:
    """创建 Qwen 模型。"""
    return ChatOpenAI(
        model="qwen-turbo",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0.5,
        max_tokens=600,
    )


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H%M%S")


def message_priority(text: str) -> str:
    high_keywords = ["必须记住", "关键", "重要", "重点"]
    medium_keywords = ["偏好", "希望", "要求", "喜欢", "记住"]

    if any(word in text for word in high_keywords):
        return "high"
    if any(word in text for word in medium_keywords):
        return "medium"
    return "low"


def is_topic_changed(text: str) -> bool:
    switch_keywords = ["换个话题", "另一个问题", "不聊这个了", "重新开始"]
    return any(keyword in text for keyword in switch_keywords)


@dataclass
class DynamicChatMemory:
    """
    动态聊天记忆，支持优先级，窗口剪裁，话题切换清理
    """
    max_messages: int = 8
    messages: list = field(default_factory=list)

    def add_user_message(self, content: str) -> None:
        if is_topic_changed(content):
            self.clear()

        priority = message_priority(content)
        if priority in ["high", "medium"]:
            self.messages.append(
                HumanMessage(content=f"[{now_text()}][{priority}] {content}")
            )
            self._trim()

    def add_ai_message(self, content: str) -> None:
        self.messages.append(AIMessage(content=f"[{now_text()}] {content}"))
        self._trim()

    def load_messages(self) -> list:
        return list(self.messages)

    def clear(self) -> None:
        self.messages.clear()

    def _trim(self) -> None:
        if len(self.messages) >= self.max_messages:
            self.messages = self.messages[-self.max_messages]


def chat(llm: ChatOpenAI, memory: DynamicChatMemory, user_input: str) -> str:
    """
    动态记忆聊天，根据策略保存重要内容
    :param llm:
    :param memory:
    :param user_input:
    :return:
    """
    messages = [
        SystemMessage(content="你是一个耐心的中文AI学习助手"),
        *memory.load_messages(),
        HumanMessage(content=user_input)
    ]

    response = llm.invoke(messages)
    answer = response.content

    memory.add_user_message(user_input)
    memory.add_ai_message(answer)

    return answer


def main():
    llm = create_qwen_llm()
    memory = DynamicChatMemory(max_messages=8)

    print(chat(llm, memory, "你好，我叫小明"))
    print(chat(llm, memory, "请记住，我正在学习 LangChain 的 Memory 模块"))
    print(chat(llm, memory, "我正在学习什么？"))
    print(chat(llm, memory, "换个话题，帮我解释一下什么是提示词"))


if __name__ == "__main__":
    main()
