import os
from http.client import responses

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI


class SimpleChatMemory:
    """
    最小聊天消息记忆：保存用户消息和模型消息
    """

    def __init__(self) -> None:
        self.messages = []

    def add_user_message(self, content: str) -> None:
        self.messages.append(HumanMessage(content=content))

    def add_ai_message(self, content: str) -> None:
        self.messages.append(AIMessage(content=content))

    def clear(self) -> None:
        self.messages.clear()

    def load_messages(self):
        return list(self.messages)


def create_qwen_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model="qwen-turbo",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0.5,
    )


def chat(llm: ChatOpenAI, memory: SimpleChatMemory, user_input: str) -> str:
    """
    带记忆的聊天
    :param llm:chat model
    :param memory:会话记忆
    :param user_input:用户输入
    :return:llm回复
    """
    messages = [
        SystemMessage(content="你是一个耐心的中文AI学习助手"),
        *memory.load_messages(),
        HumanMessage(content=user_input),
    ]

    response = llm.invoke(messages)
    answer = response.content

    memory.add_user_message(user_input)
    memory.add_ai_message(answer)

    return answer


def main() -> None:
    llm = create_qwen_llm()
    memory = SimpleChatMemory()

    print(chat(llm, memory, "你好，我的小猫叫fafa"))
    print(chat(llm, memory, "我的小猫叫什么名字？"))


if __name__ == "__main__":
    main()
