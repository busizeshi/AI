import os

from langchain_classic.chains import ConversationChain
from langchain_classic.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI


def create_qwen_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model="qwen-turbo",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0.3,
        max_tokens=600,
    )


def main() -> None:
    llm = create_qwen_llm()

    memory = ConversationSummaryMemory(
        llm=llm,
        memory_key="history",
    )

    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True,
    )

    print(conversation.invoke({"input": "你好，我正在学习 LangChain"})["response"])
    print(conversation.invoke({"input": "我已经学完了链，现在学习 Memory"})["response"])
    print(conversation.invoke({"input": "请问我现在学习到了哪个模块？"})["response"])

    print("当前摘要：")
    print(memory.load_memory_variables({})["history"])


if __name__ == "__main__":
    main()
