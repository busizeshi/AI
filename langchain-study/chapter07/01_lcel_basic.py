import sys

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from llm_client import call_prompt_value


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def codex_chat_model(prompt_value) -> str:
    return call_prompt_value(prompt_value, temperature=0.3)


def local_retriever(question: str) -> str:
    knowledge = {
        "熊猫喜欢吃什么？": "熊猫喜欢吃竹子，也会少量摄入其他植物。",
        "LCEL是什么？": "LCEL 是 LangChain Expression Language，用于组合 Runnable 任务流。",
    }
    return knowledge.get(question, "没有检索到相关资料。")


def build_basic_chain():
    prompt = ChatPromptTemplate.from_template("请围绕「{topic}」讲一个简短笑话。")
    model = RunnableLambda(codex_chat_model)
    output_parser = StrOutputParser()
    return prompt | model | output_parser


def build_rag_chain():
    model = RunnableLambda(codex_chat_model)
    output_parser = StrOutputParser()

    def add_context(question: str) -> dict:
        return {
            "question": question,
            "context": local_retriever(question),
        }

    prompt = ChatPromptTemplate.from_template(
        "请基于资料回答问题。\n资料：{context}\n问题：{question}"
    )
    return RunnableLambda(add_context) | prompt | model | output_parser


def main():
    print("== 基础 LCEL 链 ==")
    basic_chain = build_basic_chain()
    print(basic_chain.invoke({"topic": "冰淇淋"}))

    print("\n== 分步调试 ==")
    prompt = ChatPromptTemplate.from_template("请围绕「{topic}」讲一个简短笑话。")
    formatted_prompt = prompt.invoke({"topic": "冰淇淋"})
    print(formatted_prompt.to_string())

    print("\n== 简化 RAG 链 ==")
    rag_chain = build_rag_chain()
    print(rag_chain.invoke("熊猫喜欢吃什么？"))


if __name__ == "__main__":
    main()
