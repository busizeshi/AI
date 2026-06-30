import os

# langchain 1.x 已移除 langchain.chains，旧版链式 API 需从兼容层导入
from langchain_classic.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.channels import topic


def create_qwen_llm() -> ChatOpenAI:
    """创建 Qwen 模型，使用 DashScope OpenAI-compatible 接口。"""
    return ChatOpenAI(
        model="qwen-turbo",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0.5,
        max_tokens=600,
        top_p=0.9,
    )


def preprocess_input(value: str) -> str:
    """
    清理输入内容
    :param value:
    :return:
    """
    cleaned = value.strip()
    cleaned = cleaned.replace("\n", "")
    cleaned = " ".join(cleaned.split())
    return cleaned


def postprocess_output(text: str) -> str:
    """
    清理输出内容
    :param text:
    :return:
    """
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return "\n".join(lines)


def create_application_chain() -> LLMChain:
    """创建一个用于讲解技术应用场景的 LLM链。"""
    llm = create_qwen_llm()

    prompt = PromptTemplate.from_template(
        """
你是一名耐心的 AI 学习教练。
请用适合初学者的方式说明{topic}在{field}领域的应用。

要求：
1. 先用一句话解释核心概念
2. 再列出 3 个典型应用
3. 最后给出一个具体例子
4. 语言简洁，不要堆术语
"""
    )

    return LLMChain(llm=llm, prompt=prompt)


def lcel() -> str:
    """
    lcel链式
    :return:
    """
    llm = create_qwen_llm()

    prompt = PromptTemplate.from_template(
        """
你是一名耐心的 AI 学习教练。
请用适合初学者的方式说明{topic}在{field}领域的应用。

要求：
1. 先用一句话解释核心概念
2. 再列出 3 个典型应用
3. 最后给出一个具体例子
4. 语言简洁，不要堆术语
"""
    )

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "topic": "机器学习",
        "field": "医疗领域",
    })


def main() -> None:
    """执行 LLM链：预处理输入，调用模型，后处理输出。"""
    # topic = preprocess_input("  机器学习  ")
    # field = preprocess_input("  医疗  ")
    #
    # chain = create_application_chain()
    # result = chain.invoke({
    #     "topic": topic,
    #     "field": field,
    # })
    #
    # final_text = postprocess_output(result["text"])
    # print(final_text)
    print(lcel())


if __name__ == "__main__":
    main()
