import os

from langchain_classic.chains import LLMChain, SequentialChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


def create_qwen_llm(temperature: float = 0.5, max_tokens: int = 600) -> ChatOpenAI:
    """创建 Qwen 模型实例。"""
    return ChatOpenAI(
        model="qwen-turbo",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=temperature,
        max_tokens=max_tokens,
    )


llm = create_qwen_llm()

# 步骤1：生成主题概述
intro_prompt = PromptTemplate.from_template(
    "请用 3 句话介绍{topic}的基本概念。"
)

intro_chain = LLMChain(
    llm=llm,
    prompt=intro_prompt,
    output_key="intro",
)

# 步骤2：基于概述生成应用场景
application_prompt = PromptTemplate.from_template(
    """
已知主题：{topic}
已知概述：
{intro}

请列出这个主题的 3 个典型应用场景，并分别用一句话说明。
"""
)

application_chain = LLMChain(
    llm=llm,
    prompt=application_prompt,
    output_key="applications",
)

# 步骤3：基于应用场景生成详细说明
detail_prompt = PromptTemplate.from_template(
    """
主题：{topic}
应用场景：
{applications}

请选择其中一个应用场景，写一个适合初学者理解的详细案例。
"""
)

detail_chain = LLMChain(
    llm=llm,
    prompt=detail_prompt,
    output_key="detail_report",
)

overall_chain = SequentialChain(
    chains=[intro_chain, application_chain, detail_chain],
    input_variables=["topic"],
    output_variables=["intro", "applications", "detail_report"],
    verbose=True,
)

result = overall_chain.invoke({"topic": "机器学习"})

print("=== 主题概述 ===")
print(result["intro"])

print("=== 应用场景 ===")
print(result["applications"])

print("=== 详细案例 ===")
print(result["detail_report"])