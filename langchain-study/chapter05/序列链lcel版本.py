import os

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
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

# ==========================================
# 1. 定义三个步骤的 提示词
# ==========================================
intro_prompt = PromptTemplate.from_template("请用 3 句话介绍{topic}的基本概念。")

application_prompt = PromptTemplate.from_template(
    """
已知主题：{topic}
已知概述：
{intro}

请列出这个主题的 3 个典型应用场景，并分别用一句话说明。
"""
)

detail_prompt = PromptTemplate.from_template(
    """
主题：{topic}
应用场景：
{applications}

请选择其中一个应用场景，写一个适合初学者理解的详细案例。
"""
)


# ==========================================
# 2. 用 LCEL 组装顺序链 (核心部分)
# ==========================================

# 步骤 1：输入只有 topic，输出 {"topic": ..., "intro": str}
step_1 = RunnablePassthrough.assign(
    intro=intro_prompt | llm | (lambda x: x.content)
)

# 步骤 2：在上一步的结果上，追加 applications，输出 {"topic": ..., "intro": ..., "applications": str}
step_2 = RunnablePassthrough.assign(
    applications=application_prompt | llm | (lambda x: x.content)
)

# 步骤 3：在上一步的结果上，追加 detail_report，输出最终包含所有 key 的大字典
step_3 = RunnablePassthrough.assign(
    detail_report=detail_prompt | llm | (lambda x: x.content)
)

# 将整个流水线串联起来
overall_chain = step_1 | step_2 | step_3


# ==========================================
# 3. 运行并输出结果
# ==========================================
result = overall_chain.invoke({"topic": "机器学习"})

print("=== 主题概述 ===")
print(result["intro"])

print("=== 应用场景 ===")
print(result["applications"])

print("=== 详细案例 ===")
print(result["detail_report"])