import os
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


# 1. 结构化定义：强制让大模型选择其中一种标签
class RouteClassifier(BaseModel):
    """判断用户输入的意图分类标签"""
    destination: Literal["complaint", "inquiry", "suggestion", "default"] = Field(
        description="用户意图分类标签：投诉(complaint), 咨询(inquiry), 建议(suggestion), 其他(default)"
    )


# 2. 初始化用于分类的低成本千问模型 (temperature 设为 0 保证稳定)
classifier_llm = ChatOpenAI(
    model="qwen-turbo",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.0,
    max_tokens=20
).with_structured_output(RouteClassifier)  # 强制结构化输出


# 3. 核心路由分发函数
def smart_route(inputs: dict):
    user_input = inputs["input"]

    # 步骤 A：先让分类器模型给输入贴标签
    classifier_prompt = PromptTemplate.from_template(
        "请判断下面用户输入属于哪一类，必须严格分类：\n\n用户输入：{input}"
    )
    classification_chain = classifier_prompt | classifier_llm

    # 获取分类结果 (是一个 Pydantic 对象)
    decision = classification_chain.invoke({"input": user_input})
    route = decision.destination
    print(f"🤖 [模型AI决策] 识别到该输入意图为 -> {route}")

    # 步骤 B：初始化具体的业务模型
    biz_llm = ChatOpenAI(
        model="qwen-turbo",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0.5
    )

    # 步骤 C：根据标签，动态组装并直接返回对应的业务链
    if route == "complaint":
        prompt = PromptTemplate.from_template(
            "你是一名耐心的客服专员。用户投诉：{input}\n请先表达理解和歉意，再给出3个明确的处理步骤。")
    elif route == "inquiry":
        prompt = PromptTemplate.from_template("你是一名专业客服。用户咨询：{input}\n请直接、清晰地回答用户问题。")
    elif route == "suggestion":
        prompt = PromptTemplate.from_template(
            "你是一名产品经理助理。用户提出建议：{input}\n请感谢用户，并整理成“建议内容、可能价值”两部分。")
    else:
        prompt = PromptTemplate.from_template(
            "用户输入：{input}\n请礼貌回应，并引导用户说明他是要咨询、投诉，还是提出建议。")

    final_chain = prompt | biz_llm
    return final_chain.invoke({"input": user_input})


# 4. 测试运行
if __name__ == "__main__":
    # 故意不用关键字测试模糊语义
    fuzzy_inputs = [
        "买的东西快一个星期了还没发货，搞什么鬼啊？",  # 语义：投诉
        "我想了解一下怎么开发票",  # 语义：咨询
        "如果能在主页加一个搜索框体验会好很多",  # 语义：建议
        "今天天气真不错"  # 语义：默认
    ]

    for user_input in fuzzy_inputs:
        print(f"\n👉 原始输入: {user_input}")
        response = smart_route({"input": user_input})
        print(response.content)
        print("-" * 40)