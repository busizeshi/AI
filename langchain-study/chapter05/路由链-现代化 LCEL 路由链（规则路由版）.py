import os
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
from langchain_openai import ChatOpenAI


# 1. 初始化通义千问模型
def create_qwen_llm(temperature: float = 0.4, max_tokens: int = 500) -> ChatOpenAI:
    return ChatOpenAI(
        model="qwen-turbo",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=temperature,
        max_tokens=max_tokens,
    )


llm = create_qwen_llm()

# 2. 定义各个分支的 Prompt 链 (使用管道符 | )
complaint_chain = (
        PromptTemplate.from_template(
            "你是一名耐心的客服专员。用户正在投诉：{input}\n\n请先表达理解和歉意，再给出 3 个明确的处理步骤。")
        | llm
)

inquiry_chain = (
        PromptTemplate.from_template(
            "你是一名专业客服。用户正在咨询：{input}\n\n请直接、清晰地回答用户问题，如果信息不足，请说明需要补充什么。")
        | llm
)

suggestion_chain = (
        PromptTemplate.from_template(
            "你是一名产品经理助理。用户提出建议：{input}\n\n请感谢用户，并把建议整理成“建议内容、可能价值、后续动作”三部分。")
        | llm
)

default_chain = (
        PromptTemplate.from_template("用户输入：{input}\n\n请礼貌回应，并引导用户说明他是要咨询、投诉，还是提出建议。")
        | llm
)


# 3. 增强版规则判断函数
def determine_route(x: dict) -> str:
    """根据输入的字典数据提取 input 进行路由选择"""
    text = x["input"].strip()
    urgent_words = ["马上", "立刻", "紧急", "严重", "无法使用"]

    if "投诉" in text:
        return "complaint"
    if "咨询" in text or "请问" in text:
        return "inquiry"
    if "建议" in text:
        return "suggestion"
    return "default"


# 4. 【核心】使用高级条件分支函数将它们组合成一条大链
# 它会根据 determine_route 返回的字符串，动态路由到对应的子链
router_chain = RunnablePassthrough() | (lambda x: {
    "complaint": complaint_chain,
    "inquiry": inquiry_chain,
    "suggestion": suggestion_chain,
    "default": default_chain
}.get(determine_route(x), default_chain))

# 5. 测试运行
if __name__ == "__main__":
    test_inputs = [
        "请问怎么申请退款？",
        "我要投诉物流太慢了！",
        "我建议增加夜间模式",
        "你好啊"
    ]

    for user_input in test_inputs:
        print(f"\n👉 用户输入: {user_input}")
        # LCEL 默认返回消息对象，通过 .content 获取文本内容
        response = router_chain.invoke({"input": user_input})
        print(response.content)
        print("-" * 30)