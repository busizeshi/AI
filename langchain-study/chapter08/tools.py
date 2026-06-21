import ast
import operator
from dataclasses import dataclass


KNOWLEDGE_BASE = {
    "太空探索": "太空探索是人类通过探测器、卫星和载人航天研究外太空的活动，目标包括理解行星、恒星、星系和宇宙环境。",
    "智能手机": "智能手机适合需要多任务处理、移动办公、即时通信和高质量拍摄的用户，是实用的日常设备。",
    "LCEL": "LCEL 是 LangChain Expression Language，用于把提示词、模型、解析器和工具组合成可运行任务流。",
    "ReAct": "ReAct 是 Reasoning and Acting 的缩写，让模型通过推理选择工具，并根据工具观察结果继续行动。",
    "北京空气质量": "示例数据：北京空气质量需要查询实时服务。本地知识库只提供演示文本，不代表实时数据。",
    "东京天气": "示例数据：东京天气需要查询实时服务。本地知识库只提供演示文本，不代表实时天气。",
}


DOCSTORE = {
    "莎士比亚": (
        "威廉·莎士比亚是英国剧作家和诗人。关于他的第一部作品，学界常提到早期剧作"
        "《亨利六世》系列或《泰特斯·安德洛尼克斯》，具体排序存在争议。"
    ),
    "ReAct文档存储库": (
        "ReAct 文档存储库通过 Search 和 Lookup 工具支持 Agent 先搜索相关条目，"
        "再在条目中精确查找信息。"
    ),
}


PRODUCTS = {
    "智能手机": "适合移动办公、拍照、社交和多任务处理，推荐给需要高频移动使用的用户。",
    "轻薄笔记本": "适合编程、写作、学习和出差办公，推荐给重视便携性的用户。",
    "降噪耳机": "适合通勤、学习和开放办公环境，推荐给需要专注和低噪声体验的用户。",
}


_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def search(query: str) -> str:
    for key, value in KNOWLEDGE_BASE.items():
        if key.lower() in query.lower() or query.lower() in key.lower():
            return value
    return "没有找到精确资料。请换一个关键词，或接入真实搜索 API。"


def lookup(topic: str) -> str:
    for key, value in DOCSTORE.items():
        if key in topic or topic in key:
            return value
    return "文档库中没有该条目的精确资料。"


def search_product_info(product_name: str) -> str:
    return PRODUCTS.get(product_name, "没有找到该产品信息。")


def safe_calculator(expression: str) -> str:
    def eval_node(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
            return _OPERATORS[type(node.op)](eval_node(node.left), eval_node(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
            return _OPERATORS[type(node.op)](eval_node(node.operand))
        raise ValueError("只支持基础数学表达式。")

    tree = ast.parse(expression, mode="eval")
    return str(eval_node(tree.body))


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    func: callable


TOOLS = {
    "search": Tool("search", "根据关键词检索本地知识库。", search),
    "calculator": Tool("calculator", "计算基础数学表达式，例如 17 * 23。", safe_calculator),
    "lookup": Tool("lookup", "在文档存储库中精确查找条目。", lookup),
}

