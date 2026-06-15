import json
import openai
from openai import OpenAI
import redis


# =====================================================================
# 1. 轻量化 Redis 缓存组件
# =====================================================================
class RedisCache:

    def __init__(self, host="192.168.1.246", port=6379, db=0, password="localdeploymentegysaveredis222345", ttl=3600):
        # 初始化 Redis 客户端，decode_responses=True 确保拿到的结果直接是字符串
        self.client = redis.Redis(
            host=host, port=port, db=db, password=password, decode_responses=True
        )
        self.ttl = ttl

    def get(self, key: str):
        value = self.client.get(key)
        if value is None:
            return None
        # 将缓存中的 JSON 字符串还原
        return json.loads(value)

    def set(self, key: str, value: any):
        # 存入缓存并设置过期时间（TTL）
        self.client.setex(key, self.ttl, json.dumps(value, ensure_ascii=False))


# =====================================================================
# 2. 接入 Redis 缓存的自定义 Chat 模型类
# =====================================================================
class CustomChatModel:

    def __init__(self, model_name="gpt-5.5", temperature=0.3, cache=None):
        self.client = OpenAI()
        self.model_name = model_name
        self.temperature = temperature
        self.cache = cache  # 传入上面的 RedisCache 实例

    def generate(self, messages: list) -> str | None:
        """接收消息列表（包含上下文历史），返回模型的文本回应"""

        # 核心逻辑：Chat 类模型的缓存键通常由“模型名 + 整个消息队列”决定
        # 这样能确保即使同一个问题，但在不同的上下文历史（多轮对话）下，不会错误命中
        cache_key = f"{self.model_name}:{json.dumps(messages, ensure_ascii=False)}"

        # 【步骤 1】先查缓存
        if self.cache:
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                print("\n[⚡ 缓存命中] 直接从 Redis 读取结果，未消耗 Token！")
                return cached_result

        # 【步骤 2】缓存未命中，调用大模型 API
        print("\n[🤖 缓存未命中] 正在请求大模型 API...")
        try:
            response = self.client.responses.create(
                model=self.model_name,
                input=messages,  # Chat 类模型输入的是消息数组（包含开发者指令、历史上下文等）
                temperature=self.temperature,
            )
            result_text = response.output_text

            # 【步骤 3】将生成的新结果写回缓存
            if self.cache and result_text:
                self.cache.set(cache_key, result_text)

            return result_text

        except openai.OpenAIError as e:
            print(f"请求失败: {e}")
            return None
