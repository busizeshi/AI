import os

import openai
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def generate_ai_text(topic: str) -> str | None:
    prompt = f"""
    请写一段关于“{topic}”的介绍性文字。
    要求：
    1. 使用中文
    2. 面向初学者
    3. 长度控制在 120 到 180 字
    4. 风格正式、清晰
    """
    try:
        response = client.chat.completions.create(
            model="qwen-turbo",
            messages=[{"role": "user", "content": prompt}],
        )

        return response.choices[0].message.content

    except openai.AuthenticationError as e:
        print(f"认证失败 (API Key 错误): {e}")
    except openai.BadRequestError as e:
        print(f"请求格式错误: {e}")
    except openai.APIConnectionError as e:
        print(f"网络连接失败: {e}")
    except openai.RateLimitError as e:
        print(f"请求频率过高 (限流或欠费): {e}")
    except openai.APIError as e:
        print(f"OpenAI 格式 API 错误: {e}")

    return None


result = generate_ai_text("给我写一段快速排序")
if result:
    print(result)

