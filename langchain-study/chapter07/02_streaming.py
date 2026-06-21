import sys
import time
from collections.abc import Iterator

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from llm_client import stream_prompt_value


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def codex_streaming_model(prompt_value) -> Iterator[str]:
    yield from stream_prompt_value(prompt_value, temperature=0.7)


def build_streaming_chain():
    prompt = ChatPromptTemplate.from_template("请用一句话解释：{topic}")
    model = RunnableLambda(codex_streaming_model)
    return prompt | model | StrOutputParser()


def stream_callback(chunk: str) -> None:
    print(chunk, end="", flush=True)


def main():
    chain = build_streaming_chain()

    print("== 实时打印 chunk ==")
    chunks = []
    for chunk in chain.stream({"topic": "哈士奇"}):
        chunks.append(chunk)
        stream_callback(chunk)
        time.sleep(0.02)

    print("\n\n== 聚合后的完整结果 ==")
    print("".join(chunks))


if __name__ == "__main__":
    main()
