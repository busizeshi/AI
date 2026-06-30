import asyncio
import socket
import sys
from pathlib import Path

from langgraph_sdk import get_client

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from src.agent import graph


SERVER_URL = "http://localhost:2024"
INPUT_PAYLOAD = {
    "messages": [{"role": "human", "content": "你好，请问你是？"}]
}


def is_server_running(host: str = "127.0.0.1", port: int = 2024) -> bool:
    with socket.socket() as sock:
        sock.settimeout(1)
        return sock.connect_ex((host, port)) == 0


def print_last_message(messages):
    if not messages:
        return

    last_msg = messages[-1]
    msg_type = getattr(last_msg, "type", None)
    content = getattr(last_msg, "content", None)

    if isinstance(last_msg, dict):
        msg_type = last_msg.get("type") or last_msg.get("role")
        content = last_msg.get("content")

    print(f"   ├─ 角色: {msg_type}")
    print(f"   └─ 内容: {content}\n")


async def stream_from_server():
    client = get_client(url=SERVER_URL)
    print("====== 正在向 LangGraph Server 发起异步流式请求 ======\n")

    async for chunk in client.runs.stream(
        thread_id=None,
        assistant_id="agent",
        input=INPUT_PAYLOAD,
        stream_mode="values",
    ):
        print(f"【收到事件】: {chunk.event}")
        if chunk.event == "values" and "messages" in chunk.data:
            print_last_message(chunk.data["messages"])


async def stream_locally():
    print("====== 未检测到 LangGraph Server，改为本地直接运行 graph ======\n")
    print("如需体验 SDK + Server 方式，请先安装并启动 langgraph-cli：")
    print('   pip install -U "langgraph-cli[inmem]"')
    print("   langgraph dev\n")

    async for state in graph.astream(INPUT_PAYLOAD, stream_mode="values"):
        print("【收到事件】: values")
        if "messages" in state:
            print_last_message(state["messages"])


async def main():
    if is_server_running():
        await stream_from_server()
    else:
        await stream_locally()


if __name__ == "__main__":
    asyncio.run(main())
