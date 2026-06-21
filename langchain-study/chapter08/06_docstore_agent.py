import sys

from react_runtime import SimpleReActAgent, print_result
from tools import TOOLS


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    docstore_tools = {
        "search": TOOLS["search"],
        "lookup": TOOLS["lookup"],
    }
    agent = SimpleReActAgent(tools=docstore_tools, max_steps=4)
    result = agent.run("请通过文档存储库查找：莎士比亚的第一部作品是什么？")
    print_result(result)


if __name__ == "__main__":
    main()

