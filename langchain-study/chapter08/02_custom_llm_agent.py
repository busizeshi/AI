import sys

from react_runtime import SimpleReActAgent, print_result


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    agent = SimpleReActAgent(max_steps=3)
    result = agent.run("请先检索 ReAct 是什么，再用一句话解释它适合什么任务。")
    print_result(result)


if __name__ == "__main__":
    main()

