import sys

from react_runtime import SimpleReActAgent, print_result


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    agent = SimpleReActAgent(max_steps=4)

    print("== 信息检索 ==")
    print_result(agent.run("请查询 LCEL 是什么，并给出一句话说明。"))

    print("\n== 数学计算 ==")
    print_result(agent.run("请计算 17 * 23，并说明计算结果。"))


if __name__ == "__main__":
    main()

