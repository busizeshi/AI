import sys

from react_runtime import SimpleReActAgent, print_result


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    agent = SimpleReActAgent(max_steps=4)
    questions = [
        "东京当前天气如何？",
        "请问北京的当前空气质量指数是多少？",
        "请计算 (12 + 8) * 5。",
    ]
    for question in questions:
        print(f"\n== {question} ==")
        print_result(agent.run(question))


if __name__ == "__main__":
    main()

