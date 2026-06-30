import re
from dataclasses import dataclass

from llm_client import call_responses
from tools import TOOLS, Tool


@dataclass
class AgentStep:
    thought: str
    action: str | None
    action_input: str | None
    observation: str | None


class SimpleReActAgent:
    def __init__(self, tools: dict[str, Tool] | None = None, max_steps: int = 4):
        self.tools = tools or TOOLS
        self.max_steps = max_steps

    def run(self, question: str) -> dict:
        scratchpad = ""
        steps: list[AgentStep] = []

        for _ in range(self.max_steps):
            response = call_responses(
                self._build_prompt(question, scratchpad),
                temperature=0,
            )
            final = self._parse_final(response)
            if final:
                return {"answer": final, "steps": steps}

            thought, action, action_input = self._parse_action(response)
            if not action or not action_input:
                return {"answer": response.strip(), "steps": steps}

            tool = self.tools.get(action)
            if not tool:
                observation = f"未知工具：{action}"
            else:
                observation = tool.func(action_input)

            steps.append(AgentStep(thought, action, action_input, observation))
            scratchpad += (
                f"\nThought: {thought}\n"
                f"Action: {action}\n"
                f"Action Input: {action_input}\n"
                f"Observation: {observation}\n"
            )

        final_response = call_responses(
            f"请根据以下中间过程回答用户问题。\n问题：{question}\n过程：{scratchpad}",
            temperature=0.2,
        )
        return {"answer": final_response, "steps": steps}

    def _build_prompt(self, question: str, scratchpad: str) -> str:
        tool_descriptions = "\n".join(
            f"- {tool.name}: {tool.description}" for tool in self.tools.values()
        )
        return f"""
你是一个 ReAct Agent。你可以使用这些工具：
{tool_descriptions}

请严格使用以下两种格式之一。

如果需要调用工具：
Thought: 你的简短思考
Action: 工具名
Action Input: 工具输入

如果可以回答：
Final Answer: 最终答案

用户问题：{question}
已有过程：{scratchpad or "无"}
""".strip()

    @staticmethod
    def _parse_final(text: str) -> str | None:
        match = re.search(r"Final Answer\s*[:：]\s*(.*)", text, flags=re.S)
        if match:
            return match.group(1).strip()
        return None

    @staticmethod
    def _parse_action(text: str) -> tuple[str, str | None, str | None]:
        thought_match = re.search(r"Thought\s*[:：]\s*(.*)", text)
        action_match = re.search(r"Action\s*[:：]\s*([A-Za-z_][\w-]*)", text)
        input_match = re.search(r"Action Input\s*[:：]\s*(.*)", text, flags=re.S)

        thought = thought_match.group(1).strip() if thought_match else text.strip()
        action = action_match.group(1).strip() if action_match else None
        action_input = input_match.group(1).strip() if input_match else None
        return thought, action, action_input


def print_result(result: dict) -> None:
    for index, step in enumerate(result["steps"], 1):
        print(f"[Step {index}]")
        print(f"Thought: {step.thought}")
        print(f"Action: {step.action}")
        print(f"Action Input: {step.action_input}")
        print(f"Observation: {step.observation}")
    print("\nFinal Answer:")
    print(result["answer"])

