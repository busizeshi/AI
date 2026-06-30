"""
LangSmith tracing template.

Install and configure before running:

PowerShell:
    pip install -U langsmith
    $env:LANGCHAIN_TRACING_V2 = "true"
    $env:LANGCHAIN_API_KEY = "<your-langsmith-api-key>"
    $env:K_CODEX = "<your-codex-api-key>"

This file makes a real call to your OpenAI-compatible LLM API and traces it
with LangSmith after LangSmith is configured.
"""

import sys

from llm_client import call_responses

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

try:
    from langsmith import traceable
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: langsmith. Install it with `pip install -U langsmith`."
    ) from exc


@traceable
def pipeline(question: str) -> str:
    return call_responses(
        f"请用 80 字以内回答：{question}",
        temperature=0.3,
    )


def main():
    print(pipeline("LCEL 如何和 LangSmith 集成？"))


if __name__ == "__main__":
    main()
