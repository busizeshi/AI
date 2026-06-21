# Chapter 08 - Agents

本目录对应 `docs/08-langchain-agents.md`。

示例统一使用你自建的 OpenAI 兼容接口：

- `base_url`: `https://codex.ysaikeji.cn/v1`
- `wire_api`: `responses`
- `api_key`: 从环境变量 `K_CODEX` 读取
- `model`: 默认 `gpt-5.4`，可用环境变量 `K_CODEX_MODEL` 覆盖

运行前设置：

```powershell
$env:K_CODEX = "你的 API Key"
# 可选：$env:K_CODEX_MODEL = "你的模型名"
```

运行示例：

```powershell
python .\langchain-study\chapter08\01_agent_basics.py
python .\langchain-study\chapter08\02_custom_llm_agent.py
python .\langchain-study\chapter08\03_react_agent.py
python .\langchain-study\chapter08\04_zero_shot_react.py
python .\langchain-study\chapter08\05_structured_react.py
python .\langchain-study\chapter08\06_docstore_agent.py
```

为了专注理解 Agent 原理，本章工具使用本地知识库、受限计算器和本地文档库模拟。后续可以替换成 SerpAPI、浏览器工具、向量数据库或企业知识库。

