# Chapter 07 - LCEL

本目录对应 `docs/07-langchain-expression-language.md`。

这些示例使用你自建的 OpenAI 兼容接口，并统一走 Responses API。

接口配置集中在 `llm_client.py`：

- `base_url`: `https://codex.ysaikeji.cn/v1`
- `wire_api`: `responses`
- `api_key`: 从环境变量 `K_CODEX` 读取
- `model`: 默认 `gpt-5.5`，可用环境变量 `K_CODEX_MODEL` 覆盖

运行前先设置 API Key：

```powershell
$env:K_CODEX = "你的 API Key"
# 可选：$env:K_CODEX_MODEL = "你的模型名"
```

运行方式：

```powershell
python .\langchain-study\chapter07\01_lcel_basic.py
python .\langchain-study\chapter07\02_streaming.py
python .\langchain-study\chapter07\03_parallel.py
python .\langchain-study\chapter07\04_fallbacks.py
python .\langchain-study\chapter07\05_langsmith_template.py
```

`05_langsmith_template.py` 会真实调用你的 LLM；如果还要把调用记录上报到 LangSmith，需要额外配置 `LANGCHAIN_API_KEY`。
