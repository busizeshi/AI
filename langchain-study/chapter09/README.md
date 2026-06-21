# Chapter 09 - Callbacks

本目录对应 `docs/09-langchain-callbacks.md`。

本章只保留一个轻量示例：

- `callbacks_demo.py`：自定义回调、多回调、流式 token、文件日志、简单 token 估算。

运行前设置：

```powershell
$env:K_CODEX = "你的 API Key"
# 可选
$env:K_CODEX_MODEL = "你的模型名"
```

运行：

```powershell
python .\langchain-study\chapter09\callbacks_demo.py
```

日志会写入：

```text
langchain-study/chapter09/output.log
```

